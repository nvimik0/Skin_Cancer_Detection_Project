"""
train.py — Production Training Script for Skin Cancer Classification

Two-phase transfer learning with EfficientNetB0:
  Phase 1: Feature extraction (frozen base, train custom head)
  Phase 2: Fine-tuning (unfreeze top layers, low learning rate)

Handles class imbalance via computed class weights + data augmentation.

Usage:
    python src/train.py
    python src/train.py --epochs 30 --batch_size 16 --data_dir data
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224           # EfficientNetB0 native input size
AUTOTUNE = tf.data.AUTOTUNE

CLASS_INFO = {
    "akiec": "Actinic Keratoses & Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Augmentation Layer
# ──────────────────────────────────────────────────────────────────────────────

def build_augmentation_layer():
    """
    Build a Keras augmentation pipeline.
    Applied only during training to artificially expand minority classes.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


# ──────────────────────────────────────────────────────────────────────────────
# Data Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def load_datasets(data_dir: Path, batch_size: int):
    """
    Load train and validation datasets using image_dataset_from_directory.

    Returns:
        train_ds:    Augmented + prefetched training dataset
        val_ds:      Prefetched validation dataset (no augmentation)
        class_names: Sorted list of class labels
        train_labels: All training labels (for computing class weights)
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        print("❌ Data directories not found!")
        print(f"   Expected: {train_dir}")
        print(f"   Expected: {val_dir}")
        print("\n💡 Run: python src/prepare_data.py --help")
        sys.exit(1)

    # Load datasets
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"\n📋 Classes found: {class_names}")

    # Extract all training labels for class weight computation
    train_labels = []
    for _, labels in train_ds.unbatch():
        train_labels.append(np.argmax(labels.numpy()))
    train_labels = np.array(train_labels)

    # Print class distribution
    print("\n📊 Training set class distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(train_labels == i)
        print(f"   {name}: {count} images")

    # Build augmentation layer
    augmentation = build_augmentation_layer()

    # Apply augmentation to training set only
    train_ds_aug = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # Optimize pipeline performance
    train_ds_aug = train_ds_aug.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds_aug, val_ds, class_names, train_labels


# ──────────────────────────────────────────────────────────────────────────────
# Class Weight Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_weights(train_labels: np.ndarray, class_names: list[str]) -> dict:
    """
    Compute balanced class weights to handle the severe imbalance.
    Uses sklearn's compute_class_weight with 'balanced' strategy.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )

    class_weight_dict = {i: w for i, w in enumerate(weights)}

    print("\n⚖️  Computed class weights (higher = rarer class):")
    for i, name in enumerate(class_names):
        print(f"   {name}: {class_weight_dict[i]:.3f}")

    return class_weight_dict


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    """
    Build EfficientNetB0-based classifier with custom head.

    Architecture:
        Input (224x224x3)
        → EfficientNetB0 (frozen, ImageNet weights)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu)
        → Dropout(0.5)
        → Dense(128, relu)
        → Dropout(0.3)
        → Dense(num_classes, softmax)
    """
    # Load pretrained base
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze all layers in base model
    base_model.trainable = False

    # Build full model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # EfficientNetB0 includes its own preprocessing (scaling to [0,1])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="SkinCancer_EfficientNetB0")

    print("\n🏗️  Model Architecture:")
    model.summary(print_fn=lambda x: print(f"   {x}"))
    print(f"\n   Base model layers (frozen): {len(base_model.layers)}")
    print(f"   Total trainable params:     {model.count_params():,}")

    return model, base_model


# ──────────────────────────────────────────────────────────────────────────────
# Training Callbacks
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(models_dir: Path, phase: str) -> list:
    """Build callbacks for training."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(models_dir / f"best_model_{phase}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_phase1(
    model: keras.Model,
    train_ds,
    val_ds,
    class_weights: dict,
    models_dir: Path,
    epochs: int,
) -> keras.callbacks.History:
    """
    Phase 1: Feature Extraction
    Train only the custom head layers with frozen base model.
    """
    print("\n" + "=" * 60)
    print("🔒 PHASE 1: Feature Extraction (Frozen Base)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(models_dir, "phase1"),
        verbose=1,
    )

    return history


def train_phase2(
    model: keras.Model,
    base_model: keras.Model,
    train_ds,
    val_ds,
    class_weights: dict,
    models_dir: Path,
    epochs: int,
    fine_tune_from: int = 100,
) -> keras.callbacks.History:
    """
    Phase 2: Fine-Tuning
    Unfreeze top layers of base model and train with very low learning rate.
    """
    print("\n" + "=" * 60)
    print("🔓 PHASE 2: Fine-Tuning (Unfreeze Top Layers)")
    print("=" * 60)

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze layers up to `fine_tune_from`
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_layers = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Total base layers: {total_layers}")
    print(f"   Frozen layers:     {fine_tune_from}")
    print(f"   Trainable layers:  {trainable_layers}")

    # Recompile with much lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(models_dir, "phase2"),
        verbose=1,
    )

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation & Visualization
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    val_ds,
    class_names: list[str],
    models_dir: Path,
) -> None:
    """Generate classification report and confusion matrix."""
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION")
    print("=" * 60)

    # Get predictions
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\n📋 Classification Report:")
    print(report)

    # Save report
    report_path = models_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Skin Cancer Classification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(report)
    print(f"   Saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix — Skin Cancer Classification", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = models_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Confusion matrix saved to: {cm_path}")


def plot_training_history(
    history1: keras.callbacks.History,
    history2: keras.callbacks.History | None,
    models_dir: Path,
) -> None:
    """Plot training & validation accuracy/loss curves."""
    # Combine histories
    acc = history1.history["accuracy"]
    val_acc = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    phase1_epochs = len(acc)
    phase_boundary = phase1_epochs  # Vertical line between phases

    if history2 is not None:
        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — EfficientNetB0", fontsize=14, y=1.02)

    # Accuracy plot
    ax1.plot(epochs_range, acc, "b-", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    if history2 is not None:
        ax1.axvline(x=phase_boundary, color="gray", linestyle="--", alpha=0.7,
                     label="Fine-tuning starts")
    ax1.set_title("Accuracy", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs_range, loss, "b-", label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, "r-", label="Validation Loss", linewidth=2)
    if history2 is not None:
        ax2.axvline(x=phase_boundary, color="gray", linestyle="--", alpha=0.7,
                     label="Fine-tuning starts")
    ax2.set_title("Loss", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    history_path = models_dir / "training_history.png"
    fig.savefig(history_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📈 Training history saved to: {history_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 for skin cancer classification"
    )
    parser.add_argument("--data_dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--epochs_phase1", type=int, default=20, help="Epochs for Phase 1 (default: 20)")
    parser.add_argument("--epochs_phase2", type=int, default=15, help="Epochs for Phase 2 (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--fine_tune_from", type=int, default=100,
                        help="Freeze base layers up to this index (default: 100)")
    parser.add_argument("--skip_phase2", action="store_true", help="Skip fine-tuning phase")

    args = parser.parse_args()

    # ── Paths ──
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── GPU Check ──
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU detected: {gpus}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU detected — training will be slower on CPU")

    print("\n" + "=" * 60)
    print("🔬 Skin Cancer Detection — Model Training")
    print("=" * 60)
    print(f"  Data directory  : {data_dir}")
    print(f"  Models directory: {models_dir}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Phase 1 epochs  : {args.epochs_phase1}")
    print(f"  Phase 2 epochs  : {args.epochs_phase2 if not args.skip_phase2 else 'SKIPPED'}")
    print(f"  Fine-tune from  : layer {args.fine_tune_from}")

    # ── Load Data ──
    train_ds, val_ds, class_names, train_labels = load_datasets(
        data_dir, args.batch_size
    )

    # ── Save class names ──
    class_names_path = models_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"\n💾 Class names saved to: {class_names_path}")

    # ── Compute Class Weights ──
    class_weights = compute_weights(train_labels, class_names)

    # ── Build Model ──
    model, base_model = build_model(num_classes=len(class_names))

    # ── Phase 1: Feature Extraction ──
    history1 = train_phase1(
        model, train_ds, val_ds, class_weights, models_dir, args.epochs_phase1
    )

    # ── Phase 2: Fine-Tuning ──
    history2 = None
    if not args.skip_phase2:
        history2 = train_phase2(
            model, base_model, train_ds, val_ds, class_weights,
            models_dir, args.epochs_phase2, args.fine_tune_from,
        )

    # ── Save Final Model ──
    final_model_path = models_dir / "best_model.keras"
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")

    # ── Evaluate ──
    evaluate_model(model, val_ds, class_names, models_dir)

    # ── Plot Training History ──
    plot_training_history(history1, history2, models_dir)

    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)
    print(f"  Model:            {final_model_path}")
    print(f"  Confusion matrix: {models_dir / 'confusion_matrix.png'}")
    print(f"  Training curves:  {models_dir / 'training_history.png'}")
    print(f"  Class report:     {models_dir / 'classification_report.txt'}")
    print(f"\n💡 Next step: streamlit run app.py")


if __name__ == "__main__":
    main()
