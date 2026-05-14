"""
app.py — Streamlit Web Application for Skin Cancer Detection

A professional-grade dermatoscopy image classifier powered by
EfficientNetB0 transfer learning on the HAM10000 dataset.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DermaScan AI — Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224
MODEL_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
CONFUSION_MATRIX_PATH = Path("models/confusion_matrix.png")
TRAINING_HISTORY_PATH = Path("models/training_history.png")

CLASS_DETAILS = {
    "akiec": {
        "full_name": "Actinic Keratoses & Intraepithelial Carcinoma",
        "risk": "⚠️ Pre-cancerous / Cancerous",
        "color": "#FF6B6B",
        "description": (
            "Actinic keratoses are rough, scaly patches caused by years of sun exposure. "
            "They are considered pre-cancerous and can progress to squamous cell carcinoma "
            "if left untreated. Intraepithelial carcinoma (Bowen's disease) is an early form "
            "of squamous cell carcinoma confined to the epidermis."
        ),
        "action": "Consult a dermatologist promptly for evaluation and treatment options.",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk": "🔴 Cancerous",
        "color": "#FF4444",
        "description": (
            "Basal cell carcinoma (BCC) is the most common form of skin cancer. It arises from "
            "the basal cells in the deepest layer of the epidermis. BCCs rarely metastasize but "
            "can cause significant local tissue destruction if untreated. They often appear as "
            "pearly or waxy bumps, or flat, flesh-colored lesions."
        ),
        "action": "Seek immediate dermatological consultation. Treatment is highly effective when caught early.",
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "risk": "✅ Benign",
        "color": "#4CAF50",
        "description": (
            "This category includes solar lentigines (age spots), seborrheic keratoses, and "
            "lichen-planus-like keratoses. These are non-cancerous growths that are common in "
            "older adults. While benign, they can sometimes mimic melanoma in appearance."
        ),
        "action": "Generally no treatment needed. Monitor for changes in size, shape, or color.",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk": "✅ Benign",
        "color": "#4CAF50",
        "description": (
            "Dermatofibromas are common, harmless skin growths that usually appear as firm, "
            "small, raised bumps. They are most often found on the legs and can feel like a "
            "hard lump under the skin. They may be brownish or reddish in color."
        ),
        "action": "No treatment typically required. Removal is optional and cosmetic.",
    },
    "mel": {
        "full_name": "Melanoma",
        "risk": "🔴 Cancerous (High Risk)",
        "color": "#D32F2F",
        "description": (
            "Melanoma is the most dangerous form of skin cancer. It develops from melanocytes, "
            "the cells that give skin its color. Melanoma can spread (metastasize) to other parts "
            "of the body if not detected and treated early. Look for the ABCDE signs: Asymmetry, "
            "Border irregularity, Color variation, Diameter >6mm, and Evolving appearance."
        ),
        "action": "⚡ URGENT: Seek immediate medical evaluation. Early detection saves lives.",
    },
    "nv": {
        "full_name": "Melanocytic Nevi (Moles)",
        "risk": "✅ Benign",
        "color": "#4CAF50",
        "description": (
            "Melanocytic nevi (moles) are benign neoplasms composed of melanocytes. They are "
            "extremely common and most adults have 10-40 moles. Most moles are harmless, but "
            "atypical moles (dysplastic nevi) may have a slightly increased risk of developing "
            "into melanoma."
        ),
        "action": "Monitor regularly using ABCDE criteria. Annual skin checks recommended.",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "risk": "✅ Benign",
        "color": "#4CAF50",
        "description": (
            "Vascular lesions include cherry angiomas, angiokeratomas, pyogenic granulomas, "
            "and hemorrhage. These are non-cancerous growths involving blood vessels. They can "
            "appear as red, purple, or blue spots or bumps on the skin."
        ),
        "action": "Generally benign. Treatment available for cosmetic reasons if desired.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

def inject_css():
    """Inject custom CSS for premium styling."""
    st.markdown("""
    <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ── Hero Section ── */
        .hero-container {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid rgba(79, 195, 247, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #4FC3F7, #81D4FA, #B3E5FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #B0BEC5;
            font-weight: 300;
            margin-bottom: 0;
        }

        /* ── Prediction Card ── */
        .prediction-card {
            background: linear-gradient(145deg, #1a1d23, #22252c);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .prediction-label {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .prediction-confidence {
            font-size: 1rem;
            color: #90A4AE;
        }

        /* ── Risk Badge ── */
        .risk-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }

        .risk-high {
            background: rgba(211, 47, 47, 0.2);
            color: #EF5350;
            border: 1px solid rgba(211, 47, 47, 0.3);
        }

        .risk-precancer {
            background: rgba(255, 107, 107, 0.2);
            color: #FF6B6B;
            border: 1px solid rgba(255, 107, 107, 0.3);
        }

        .risk-benign {
            background: rgba(76, 175, 80, 0.2);
            color: #66BB6A;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        /* ── Info Cards ── */
        .info-card {
            background: #1a1d23;
            border-radius: 10px;
            padding: 1.2rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* ── Disclaimer ── */
        .disclaimer {
            background: rgba(255, 152, 0, 0.1);
            border: 1px solid rgba(255, 152, 0, 0.3);
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            color: #FFB74D;
            font-size: 0.9rem;
        }

        /* ── Sidebar ── */
        .sidebar-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #4FC3F7;
            margin-bottom: 0.5rem;
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            color: #546E7A;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* ── Progress bars ── */
        .confidence-bar-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin: 0.3rem 0;
            overflow: hidden;
        }

        .confidence-bar {
            height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            font-size: 0.8rem;
            font-weight: 500;
            color: white;
            transition: width 0.5s ease;
        }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

MODEL_URL = (
    "https://github.com/nvimik0/Skin_Cancer_Detection_Project"
    "/raw/main/models/best_model.keras"
)


def _ensure_model_available():
    """Download the model from GitHub if it's missing or is an LFS pointer."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist or is a tiny LFS pointer (<1 MB), download it
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return  # Real model already present

    import urllib.request
    st.info("⬇️ Downloading model (≈50 MB) — this only happens once…")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")


@st.cache_resource
def load_model():
    """Load the trained Keras model (cached across sessions)."""
    import tensorflow as tf
    _ensure_model_available()
    if not MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_class_names():
    """Load ordered class names from JSON."""
    if not CLASS_NAMES_PATH.exists():
        return list(CLASS_DETAILS.keys())
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array.astype("float32")


def predict(model, image: Image.Image, class_names: list[str]):
    """Run prediction and return sorted results."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]

    results = []
    for i, prob in enumerate(predictions):
        results.append({
            "class": class_names[i],
            "probability": float(prob),
            "details": CLASS_DETAILS.get(class_names[i], {}),
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────────────────────────────────────

def render_hero():
    """Render the hero/header section."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔬 DermaScan AI</div>
        <div class="hero-subtitle">
            Advanced Skin Lesion Classification powered by EfficientNetB0 Deep Learning
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render medical disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for skin concerns.
    </div>
    """, unsafe_allow_html=True)


def get_risk_class(risk_text: str) -> str:
    """Map risk text to CSS class."""
    if "High Risk" in risk_text or "Cancerous" in risk_text:
        return "risk-high"
    elif "Pre-cancerous" in risk_text:
        return "risk-precancer"
    return "risk-benign"


def render_prediction_results(results: list):
    """Render prediction results with confidence bars."""
    top = results[0]
    details = top["details"]
    confidence = top["probability"] * 100

    # Main prediction card
    risk_class = get_risk_class(details.get("risk", ""))
    border_color = details.get("color", "#4FC3F7")

    st.markdown(f"""
    <div class="prediction-card" style="border-left-color: {border_color};">
        <div class="prediction-label" style="color: {border_color};">
            {details.get('full_name', top['class'])}
        </div>
        <div class="prediction-confidence">
            Confidence: {confidence:.1f}%
        </div>
        <span class="risk-badge {risk_class}">
            {details.get('risk', 'Unknown')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Description
    st.markdown(f"""
    <div class="info-card">
        <p style="color: #CFD8DC; line-height: 1.6;">{details.get('description', '')}</p>
        <p style="color: #4FC3F7; font-weight: 500; margin-bottom: 0;">
            🩺 {details.get('action', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars for all classes
    st.markdown("#### 📊 All Class Probabilities")
    for result in results:
        prob = result["probability"] * 100
        color = result["details"].get("color", "#4FC3F7")
        name = result["class"]

        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 4px 0;">
            <span style="width: 60px; font-size: 0.85rem; color: #90A4AE; font-weight: 500;">{name}</span>
            <div class="confidence-bar-container" style="flex: 1;">
                <div class="confidence-bar"
                     style="width: {max(prob, 2)}%; background: linear-gradient(90deg, {color}, {color}88);">
                    {prob:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with project info."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔬 About DermaScan AI</div>', unsafe_allow_html=True)
        st.markdown("""
        This application uses a deep learning model trained on the
        **HAM10000** dataset to classify dermoscopic images into
        **7 categories** of skin lesions.
        """)

        st.divider()

        st.markdown('<div class="sidebar-header">🧠 Model Details</div>', unsafe_allow_html=True)
        st.markdown("""
        | Feature | Value |
        |---------|-------|
        | Architecture | EfficientNetB0 |
        | Input Size | 224 × 224 px |
        | Classes | 7 |
        | Transfer Learning | ImageNet |
        | Framework | TensorFlow / Keras |
        """)

        st.divider()

        st.markdown('<div class="sidebar-header">📋 Supported Classes</div>', unsafe_allow_html=True)
        for code, info in CLASS_DETAILS.items():
            risk_emoji = "🟢" if "Benign" in info["risk"] else "🔴"
            st.markdown(f"{risk_emoji} **{code}** — {info['full_name']}")

        st.divider()

        st.markdown('<div class="sidebar-header">📷 Tips for Best Results</div>', unsafe_allow_html=True)
        st.markdown("""
        - Use **dermoscopic** images when possible
        - Ensure good **lighting** and **focus**
        - Crop to show primarily the **lesion**
        - Image formats: **JPG, PNG, JPEG**
        """)

        st.divider()
        st.caption("Built with ❤️ using Streamlit & TensorFlow")


def render_model_performance():
    """Render model performance section with saved charts."""
    st.markdown("---")
    st.markdown("### 📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        if CONFUSION_MATRIX_PATH.exists():
            st.image(
                str(CONFUSION_MATRIX_PATH),
                caption="Confusion Matrix",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix will appear after model training.")

    with col2:
        if TRAINING_HISTORY_PATH.exists():
            st.image(
                str(TRAINING_HISTORY_PATH),
                caption="Training History (Accuracy & Loss)",
                use_container_width=True,
            )
        else:
            st.info("Training history will appear after model training.")

    # Classification report
    report_path = Path("models/classification_report.txt")
    if report_path.exists():
        with open(report_path) as f:
            report = f.read()
        with st.expander("📋 Full Classification Report"):
            st.code(report, language="text")


# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    render_sidebar()
    render_hero()
    render_disclaimer()

    # ── Load model ──
    model = load_model()
    class_names = load_class_names()

    if model is None:
        st.warning(
            "⚠️ **Model not found!** Please train the model first:\n\n"
            "```bash\npython src/train.py\n```\n\n"
            "The model file should be saved at `models/best_model.keras`."
        )
        st.stop()

    # ── Main Content ──
    st.markdown("### 📤 Upload a Dermoscopic Image")

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the skin lesion. Dermoscopic images work best.",
            key="skin_image_uploader",
        )

        if uploaded_file is not None:
            st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    with col_preview:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── Run Prediction ──
    if uploaded_file is not None:
        st.markdown("---")

        with st.spinner("🔍 Analyzing image with EfficientNetB0..."):
            results = predict(model, image, class_names)

        st.markdown("### 🎯 Prediction Results")
        render_prediction_results(results)

    # ── Model Performance Section ──
    render_model_performance()

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        <p>DermaScan AI — Skin Cancer Detection using Deep Learning</p>
        <p>Powered by EfficientNetB0 · Trained on HAM10000 · Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
