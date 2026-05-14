"""
prepare_data.py — HAM10000 Dataset Organizer

Reads the locally downloaded HAM10000 dataset, organizes images into
class-specific subfolders, and splits into train/val sets.

Usage:
    python src/prepare_data.py --raw_dir <path_to_raw_images> --metadata <path_to_csv>

Example:
    python src/prepare_data.py \
        --raw_dir "C:/Users/YourName/Downloads/HAM10000" \
        --metadata "C:/Users/YourName/Downloads/HAM10000_metadata.csv"
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    "akiec": "Actinic Keratoses (akiec)",
    "bcc":   "Basal Cell Carcinoma (bcc)",
    "bkl":   "Benign Keratosis (bkl)",
    "df":    "Dermatofibroma (df)",
    "mel":   "Melanoma (mel)",
    "nv":    "Melanocytic Nevi (nv)",
    "vasc":  "Vascular Lesions (vasc)",
}

SPLIT_RATIO = (0.8, 0.2)  # train, val


def find_image(image_id: str, search_dirs: list[Path]) -> Path | None:
    """Search for an image file across multiple directories."""
    for directory in search_dirs:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = directory / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def organize_by_class(
    metadata_path: Path,
    raw_dirs: list[Path],
    organized_dir: Path,
) -> dict[str, int]:
    """
    Read metadata CSV and copy images into class-specific subfolders.

    Args:
        metadata_path: Path to HAM10000_metadata.csv
        raw_dirs:      List of directories containing raw images
        organized_dir: Output directory (e.g., data/organized/)

    Returns:
        Dictionary of class_name → count
    """
    print(f"\n📂 Reading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Validate expected columns
    required_cols = {"image_id", "dx"}
    if not required_cols.issubset(df.columns):
        print(f"❌ Metadata CSV must contain columns: {required_cols}")
        print(f"   Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"   Found {len(df)} entries across {df['dx'].nunique()} classes")

    # Create class subdirectories
    for class_name in df["dx"].unique():
        (organized_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images into class folders
    counts: dict[str, int] = Counter()
    skipped = 0

    for idx, row in df.iterrows():
        image_id = row["image_id"]
        dx = row["dx"]

        src = find_image(image_id, raw_dirs)
        if src is None:
            skipped += 1
            continue

        dst = organized_dir / dx / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[dx] += 1

        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{len(df)} images...")

    print(f"\n✅ Organized {sum(counts.values())} images into {organized_dir}")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} images (not found in raw directories)")

    return dict(counts)


def split_dataset(organized_dir: Path, output_dir: Path, ratio: tuple) -> None:
    """
    Split organized dataset into train/val using splitfolders.

    Args:
        organized_dir: Directory with class subfolders
        output_dir:    Output directory (will contain train/ and val/)
        ratio:         Tuple of (train_ratio, val_ratio)
    """
    try:
        import splitfolders
    except ImportError:
        print("❌ Please install split-folders: pip install split-folders")
        sys.exit(1)

    print(f"\n✂️  Splitting dataset with ratio train={ratio[0]}, val={ratio[1]}...")
    splitfolders.ratio(
        str(organized_dir),
        output=str(output_dir),
        seed=42,
        ratio=ratio,
        group_prefix=None,
        move=False,  # Copy, don't move (keep organized dir intact)
    )
    print(f"✅ Split complete → {output_dir}")


def print_statistics(data_dir: Path) -> None:
    """Print class distribution for train and val sets."""
    print("\n" + "=" * 60)
    print("📊 DATASET STATISTICS")
    print("=" * 60)

    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        print(f"\n{'─' * 40}")
        print(f"  {split.upper()} SET")
        print(f"{'─' * 40}")

        total = 0
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                class_counts[class_dir.name] = count
                total += count

        for class_name, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(count / max(class_counts.values()) * 30)
            pct = count / total * 100
            full_name = CLASS_NAMES.get(class_name, class_name)
            print(f"  {full_name:<35} {count:>5}  ({pct:5.1f}%)  {bar}")

        print(f"  {'TOTAL':<35} {total:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize HAM10000 dataset into train/val splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # If images are in one folder:
  python src/prepare_data.py --raw_dir "C:/Downloads/HAM10000_images" --metadata "C:/Downloads/HAM10000_metadata.csv"

  # If images are in two folders (part1 + part2):
  python src/prepare_data.py --raw_dir "C:/Downloads/HAM10000_images_part_1" "C:/Downloads/HAM10000_images_part_2" --metadata "C:/Downloads/HAM10000_metadata.csv"
        """,
    )

    parser.add_argument(
        "--raw_dir",
        nargs="+",
        required=True,
        help="Path(s) to directory(ies) containing raw HAM10000 images. "
             "Provide multiple paths if images are split across folders.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to HAM10000_metadata.csv",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory for train/val splits (default: data/)",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    raw_dirs = [Path(d).resolve() for d in args.raw_dir]
    metadata_path = Path(args.metadata).resolve()
    output_dir = (project_root / args.output).resolve()
    organized_dir = output_dir / "_organized"

    # Validate inputs
    for d in raw_dirs:
        if not d.exists():
            print(f"❌ Raw image directory not found: {d}")
            sys.exit(1)

    if not metadata_path.exists():
        print(f"❌ Metadata CSV not found: {metadata_path}")
        sys.exit(1)

    print("=" * 60)
    print("🔬 HAM10000 Dataset Preparation")
    print("=" * 60)
    print(f"  Raw image dirs : {[str(d) for d in raw_dirs]}")
    print(f"  Metadata CSV   : {metadata_path}")
    print(f"  Output dir     : {output_dir}")

    # Step 1: Organize by class
    counts = organize_by_class(metadata_path, raw_dirs, organized_dir)

    # Step 2: Split into train/val
    split_dataset(organized_dir, output_dir, SPLIT_RATIO)

    # Step 3: Print statistics
    print_statistics(output_dir)

    # Step 4: Cleanup organized temp dir
    if organized_dir.exists():
        shutil.rmtree(organized_dir)
        print(f"\n🧹 Cleaned up temporary directory: {organized_dir}")

    print("\n✅ Data preparation complete!")
    print(f"   Train: {output_dir / 'train'}")
    print(f"   Val:   {output_dir / 'val'}")
    print("\n💡 Next step: python src/train.py")


if __name__ == "__main__":
    main()
