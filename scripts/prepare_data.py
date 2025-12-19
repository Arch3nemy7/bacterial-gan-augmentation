"""
Prepare bacterial dataset for training.

This script:
1. Optionally organizes DeepDataSet images by Gram type
2. Resizes/crops images to target size
3. Applies traditional augmentation (rotations, flips)
4. Splits data into train/val/test sets (70/15/15)
5. Saves processed images to data/02_processed/

Usage:
    poetry run python scripts/prepare_data.py [--organize] [--skip-organize]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.data.data_processing import create_data_splits


def organize_deepdataset(dataset_dir: Path, output_dir: Path) -> dict:
    """
    Organize DeepDataSet images by Gram type based on JSON annotations.

    Label mapping:
        G+ → gram_positive
        G  → gram_negative
    """
    images_dir = dataset_dir / "images"
    json_dir = dataset_dir / "json"

    if not images_dir.exists() or not json_dir.exists():
        return None

    gram_positive_dir = output_dir / "gram_positive"
    gram_negative_dir = output_dir / "gram_negative"
    gram_positive_dir.mkdir(parents=True, exist_ok=True)
    gram_negative_dir.mkdir(parents=True, exist_ok=True)

    stats = {"gram_positive": 0, "gram_negative": 0, "skipped": 0}
    json_files = sorted(json_dir.glob("*.json"))

    print(f"\n📁 Organizing {len(json_files)} images from DeepDataSet...")

    for idx, json_path in enumerate(json_files, 1):
        image_name = json_path.stem + ".jpg"
        image_path = images_dir / image_name

        if not image_path.exists():
            for ext in [".png", ".jpeg", ".JPG"]:
                alt = images_dir / (json_path.stem + ext)
                if alt.exists():
                    image_path = alt
                    break

        if not image_path.exists():
            stats["skipped"] += 1
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)

            labels = set()
            for shape in data.get("shapes", []):
                labels.add(shape.get("label", "").upper())

            if "G+" in labels:
                dest = gram_positive_dir / image_path.name
                stats["gram_positive"] += 1
            elif "G" in labels:
                dest = gram_negative_dir / image_path.name
                stats["gram_negative"] += 1
            else:
                stats["skipped"] += 1
                continue

            shutil.copy2(image_path, dest)

        except Exception:
            stats["skipped"] += 1

        if idx % 1000 == 0:
            print(f"  [{idx}/{len(json_files)}] Processing...")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare bacterial dataset")
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Force organize DeepDataSet before preprocessing",
    )
    parser.add_argument(
        "--skip-organize",
        action="store_true",
        help="Skip DeepDataSet organization",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BACTERIAL DATASET PREPROCESSING")
    print("=" * 80)

    settings = get_settings("configs/config.yaml")
    raw_path = Path(settings.data.raw_data_dir)
    gram_positive = raw_path / "gram_positive"
    gram_negative = raw_path / "gram_negative"

    has_organized = gram_positive.exists() and gram_negative.exists()
    has_deepdataset = (raw_path / "DeepDataSet" / "640DataSet").exists()

    if has_deepdataset and (args.organize or not has_organized):
        if not args.skip_organize:
            print("\n📦 DeepDataSet detected - organizing by Gram type...")
            stats = organize_deepdataset(
                dataset_dir=raw_path / "DeepDataSet" / "640DataSet",
                output_dir=raw_path,
            )
            if stats:
                print(f"  ✅ Gram-positive: {stats['gram_positive']}")
                print(f"  ✅ Gram-negative: {stats['gram_negative']}")
                print(f"  ⚠️  Skipped: {stats['skipped']}")
            else:
                print("  ⚠️  Could not organize DeepDataSet")

    gram_positive = raw_path / "gram_positive"
    gram_negative = raw_path / "gram_negative"

    if not gram_positive.exists() or not gram_negative.exists():
        print(f"\n❌ Raw data directories not found!")
        print(f"   Expected: {gram_positive}")
        print(f"            {gram_negative}")
        print(f"\n💡 Run with --organize to organize DeepDataSet")
        return

    pos_images = []
    neg_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        pos_images.extend(list(gram_positive.glob(ext)))
        neg_images.extend(list(gram_negative.glob(ext)))

    print(f"\nConfiguration:")
    print(f"  Raw data: {settings.data.raw_data_dir}")
    print(f"  Processed: {settings.data.processed_data_dir}")
    if settings.preprocessing.use_patch_extraction:
        print("  Mode: Patch Extraction")
    else:
        print(f"  Mode: {settings.preprocessing.preprocess_mode.title()}")
    print(f"  Size: {settings.preprocessing.image_size}x{settings.preprocessing.image_size}")
    print(f"  Augmentation: {'8x' if settings.preprocessing.apply_augmentation else 'Off'}")

    print(f"\n📊 Dataset:")
    print(f"  Gram-positive: {len(pos_images)}")
    print(f"  Gram-negative: {len(neg_images)}")
    print(f"  Total: {len(pos_images) + len(neg_images)}")

    processed_path = Path(settings.data.processed_data_dir)
    if processed_path.exists():
        print(f"\n🧹 Cleaning {processed_path}...")
        shutil.rmtree(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    train_ratio = settings.preprocessing.train_ratio
    val_ratio = settings.preprocessing.val_ratio
    test_ratio = settings.preprocessing.test_ratio

    print(f"\n📂 Split: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%")

    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)

    try:
        create_data_splits(
            data_config=settings.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=settings.preprocessing.random_seed,
            image_size=settings.preprocessing.image_size,
            apply_augmentation=settings.preprocessing.apply_augmentation,
            bg_threshold=settings.preprocessing.bg_threshold,
            use_patch_extraction=settings.preprocessing.use_patch_extraction,
            preprocess_mode=settings.preprocessing.preprocess_mode,
            max_patches_per_split=settings.preprocessing.max_patches_per_split,
        )

        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)

        for split in ["train", "val", "test"]:
            pos = len(list((processed_path / split / "gram_positive").glob("*.png")))
            neg = len(list((processed_path / split / "gram_negative").glob("*.png")))
            print(f"\n{split.upper()}: {pos} pos, {neg} neg, {pos+neg} total")

        print("\n📝 Next: bacterial-gan train")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
