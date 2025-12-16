"""
Prepare bacterial dataset for training.

This script:
1. Extracts patches from high-resolution images
2. Applies traditional augmentation (rotations, flips)
3. Splits data into train/val/test sets (70/15/15)
4. Saves processed images to data/02_processed/
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.data.data_processing import create_data_splits


def main():
    """Prepare and split the bacterial dataset."""
    print("=" * 80)
    print("BACTERIAL DATASET PREPROCESSING")
    print("=" * 80)

    config_path = Path("configs/config.yaml")
    settings = get_settings(str(config_path))

    print(f"\nConfiguration:")
    print(f"  Raw data: {settings.data.raw_data_dir}")
    print(f"  Processed data: {settings.data.processed_data_dir}")
    print(
        f"  Mode: {'Patch Extraction' if settings.preprocessing.use_patch_extraction else 'Random Crop'}"
    )
    print(f"  Target size: {settings.preprocessing.image_size}x{settings.preprocessing.image_size}")
    print(f"  Augmentation: {'8x' if settings.preprocessing.apply_augmentation else 'Disabled'}")
    print(f"  Background threshold: >{settings.preprocessing.bg_threshold*100:.0f}%")

    raw_path = Path(settings.data.raw_data_dir)
    gram_positive = raw_path / "gram_positive"
    gram_negative = raw_path / "gram_negative"

    if not gram_positive.exists() or not gram_negative.exists():
        print(f"\n❌ Raw data directories not found!")
        print(f"   Expected: {gram_positive} and {gram_negative}")
        return

    processed_path = Path(settings.data.processed_data_dir)
    if processed_path.exists():
        print(f"\n🧹 Cleaning {processed_path}...")
        shutil.rmtree(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    pos_images = []
    neg_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.JPG", "*.PNG"]:
        pos_images.extend(list(gram_positive.glob(ext)))
        neg_images.extend(list(gram_negative.glob(ext)))

    total = len(pos_images) + len(neg_images)

    print(f"\n📊 Dataset:")
    print(f"  Gram-positive: {len(pos_images)}")
    print(f"  Gram-negative: {len(neg_images)}")
    print(f"  Total: {total}")

    train_ratio = settings.preprocessing.train_ratio
    val_ratio = settings.preprocessing.val_ratio
    test_ratio = settings.preprocessing.test_ratio

    print(f"\n📂 Split: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%")

    print(f"\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)

    try:
        create_data_splits(
            data_config=settings.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=settings.preprocessing.random_seed,
            patch_size=settings.preprocessing.image_size,
            apply_augmentation=settings.preprocessing.apply_augmentation,
            bg_threshold=settings.preprocessing.bg_threshold,
            use_patch_extraction=settings.preprocessing.use_patch_extraction,
            crop_mode=settings.preprocessing.crop_mode,
            max_patches_per_split=settings.preprocessing.max_patches_per_split,
        )

        print(f"\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)

        for split in ["train", "val", "test"]:
            pos = len(list((processed_path / split / "gram_positive").glob("*.png")))
            neg = len(list((processed_path / split / "gram_negative").glob("*.png")))
            print(f"\n{split.upper()}: {pos} pos, {neg} neg, {pos+neg} total")

        print(f"\n📝 Next: bacterial-gan train")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
