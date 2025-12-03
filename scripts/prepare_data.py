"""
Prepare bacterial dataset for training.

This script:
1. Applies Macenko color normalization to images
2. Splits data into train/val/test sets (70/15/15)
3. Saves processed images to data/02_processed/
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.data.data_processing import create_data_splits


def main():
    """Prepare and split the bacterial dataset."""
    print("=" * 80)
    print("BACTERIAL DATASET PREPROCESSING")
    print("=" * 80)

    # Load configuration
    config_path = Path("configs/config.yaml")
    settings = get_settings(str(config_path))

    print(f"\nConfiguration:")
    print(f"  Raw data directory: {settings.data.raw_data_dir}")
    print(f"  Processed data directory: {settings.data.processed_data_dir}")
    print(f"  Image size: {settings.preprocessing.image_size}x{settings.preprocessing.image_size}")
    print(f"  Macenko normalization: {'Enabled' if settings.preprocessing.apply_macenko_normalization else 'Disabled'}")

    # Check if raw data exists
    raw_path = Path(settings.data.raw_data_dir)
    gram_positive = raw_path / "gram_positive"
    gram_negative = raw_path / "gram_negative"

    if not gram_positive.exists() or not gram_negative.exists():
        print(f"\n❌ Error: Raw data directories not found!")
        print(f"   Expected: {gram_positive} and {gram_negative}")
        print(f"\n   Please run: poetry run python scripts/organize_dataset.py")
        return

    # Count images
    pos_count = len(list(gram_positive.glob("*.jpg")))
    neg_count = len(list(gram_negative.glob("*.jpg")))
    total = pos_count + neg_count

    print(f"\n📊 Dataset Statistics:")
    print(f"  Gram-positive images: {pos_count}")
    print(f"  Gram-negative images: {neg_count}")
    print(f"  Total images: {total}")

    # Get split ratios from config
    train_ratio = settings.preprocessing.train_ratio
    val_ratio = settings.preprocessing.val_ratio
    test_ratio = settings.preprocessing.test_ratio

    print(f"\n📂 Data Split:")
    print(f"  Training: {train_ratio*100:.0f}% (~{int(total*train_ratio)} images)")
    print(f"  Validation: {val_ratio*100:.0f}% (~{int(total*val_ratio)} images)")
    print(f"  Test: {test_ratio*100:.0f}% (~{int(total*test_ratio)} images)")

    # Process and split data
    print(f"\n" + "=" * 80)
    print("PROCESSING AND SPLITTING DATA")
    print("=" * 80)
    print("\nThis will:")
    if settings.preprocessing.apply_macenko_normalization:
        print("  1. Apply Macenko color normalization")
        print(f"     - Background intensity (Io): {settings.preprocessing.macenko_io}")
        print(f"     - Alpha percentile: {settings.preprocessing.macenko_alpha}")
        print(f"     - OD threshold (beta): {settings.preprocessing.macenko_beta}")
    else:
        print("  1. Skip Macenko color normalization (disabled in config)")
    print(f"  2. Resize images to {settings.preprocessing.image_size}x{settings.preprocessing.image_size}")
    print("  3. Split into train/val/test sets")
    print("  4. Save to data/02_processed/")

    try:
        create_data_splits(
            data_config=settings.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=settings.preprocessing.random_seed,
            target_size=settings.preprocessing.image_size,
            apply_normalization=settings.preprocessing.apply_macenko_normalization,
            macenko_io=settings.preprocessing.macenko_io,
            macenko_alpha=settings.preprocessing.macenko_alpha,
            macenko_beta=settings.preprocessing.macenko_beta
        )

        print(f"\n" + "=" * 80)
        print("✅ DATA PREPROCESSING COMPLETE!")
        print("=" * 80)

        # Verify processed data
        processed_path = Path(settings.data.processed_data_dir)
        for split in ['train', 'val', 'test']:
            pos_processed = len(list((processed_path / split / "gram_positive").glob("*.jpg")))
            neg_processed = len(list((processed_path / split / "gram_negative").glob("*.jpg")))
            print(f"\n{split.upper()}:")
            print(f"  Gram-positive: {pos_processed}")
            print(f"  Gram-negative: {neg_processed}")
            print(f"  Total: {pos_processed + neg_processed}")

        print(f"\n📝 Next steps:")
        print(f"  1. Test training: poetry run python scripts/test_training.py")
        print(f"  2. Full training: bacterial-gan train")

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
