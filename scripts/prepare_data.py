"""
Prepare bacterial dataset for training.

This script:
1. Applies Macenko color normalization to images
2. Splits data into train/val/test sets (70/15/15)
3. Saves processed images to data/02_processed/
"""
from pathlib import Path
import shutil
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
    print(f"  Mode: {'All Patches Extraction' if settings.preprocessing.use_patch_extraction else 'Single Random Patch'}")
    print(f"  Target size: {settings.preprocessing.image_size}x{settings.preprocessing.image_size}")
    print(f"  Augmentation: {'Enabled (8x)' if settings.preprocessing.apply_augmentation else 'Disabled'}")
    print(f"  Background threshold: >{settings.preprocessing.bg_threshold*100:.0f}%")
    print(f"  Macenko normalization: {'Enabled' if settings.preprocessing.apply_macenko_normalization else 'Disabled'}")

    # Check if raw data exists
    raw_path = Path(settings.data.raw_data_dir)
    gram_positive = raw_path / "gram_positive"
    gram_negative = raw_path / "gram_negative"

    if not gram_positive.exists() or not gram_negative.exists():
        print(f"\n❌ Error: Raw data directories not found!")
        print(f"   Expected: {gram_positive} and {gram_negative}")
        print(f"\n   Please run: poetry run python scripts/organize_dataset.py")
        print(f"\n   Please run: poetry run python scripts/organize_dataset.py")
        return

    # Clear processed data directory
    processed_path = Path(settings.data.processed_data_dir)
    if processed_path.exists():
        print(f"\n🧹 Cleaning up {processed_path}...")
        shutil.rmtree(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created fresh directory: {processed_path}")

    # Count images (all formats including TIFF)
    pos_images = []
    neg_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.PNG', '*.TIF', '*.TIFF']:
        pos_images.extend(list(gram_positive.glob(ext)))
        neg_images.extend(list(gram_negative.glob(ext)))

    pos_count = len(pos_images)
    neg_count = len(neg_images)
    total = pos_count + neg_count

    # Detect image format
    if pos_images:
        sample_format = pos_images[0].suffix.upper()
        print(f"\n📁 Detected image format: {sample_format}")

    print(f"\n📊 Dataset Statistics:")
    print(f"  Gram-positive images: {pos_count}")
    print(f"  Gram-negative images: {neg_count}")
    print(f"  Total images: {total}")

    val_pct_of_train = settings.preprocessing.val_ratio
    test_pct_of_train = settings.preprocessing.test_ratio
    
    train_ratio = 1.0 / (1.0 + val_pct_of_train + test_pct_of_train)
    val_ratio = train_ratio * val_pct_of_train
    test_ratio = train_ratio * test_pct_of_train

    print(f"\n📂 Data Split (Calculated to be {val_pct_of_train*100:.0f}% of Train):")
    print(f"  Training: {train_ratio*100:.1f}% (~{int(total*train_ratio)} images)")
    print(f"  Validation: {val_ratio*100:.1f}% (~{int(total*val_ratio)} images)")
    print(f"  Test: {test_ratio*100:.1f}% (~{int(total*test_ratio)} images)")

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

    
    if settings.preprocessing.use_patch_extraction:
        print(f"  2. Extract {settings.preprocessing.image_size}x{settings.preprocessing.image_size} patches from high-res images (All patches)")
        print(f"  3. Filter patches with >{settings.preprocessing.bg_threshold*100:.0f}% background (white space)")
    else:
        print(f"  2. Extract 1 random {settings.preprocessing.image_size}x{settings.preprocessing.image_size} patch per image")
        print(f"  3. Skip background filtering")
        
    print(f"  4. Apply traditional augmentation: {'Yes (8x multiplier)' if settings.preprocessing.apply_augmentation else 'No'}")
    print("  5. Split into train/val/test sets")
    print("  6. Save to data/02_processed/")

    try:
        create_data_splits(
            data_config=settings.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=settings.preprocessing.random_seed,
            patch_size=settings.preprocessing.image_size,
            apply_normalization=settings.preprocessing.apply_macenko_normalization,
            apply_augmentation=settings.preprocessing.apply_augmentation,
            bg_threshold=settings.preprocessing.bg_threshold,
            macenko_io=settings.preprocessing.macenko_io,
            macenko_alpha=settings.preprocessing.macenko_alpha,
            macenko_beta=settings.preprocessing.macenko_beta,
            use_patch_extraction=settings.preprocessing.use_patch_extraction,
            max_patches_per_split=settings.preprocessing.max_patches_per_split
        )

        print(f"\n" + "=" * 80)
        print("✅ DATA PREPROCESSING COMPLETE!")
        print("=" * 80)

        # Verify processed data (patches are saved as PNG)
        processed_path = Path(settings.data.processed_data_dir)
        for split in ['train', 'val', 'test']:
            pos_processed = len(list((processed_path / split / "gram_positive").glob("*.png")))
            neg_processed = len(list((processed_path / split / "gram_negative").glob("*.png")))
            print(f"\n{split.upper()}:")
            print(f"  Gram-positive patches: {pos_processed}")
            print(f"  Gram-negative patches: {neg_processed}")
            print(f"  Total patches: {pos_processed + neg_processed}")

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
