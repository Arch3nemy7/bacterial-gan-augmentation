"""Test data pipeline with sample images."""

from pathlib import Path

from bacterial_gan.config import get_settings
from bacterial_gan.data.data_processing import create_data_splits, validate_dataset_integrity
from bacterial_gan.data.dataset import create_datasets


def main():
    """Test data pipeline."""
    settings = get_settings()

    print("=" * 80)
    print("BACTERIAL GAN DATA PIPELINE TEST")
    print("=" * 80)

    print("\nStep 1: Validating raw dataset...")
    raw_path = Path(settings.data.raw_data_dir)
    stats = validate_dataset_integrity(raw_path)

    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Class distribution: {stats['class_distribution']}")
    print(f"  Size distribution: {stats['size_distribution']}")
    print(f"  Corrupt images: {len(stats['corrupt_images'])}")

    if stats["total_images"] == 0:
        print("\n" + "=" * 80)
        print("⚠️  NO IMAGES FOUND!")
        print("=" * 80)
        print("\nPlease add images to the data/01_raw/ directory:")
        print("  data/01_raw/gram_positive/  <- Add Gram-positive bacterial images here")
        print("  data/01_raw/gram_negative/  <- Add Gram-negative bacterial images here")
        print("\nRequired format:")
        print("  - JPG or PNG files")
        print("  - At least 500 images per class (more is better)")
        print("  - Consistent image quality")
        print("\nOnce you have your dataset, run this script again!")
        return

    print("\n" + "-" * 80)
    print("Step 2: Creating train/val/test splits with patch extraction...")
    print("-" * 80)
    create_data_splits(
        settings.data,
        train_ratio=settings.preprocessing.train_ratio,
        val_ratio=settings.preprocessing.val_ratio,
        test_ratio=settings.preprocessing.test_ratio,
        random_seed=settings.preprocessing.random_seed,
        patch_size=settings.preprocessing.image_size,
        apply_augmentation=settings.preprocessing.apply_augmentation,
        bg_threshold=settings.preprocessing.bg_threshold,
        use_patch_extraction=settings.preprocessing.use_patch_extraction,
        crop_mode=settings.preprocessing.crop_mode,
    )

    print("\n" + "-" * 80)
    print("Step 3: Loading datasets with TensorFlow...")
    print("-" * 80)

    try:
        datasets = create_datasets(
            settings.data,
            batch_size=4,
            image_size=(128, 128),
        )

        for split_name, dataset in datasets.items():
            print(f"\n{split_name.upper()} dataset:")
            for images, labels in dataset.take(1):
                print(f"  Batch shape: {images.shape}")
                print(f"  Labels: {labels.numpy()}")
                print(f"  Value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
                print(f"  Expected: [-1.0, 1.0] for GAN training")

        print("\n" + "=" * 80)
        print("✅ DATA PIPELINE TEST COMPLETE!")
        print("=" * 80)
        print("\nYour data is ready for training!")
        print("Next steps:")
        print("  1. Test the GAN architecture: poetry run python scripts/test_architecture.py")
        print("  2. Start training: bacterial-gan train")

    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
