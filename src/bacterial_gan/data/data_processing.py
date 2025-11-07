"""Data preprocessing and splitting utilities."""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from bacterial_gan.config import DataConfig
from bacterial_gan.constants import CLASS_LABELS
from bacterial_gan.utils import normalize_macenko


def create_data_splits(
    data_config: DataConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> None:
    """
    Split raw data into train/val/test sets with stratified sampling.

    Args:
        data_config: Data configuration
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    raw_path = Path(data_config.raw_data_dir)
    processed_path = Path(data_config.processed_data_dir)

    # Create split directories
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_LABELS.keys():
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in CLASS_LABELS.keys():
        class_dir = raw_path / class_name

        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist. Skipping.")
            continue

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))

        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue

        print(f"Processing {len(image_files)} images for class '{class_name}'...")

        # Split data
        train_files, temp_files = train_test_split(
            image_files,
            test_size=(1 - train_ratio),
            random_state=random_seed
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )

        # Copy files to respective directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} images")
            for src_file in files:
                dst_file = processed_path / split_name / class_name / src_file.name
                shutil.copy2(src_file, dst_file)

    print("Data splitting complete!")


def apply_macenko_normalization(
    input_dir: Path,
    output_dir: Path,
    reference_image_path: str = None,
) -> None:
    """
    Apply Macenko color normalization to all images in directory.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for normalized images
        reference_image_path: Path to reference image (if None, uses first image)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference image
    if reference_image_path:
        ref_image = Image.open(reference_image_path)
    else:
        # Use first image as reference
        first_image = next(input_dir.glob('*.jpg'), None)
        if first_image is None:
            first_image = next(input_dir.glob('*.png'), None)
        if first_image is None:
            print(f"No images found in {input_dir}")
            return
        ref_image = Image.open(first_image)

    ref_array = np.array(ref_image)

    # Process all images
    image_count = 0
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        try:
            img = Image.open(img_path)
            img_array = np.array(img)

            # Apply Macenko normalization
            normalized = normalize_macenko(img_array, ref_array)

            # Save
            output_path = output_dir / img_path.name
            Image.fromarray(normalized.astype(np.uint8)).save(output_path)

            image_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Normalized {image_count} images from {input_dir} to {output_dir}")


def validate_dataset_integrity(data_path: Path) -> Dict[str, any]:
    """
    Validate dataset integrity and return statistics.

    Args:
        data_path: Path to dataset directory

    Returns:
        Dictionary with validation results
    """
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'corrupt_images': [],
        'size_distribution': {},
    }

    for class_name in CLASS_LABELS.keys():
        class_dir = data_path / class_name

        if not class_dir.exists():
            continue

        class_count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            for img_path in class_dir.glob(ext):
                try:
                    # Try to open image
                    img = Image.open(img_path)
                    img.verify()  # Verify it's a valid image

                    # Re-open for size check (verify closes the file)
                    img = Image.open(img_path)

                    # Check size
                    size_key = f"{img.size[0]}x{img.size[1]}"
                    stats['size_distribution'][size_key] = \
                        stats['size_distribution'].get(size_key, 0) + 1

                    class_count += 1
                    stats['total_images'] += 1

                except Exception as e:
                    stats['corrupt_images'].append(str(img_path))
                    print(f"Corrupt image: {img_path} - {e}")

        stats['class_distribution'][class_name] = class_count

    return stats


def calculate_dataset_statistics(data_path: Path) -> Dict[str, np.ndarray]:
    """
    Calculate mean and std of dataset for normalization.

    Args:
        data_path: Path to dataset

    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    for class_name in CLASS_LABELS.keys():
        class_dir = data_path / class_name

        if not class_dir.exists():
            continue

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            for img_path in class_dir.glob(ext):
                try:
                    img = np.array(Image.open(img_path)) / 255.0
                    pixel_sum += img.sum(axis=(0, 1))
                    pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
                    n_pixels += img.shape[0] * img.shape[1]
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

    if n_pixels == 0:
        return {'mean': np.zeros(3), 'std': np.ones(3)}

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    return {'mean': mean, 'std': std}


def preprocess_image_batch(
    image_paths: List[str],
    target_size: Tuple[int, int] = (128, 128),
    normalize: bool = True
) -> np.ndarray:
    """
    Batch processing for image preprocessing.

    Args:
        image_paths: List of image file paths
        target_size: Target size (height, width)
        normalize: Whether to normalize to [-1, 1]

    Returns:
        Numpy array of preprocessed images
    """
    images = []

    for img_path in image_paths:
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Resize
            img = img.resize(target_size, Image.BILINEAR)

            # Convert to numpy array
            img_array = np.array(img)

            # Normalize if requested
            if normalize:
                img_array = (img_array - 127.5) / 127.5

            images.append(img_array)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(images)
