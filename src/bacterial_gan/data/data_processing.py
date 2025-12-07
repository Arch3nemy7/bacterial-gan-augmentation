"""Data preprocessing and splitting utilities with patch extraction."""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from bacterial_gan.config import DataConfig
from bacterial_gan.constants import CLASS_LABELS
from bacterial_gan.utils import normalize_macenko


def is_background_patch(patch: np.ndarray, threshold: float = 0.9, bg_value: int = 240) -> bool:
    """
    Check if a patch is predominantly background (white space).

    Args:
        patch: Image patch as numpy array (H, W, C)
        threshold: Percentage threshold for background (default: 0.9 = 90%)
        bg_value: Background intensity threshold (default: 240 for white)

    Returns:
        True if patch is >threshold% background, False otherwise
    """
    # Convert to grayscale for simpler background detection
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
    else:
        gray = patch

    # Count pixels above background threshold
    background_pixels = np.sum(gray >= bg_value)
    total_pixels = gray.size

    background_ratio = background_pixels / total_pixels
    return background_ratio > threshold


def extract_patches_from_image(
    img: np.ndarray,
    patch_size: int = 128,
    stride: int = None,
    filter_background: bool = True,
    bg_threshold: float = 0.9
) -> List[np.ndarray]:
    """
    Extract non-overlapping or strided patches from an image.

    Args:
        img: Input image as numpy array (H, W, C)
        patch_size: Size of square patches to extract
        stride: Step size between patches (None = non-overlapping, i.e., stride=patch_size)
        filter_background: Whether to filter out background patches
        bg_threshold: Background threshold for filtering (default: 0.9 = 90%)

    Returns:
        List of image patches as numpy arrays
    """
    if stride is None:
        stride = patch_size  # Non-overlapping patches

    height, width = img.shape[:2]
    patches = []

    # Extract patches with striding
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]

            # Filter background patches if requested
            if filter_background:
                if not is_background_patch(patch, threshold=bg_threshold):
                    patches.append(patch)
            else:
                patches.append(patch)

    return patches


def apply_traditional_augmentation(patch: np.ndarray) -> List[np.ndarray]:
    """
    Apply traditional augmentation (rotations and flips) to a patch.

    Generates 8 augmented versions:
    - Original
    - Rotated 90°, 180°, 270°
    - Horizontal flip
    - Vertical flip
    - Horizontal flip + 90° rotation
    - Horizontal flip + 270° rotation

    Args:
        patch: Input patch as numpy array (H, W, C)

    Returns:
        List of augmented patches (8 versions)
    """
    augmented = []

    # Original
    augmented.append(patch)

    # Rotations
    augmented.append(np.rot90(patch, k=1))  # 90°
    augmented.append(np.rot90(patch, k=2))  # 180°
    augmented.append(np.rot90(patch, k=3))  # 270°

    # Flips
    augmented.append(np.fliplr(patch))  # Horizontal flip
    augmented.append(np.flipud(patch))  # Vertical flip

    # Combined flips and rotations
    augmented.append(np.rot90(np.fliplr(patch), k=1))  # H-flip + 90°
    augmented.append(np.rot90(np.fliplr(patch), k=3))  # H-flip + 270°

    return augmented


def create_data_splits(
    data_config: DataConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    patch_size: int = 128,
    apply_normalization: bool = False,
    apply_augmentation: bool = True,
    bg_threshold: float = 0.9,
    macenko_io: int = 240,
    macenko_alpha: float = 1.0,
    macenko_beta: float = 0.15,
    use_patch_extraction: bool = True,
    max_patches_per_split: int | None = None,
) -> None:
    """
    Split raw data into train/val/test sets with patch-based preprocessing.

    NEW APPROACH: Extracts patches from high-res images instead of resizing.
    This preserves bacterial structure and avoids stretching artifacts.

    Args:
        data_config: Data configuration
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        patch_size: Size of patches to extract (default: 128x128)
        apply_normalization: Whether to apply Macenko normalization (default: False)
        apply_augmentation: Whether to apply traditional augmentation (default: True)
        bg_threshold: Background filtering threshold (default: 0.9 = 90%)
        macenko_io: Background light intensity for Macenko
        macenko_alpha: Percentile for angle calculation in Macenko
        macenko_io: Background light intensity for Macenko
        macenko_alpha: Percentile for angle calculation in Macenko
        macenko_beta: Optical density threshold for Macenko
        use_patch_extraction: Whether to extract patches (True) or resize images (False)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    raw_path = Path(data_config.raw_data_dir)
    processed_path = Path(data_config.processed_data_dir)

    # Create split directories
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_LABELS.values():
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Print preprocessing settings
    print("\n" + "="*80)
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    print(f"  Mode: {'All Patches Extraction' if use_patch_extraction else 'Single Random Patch'}")
    print(f"  Target size: {patch_size}x{patch_size}")
    if use_patch_extraction:
        print(f"  Background filtering: >={bg_threshold*100}% white pixels")
    print(f"  Traditional augmentation: {'Enabled (8x multiplier)' if apply_augmentation else 'Disabled'}")
    print(f"  Traditional augmentation: {'Enabled (8x multiplier)' if apply_augmentation else 'Disabled'}")
    print(f"  Macenko normalization: {'Enabled' if apply_normalization else 'Disabled'}")
    if max_patches_per_split:
        print(f"  Max patches per split: {max_patches_per_split} (approx {max_patches_per_split // len(CLASS_LABELS)} per class)")
    print("="*80)

    # Calculate limit per class
    limit_per_class = None
    if max_patches_per_split:
        limit_per_class = max_patches_per_split // len(CLASS_LABELS)

    # Process each class
    total_patches = 0
    total_images = 0
    total_failed = 0

    for class_name in CLASS_LABELS.values():
        class_dir = raw_path / class_name

        if not class_dir.exists():
            print(f"\nWarning: {class_dir} does not exist. Skipping.")
            continue

        # Get all image files (including TIFF format)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.PNG', '*.TIF', '*.TIFF']:
            image_files.extend(list(class_dir.glob(ext)))

        if len(image_files) == 0:
            print(f"\nWarning: No images found in {class_dir}")
            continue

        print(f"\nProcessing {len(image_files)} images for class '{class_name}'...")

        # Split data at image level (before patch extraction)
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

        # Process and save files to respective directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, files in splits.items():
            # Determine limit for this specific split relative to training ratio
            # This ensures validation and test sets are proportionally smaller than training set
            split_limit_per_class = limit_per_class
            if limit_per_class:
                if split_name == 'val' and train_ratio > 0:
                     split_limit_per_class = int(limit_per_class * (val_ratio / train_ratio))
                elif split_name == 'test' and train_ratio > 0:
                     split_limit_per_class = int(limit_per_class * (test_ratio / train_ratio))

            print(f"\n  {split_name.upper()}: Processing {len(files)} images...")
            print(f"  {'='*76}")
            split_patches = 0
            split_images = 0
            failed = 0
            
            # Calculate patches budget per image to ensure variability across the dataset
            patches_budget_per_image = None
            if split_limit_per_class and len(files) > 0:
                patches_budget_per_image = max(1, split_limit_per_class // len(files))
                print(f"  ℹ️  Budget per image: {patches_budget_per_image} patches (to cover {len(files)} images)")

            # Check if limit already reached for this class in this split (unlikely but good for safety)
            if split_limit_per_class and split_patches >= split_limit_per_class:
                print(f"  ✓ {split_name.upper()}: Limit of {split_limit_per_class} patches reached for class {class_name}. Skipping.")
                continue

            for idx, src_file in enumerate(files, 1):
                try:
                    # Progress indicator
                    print(f"  [{idx}/{len(files)}] {src_file.name[:50]:<50}", end='', flush=True)

                    # Load image
                    img = Image.open(src_file).convert('RGB')
                    img_array = np.array(img)
                    img_size = f"{img.width}x{img.height}"

                    # Apply Macenko normalization if requested (but project doesn't use it)
                    if apply_normalization:
                        try:
                            img_array = normalize_macenko(
                                img_array,
                                Io=macenko_io,
                                alpha=macenko_alpha,
                                beta=macenko_beta
                            )
                        except Exception as e:
                            print(f"\n      ⚠️  Macenko normalization failed: {e}")
                            # Continue with unnormalized image

                    if use_patch_extraction:
                        # Extract patches from high-res image
                        patches = extract_patches_from_image(
                            img_array,
                            patch_size=patch_size,
                            stride=None,  # Non-overlapping
                            filter_background=True,
                            bg_threshold=bg_threshold
                        )
                    else:
                        # Extract a single random patch
                        h, w = img_array.shape[:2]
                        if h >= patch_size and w >= patch_size:
                            # Random crop
                            # Use a deterministic random based on file name hash to ensure reproducibility across runs if needed
                            # or just use the global random state if seeded
                            # Here we use numpy's random which should be seeded if create_data_splits called np.random.seed
                            # But create_data_splits doesn't seem to set np.random.seed globally, it uses random_state for train_test_split
                            
                            # Let's use a local random generator seeded with the file name to be deterministic per file
                            local_seed = int(int(src_file.stat().st_size) + idx) % (2**32)
                            rng = np.random.RandomState(local_seed)
                            
                            y = rng.randint(0, h - patch_size + 1)
                            x = rng.randint(0, w - patch_size + 1)
                            patch = img_array[y:y+patch_size, x:x+patch_size]
                            patches = [patch]
                        else:
                            # Image too small, resize to patch_size
                            img_resized = Image.fromarray(img_array).resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                            patches = [np.array(img_resized)]

                    if len(patches) == 0:
                        print(f" ❌ No valid patches")
                        failed += 1
                        total_failed += 1
                        continue

                    # Apply traditional augmentation to each patch
                    augmented_patches = []
                    
                    # Optimization: If we have a budget, only augment what we need
                    # But to ensure spatial diversity, we should shuffle the original patches first
                    if patches_budget_per_image:
                        np.random.shuffle(patches)
                        # We need enough original patches to generate the budget
                        # Each original patch generates 8 augmented patches (if enabled)
                        multiplier = 8 if apply_augmentation else 1
                        needed_originals = int(np.ceil(patches_budget_per_image / multiplier))
                        # Take a few more to be safe/diverse, but don't process everything if we have 1000 patches
                        needed_originals = min(len(patches), needed_originals * 2) 
                        patches_to_augment = patches[:needed_originals]
                    else:
                        patches_to_augment = patches

                    for patch in patches_to_augment:
                        if apply_augmentation:
                            # Generate 8 augmented versions per patch
                            augmented_patches.extend(apply_traditional_augmentation(patch))
                        else:
                            # Keep only original patch
                            augmented_patches.append(patch)
                            
                    # If we have a budget, shuffle and trim the final list
                    if patches_budget_per_image:
                        np.random.shuffle(augmented_patches)
                        augmented_patches = augmented_patches[:patches_budget_per_image]

                    # Save all augmented patches
                    for patch_idx, patch in enumerate(augmented_patches):
                        # Create unique filename: original_name_patchXXX.png
                        patch_filename = f"{src_file.stem}_patch{patch_idx:04d}.png"
                        dst_file = processed_path / split_name / class_name / patch_filename

                        # Convert to PIL Image and save
                        patch_img = Image.fromarray(patch.astype(np.uint8))
                        patch_img.save(dst_file, "PNG")

                        split_patches += 1
                        total_patches += 1

                    split_images += 1
                    total_images += 1

                    # Check limit
                    if split_limit_per_class and split_patches >= split_limit_per_class:
                        print(f"    ⚠️  Limit of {split_limit_per_class} patches reached for class {class_name}. Stopping.")
                        break

                    # Print success with stats
                    raw_patches = len(patches)
                    print(f" ✓ {img_size} → {raw_patches}p {'(sampled)' if patches_budget_per_image else ''} × {'8x' if apply_augmentation else '1x'} = {len(augmented_patches)} patches")

                except Exception as e:
                    print(f" ❌ Error: {str(e)[:30]}")
                    failed += 1
                    total_failed += 1
                    continue

            # Print summary for this split
            print(f"  {'='*76}")
            avg_patches = split_patches / split_images if split_images > 0 else 0
            if failed > 0:
                print(f"  ✓ {split_name.upper()}: {split_patches:,} patches from {split_images} images "
                      f"(avg: {avg_patches:.1f}/img, {failed} failed)")
            else:
                print(f"  ✓ {split_name.upper()}: {split_patches:,} patches from {split_images} images "
                      f"(avg: {avg_patches:.1f}/img)")

    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"  Total images processed: {total_images}")
    print(f"  Total patches generated: {total_patches}")
    print(f"  Total failed: {total_failed}")
    if total_images > 0:
        print(f"  Average patches per image: {total_patches/total_images:.1f}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Background filtering: >={bg_threshold*100}% threshold")
    print(f"  Augmentation multiplier: {'8x' if apply_augmentation else '1x'}")
    print(f"{'='*80}")


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
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.PNG', '*.TIF', '*.TIFF']:
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

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.PNG', '*.TIF', '*.TIFF']:
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
