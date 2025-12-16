"""Data preprocessing and splitting utilities with patch extraction."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from bacterial_gan.config import DataConfig
from bacterial_gan.constants import CLASS_LABELS


def is_background_patch(patch: np.ndarray, threshold: float = 0.9, bg_value: int = 240) -> bool:
    """Check if a patch is predominantly background (white space)."""
    if len(patch.shape) == 3:
        gray = np.mean(patch, axis=2)
    else:
        gray = patch

    background_pixels = np.sum(gray >= bg_value)
    total_pixels = gray.size

    return (background_pixels / total_pixels) > threshold


def extract_patches_from_image(
    img: np.ndarray,
    patch_size: int = 128,
    stride: int = None,
    filter_background: bool = True,
    bg_threshold: float = 0.9,
) -> List[np.ndarray]:
    """Extract non-overlapping patches from an image."""
    if stride is None:
        stride = patch_size

    height, width = img.shape[:2]
    patches = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]

            if filter_background:
                if not is_background_patch(patch, threshold=bg_threshold):
                    patches.append(patch)
            else:
                patches.append(patch)

    return patches


def apply_traditional_augmentation(patch: np.ndarray) -> List[np.ndarray]:
    """Apply rotations and flips (8 versions)."""
    return [
        patch,
        np.rot90(patch, k=1),
        np.rot90(patch, k=2),
        np.rot90(patch, k=3),
        np.fliplr(patch),
        np.flipud(patch),
        np.rot90(np.fliplr(patch), k=1),
        np.rot90(np.fliplr(patch), k=3),
    ]


def create_data_splits(
    data_config: DataConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    patch_size: int = 128,
    apply_augmentation: bool = True,
    bg_threshold: float = 0.9,
    use_patch_extraction: bool = True,
    crop_mode: str = "resize",  # 'resize', 'center', or 'random'
    max_patches_per_split: int | None = None,
) -> None:
    """Split raw data into train/val/test sets with patch extraction."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    raw_path = Path(data_config.raw_data_dir)
    processed_path = Path(data_config.processed_data_dir)

    for split in ["train", "val", "test"]:
        for class_name in CLASS_LABELS.values():
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE")
    print("=" * 80)
    if use_patch_extraction:
        print("  Mode: Patch Extraction")
        print(f"  Background filtering: >={bg_threshold*100}% white pixels")
    else:
        mode_names = {"resize": "Resize", "center": "Center Crop", "random": "Random Crop"}
        print(f"  Mode: {mode_names.get(crop_mode, crop_mode)}")
    print(f"  Target size: {patch_size}x{patch_size}")
    print(f"  Augmentation: {'8x' if apply_augmentation else 'Disabled'}")
    if max_patches_per_split:
        print(f"  Max patches per split: {max_patches_per_split}")
    print("=" * 80)

    limit_per_class = max_patches_per_split // len(CLASS_LABELS) if max_patches_per_split else None

    total_patches = 0
    total_images = 0
    total_failed = 0

    for class_name in CLASS_LABELS.values():
        class_dir = raw_path / class_name

        if not class_dir.exists():
            print(f"\nWarning: {class_dir} does not exist. Skipping.")
            continue

        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.JPG", "*.PNG"]:
            image_files.extend(list(class_dir.glob(ext)))

        if len(image_files) == 0:
            print(f"\nWarning: No images found in {class_dir}")
            continue

        print(f"\nProcessing {len(image_files)} images for '{class_name}'...")

        train_files, temp_files = train_test_split(
            image_files, test_size=(1 - train_ratio), random_state=random_seed
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            split_limit = limit_per_class
            if limit_per_class and train_ratio > 0:
                if split_name == "val":
                    split_limit = int(limit_per_class * (val_ratio / train_ratio))
                elif split_name == "test":
                    split_limit = int(limit_per_class * (test_ratio / train_ratio))

            print(f"\n  {split_name.upper()}: {len(files)} images")
            split_patches = 0
            split_images = 0
            failed = 0

            budget_per_image = None
            if split_limit and len(files) > 0:
                budget_per_image = max(1, split_limit // len(files))

            for idx, src_file in enumerate(files, 1):
                try:
                    print(f"  [{idx}/{len(files)}] {src_file.name[:40]:<40}", end="", flush=True)

                    img = Image.open(src_file).convert("RGB")
                    img_array = np.array(img)
                    img_size = f"{img.width}x{img.height}"

                    if use_patch_extraction:
                        patches = extract_patches_from_image(
                            img_array,
                            patch_size=patch_size,
                            filter_background=True,
                            bg_threshold=bg_threshold,
                        )
                    else:
                        h, w = img_array.shape[:2]

                        if crop_mode == "resize":
                            # Resize entire image to target size
                            img_resized = Image.fromarray(img_array).resize(
                                (patch_size, patch_size), Image.Resampling.LANCZOS
                            )
                            patches = [np.array(img_resized)]

                        elif crop_mode == "center":
                            # Center crop
                            if h >= patch_size and w >= patch_size:
                                y = (h - patch_size) // 2
                                x = (w - patch_size) // 2
                                patches = [img_array[y : y + patch_size, x : x + patch_size]]
                            else:
                                # Image too small, resize instead
                                img_resized = Image.fromarray(img_array).resize(
                                    (patch_size, patch_size), Image.Resampling.LANCZOS
                                )
                                patches = [np.array(img_resized)]

                        else:  # crop_mode == "random"
                            # Random crop
                            if h >= patch_size and w >= patch_size:
                                local_seed = int(src_file.stat().st_size + idx) % (2**32)
                                rng = np.random.RandomState(local_seed)
                                y = rng.randint(0, h - patch_size + 1)
                                x = rng.randint(0, w - patch_size + 1)
                                patches = [img_array[y : y + patch_size, x : x + patch_size]]
                            else:
                                # Image too small, resize instead
                                img_resized = Image.fromarray(img_array).resize(
                                    (patch_size, patch_size), Image.Resampling.LANCZOS
                                )
                                patches = [np.array(img_resized)]

                    if len(patches) == 0:
                        print(" ❌ No valid patches")
                        failed += 1
                        total_failed += 1
                        continue

                    augmented = []
                    if budget_per_image:
                        np.random.shuffle(patches)
                        multiplier = 8 if apply_augmentation else 1
                        needed = min(len(patches), int(np.ceil(budget_per_image / multiplier)) * 2)
                        patches = patches[:needed]

                    for patch in patches:
                        if apply_augmentation:
                            augmented.extend(apply_traditional_augmentation(patch))
                        else:
                            augmented.append(patch)

                    if budget_per_image:
                        np.random.shuffle(augmented)
                        augmented = augmented[:budget_per_image]

                    for patch_idx, patch in enumerate(augmented):
                        patch_filename = f"{src_file.stem}_patch{patch_idx:04d}.png"
                        dst_file = processed_path / split_name / class_name / patch_filename
                        Image.fromarray(patch.astype(np.uint8)).save(dst_file, "PNG")
                        split_patches += 1
                        total_patches += 1

                    split_images += 1
                    total_images += 1

                    if split_limit and split_patches >= split_limit:
                        print(f"    ✓ Limit {split_limit} reached")
                        break

                    print(
                        f" ✓ {img_size} → {len(patches)}p × {'8x' if apply_augmentation else '1x'} = {len(augmented)}"
                    )

                except Exception as e:
                    print(f" ❌ {str(e)[:30]}")
                    failed += 1
                    total_failed += 1

            avg = split_patches / split_images if split_images > 0 else 0
            print(f"  ✓ {split_name.upper()}: {split_patches:,} patches ({avg:.1f}/img)")

    print(f"\n{'='*80}")
    print(f"COMPLETE: {total_patches:,} patches from {total_images} images")
    if total_failed > 0:
        print(f"  Failed: {total_failed}")
    print(f"{'='*80}")


def validate_dataset_integrity(data_path: Path) -> Dict[str, any]:
    """Validate dataset integrity."""
    stats = {
        "total_images": 0,
        "class_distribution": {},
        "corrupt_images": [],
        "size_distribution": {},
    }

    for class_name in CLASS_LABELS.keys():
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue

        class_count = 0
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
            for img_path in class_dir.glob(ext):
                try:
                    img = Image.open(img_path)
                    img.verify()
                    img = Image.open(img_path)

                    size_key = f"{img.size[0]}x{img.size[1]}"
                    stats["size_distribution"][size_key] = (
                        stats["size_distribution"].get(size_key, 0) + 1
                    )

                    class_count += 1
                    stats["total_images"] += 1
                except Exception as e:
                    stats["corrupt_images"].append(str(img_path))

        stats["class_distribution"][class_name] = class_count

    return stats


def calculate_dataset_statistics(data_path: Path) -> Dict[str, np.ndarray]:
    """Calculate mean and std of dataset."""
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    for class_name in CLASS_LABELS.keys():
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in class_dir.glob(ext):
                try:
                    img = np.array(Image.open(img_path)) / 255.0
                    pixel_sum += img.sum(axis=(0, 1))
                    pixel_sq_sum += (img**2).sum(axis=(0, 1))
                    n_pixels += img.shape[0] * img.shape[1]
                except Exception:
                    pass

    if n_pixels == 0:
        return {"mean": np.zeros(3), "std": np.ones(3)}

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean**2)

    return {"mean": mean, "std": std}


def preprocess_image_batch(
    image_paths: List[str], target_size: Tuple[int, int] = (128, 128), normalize: bool = True
) -> np.ndarray:
    """Batch processing for image preprocessing."""
    images = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size, Image.BILINEAR)
            img_array = np.array(img)

            if normalize:
                img_array = (img_array - 127.5) / 127.5

            images.append(img_array)
        except Exception:
            pass

    return np.array(images)
