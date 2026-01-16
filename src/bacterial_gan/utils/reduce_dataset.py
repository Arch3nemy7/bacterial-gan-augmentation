"""
Dataset reduction with diversity preservation using feature clustering.

Reduces a large dataset to a smaller representative subset by:
1. Extracting feature embeddings using a pretrained model
2. Clustering images using K-Means
3. Selecting the most representative image from each cluster

Usage:
    python -m bacterial_gan.utils.reduce_dataset --input data/02_processed/train --output data/reduced --target-size 5000
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(image_path: Path, target_size: tuple = (128, 128)) -> np.ndarray:
    """Load and preprocess image for feature extraction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def extract_simple_features(image: np.ndarray) -> np.ndarray:
    """
    Extract simple but effective features without requiring heavy dependencies.
    
    Features include:
    - Color histogram (RGB)
    - Spatial color distribution (grid-based)
    - Edge density approximation
    """
    features = []
    
    # 1. Color histogram (32 bins per channel = 96 features)
    for c in range(3):
        hist, _ = np.histogram(image[:, :, c], bins=32, range=(0, 1))
        features.extend(hist / hist.sum())  # Normalize
    
    # 2. Spatial grid features (4x4 grid = 48 features)
    h, w = image.shape[:2]
    grid_h, grid_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            patch = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            features.extend([patch[:, :, c].mean() for c in range(3)])
    
    # 3. Edge density approximation using gradients (2 features)
    gray = np.mean(image, axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    features.extend([grad_x, grad_y])
    
    return np.array(features, dtype=np.float32)


def extract_all_features(
    image_paths: list[Path],
    feature_size: tuple = (128, 128),
) -> np.ndarray:
    """Extract features from all images."""
    print(f"📊 Extracting features from {len(image_paths)} images...")
    
    features_list = []
    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            img = load_image(path, feature_size)
            features = extract_simple_features(img)
            features_list.append(features)
        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")
            # Use zero features for failed images
            features_list.append(np.zeros(146, dtype=np.float32))
    
    return np.array(features_list)


def cluster_and_select(
    features: np.ndarray,
    image_paths: list[Path],
    target_size: int,
    random_seed: int = 42,
) -> list[Path]:
    """
    Cluster features and select representative images.
    
    Uses MiniBatchKMeans for efficiency with large datasets.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    
    print(f"🔄 Clustering into {target_size} groups...")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Handle case where target_size > num_images
    actual_target = min(target_size, len(image_paths))
    
    # Cluster using MiniBatchKMeans (memory efficient)
    kmeans = MiniBatchKMeans(
        n_clusters=actual_target,
        random_state=random_seed,
        batch_size=min(1024, len(image_paths)),
        n_init=3,
        max_iter=100,
    )
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Find closest image to each cluster centroid
    print("🎯 Selecting representative images...")
    selected_paths = []
    
    for cluster_id in tqdm(range(actual_target), desc="Selecting"):
        # Get indices of images in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get features for this cluster
        cluster_features = features_normalized[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Find image closest to centroid
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        
        selected_paths.append(image_paths[closest_idx])
    
    return selected_paths


def copy_selected_images(
    selected_paths: list[Path],
    output_dir: Path,
    preserve_structure: bool = True,
) -> None:
    """Copy selected images to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Copying {len(selected_paths)} images to {output_dir}...")
    
    for path in tqdm(selected_paths, desc="Copying"):
        if preserve_structure:
            # Preserve class subdirectory (e.g., gram_positive, gram_negative)
            relative_parent = path.parent.name
            dest_dir = output_dir / relative_parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / path.name
        else:
            dest_path = output_dir / path.name
        
        shutil.copy2(path, dest_path)


def reduce_dataset(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 5000,
    random_seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Main function to reduce dataset while preserving diversity.
    
    Args:
        input_dir: Directory containing images (with class subdirectories)
        output_dir: Directory to save reduced dataset
        target_size: Target number of images in reduced dataset
        random_seed: Random seed for reproducibility
        dry_run: If True, only report what would be done
        
    Returns:
        Dictionary with statistics about the reduction
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    print(f"📊 Found {len(image_paths)} images in {input_dir}")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return {"original": 0, "reduced": 0}
    
    if len(image_paths) <= target_size:
        print(f"ℹ️  Dataset already smaller than target ({len(image_paths)} <= {target_size})")
        if not dry_run:
            print("Copying all images...")
            copy_selected_images(image_paths, output_dir)
        return {"original": len(image_paths), "reduced": len(image_paths)}
    
    # Count by class
    class_counts = {}
    for path in image_paths:
        class_name = path.parent.name
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("📊 Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count}")
    
    # Extract features
    features = extract_all_features(image_paths)
    
    # Cluster and select
    selected_paths = cluster_and_select(
        features, image_paths, target_size, random_seed
    )
    
    # Report statistics
    reduced_class_counts = {}
    for path in selected_paths:
        class_name = path.parent.name
        reduced_class_counts[class_name] = reduced_class_counts.get(class_name, 0) + 1
    
    print("\n📊 Reduced class distribution:")
    for class_name, count in sorted(reduced_class_counts.items()):
        original = class_counts.get(class_name, 0)
        pct = (count / original * 100) if original > 0 else 0
        print(f"   {class_name}: {count} ({pct:.1f}% of original)")
    
    print(f"\n✅ Reduced: {len(image_paths)} → {len(selected_paths)} images")
    print(f"   Reduction: {(1 - len(selected_paths)/len(image_paths))*100:.1f}%")
    
    if not dry_run:
        copy_selected_images(selected_paths, output_dir)
        print(f"\n✅ Saved to {output_dir}")
    else:
        print("\n🔍 Dry run - no files copied")
    
    return {
        "original": len(image_paths),
        "reduced": len(selected_paths),
        "class_counts_original": class_counts,
        "class_counts_reduced": reduced_class_counts,
    }


def reduce_dataset_with_splits(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 5000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Reduce dataset while preserving diversity, then split into train/val/test.
    
    Args:
        input_dir: Directory containing images (can have train/val/test subdirs or class subdirs)
        output_dir: Directory to save reduced dataset with train/val/test splits
        target_size: Target TOTAL number of images
        train_ratio: Fraction for training set (default: 0.7)
        val_ratio: Fraction for validation set (default: 0.15)
        test_ratio: Fraction for test set (default: 0.15)
        random_seed: Random seed for reproducibility
        dry_run: If True, only report what would be done
        
    Returns:
        Dictionary with statistics about the reduction
    """
    import random
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️  Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Find all images (from all subdirectories)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    print(f"📊 Found {len(image_paths)} total images in {input_dir}")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return {"original": 0, "reduced": 0}
    
    # Determine class from path (handle both train/class and class structures)
    def get_class_name(path: Path) -> str:
        # Check if parent is a split folder (train/val/test)
        parent = path.parent.name
        grandparent = path.parent.parent.name if path.parent.parent else ""
        
        if grandparent in ("train", "val", "test"):
            return parent  # Class is the direct parent
        elif parent in ("train", "val", "test"):
            return "unknown"  # No class subdir
        else:
            return parent  # Assume parent is class
    
    # Count by class
    class_counts = {}
    for path in image_paths:
        class_name = get_class_name(path)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("📊 Original class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count}")
    
    # Reduce dataset
    if len(image_paths) <= target_size:
        print(f"ℹ️  Dataset already smaller than target ({len(image_paths)} <= {target_size})")
        selected_paths = image_paths
    else:
        # Extract features and cluster
        features = extract_all_features(image_paths)
        selected_paths = cluster_and_select(features, image_paths, target_size, random_seed)
    
    # Calculate split sizes
    n_train = int(len(selected_paths) * train_ratio)
    n_val = int(len(selected_paths) * val_ratio)
    n_test = len(selected_paths) - n_train - n_val
    
    print(f"\n📊 Split sizes (total: {len(selected_paths)}):")
    print(f"   train: {n_train} ({train_ratio*100:.0f}%)")
    print(f"   val:   {n_val} ({val_ratio*100:.0f}%)")
    print(f"   test:  {n_test} ({test_ratio*100:.0f}%)")
    
    # Shuffle and split, maintaining class balance as much as possible
    random.seed(random_seed)
    
    # Group by class
    class_to_paths = {}
    for path in selected_paths:
        class_name = get_class_name(path)
        if class_name not in class_to_paths:
            class_to_paths[class_name] = []
        class_to_paths[class_name].append(path)
    
    # Shuffle within each class
    for class_name in class_to_paths:
        random.shuffle(class_to_paths[class_name])
    
    # Stratified split
    train_paths = []
    val_paths = []
    test_paths = []
    
    for class_name, paths in class_to_paths.items():
        n_class = len(paths)
        n_class_train = int(n_class * train_ratio)
        n_class_val = int(n_class * val_ratio)
        
        train_paths.extend(paths[:n_class_train])
        val_paths.extend(paths[n_class_train:n_class_train + n_class_val])
        test_paths.extend(paths[n_class_train + n_class_val:])
    
    # Report final distribution
    def count_classes(paths):
        counts = {}
        for p in paths:
            c = get_class_name(p)
            counts[c] = counts.get(c, 0) + 1
        return counts
    
    print("\n📊 Final distribution:")
    for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        print(f"\n   {split_name} ({len(split_paths)} images):")
        for class_name, count in sorted(count_classes(split_paths).items()):
            print(f"      {class_name}: {count}")
    
    if not dry_run:
        # Copy files to split directories
        for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            split_dir = output_dir / split_name
            print(f"\n📁 Copying {len(split_paths)} images to {split_dir}...")
            
            for path in tqdm(split_paths, desc=f"Copying {split_name}"):
                class_name = get_class_name(path)
                dest_dir = split_dir / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / path.name
                shutil.copy2(path, dest_path)
        
        print(f"\n✅ Saved to {output_dir}")
    else:
        print("\n🔍 Dry run - no files copied")
    
    return {
        "original": len(image_paths),
        "reduced": len(selected_paths),
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reduce dataset size while preserving diversity using clustering"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for reduced dataset",
    )
    parser.add_argument(
        "--target-size", "-n",
        type=int,
        default=5000,
        help="Target number of images (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done, don't copy files",
    )
    
    args = parser.parse_args()
    
    reduce_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_size=args.target_size,
        random_seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
