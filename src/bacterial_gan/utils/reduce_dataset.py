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
