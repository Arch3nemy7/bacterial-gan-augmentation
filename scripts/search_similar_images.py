#!/usr/bin/env python3
"""
Microscopic image similarity search script optimized for bacterial images.

Searches for similar images using methods tailored for microscopy:
- Texture-based: LBP (Local Binary Patterns), Haralick features
- Shape-based: Contour matching, Hu moments
- Multi-scale SSIM: Better for different magnifications
- Feature-based: ORB/SIFT descriptors
- Combined ensemble method for best accuracy
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_image(image_path: Path, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Load and optionally resize an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    if target_size:
        img = cv2.resize(img, target_size)

    return img


def preprocess_microscopy_image(img: np.ndarray, denoise: bool = True) -> np.ndarray:
    """Preprocess microscopy image to enhance features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if denoise:
        # Non-local means denoising for microscopy images
        gray = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

    # CLAHE for contrast enhancement (common in microscopy)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def compute_lbp_histogram(
    img: np.ndarray, num_points: int = 24, radius: int = 3
) -> np.ndarray:
    """
    Compute Local Binary Pattern histogram for texture analysis.
    Excellent for bacterial morphology patterns.
    """
    gray = preprocess_microscopy_image(img)

    # Compute LBP
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")

    # Calculate histogram
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist


def compute_haralick_features(img: np.ndarray) -> np.ndarray:
    """
    Compute Haralick texture features from Gray-Level Co-occurrence Matrix.
    Captures texture properties important in microscopy.
    """
    gray = preprocess_microscopy_image(img)

    # Quantize to reduce computational cost
    quantized = (gray / 16).astype(np.uint8)

    # Compute GLCM for multiple directions
    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        quantized, distances, angles, levels=16, symmetric=True, normed=True
    )

    # Extract properties
    features = []
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    for prop in properties:
        feature = graycoprops(glcm, prop)
        features.append(feature.mean())

    return np.array(features)


def compute_shape_features(img: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Extract shape-based features using contour detection and Hu moments.
    Good for bacterial colony shapes.
    """
    gray = preprocess_microscopy_image(img)

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(7), []

    # Get largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute Hu moments (scale, rotation, translation invariant)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Log transform for better scale
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return hu_moments, contours


def compute_multiscale_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Multi-scale SSIM for better structural similarity at different scales.
    Important for microscopy images with varying magnifications.
    """
    gray1 = preprocess_microscopy_image(img1, denoise=False)
    gray2 = preprocess_microscopy_image(img2, denoise=False)

    # Ensure images are same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute SSIM at multiple scales
    scales = []
    current1, current2 = gray1.copy(), gray2.copy()

    for i in range(3):  # 3 scales
        if min(current1.shape) < 11:  # Minimum size for SSIM
            break

        score, _ = ssim(current1, current2, full=True)
        scales.append(score)

        # Downsample for next scale
        current1 = cv2.pyrDown(current1)
        current2 = cv2.pyrDown(current2)

    # Weighted average (give more weight to original scale)
    weights = [0.5, 0.3, 0.2][: len(scales)]
    return np.average(scales, weights=weights)


def compute_feature_matching(
    img1: np.ndarray, img2: np.ndarray, method: str = "orb"
) -> float:
    """
    Feature descriptor matching using ORB or SIFT.
    Good for finding similar structures in bacterial images.
    """
    gray1 = preprocess_microscopy_image(img1)
    gray2 = preprocess_microscopy_image(img2)

    # Create detector
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=500)
    else:  # SIFT
        try:
            detector = cv2.SIFT_create(nfeatures=500)
        except AttributeError:
            # SIFT not available, fall back to ORB
            detector = cv2.ORB_create(nfeatures=500)

    # Detect and compute
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0

    # Match features
    if method == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Normalize by number of keypoints
    score = len(good_matches) / max(len(kp1), len(kp2))
    return min(score, 1.0)


def compute_texture_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compare texture histograms using chi-square distance."""
    # Chi-square distance
    chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

    # Convert to similarity (0 to 1)
    similarity = 1.0 / (1.0 + chi_square)
    return similarity


def compute_ensemble_similarity(
    img1: np.ndarray, img2: np.ndarray, weights: dict = None
) -> float:
    """
    Ensemble method combining multiple similarity metrics.
    Provides the most robust results for microscopy images.
    """
    if weights is None:
        weights = {
            "multiscale_ssim": 0.30,
            "lbp_texture": 0.25,
            "haralick": 0.20,
            "feature_matching": 0.15,
            "shape": 0.10,
        }

    scores = {}

    # Multi-scale SSIM
    scores["multiscale_ssim"] = compute_multiscale_ssim(img1, img2)

    # LBP texture similarity
    lbp1 = compute_lbp_histogram(img1)
    lbp2 = compute_lbp_histogram(img2)
    scores["lbp_texture"] = compute_texture_similarity(lbp1, lbp2)

    # Haralick features
    haralick1 = compute_haralick_features(img1)
    haralick2 = compute_haralick_features(img2)
    haralick_dist = np.linalg.norm(haralick1 - haralick2)
    scores["haralick"] = 1.0 / (1.0 + haralick_dist)

    # Feature matching
    scores["feature_matching"] = compute_feature_matching(img1, img2)

    # Shape similarity
    hu1, _ = compute_shape_features(img1)
    hu2, _ = compute_shape_features(img2)
    shape_dist = np.linalg.norm(hu1 - hu2)
    scores["shape"] = 1.0 / (1.0 + shape_dist)

    # Weighted ensemble
    final_score = sum(scores[key] * weights[key] for key in weights.keys())

    return final_score


def search_similar_images(
    query_image_path: Path,
    search_folder: Path,
    method: str = "ensemble",
    top_k: int = 10,
    threshold: float = None,
    image_extensions: List[str] = None,
    denoise: bool = True,
) -> List[Tuple[Path, float]]:
    """
    Search for similar microscopy images in a folder.

    Args:
        query_image_path: Path to the query image
        search_folder: Folder to search in
        method: Similarity method
        top_k: Number of top similar images to return
        threshold: Minimum similarity threshold
        image_extensions: List of image file extensions
        denoise: Apply denoising preprocessing

    Returns:
        List of (image_path, similarity_score) tuples
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    # Load query image
    print(f"Loading query image: {query_image_path}")
    query_img = load_image(query_image_path)
    target_size = (query_img.shape[1], query_img.shape[0])

    # Precompute query features for applicable methods
    query_features = {}
    if method == "lbp":
        query_features["lbp"] = compute_lbp_histogram(query_img)
    elif method == "haralick":
        query_features["haralick"] = compute_haralick_features(query_img)
    elif method == "shape":
        query_features["shape"], _ = compute_shape_features(query_img)

    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(search_folder.glob(f"**/*{ext}"))
        image_files.extend(search_folder.glob(f"**/*{ext.upper()}"))

    print(f"Found {len(image_files)} images in {search_folder}")
    print(f"Using method: {method}")

    # Compute similarities
    results = []

    for img_path in tqdm(image_files, desc="Computing similarities"):
        # Skip the query image itself
        if img_path.resolve() == query_image_path.resolve():
            continue

        try:
            # Load and resize image
            img = load_image(img_path, target_size)

            # Compute similarity based on method
            if method == "multiscale_ssim":
                score = compute_multiscale_ssim(query_img, img)
            elif method == "lbp":
                hist = compute_lbp_histogram(img)
                score = compute_texture_similarity(query_features["lbp"], hist)
            elif method == "haralick":
                features = compute_haralick_features(img)
                dist = np.linalg.norm(query_features["haralick"] - features)
                score = 1.0 / (1.0 + dist)
            elif method == "shape":
                hu_moments, _ = compute_shape_features(img)
                dist = np.linalg.norm(query_features["shape"] - hu_moments)
                score = 1.0 / (1.0 + dist)
            elif method == "orb":
                score = compute_feature_matching(query_img, img, method="orb")
            elif method == "ensemble":
                score = compute_ensemble_similarity(query_img, img)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Apply threshold if specified
            if threshold is None or score >= threshold:
                results.append((img_path, score))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Search for similar microscopic bacterial images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods optimized for microscopy:
  ensemble         - Combined method (RECOMMENDED for best accuracy)
  multiscale_ssim  - Multi-scale structural similarity
  lbp              - Local Binary Patterns (texture analysis)
  haralick         - Haralick features (texture properties)
  shape            - Shape-based matching (Hu moments)
  orb              - ORB feature matching

Examples:
  # Best accuracy: ensemble method
  python search_similar_images.py query.jpg ./images/ --method ensemble

  # Fast texture-based search
  python search_similar_images.py query.jpg ./images/ --method lbp --top-k 20

  # Shape-based search for colony morphology
  python search_similar_images.py query.jpg ./images/ --method shape

  # With threshold
  python search_similar_images.py query.jpg ./images/ --threshold 0.7
        """,
    )

    parser.add_argument("query_image", type=Path, help="Path to query image")
    parser.add_argument("search_folder", type=Path, help="Folder to search in")
    parser.add_argument(
        "--method",
        choices=["ensemble", "multiscale_ssim", "lbp", "haralick", "shape", "orb"],
        default="ensemble",
        help="Similarity method (default: ensemble)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top similar images to return (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum similarity threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Save results to a text file",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising preprocessing",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.query_image.exists():
        print(f"Error: Query image not found: {args.query_image}")
        return

    if not args.search_folder.exists():
        print(f"Error: Search folder not found: {args.search_folder}")
        return

    # Search for similar images
    results = search_similar_images(
        query_image_path=args.query_image,
        search_folder=args.search_folder,
        method=args.method,
        top_k=args.top_k,
        threshold=args.threshold,
        denoise=not args.no_denoise,
    )

    # Display results
    print(f"\n{'=' * 80}")
    print(f"Top {len(results)} similar images (using {args.method}):")
    print(f"{'=' * 80}\n")

    for i, (img_path, score) in enumerate(results, 1):
        rel_path = (
            img_path.relative_to(args.search_folder)
            if img_path.is_relative_to(args.search_folder)
            else img_path
        )
        print(f"{i}. Score: {score:.4f} - {rel_path}")

    # Save results if requested
    if args.save_results:
        with open(args.save_results, "w") as f:
            f.write(f"Query: {args.query_image}\n")
            f.write(f"Method: {args.method}\n")
            f.write(f"Search folder: {args.search_folder}\n")
            f.write(f"Denoising: {not args.no_denoise}\n\n")

            for i, (img_path, score) in enumerate(results, 1):
                f.write(f"{i}. {score:.4f} - {img_path}\n")

        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
