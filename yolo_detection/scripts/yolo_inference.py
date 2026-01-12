"""
YOLO Inference Script for Bacterial Detection and Cropping

This script provides functionality to:
1. Run inference using trained YOLO model on bacterial images
2. Visualize detections with color-coded bounding boxes
3. Crop detected bacteria in 1:1 ratio (square crops)
4. Process images based on ground truth labels or model predictions

Class mapping:
- 0: Negative cocci (red bounding box)
- 1: Positive cocci (green bounding box)
- 2: Negative bacilli (dark blue bounding box)
- 3: Positive bacilli (light blue bounding box)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Color mapping for each class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),  # Negative cocci - Red
    1: (0, 255, 0),  # Positive cocci - Green
    2: (139, 0, 0),  # Negative bacilli - Dark Blue
    3: (255, 191, 0),  # Positive bacilli - Light Blue (cyan-ish)
}

CLASS_NAMES = {
    0: "negative_cocci",
    1: "positive_cocci",
    2: "negative_bacilli",
    3: "positive_bacilli",
}


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse YOLO format label file.

    Args:
        label_path: Path to .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of dictionaries containing class_id and bounding box coordinates
    """
    detections = []

    if not label_path.exists():
        return detections

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert from YOLO format (normalized) to pixel coordinates
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height

            # Calculate corner coordinates
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            detections.append(
                {
                    "class_id": class_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": 1.0,  # Ground truth has confidence 1.0
                }
            )

    return detections


def crop_to_square(
    image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
) -> Optional[np.ndarray]:
    """
    Crop image to 1:1 (square) ratio based on bounding box.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Additional padding around the bounding box

    Returns:
        Cropped square image or None if invalid
    """
    x1, y1, x2, y2 = bbox
    img_height, img_width = image.shape[:2]

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_width, x2 + padding)
    y2 = min(img_height, y2 + padding)

    # Calculate center
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate size for square crop (use max of width/height)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    square_size = max(bbox_width, bbox_height)

    # Calculate square boundaries
    half_size = square_size // 2
    crop_x1 = max(0, center_x - half_size)
    crop_y1 = max(0, center_y - half_size)
    crop_x2 = min(img_width, center_x + half_size)
    crop_y2 = min(img_height, center_y + half_size)

    # Adjust if we're at image boundaries
    if crop_x2 - crop_x1 < square_size:
        if crop_x1 == 0:
            crop_x2 = min(img_width, crop_x1 + square_size)
        else:
            crop_x1 = max(0, crop_x2 - square_size)

    if crop_y2 - crop_y1 < square_size:
        if crop_y1 == 0:
            crop_y2 = min(img_height, crop_y1 + square_size)
        else:
            crop_y1 = max(0, crop_y2 - square_size)

    # Crop the image
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Ensure it's square by padding if needed (edge cases)
    if cropped.shape[0] != cropped.shape[1]:
        target_size = max(cropped.shape[0], cropped.shape[1])
        squared = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - cropped.shape[0]) // 2
        x_offset = (target_size - cropped.shape[1]) // 2
        squared[
            y_offset : y_offset + cropped.shape[0],
            x_offset : x_offset + cropped.shape[1],
        ] = cropped
        cropped = squared

    return cropped


def draw_detections(
    image: np.ndarray, detections: List[Dict], show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes with class-specific colors on image.

    Args:
        image: Input image
        detections: List of detection dictionaries
        show_confidence: Whether to show confidence scores

    Returns:
        Image with drawn bounding boxes
    """
    img_vis = image.copy()

    for idx, det in enumerate(detections):
        class_id = det["class_id"]
        x1, y1, x2, y2 = det["bbox"]
        confidence = det.get("confidence", 1.0)

        # Get color for this class
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        # Draw class number (use class_id for ground truth, index for predictions)
        if show_confidence and confidence < 1.0:
            # For model predictions: show index number
            label = str(idx)
        else:
            # For ground truth: show class_id
            label = str(class_id)

        # Position for number (top-left corner of bbox)
        text_position = (x1 + 2, y1 - 5 if y1 > 20 else y1 + 15)

        # Put text (simple, no background)
        cv2.putText(
            img_vis,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return img_vis


def run_model_inference(
    model: YOLO, image_path: Path, conf_threshold: float = 0.25
) -> List[Dict]:
    """
    Run YOLO model inference on an image.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections

    Returns:
        List of detection dictionaries
    """
    results = model(str(image_path), conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())

            detections.append(
                {
                    "class_id": class_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": confidence,
                }
            )

    return detections


def process_images(
    image_dir: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    label_dir: Optional[Path] = None,
    use_ground_truth: bool = False,
    conf_threshold: float = 0.25,
    save_crops: bool = True,
    save_visualization: bool = True,
    crop_padding: int = 10,
    image_list: Optional[List[str]] = None,
):
    """
    Process images for bacterial detection and cropping.

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save outputs
        model_path: Path to YOLO model weights (required if not using ground truth)
        label_dir: Directory containing YOLO label files
        use_ground_truth: Use ground truth labels instead of model predictions
        conf_threshold: Confidence threshold for model predictions
        save_crops: Whether to save individual crops
        save_visualization: Whether to save visualization images
        crop_padding: Padding around crops in pixels
        image_list: Optional list of specific image filenames to process
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_crops:
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        # Create subdirectories for each class
        for class_id in CLASS_NAMES:
            (crops_dir / CLASS_NAMES[class_id]).mkdir(exist_ok=True)

    if save_visualization:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

    # Load model if needed
    model = None
    if not use_ground_truth:
        if model_path is None:
            raise ValueError("model_path is required when not using ground truth")
        print(f"Loading YOLO model from {model_path}...")
        model = YOLO(str(model_path))

    # Get list of images to process
    if image_list:
        image_files = [
            image_dir / img for img in image_list if (image_dir / img).exists()
        ]
    else:
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    print(f"Processing {len(image_files)} images...")

    # Statistics
    total_detections = 0
    total_crops = 0

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_height, img_width = image.shape[:2]

        # Get detections
        if use_ground_truth and label_dir:
            label_path = label_dir / f"{img_path.stem}.txt"
            detections = parse_yolo_label(label_path, img_width, img_height)
        else:
            detections = run_model_inference(model, img_path, conf_threshold)

        total_detections += len(detections)

        # Draw visualization
        if save_visualization and len(detections) > 0:
            img_vis = draw_detections(
                image, detections, show_confidence=not use_ground_truth
            )
            vis_path = vis_dir / f"{img_path.stem}_detect.jpg"
            cv2.imwrite(str(vis_path), img_vis)

        # Save crops
        if save_crops:
            for idx, det in enumerate(detections):
                cropped = crop_to_square(image, det["bbox"], padding=crop_padding)
                if cropped is not None and cropped.size > 0:
                    class_name = CLASS_NAMES[det["class_id"]]
                    crop_filename = f"{img_path.stem}_crop{idx}_{class_name}.jpg"
                    crop_path = crops_dir / class_name / crop_filename
                    cv2.imwrite(str(crop_path), cropped)
                    total_crops += 1

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Total crops saved: {total_crops}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference for bacterial detection and cropping"
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/01_raw/1. DeepDataSet/DetectionDataSet/images"),
        help="Directory containing input images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/02_processed/bacteria_crops"),
        help="Directory to save outputs",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("yolo_detection/runs/bacteria_detection/weights/best.pt"),
        help="Path to YOLO model weights",
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("data/01_raw/1. DeepDataSet/DetectionDataSet/labels"),
        help="Directory containing YOLO label files",
    )

    parser.add_argument(
        "--use-ground-truth",
        action="store_true",
        help="Use ground truth labels instead of model predictions",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for model predictions (default: 0.25)",
    )

    parser.add_argument(
        "--crop-padding",
        type=int,
        default=10,
        help="Padding around crops in pixels (default: 10)",
    )

    parser.add_argument(
        "--no-crops", action="store_true", help="Don't save individual crops"
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Don't save visualization images",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which dataset split to process (default: all)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )

    args = parser.parse_args()

    # Get image list based on split
    image_list = None
    if args.split != "all":
        txt_path = Path(
            f"data/01_raw/1. DeepDataSet/DetectionDataSet/txt/{args.split}_new.txt"
        )
        if txt_path.exists():
            with open(txt_path, "r") as f:
                image_list = [Path(line.strip()).name for line in f]
                if args.limit:
                    image_list = image_list[: args.limit]
        else:
            print(f"Warning: {txt_path} not found, processing all images")

    # Process images
    process_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path if not args.use_ground_truth else None,
        label_dir=args.label_dir,
        use_ground_truth=args.use_ground_truth,
        conf_threshold=args.conf_threshold,
        save_crops=not args.no_crops,
        save_visualization=not args.no_visualization,
        crop_padding=args.crop_padding,
        image_list=image_list,
    )


if __name__ == "__main__":
    main()
