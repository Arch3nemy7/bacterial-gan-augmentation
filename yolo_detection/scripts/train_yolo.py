import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on DetectionDataSet")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model variant (n/s/m/l/x)",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Config and output paths relative to YOLO project root
    config_path = project_root / "configs" / "yolo_detection_dataset.yaml"
    project_dir = project_root / "runs"

    print(f"Loading YOLOv8 model: {args.model}")
    print(f"Dataset config: {config_path}")
    print(
        f"Classes: 0=negative_cocci, 1=positive_cocci, 2=negative_bacilli, 3=positive_bacilli"
    )

    model = YOLO(args.model)

    print(f"\nStarting training for {args.epochs} epochs...")
    try:
        results = model.train(
            data=str(config_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(project_dir),
            name="bacteria_detection",
            exist_ok=True,
            device=args.device,
            patience=50,
            save=True,
            plots=True,
            verbose=True,
        )
        print("\nTraining complete!")
        print(f"Results saved to: {project_dir}/bacteria_detection")

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
