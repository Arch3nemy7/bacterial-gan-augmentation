from ultralytics import YOLO
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on DeepDataSet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    
    args = parser.parse_args()

    config_path = "/home/arch3nemy7/Documents/bacterial-gan-augmentation/configs/yolo_deep_dataset.yaml"
    project_dir = "/home/arch3nemy7/Documents/bacterial-gan-augmentation/runs/mlflow"
    
    print(f"Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt") 
    
    print("Starting training...")
    try:
        model.train(
            data=config_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=project_dir,
            name="bacteria_yolo_model",
            exist_ok=True,
            device=0
        )
        print("Training complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
