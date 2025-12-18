from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def process_hd_images(model_path, source_dir, output_dir, conf=0.25, crop_size=None):
    """
    Apply trained YOLO model to HD images to detect/crop bacteria.
    If crop_size is set (e.g. 128), output crops will be 1:1 square resized to that dimension.
    """
    model = YOLO(model_path)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    crops_dir = output_path / "bacteria_crops"
    vis_dir = output_path / "visualizations"
    crops_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    print(f"Found {len(images)} HD images. Starting inference...")
    
    for img_file in tqdm(images):
        results = model.predict(
            source=str(img_file),
            conf=conf,
            imgsz=1280, 
            save=False,
            verbose=False,
            device=0
        )
        
        result = results[0]
        
        res_plotted = result.plot()
        cv2.imwrite(str(vis_dir / img_file.name), res_plotted)
        
        img_cv = cv2.imread(str(img_file))
        h_img, w_img, _ = img_cv.shape
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf)
            
            target_w = x2 - x1
            target_h = y2 - y1
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            max_dim = max(target_w, target_h)
            
            padding = int(max_dim * 0.1) 
            crop_dim = max_dim + padding
            half_dim = crop_dim // 2
            
            sx1 = max(0, cx - half_dim)
            sy1 = max(0, cy - half_dim)
            sx2 = min(w_img, cx + half_dim)
            sy2 = min(h_img, cy + half_dim)
            
            square_crop = img_cv[sy1:sy2, sx1:sx2]
            
            if square_crop.size == 0: continue
            
            if crop_size:
                final_crop = cv2.resize(square_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            else:
                final_crop = square_crop

            crop_name = f"{img_file.stem}_crop_{i}_conf{conf_score:.2f}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), final_crop)
            
    print(f"Processing complete. Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=128, help="Target output size (square), e.g. 128 or 256")
    
    args = parser.parse_args()
    process_hd_images(args.model, args.source, args.output, args.conf, args.size)

if __name__ == "__main__":
    main()
