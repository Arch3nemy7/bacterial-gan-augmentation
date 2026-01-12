# YOLO Bacterial Detection Project

This is a standalone YOLO detection project for identifying and classifying bacterial types in microscopy images.

## 🎯 Overview

This project uses YOLOv8 to detect and classify 4 types of bacteria:
- **Class 0**: Gram-negative cocci (RED bounding box)
- **Class 1**: Gram-positive cocci (GREEN bounding box)
- **Class 2**: Gram-negative bacilli (DARK BLUE bounding box)
- **Class 3**: Gram-positive bacilli (LIGHT BLUE bounding box)

## 📂 Project Structure

```
yolo_detection/
├── configs/
│   └── yolo_detection_dataset.yaml    # Dataset configuration
├── data/
│   └── DetectionDataSet/               # Symlink to actual dataset
├── docs/
│   └── README_YOLO.md                  # Detailed documentation
├── runs/
│   └── bacteria_detection/             # Training outputs
└── scripts/
    ├── train_yolo.py                   # Training script
    ├── visualize_yolo_labels.py        # Visualization script
    └── create_yolo_splits.py           # Split generation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install YOLOv8
pip install ultralytics opencv-python tqdm

# Or install all dependencies from parent project
cd ..
poetry install
```

### 2. Visualize Dataset

```bash
cd scripts
python visualize_yolo_labels.py --split train --max-images 50
```

This creates visualized images in `data/DetectionDataSet/visualized_train/`

### 3. Train Model

```bash
cd scripts
python train_yolo.py --epochs 100 --batch 16 --device 0
```

Results are saved to `runs/bacteria_detection/`

### 4. Evaluate Model

YOLOv8 automatically evaluates on the validation set during training.

## 📊 Dataset Statistics

- **Total Images**: 6,005 with labels
- **Training**: 4,203 images (70%)
- **Validation**: 1,201 images (20%)
- **Test**: 601 images (10%)

## 🎨 Color Coding

| Class ID | Name | Color | Description |
|----------|------|-------|-------------|
| 0 | negative_cocci | RED | Gram-negative cocci bacteria |
| 1 | positive_cocci | GREEN | Gram-positive cocci bacteria |
| 2 | negative_bacilli | DARK BLUE | Gram-negative bacilli bacteria |
| 3 | positive_bacilli | LIGHT BLUE | Gram-positive bacilli bacteria |

## 🔧 Training Commands

### Basic Training
```bash
cd scripts
python train_yolo.py
```

### Advanced Training
```bash
cd scripts
python train_yolo.py \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --model yolov8s.pt \
    --device 0
```

### Available Models
- `yolov8n.pt` - Nano (fastest, default)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large (most accurate)

## 📈 Monitoring Training

Training outputs are saved to `runs/bacteria_detection/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training metrics
- `confusion_matrix.png` - Classification performance
- `val_batch*.jpg` - Validation predictions

## 🔍 Visualization

### Visualize All Splits
```bash
cd scripts

# Training set
python visualize_yolo_labels.py --split train

# Validation set
python visualize_yolo_labels.py --split val

# Test set
python visualize_yolo_labels.py --split test
```

### Custom Output Directory
```bash
cd scripts
python visualize_yolo_labels.py \
    --split train \
    --output-dir /path/to/output \
    --max-images 100
```

## 🧪 Inference

After training, use the model for predictions:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/bacteria_detection/weights/best.pt')

# Predict on new images
results = model.predict('path/to/image.jpg')

# Show results
results[0].show()

# Save results
results[0].save('output.jpg')
```

## 📝 Notes

- **GPU Recommended**: Training on CPU is very slow
- **Memory**: Reduce batch size if you encounter OOM errors
- **Image Size**: 640x640 is the default, can be adjusted
- **Label Format**: YOLO format (class x_center y_center width height, normalized)

## 🐛 Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train_yolo.py --batch 8
```

### Slow Training
Use smaller model or reduce image size:
```bash
python train_yolo.py --model yolov8n.pt --imgsz 416
```

### Dataset Not Found
The `data/DetectionDataSet` should be a symlink to the actual dataset. If broken:
```bash
cd data
ln -s "../../data/01_raw/1. DeepDataSet/DetectionDataSet" DetectionDataSet
```

## 📚 Documentation

See `docs/README_YOLO.md` for detailed documentation and advanced usage.

## 🔗 Related Projects

This YOLO detection project is separate from the StyleGAN2-ADA augmentation project. See the parent directory for:
- **StyleGAN2-ADA**: Synthetic bacterial image generation
- **Main Project**: Complete pipeline with both augmentation and detection

## 📧 Support

For issues or questions, refer to the main project documentation or create an issue in the project repository.
