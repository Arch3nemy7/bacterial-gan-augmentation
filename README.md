# Bacterial Image Analysis - Multi-Project Repository

This repository contains **two separate but related projects** for bacterial image analysis:

1. **StyleGAN2-ADA** - Synthetic bacterial image generation
2. **YOLO Detection** - Bacterial detection and classification

## 🎯 Overview

### StyleGAN2-ADA Project (Main Directory)
Generate synthetic bacterial images using StyleGAN2-ADA to augment limited medical imaging datasets.

- **Purpose**: Data augmentation for bacterial classification
- **Technology**: StyleGAN2 with Adaptive Discriminator Augmentation
- **Output**: Synthetic 256×256 RGB bacterial images
- **Classes**: Gram-positive and Gram-negative bacteria

### YOLO Detection Project (`yolo_detection/`)
Detect and classify bacterial types in microscopy images using YOLOv8.

- **Purpose**: Object detection and classification
- **Technology**: YOLOv8 with 4-class detection
- **Output**: Bounding boxes with bacterial type classification
- **Classes**: negative_cocci, positive_cocci, negative_bacilli, positive_bacilli

## 📂 Repository Structure

```
bacterial-gan-augmentation/
├── yolo_detection/              # ⭐ YOLO Detection Project (SEPARATE)
│   ├── configs/                 # YOLO dataset configs
│   ├── data/                    # Symlink to detection dataset
│   ├── docs/                    # YOLO documentation
│   ├── runs/                    # Training outputs
│   ├── scripts/                 # YOLO training & visualization
│   └── README.md                # YOLO project README
│
├── src/bacterial_gan/           # 🧬 StyleGAN2-ADA Project (MAIN)
│   ├── models/                  # Generator, Discriminator, Losses
│   ├── pipelines/               # Training, evaluation, generation
│   ├── data/                    # Data loading and processing
│   └── cli.py                   # CLI interface
│
├── configs/                     # StyleGAN2-ADA configurations
│   └── config.yaml              # Main training config
├── data/                        # Shared dataset directory
│   ├── 01_raw/                  # Raw images
│   ├── 02_processed/            # Processed patches
│   └── 03_synthetic/            # Generated images
├── app/                         # FastAPI inference server
├── scripts/                     # StyleGAN2-ADA scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
└── README.md                    # This file
```

## 🚀 Quick Start

### StyleGAN2-ADA (Data Augmentation)

```bash
# Install dependencies
poetry install

# Prepare training data
poetry run python scripts/prepare_data.py

# Train GAN
bacterial-gan train --config configs/config.yaml

# Generate synthetic images
bacterial-gan generate-data --run-id <mlflow-run-id> --num-images 1000

# Evaluate quality
bacterial-gan evaluate --run-id <mlflow-run-id>
```

### YOLO Detection (Bacterial Classification)

```bash
# Navigate to YOLO project
cd yolo_detection/

# Install YOLO dependencies (if not already installed)
pip install ultralytics opencv-python tqdm

# Visualize dataset
cd scripts
python visualize_yolo_labels.py --split train --max-images 50

# Train YOLO model
python train_yolo.py --epochs 100 --batch 16

# Results in: yolo_detection/runs/bacteria_detection/
```

## 📊 Datasets

### StyleGAN2-ADA Dataset
- **Location**: `data/01_raw/`
- **Structure**:
  - `gram_positive/` - Gram-positive bacterial images
  - `gram_negative/` - Gram-negative bacterial images
- **Format**: RGB images, various sizes
- **Processing**: Extracted to 256×256 patches

### YOLO Detection Dataset
- **Location**: `data/01_raw/1. DeepDataSet/DetectionDataSet/`
- **Structure**:
  - `images/` - 6,005 bacterial microscopy images
  - `labels/` - YOLO format labels
  - `txt/` - Train/val/test splits
- **Classes**: 4 types (negative/positive × cocci/bacilli)
- **Format**: YOLO detection format

## 🔧 Technologies

### StyleGAN2-ADA
- **Framework**: TensorFlow/Keras
- **Package Management**: Poetry
- **Experiment Tracking**: MLflow
- **Pipeline Management**: DVC
- **API**: FastAPI
- **CLI**: Typer

### YOLO Detection
- **Framework**: Ultralytics YOLOv8
- **Dependencies**: OpenCV, NumPy, tqdm
- **Format**: YOLO detection format

## 📖 Documentation

### StyleGAN2-ADA Docs
- `CLAUDE.md` - Project instructions and architecture
- `QUICK_START.md` - Quick start guide
- `INSTALLATION.md` - Installation instructions
- `docs/architecture.md` - Detailed architecture

### YOLO Detection Docs
- `yolo_detection/README.md` - YOLO project overview
- `yolo_detection/docs/README_YOLO.md` - Detailed usage guide

## 🎯 Use Cases

### Combined Workflow
1. **Train StyleGAN2-ADA** to generate synthetic bacterial images
2. **Augment training dataset** with generated images
3. **Train YOLO detector** on augmented dataset
4. **Deploy YOLO model** for bacterial detection

### Individual Use
- **StyleGAN2-ADA only**: Data augmentation for classification tasks
- **YOLO only**: Bacterial detection and localization

## 🔗 Project Separation

These projects are intentionally separated:

| Aspect | StyleGAN2-ADA | YOLO Detection |
|--------|---------------|----------------|
| **Purpose** | Generate synthetic images | Detect and classify bacteria |
| **Location** | Root directory | `yolo_detection/` |
| **Technology** | TensorFlow, StyleGAN2 | PyTorch, YOLOv8 |
| **Output** | Full images | Bounding boxes |
| **Dataset** | Image-level labels | Object-level annotations |

## 🚧 Development

### StyleGAN2-ADA Development
```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Start API server
make run-api
```

### YOLO Development
```bash
cd yolo_detection/scripts

# Create new splits
python create_yolo_splits.py --train-ratio 0.7

# Visualize predictions
python visualize_yolo_labels.py --split val
```

## 📦 Installation

### Full Installation (Both Projects)
```bash
# Clone repository
git clone <repo-url>
cd bacterial-gan-augmentation

# Install StyleGAN2-ADA
poetry install

# Install YOLO dependencies
pip install ultralytics opencv-python tqdm
```

### Individual Installation

**StyleGAN2-ADA only:**
```bash
poetry install
```

**YOLO only:**
```bash
cd yolo_detection
pip install ultralytics opencv-python tqdm
```

## 🐛 Troubleshooting

### StyleGAN2-ADA Issues
- **OOM Errors**: Use `use_simplified: true` in config
- **Mixed Precision**: Enabled by default for memory efficiency
- **Multi-GPU**: Automatic detection and usage

### YOLO Issues
- **OOM**: Reduce batch size (`--batch 8`)
- **Slow Training**: Use smaller model (`--model yolov8n.pt`)
- **Dataset Not Found**: Check symlink in `yolo_detection/data/`

## 📧 Support

For project-specific issues:
- **StyleGAN2-ADA**: See main documentation
- **YOLO Detection**: See `yolo_detection/README.md`

## 📝 License

See LICENSE file for details.

## ✨ Features

### StyleGAN2-ADA
- ✅ Class-conditional generation
- ✅ Adaptive augmentation (ADA)
- ✅ Mixed precision training
- ✅ Multi-GPU support
- ✅ MLflow experiment tracking
- ✅ FastAPI inference server

### YOLO Detection
- ✅ 4-class bacterial detection
- ✅ Color-coded visualizations
- ✅ Easy-to-use CLI
- ✅ Automated train/val/test splits
- ✅ GPU-accelerated training
- ✅ Real-time inference
