# Bacterial GAN Augmentation

A deep learning project for bacterial image augmentation using **StyleGAN2-ADA** to generate synthetic Gram-positive and Gram-negative bacterial images for improved classification.

## 🎯 Overview

This project uses StyleGAN2-ADA (Adaptive Discriminator Augmentation) to generate realistic synthetic bacterial images, specifically designed for:
- Limited data scenarios common in medical imaging
- Class-conditional generation (Gram-positive vs Gram-negative)
- Resource-constrained training (optimized for <16GB VRAM)

## 🏗️ Architecture

### StyleGAN2-ADA
- **Mapping Network**: Transforms z → w latent space for better disentanglement
- **Synthesis Network**: Style-modulated image generation at 256×256 resolution
- **Discriminator**: With Adaptive Discriminator Augmentation (ADA)
- **Class Conditioning**: Via projection discriminator and class embeddings

### Key Features
- **ADA**: Dynamically adjusts augmentation to prevent discriminator overfitting
- **R1 Regularization**: Gradient penalty for stable training
- **Lazy Regularization**: Efficient computation (R1 every 16 steps)
- **Simplified Mode**: For GPUs with <16GB VRAM

## 🚀 Quick Start

```bash
# Installation
git clone <repository-url>
cd bacterial-gan-augmentation
poetry install

# Prepare data
# Place images in: data/01_raw/gram_positive/ and data/01_raw/gram_negative/
poetry run python scripts/prepare_data.py

# Training
bacterial-gan train

# Generate synthetic data
bacterial-gan generate-data --run-id <mlflow-run-id> --num-images 1000

# Run API
make run-api
```

## 📁 Project Structure

```
bacterial-gan-augmentation/
├── src/bacterial_gan/
│   ├── models/
│   │   ├── stylegan2_ada.py      # Generator & Discriminator
│   │   ├── stylegan2_wrapper.py  # Training wrapper
│   │   └── losses.py             # R1, path length, logistic loss
│   ├── pipelines/
│   │   ├── train_pipeline.py     # Training with MLflow
│   │   ├── evaluate_pipeline.py  # FID, IS, accuracy
│   │   └── generate_data_pipeline.py
│   ├── data/
│   │   ├── dataset.py            # Data loading
│   │   └── data_processing.py    # Patch extraction
│   └── config.py                 # Configuration
├── app/                          # FastAPI application
├── configs/config.yaml          # Training configuration
├── scripts/                      # Utility scripts
└── tests/                        # Unit tests
```

## ⚙️ Configuration

Key settings in `configs/config.yaml`:

```yaml
training:
  use_simplified: true       # For <16GB VRAM
  image_size: 256
  batch_size: 12
  epochs: 300
  learning_rate_g: 0.0002
  learning_rate_d: 0.0002
  
  # Regularization
  r1_gamma: 10.0
  r1_interval: 16
  
  # ADA
  use_ada: true
  ada_target: 0.6
```

## 📊 MLflow Tracking

All training runs are tracked with:
- **Parameters**: Architecture settings, hyperparameters
- **Metrics**: generator_loss, discriminator_loss, r1_penalty, ada_probability
- **Artifacts**: Sample images, checkpoints, final model

View experiments: `mlflow ui`

## 📈 Evaluation Metrics

- **FID Score**: Image quality measurement
- **Inception Score**: Diversity and quality
- **Classification Accuracy**: Downstream task performance

## 🛠️ Development

```bash
# Format code
make format

# Lint
make lint

# Run tests
make test
```

## 📚 References

- [StyleGAN2-ADA Paper](https://arxiv.org/abs/2006.06676)
- [StyleGAN2 Paper](https://arxiv.org/abs/1912.04958)

## 📄 License

MIT License
