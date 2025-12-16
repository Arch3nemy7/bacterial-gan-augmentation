# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for bacterial image augmentation using StyleGAN2-ADA. The goal is to generate synthetic bacterial images (Gram-positive and Gram-negative) to augment limited training datasets and improve bacterial classification models.

**Tech Stack:**
- Python 3.11 with Poetry for dependency management
- TensorFlow/Keras for model implementation
- DVC for pipeline management
- MLflow for experiment tracking
- FastAPI for inference API
- Typer for CLI

## Common Commands

```bash
# Setup
make install

# Training
bacterial-gan train
bacterial-gan train --config-path configs/config.yaml

# Generation
bacterial-gan generate-data --run-id <mlflow-run-id> --num-images 1000

# Evaluation
bacterial-gan evaluate --run-id <mlflow-run-id>

# Code quality
make format
make lint
make test

# API
make run-api
```

## Architecture

### StyleGAN2-ADA
- **Mapping Network**: z → w latent space transformation
- **Synthesis Network**: Style-modulated image generation
- **Discriminator**: With Adaptive Discriminator Augmentation (ADA)
- **Class Conditioning**: Gram-positive/Gram-negative via embeddings
- **Image Size**: 256x256 RGB

### Key Features
- **ADA**: Prevents discriminator overfitting on limited data
- **R1 Regularization**: Gradient penalty for stable training
- **Path Length Regularization**: Smooth latent space mapping
- **Lazy Regularization**: Efficient computation (R1 every 16 steps)
- **Simplified Mode**: Resource-constrained training (<16GB VRAM)

## Project Structure

```
src/bacterial_gan/
├── models/
│   ├── stylegan2_ada.py     # Generator + Discriminator architecture
│   ├── stylegan2_wrapper.py # Training wrapper
│   └── losses.py            # R1, path length, logistic loss
├── pipelines/
│   ├── train_pipeline.py    # Training with MLflow
│   ├── evaluate_pipeline.py # FID, IS, accuracy
│   └── generate_data_pipeline.py
├── data/
│   ├── dataset.py           # Data loading
│   └── data_processing.py   # Patch extraction, augmentation
├── config.py                # Pydantic settings
└── cli.py                   # Typer CLI

app/                         # FastAPI application
configs/config.yaml          # Configuration
```

## Configuration

```yaml
training:
  # Architecture
  use_simplified: true       # For <16GB VRAM
  latent_dim: 256
  
  # Training
  batch_size: 12
  epochs: 300
  learning_rate_g: 0.0002
  learning_rate_d: 0.0002
  
  # Regularization
  r1_gamma: 10.0
  r1_interval: 16
  pl_weight: 2.0
  pl_interval: 4
  
  # ADA
  use_ada: true
  ada_target: 0.6
```

## MLflow

Training runs are tracked with:
- Parameters: All training config
- Metrics: generator_loss, discriminator_loss, r1_penalty, ada_probability
- Artifacts: Sample images, checkpoints, final model

Use run ID for evaluation and generation.

## Development

1. **Pre-commit**: black, isort, flake8
2. **DVC Pipeline**: `dvc.yaml` defines train/generate stages
3. **Package**: Import as `bacterial_gan`, CLI as `bacterial-gan`
