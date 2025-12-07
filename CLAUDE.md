# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for bacterial image augmentation using Conditional Generative Adversarial Networks (cGAN). The goal is to generate synthetic bacterial images (Gram-positive and Gram-negative) to augment limited training datasets and improve bacterial classification models.

**Tech Stack:**
- Python 3.11 with Poetry for dependency management
- TensorFlow/Keras for model implementation
- DVC (Data Version Control) for pipeline management
- MLflow for experiment tracking and model registry
- FastAPI for inference API
- Typer for CLI interface

## Common Development Commands

**Setup and Installation:**
```bash
make install                    # Install dependencies using Poetry
poetry install                  # Alternative to make install
```

**Code Quality:**
```bash
make format                     # Format code with black and isort
make lint                       # Run flake8 linter
poetry run black .              # Format specific files
poetry run isort .              # Sort imports
poetry run flake8 src/ app/ tests/  # Lint specific directories
```

**Testing:**
```bash
make test                       # Run all tests with pytest
poetry run pytest               # Alternative
poetry run pytest tests/test_data_processing.py  # Run specific test file
poetry run pytest -k test_name  # Run specific test by name
```

**Training and Pipeline:**
```bash
make train                      # Run training pipeline via DVC
poetry run dvc repro train      # Alternative
bacterial-gan train             # Direct CLI call
bacterial-gan train --config-path configs/config.yaml  # With custom config
```

**Data Generation:**
```bash
make generate-data              # Run generation pipeline via DVC
bacterial-gan generate-data --run-id <mlflow-run-id> --num-images 1000
```

**Model Evaluation:**
```bash
bacterial-gan evaluate --run-id <mlflow-run-id>
```

**API Server:**
```bash
make run-api                    # Run FastAPI server on localhost:8000
poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Cleanup:**
```bash
make clean                      # Remove cache files and __pycache__
```

## Architecture Overview

### GAN Architecture
- **Generator**: ResNet based conditional generator that takes noise + class label and produces synthetic bacterial images
- **Discriminator**: PatchGAN discriminator that focuses on local texture patterns and class conditioning
- **Model Type**: Conditional GAN (cGAN) for class-specific image generation
- **Image Size**: 256x256 pixels, 3 channels (RGB)
- **Classes**: 2 (Gram-positive and Gram-negative bacteria)

### Key Components

**Data Pipeline (`src/data/`):**
- `data_processing.py`: Macenko color normalization for stain-invariant preprocessing
- `dataset.py`: Dataset loading and augmentation utilities
- Data directories: `data/01_raw/`, `data/02_processed/`, `data/03_synthetic/`

**Models (`src/models/`):**
- `architecture.py`: Generator and Discriminator definitions (ResNet, PatchGAN)
  - Supports custom layers: SpectralNormalization, SelfAttention
  - Multiple loss functions: adversarial (WGAN-GP/LSGAN), reconstruction (L1/L2), perceptual (VGG-based)
- `gan_wrapper.py`: High-level GAN wrapper for training and inference

**Pipelines (`src/pipelines/`):**
- `train_pipeline.py`: Full training loop with MLflow tracking
- `evaluate_pipeline.py`: Model evaluation (FID, IS, classification accuracy)
- `generate_data_pipeline.py`: Synthetic data generation from trained models

**API (`app/`):**
- `main.py`: FastAPI application entry point
- `api/v1/endpoints.py`: API endpoints for inference and model management
- `api/v1/schemas.py`: Request/response schemas
- `core/dependencies.py`: Dependency injection (model registry, etc.)

**CLI (`src/cli.py`):**
- Typer-based CLI with commands: train, evaluate, generate-data
- All commands support `--config-path` for custom configurations

### Configuration System

Configuration is centralized in `configs/config.yaml`:
- `app`: API settings (title, version)
- `data`: Data directory paths
- `preprocessing`: Image size, channels
- `model`: Architecture selection (ResNet, PatchGAN, cGAN)
- `training`: Optimizer, learning rate, batch size, epochs

Access via `src/config.py` using `get_settings(config_path)`.

### MLflow Integration

All training runs are tracked in MLflow:
- Experiment name: "Bacterial GAN Augmentation"
- Logged parameters: training config, data config
- Logged metrics: generator_loss, discriminator_loss (per epoch)
- Logged artifacts: Generated sample images, model checkpoints
- Model Registry: Models registered as "bacterial-gan-generator"
- Run IDs are required for evaluation and data generation

To access trained models, use the MLflow run ID from training output.

## Development Workflow

1. **Pre-commit Hooks**: Automatically run on git commit
   - Trailing whitespace removal
   - End-of-file fixer
   - YAML syntax checking
   - Large file detection
   - isort (with black profile)
   - black formatting
   - flake8 linting

2. **Code Style**: This project follows black formatting with isort for imports (black profile)

3. **DVC Pipeline**: Training and generation pipelines are managed by DVC (see `dvc.yaml`)
   - Stage names: `train`, `generate`
   - Use `dvc repro <stage>` to reproduce pipelines

4. **Model Storage**: Trained models are stored in `models/` directory
   - Example: `models/model_cgan_hasil_training.h5`
   - MLflow also stores models in its artifact store

5. **Module Import**: The project uses package name `bacterial_gan` (with underscore)
   - Import from modules as: `from bacterial_gan.config import get_settings`
   - CLI is accessible as: `bacterial-gan` (with hyphen)

## Key Implementation Notes

- **Macenko Normalization**: Critical preprocessing step for bacterial images to normalize stain variations
- **Class Conditioning**: GAN is conditioned on bacterial class (Gram-positive vs Gram-negative)
- **Evaluation Metrics**: Use FID (Fréchet Inception Distance), IS (Inception Score), and classification accuracy
- **Training Stability**: Consider using spectral normalization and gradient penalty for stable GAN training
- **API Design**: Designed to support model inference, model management, health checks, file uploads, and real-time progress tracking

## File Locations

- Source code: `src/`
- API code: `app/`
- Tests: `tests/`
- Configs: `configs/`
- Training scripts: `scripts/`
- Notebooks: `notebooks/`
- Documentation: `docs/` (currently sparse)
- Models: `models/`
- Data: `data/` (organized as 01_raw, 02_processed, 03_synthetic)
