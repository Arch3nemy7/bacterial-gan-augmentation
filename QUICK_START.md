# Quick Start Guide

Get your Bacterial StyleGAN2-ADA project running quickly!

## Step 1: Install Dependencies

```bash
cd bacterial-gan-augmentation
poetry install
```

## Step 2: Test Architecture

```bash
poetry run python tests/test_architecture.py
```

You should see:
- ✅ Generator and Discriminator built
- ✅ ~13M parameters total
- ✅ Sample generation working

## Step 3: Add Your Dataset

```bash
# Copy your bacterial images to:
data/01_raw/gram_positive/     # Gram-positive images
data/01_raw/gram_negative/     # Gram-negative images

# At least 500 images per class (JPG, PNG, or TIFF)
```

## Step 4: Prepare Data

```bash
poetry run python scripts/prepare_data.py
```

This will:
- Extract patches from high-resolution images
- Apply augmentation (8x multiplier)
- Split into train/val/test (70/15/15)

## Step 5: Train

```bash
bacterial-gan train
```

View progress: `mlflow ui` → http://localhost:5000

---

## Quick Commands

```bash
# Test architecture
poetry run python tests/test_architecture.py

# Test training (dummy data)
poetry run python scripts/test_training.py

# Prepare dataset
poetry run python scripts/prepare_data.py

# Train
bacterial-gan train

# Generate images
bacterial-gan generate-data --run-id <RUN_ID> --num-images 1000

# View experiments
mlflow ui
```

---

## Architecture: StyleGAN2-ADA

- **Generator**: Mapping Network + Synthesis Network with style modulation
- **Discriminator**: With Adaptive Discriminator Augmentation (ADA)
- **Loss**: Non-saturating logistic + R1 regularization
- **Class Conditioning**: Gram-positive/Gram-negative via embeddings

### Key Settings

```yaml
training:
  use_simplified: true    # For <16GB VRAM
  image_size: 256
  batch_size: 12
  r1_gamma: 10.0
  use_ada: true
```

---

## Project Status

- ✅ StyleGAN2-ADA Architecture
- ✅ Data Pipeline (patch extraction, augmentation)
- ✅ Training Pipeline (MLflow, checkpointing)
- ⏳ Evaluation Metrics (FID, IS)
- ⏳ API Development
