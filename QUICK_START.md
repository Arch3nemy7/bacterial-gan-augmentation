# Quick Start Guide

Get your Bacterial cGAN project running in 4 steps!

## Step 1: Install Poetry (1 minute)

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

## Step 2: Install Dependencies (5-10 minutes)

```bash
cd /home/arch3nemy7/Documents/bacterial-gan-augmentation
poetry install
```

## Step 3: Test Architecture (30 seconds)

```bash
poetry run python scripts/test_architecture.py
```

You should see:
- ✅ GPU detected (GTX 1650 Max Q)
- ✅ Generator and Discriminator built
- ✅ ~20M parameters (~2-3GB VRAM)

## Step 4: Add Your Dataset

```bash
# Copy your bacterial images to:
data/01_raw/gram_positive/     ← Put gram-positive images here
data/01_raw/gram_negative/     ← Put gram-negative images here

# At least 500 images per class (JPG or PNG)
```

## Step 5: Test Data Pipeline

```bash
poetry run python scripts/test_data_pipeline.py
```

This will:
- ✅ Validate your images
- ✅ Split into train/val/test (70/15/15)
- ✅ Test TensorFlow loading

---

## What We Built Today (Phases 1-3):

### ✅ Foundation (Phase 1)
- Project configuration (pyproject.toml)
- Proper gitignore
- Fixed Makefile
- Optimized config for GTX 1650

### ✅ Data Pipeline (Phase 2)
- TensorFlow dataset loading
- Data augmentation
- Train/val/test splitting
- Image validation

### ✅ GAN Architecture (Phase 3)
- Conditional Generator (U-Net)
- PatchGAN Discriminator
- Spectral Normalization
- Self-Attention
- WGAN-GP loss functions
- **Optimized for 4GB VRAM**

---

## What's Next:

### Phase 4: Training Pipeline (You need this!)
Without this, you can't train your model yet.

**Will implement:**
- Training loop
- MLflow logging
- Checkpointing
- Image generation during training

**Estimated time: 1-2 weeks to implement**

### Phase 5: Evaluation Metrics
- FID score (image quality)
- Inception Score
- Classification accuracy

### Phase 6: API for Website
- `/generate` endpoint
- `/retrain` endpoint (background jobs)
- Model management
- **This is what your friend's website will use**

---

## Quick Commands Reference:

```bash
# Test architecture (no dataset needed)
poetry run python scripts/test_architecture.py

# Test data pipeline (needs dataset)
poetry run python scripts/test_data_pipeline.py

# Check version
poetry run bacterial-gan --version

# Format code
make format

# Run linter
make lint

# Clean cache
make clean
```

---

## Important Settings for GTX 1650:

In `configs/config.yaml`:
```yaml
training:
  image_size: 128        # Don't increase! GPU memory limited
  batch_size: 8          # Reduce to 4 if you get OOM errors
  epochs: 200            # Standard for GANs
  learning_rate: 0.0002  # Standard for Adam + GANs
```

---

## Need Help?

1. **Installation issues**: See [INSTALLATION.md](INSTALLATION.md)
2. **Project overview**: See [CLAUDE.md](CLAUDE.md)
3. **Implementation status**: See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
4. **Architecture details**: Check `scripts/test_architecture.py` output

---

## Current Status: 37.5% Complete ✅

- ✅ Phase 1: Foundation
- ✅ Phase 2: Data Pipeline
- ✅ Phase 3: GAN Architecture
- ⏳ Phase 4: Training Pipeline (NEXT)
- ⏳ Phase 5: Evaluation
- ⏳ Phase 6: API Development
- ⏳ Phase 7: Testing
- ⏳ Phase 8: Deployment

You can test architecture now, but need Phase 4 to actually train!
