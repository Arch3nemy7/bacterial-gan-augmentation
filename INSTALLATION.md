# Installation Guide

This guide will help you set up the Bacterial GAN Augmentation project on your system.

## Prerequisites

- **Python 3.11 or higher** (you have Python 3.12.3 ✓)
- **NVIDIA GPU** (GTX 1650 Max Q) with CUDA drivers installed
- **16GB RAM** ✓
- **Git** (for version control)

## Option 1: Install with Poetry (Recommended)

Poetry is a modern Python package manager that handles dependencies more reliably.

### Step 1: Install Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

### Step 2: Install Project Dependencies

```bash
# Navigate to project directory
cd /home/arch3nemy7/Documents/bacterial-gan-augmentation

# Install all dependencies
poetry install

# This will:
# - Create a virtual environment
# - Install all required packages
# - Install the project as a package (bacterial-gan)
```

### Step 3: Activate Environment

```bash
# Option 1: Use poetry shell
poetry shell

# Option 2: Run commands with poetry run
poetry run bacterial-gan --version
```

## Option 2: Install with pip (Alternative)

If you prefer not to use Poetry, you can use pip with the provided `requirements.txt`.

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd /home/arch3nemy7/Documents/bacterial-gan-augmentation

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

## Verify Installation

### Test 1: Check CLI

```bash
# With Poetry
poetry run bacterial-gan --version

# With pip (in activated venv)
bacterial-gan --version

# Expected output: bacterial-gan version 0.1.0
```

### Test 2: Test GAN Architecture

```bash
# Run architecture test
poetry run python scripts/test_architecture.py

# This will:
# - Check GPU availability
# - Build Generator and Discriminator
# - Test forward passes
# - Show model parameters and memory usage
```

### Test 3: Import Package

```bash
# Test Python imports
python3 -c "from bacterial_gan.config import get_settings; print('✓ Config loaded')"
python3 -c "from bacterial_gan.models.architecture import build_generator; print('✓ Architecture loaded')"
```

## Configure TensorFlow for GPU

### Install CUDA (if not already installed)

Your GTX 1650 requires CUDA 11.8 or 12.x:

```bash
# Check if CUDA is installed
nvidia-smi

# If not installed, download from:
# https://developer.nvidia.com/cuda-downloads
```

### Verify GPU Detection

```bash
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Expected output:
# GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Enable Memory Growth (Recommended for GTX 1650)

This is already configured in the test scripts, but you can also set it globally:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## Troubleshooting

### Poetry Not Found

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### TensorFlow GPU Not Detected

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Or specific version
pip install tensorflow==2.15.0
```

### Out of Memory (OOM) Errors

If you get OOM errors during training:

1. Reduce batch size in `configs/config.yaml`:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
   ```

2. Close other GPU applications:
   ```bash
   # Check GPU usage
   nvidia-smi
   ```

3. Enable mixed precision training (covered in training guide)

### Import Errors

If you get import errors like `ModuleNotFoundError: No module named 'bacterial_gan'`:

```bash
# With Poetry
poetry install

# With pip
pip install -e .
```

## Next Steps

Once installation is complete:

1. **Add your dataset**: Place bacterial images in `data/01_raw/`
   ```
   data/01_raw/
   ├── gram_positive/  <- Add images here
   └── gram_negative/  <- Add images here
   ```

2. **Test data pipeline**:
   ```bash
   poetry run python scripts/test_data_pipeline.py
   ```

3. **Start training**:
   ```bash
   bacterial-gan train
   ```

## Need Help?

- Check [CLAUDE.md](CLAUDE.md) for project overview
- Review [README.md](README.md) for usage guide
- See architecture test output for GPU memory recommendations
