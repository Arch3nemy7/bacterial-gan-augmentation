"""
Constants and enumerations for bacterial GAN augmentation project.

This file should contain:
1. Model hyperparameters defaults
2. File paths and directory structure
3. API endpoints and response codes
4. Evaluation thresholds and metrics
5. Bacterial class definitions
6. Image processing parameters
7. MLflow experiment names
8. Logging configurations
"""

import os
from enum import Enum
from pathlib import Path

# 📁 Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
LOGS_ROOT = PROJECT_ROOT / "logs"
CONFIGS_ROOT = PROJECT_ROOT / "configs"


# 🧬 Bacterial classes
class BacterialClass(Enum):
    """Enumeration for bacterial types in the dataset."""

    GRAM_POSITIVE = "gram_positive"
    GRAM_NEGATIVE = "gram_negative"


# Class labels mapping
CLASS_LABELS = {0: "gram_positive", 1: "gram_negative"}

CLASS_NAMES = ["Gram-Positive", "Gram-Negative"]

# Model parameters
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_BETA1 = 0.5
DEFAULT_EPOCHS = 200

# 📊 Evaluation thresholds
FID_THRESHOLD = 50.0  # Threshold for quality filtering
IS_THRESHOLD = 2.0  # Minimum Inception Score
EXPERT_EVAL_SAMPLE_SIZE = 25  # Number of samples for expert evaluation

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "Bacterial GAN Augmentation"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Default local SQLite

# API configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB max file size

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
MODEL_EXTENSION = ".h5"
CHECKPOINT_EXTENSION = ".ckpt"

# Logging levels
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# Training monitoring
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N epochs
VALIDATION_FREQUENCY = 5  # Validate every N epochs
SAMPLE_GENERATION_FREQUENCY = 10  # Generate samples every N epochs

# ✅ Quality control
MIN_IMAGE_SIZE = (64, 64)
MAX_IMAGE_SIZE = (1024, 1024)
MIN_DATASET_SIZE_PER_CLASS = 100
MAX_DUPLICATE_THRESHOLD = 0.95  # SSIM threshold for duplicate detection
