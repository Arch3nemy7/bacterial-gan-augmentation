# src/bacterial_gan/config.py

import pathlib
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# Define project root path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class AppConfig(BaseModel):
    """Configuration for API application."""
    title: str = "Bacterial GAN Augmentation API"
    version: str = "0.1.0"

class DataConfig(BaseModel):
    """Configuration for data paths."""
    raw_data_dir: pathlib.Path
    processed_data_dir: pathlib.Path
    synthetic_data_dir: pathlib.Path
    expert_testing_set_path: pathlib.Path

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing with patch extraction."""
    # Patch extraction settings
    use_patch_extraction: bool = True
    image_size: int = 256  # Patch size to extract from high-res images
    apply_augmentation: bool = True  # Apply traditional augmentation (8x multiplier)
    bg_threshold: float = 0.9  # Background filtering threshold (0.9 = >90% white space)
    max_patches_per_split: int | None = None  # Maximum number of patches to generate per split (Train/Val/Test)

    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Macenko normalization parameters
    apply_macenko_normalization: bool = False
    macenko_io: int = 240  # Background light intensity
    macenko_alpha: float = 1.0  # Percentile for angle calculation
    macenko_beta: float = 0.15  # Optical density threshold

class MemoryOptimizationConfig(BaseModel):
    """Configuration for memory optimization."""
    gpu_memory_growth: bool = True
    gpu_memory_limit_mb: int | None = None
    cpu_threads: int | None = 12
    enable_xla: bool = False
    dataset_prefetch_buffer: int = -1
    dataset_cache_in_memory: bool = False
    dataset_cache_filename: str | None = None

class TrainingConfig(BaseModel):
    """Hyperparameters for model training."""
    image_size: int = 256
    num_classes: int = 2
    channels: int = 3
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.9
    latent_dim: int = 100
    loss_type: str = "wgan-gp"
    n_critic: int = 5
    lambda_gp: float = 10.0
    use_mixed_precision: bool = True
    memory_optimization: MemoryOptimizationConfig = MemoryOptimizationConfig()
    checkpoint_interval: int = 50
    sample_interval: int = 10
    num_samples_during_training: int = 4
    num_samples_final: int = 8
    num_samples_grid: int = 16
    models_output_dir: str = "models"
    samples_output_dir: str = "samples"
    mlflow_experiment_name: str = "Bacterial GAN Augmentation"
    mlflow_tracking_uri: str = ""
    dummy_num_batches: int = 10

class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    num_images_for_expert_panel: int = 25

class Settings(BaseSettings):
    """
    Main configuration model that combines all sub-configurations.
    Loads configuration from YAML file and can be overridden by environment variables.
    """
    app: AppConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',  # Allow nested config override, e.g., DATA__RAW_DATA_DIR
        case_sensitive=False
    )

def load_config_from_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_settings(config_path: str = "configs/config.yaml") -> Settings:
    """
    Factory function to create a Settings instance.
    Loads from YAML file and lets Pydantic handle validation
    and environment variable overrides.
    """
    config_file_path = ROOT_DIR / config_path
    yaml_config = load_config_from_yaml(config_file_path)
    return Settings(**yaml_config)

# Singleton instance for use throughout the application
# This is executed when the module is first imported
# Default path can be overridden if needed during testing
settings = get_settings()