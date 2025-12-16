"""Configuration settings for bacterial GAN augmentation."""

import pathlib
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


class AppConfig(BaseModel):
    """API application settings."""

    title: str = "Bacterial GAN Augmentation API"
    version: str = "0.1.0"


class DataConfig(BaseModel):
    """Data paths configuration."""

    raw_data_dir: pathlib.Path
    processed_data_dir: pathlib.Path
    synthetic_data_dir: pathlib.Path
    expert_testing_set_path: pathlib.Path


class PreprocessingConfig(BaseModel):
    """Preprocessing settings."""

    use_patch_extraction: bool = True
    crop_mode: str = "resize"  # 'resize', 'center', or 'random' (when patch extraction disabled)
    image_size: int = 256
    apply_augmentation: bool = True
    bg_threshold: float = 0.9
    max_patches_per_split: int | None = None

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


class MemoryOptimizationConfig(BaseModel):
    """Memory optimization settings."""

    gpu_memory_growth: bool = True
    gpu_memory_limit_mb: int | None = None
    cpu_threads: int | None = 12
    enable_xla: bool = False
    dataset_prefetch_buffer: int = -1
    dataset_cache_in_memory: bool = False
    dataset_cache_filename: str | None = None


class TrainingConfig(BaseModel):
    """StyleGAN2-ADA training configuration."""

    # Architecture
    use_simplified: bool = True
    image_size: int = 256
    num_classes: int = 2
    channels: int = 3
    latent_dim: int = 256

    # Training
    batch_size: int = 16
    epochs: int = 200
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    beta1: float = 0.0
    beta2: float = 0.99
    n_critic: int = 1

    # Loss
    loss_type: str = "stylegan2"

    # R1 Regularization
    r1_gamma: float = 10.0
    r1_interval: int = 16

    # Path Length Regularization
    pl_weight: float = 2.0
    pl_interval: int = 4

    # Adaptive Discriminator Augmentation
    use_ada: bool = True
    ada_target: float = 0.6

    # Performance
    use_mixed_precision: bool = True
    memory_optimization: MemoryOptimizationConfig = MemoryOptimizationConfig()

    # Checkpointing
    checkpoint_interval: int = 50
    sample_interval: int = 10
    num_samples_during_training: int = 4
    num_samples_final: int = 8
    num_samples_grid: int = 16
    models_output_dir: str = "models"
    samples_output_dir: str = "samples"

    # MLflow
    mlflow_experiment_name: str = "Bacterial GAN Augmentation"
    mlflow_tracking_uri: str = ""

    # Testing
    dummy_num_batches: int = 10


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    num_images_for_expert_panel: int = 25


class Settings(BaseSettings):
    """Main configuration combining all sub-configurations."""

    app: AppConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    model_config = SettingsConfigDict(env_nested_delimiter="__", case_sensitive=False)


def load_config_from_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_settings(config_path: str = "configs/config.yaml") -> Settings:
    """Create Settings instance from YAML file."""
    config_file_path = ROOT_DIR / config_path
    yaml_config = load_config_from_yaml(config_file_path)
    return Settings(**yaml_config)


settings = get_settings()
