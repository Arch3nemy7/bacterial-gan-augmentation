# src/bacterial_gan/config.py

import pathlib
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# Definisikan path root proyek
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class AppConfig(BaseModel):
    """Konfigurasi untuk aplikasi API."""
    title: str = "Bacterial GAN Augmentation API"
    version: str = "0.1.0"

class DataConfig(BaseModel):
    """Konfigurasi terkait path data."""
    raw_data_dir: pathlib.Path
    processed_data_dir: pathlib.Path
    synthetic_data_dir: pathlib.Path
    expert_testing_set_path: pathlib.Path

class TrainingConfig(BaseModel):
    """Hyperparameter untuk pelatihan model."""
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 0.0002
    beta1: float = 0.5

class EvaluationConfig(BaseModel):
    """Konfigurasi untuk proses evaluasi."""
    num_images_for_expert_panel: int = 25

class Settings(BaseSettings):
    """
    Model konfigurasi utama yang menggabungkan semua sub-konfigurasi.
    Memuat konfigurasi dari file YAML dan dapat ditimpa oleh environment variables.
    """
    app: AppConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',  # Memungkinkan override nested config, e.g., DATA__RAW_DATA_DIR
        case_sensitive=False
    )

def load_config_from_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Memuat file konfigurasi YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan di: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_settings(config_path: str = "configs/config.yaml") -> Settings:
    """
    Fungsi factory untuk membuat instance Settings.
    Ini memuat dari file YAML dan kemudian membiarkan Pydantic
    menangani validasi dan override dari environment variables.
    """
    config_file_path = ROOT_DIR / config_path
    yaml_config = load_config_from_yaml(config_file_path)
    return Settings(**yaml_config)

# Singleton instance untuk digunakan di seluruh aplikasi
# Ini akan dieksekusi saat modul diimpor pertama kali.
# Path default dapat di-override jika diperlukan saat pengujian.
settings = get_settings()