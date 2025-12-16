"""Utility functions and memory optimization."""

from bacterial_gan.utils.helpers import (
    calculate_fid_score,
    calculate_is_score,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
    visualize_training_progress,
)
from bacterial_gan.utils.memory_optimization import (
    GradientAccumulator,
    clear_session,
    configure_tensorflow_memory,
    enable_xla_compilation,
    get_memory_info,
    optimize_dataset_pipeline,
)

__all__ = [
    "configure_tensorflow_memory",
    "enable_xla_compilation",
    "clear_session",
    "get_memory_info",
    "optimize_dataset_pipeline",
    "GradientAccumulator",
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint",
    "calculate_fid_score",
    "calculate_is_score",
    "visualize_training_progress",
]
