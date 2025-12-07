"""Utility functions and memory optimization."""

from bacterial_gan.utils.memory_optimization import (
    configure_tensorflow_memory,
    enable_xla_compilation,
    clear_session,
    get_memory_info,
    optimize_dataset_pipeline,
    GradientAccumulator,
)
from bacterial_gan.utils.normalization import (
    normalize_macenko,
    MacenkoNormalize,
)
from bacterial_gan.utils.helpers import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    calculate_fid_score,
    calculate_is_score,
    visualize_training_progress,
)

__all__ = [
    'configure_tensorflow_memory',
    'enable_xla_compilation',
    'clear_session',
    'get_memory_info',
    'optimize_dataset_pipeline',
    'GradientAccumulator',
    'normalize_macenko',
    'MacenkoNormalize',
    'setup_logging',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_fid_score',
    'calculate_is_score',
    'visualize_training_progress',
]
