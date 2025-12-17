"""Memory optimization utilities for training on limited GPU/CPU resources."""

import gc
import os
import tensorflow as tf
from typing import Optional

from bacterial_gan.config import settings


def configure_cpu_parallelism(num_threads: Optional[int] = None) -> None:
    """
    Configure TensorFlow to use CPU cores efficiently for data loading.

    This enables parallel data loading on CPU while GPU trains,
    maximizing hardware utilization.

    Args:
        num_threads: Number of CPU threads to use (None = auto-detect)
    """
    if num_threads is None:
        num_threads = settings.training.memory_optimization.cpu_threads
        if num_threads is None:
            num_threads = os.cpu_count() or 12

    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    print(f"✓ CPU parallelism: {num_threads} threads enabled")
    print(f"  ├─ Data loading will use CPU cores in parallel")
    print(f"  └─ GPU trains while CPU prepares next batch")


def configure_tensorflow_memory(gpu_memory_limit_mb: Optional[int] = None) -> None:
    """
    Configure TensorFlow for optimal memory usage on limited hardware.

    This enables:
    - GPU memory growth (allocates only what's needed) OR
    - Hard memory limit (fixed allocation)

    Note: Memory growth and hard limit are mutually exclusive.

    Args:
        gpu_memory_limit_mb: Optional GPU memory limit in MB (e.g., 3072 for 3GB)
                           If None, uses memory growth instead
    """
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            if gpu_memory_limit_mb is None:
                gpu_memory_limit_mb = settings.training.memory_optimization.gpu_memory_limit_mb

            if gpu_memory_limit_mb:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=gpu_memory_limit_mb
                        )]
                    )
                print(f"✓ GPU memory limit: {gpu_memory_limit_mb} MB ({len(gpus)} GPU(s))")
            elif settings.training.memory_optimization.gpu_memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")

        except RuntimeError as e:
            print(f"⚠️  GPU configuration warning: {e}")
    else:
        print("ℹ️  No GPU detected, using CPU mode")


def enable_xla_compilation(enable: Optional[bool] = None) -> None:
    """
    Enable XLA (Accelerated Linear Algebra) for faster execution.

    XLA compiles TensorFlow operations into optimized machine code,
    which can improve performance BUT increases memory usage during compilation.

    ⚠️ WARNING: XLA can cause OOM on limited GPUs (4GB VRAM) due to:
    - Memory spikes during graph compilation
    - Larger intermediate tensor allocations
    - Pre-allocation of execution buffers

    Args:
        enable: Whether to enable XLA (default: False for safety)
    """
    if enable is None:
        enable = settings.training.memory_optimization.enable_xla

    if enable:
        tf.config.optimizer.set_jit(True)
        print("✓ XLA compilation enabled")
        print("  ⚠️  WARNING: XLA may increase memory usage during training")
    else:
        tf.config.optimizer.set_jit(False)
        print("✓ XLA compilation disabled (memory-safe mode)")


def clear_session() -> None:
    """
    Clear TensorFlow session and run garbage collection.

    Call this between training runs or when switching models
    to free up memory.
    """
    tf.keras.backend.clear_session()
    gc.collect()


def get_memory_info() -> dict:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory statistics
    """
    info = {}

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            info['gpu_current_mb'] = gpu_info['current'] / (1024 * 1024)
            info['gpu_peak_mb'] = gpu_info['peak'] / (1024 * 1024)
        except:
            info['gpu_current_mb'] = 'N/A'
            info['gpu_peak_mb'] = 'N/A'

    return info


def optimize_dataset_pipeline(
    dataset: tf.data.Dataset,
    batch_size: int,
    prefetch_buffer: Optional[int] = None,
    cache_in_memory: Optional[bool] = None,
    cache_filename: Optional[str] = None
) -> tf.data.Dataset:
    """
    Optimize dataset pipeline for memory efficiency with CPU/GPU pipelining.

    Strategy:
    - CPU loads & preprocesses data in parallel
    - Optional caching (memory or disk) for faster epochs
    - GPU trains on batches while CPU prepares next
    - Prefetch keeps GPU fed with data (no idle time)

    ⚠️ CACHING WARNING:
    - In-memory caching can cause OOM with large patch-based datasets
    - Use disk-based caching (cache_filename) for large datasets
    - Or disable caching entirely for very limited memory

    Args:
        dataset: Input TensorFlow dataset
        batch_size: Batch size
        prefetch_buffer: Prefetch buffer size (default: AUTOTUNE)
        cache_in_memory: Cache in RAM (ONLY for small datasets <1GB)
        cache_filename: Path for disk-based cache (recommended for large datasets)

    Returns:
        Optimized dataset with CPU/GPU pipelining
    """
    if prefetch_buffer is None:
        prefetch_buffer = settings.training.memory_optimization.dataset_prefetch_buffer
        if prefetch_buffer == -1:
            prefetch_buffer = tf.data.AUTOTUNE

    if cache_in_memory is None:
        cache_in_memory = settings.training.memory_optimization.dataset_cache_in_memory

    if cache_filename is None:
        cache_filename = settings.training.memory_optimization.dataset_cache_filename

    dataset = dataset.batch(batch_size, drop_remainder=True)

    if cache_filename:
        dataset = dataset.cache(cache_filename)
        print(f"  ✓ Disk-based caching enabled: {cache_filename}")
    elif cache_in_memory:
        dataset = dataset.cache()
        print("  ✓ In-memory caching enabled (ensure dataset fits in RAM)")
        print("  ⚠️  WARNING: May cause OOM with large patch-based datasets")
    else:
        print("  ✓ Caching disabled (memory-safe mode)")

    dataset = dataset.prefetch(buffer_size=prefetch_buffer)

    return dataset


class GradientAccumulator:
    """
    Gradient accumulation for simulating larger batch sizes.

    This allows training with effective batch_size=16 while only computing
    gradients for micro_batch_size=4, reducing memory usage by 4x.

    Example:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for step in range(accumulation_steps):
            with tf.GradientTape() as tape:
                loss = compute_loss(...)
            gradients = tape.gradient(loss, model.trainable_variables)
            accumulator.accumulate(gradients)

        optimizer.apply_gradients(zip(accumulator.gradients, model.trainable_variables))
        accumulator.reset()
    """

    def __init__(self, accumulation_steps: int = 4):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.gradients = None
        self.step = 0

    def accumulate(self, gradients: list) -> None:
        """
        Accumulate gradients from one micro-batch.

        Args:
            gradients: List of gradient tensors
        """
        if self.gradients is None:
            self.gradients = [tf.Variable(tf.zeros_like(g), trainable=False)
                            for g in gradients]

        for i, g in enumerate(gradients):
            if g is not None:
                self.gradients[i].assign_add(g / self.accumulation_steps)

        self.step += 1

    def should_apply(self) -> bool:
        """Check if we should apply accumulated gradients."""
        return self.step >= self.accumulation_steps

    def reset(self) -> None:
        """Reset accumulator for next iteration."""
        if self.gradients is not None:
            for g in self.gradients:
                g.assign(tf.zeros_like(g))
        self.step = 0

    def get_gradients(self) -> list:
        """Get accumulated gradients."""
        return [g.value() for g in self.gradients]
