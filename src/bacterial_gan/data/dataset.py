"""Dataset utilities for bacterial images using TensorFlow."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from bacterial_gan.config import DataConfig
from bacterial_gan.constants import CLASS_LABELS


class GramStainDataset:
    """Dataset for Gram-stained bacterial images using TensorFlow."""

    def __init__(
        self,
        data_path: str,
        image_size: Tuple[int, int] = (128, 128),
        augment: bool = False,
        split: str = "train",
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data directory containing class subdirectories
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augment = augment
        self.split = split

        # Load image paths and labels
        self.image_paths, self.labels = self._load_image_paths()
        self.num_samples = len(self.image_paths)

        print(f"Loaded {self.num_samples} images for {split} split")

    def _load_image_paths(self) -> Tuple[List[Path], List[int]]:
        """Load all image paths and corresponding labels."""
        image_paths = []
        labels = []

        for class_name, class_idx in CLASS_LABELS.items():
            class_dir = self.data_path / class_name

            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist. Skipping.")
                continue

            # Get all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_idx)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")

        return image_paths, labels

    def _load_and_preprocess_image(self, image_path: str, label: int):
        """Load and preprocess a single image."""
        # Read image file
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize
        image = tf.image.resize(image, self.image_size)

        # Normalize to [-1, 1] for GAN training
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5

        return image, label

    def _augment_image(self, image: tf.Tensor, label: int):
        """Apply data augmentation."""
        if self.augment and self.split == "train":
            # Random flip
            image = tf.image.random_flip_left_right(image)

            # Random rotation (small angles)
            image = tf.image.rot90(
                image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            )

            # Random brightness/contrast
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

            # Clip to valid range
            image = tf.clip_by_value(image, -1.0, 1.0)

        return image, label

    def get_tf_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            tf.data.Dataset yielding (images, labels)
        """
        # Create dataset from paths
        path_ds = tf.data.Dataset.from_tensor_slices(
            ([str(p) for p in self.image_paths], self.labels)
        )

        if shuffle:
            path_ds = path_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

        # Load and preprocess
        dataset = path_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply augmentation
        dataset = dataset.map(
            self._augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def create_datasets(
    data_config: DataConfig,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (128, 128),
) -> Dict[str, tf.data.Dataset]:
    """
    Create train, validation, and test datasets.

    Args:
        data_config: Data configuration
        batch_size: Batch size for all datasets
        image_size: Target image size

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    datasets = {}

    for split in ['train', 'val', 'test']:
        split_path = Path(data_config.processed_data_dir) / split

        if not split_path.exists():
            print(f"Warning: {split_path} does not exist. Skipping {split} split.")
            continue

        dataset = GramStainDataset(
            data_path=str(split_path),
            image_size=image_size,
            augment=(split == 'train'),
            split=split,
        )

        datasets[split] = dataset.get_tf_dataset(
            batch_size=batch_size,
            shuffle=(split == 'train')
        )

    return datasets
