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

        for class_idx, class_name in CLASS_LABELS.items():
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
        """
        Apply aggressive geometric augmentation for bacterial images.

        NOTE: Color augmentation is deliberately EXCLUDED because Gram stain colors are
        diagnostic (pink=gram-positive, blue=gram-negative). Changing colors would
        confuse the GAN about correct stain appearance.

        Implements geometric augmentations to maximize dataset diversity:
        - Geometric: flips, 90° rotations, zoom/crop
        - Noise: Gaussian noise to simulate imaging artifacts (preserves color)
        """
        if self.augment and self.split == "train":
            # === GEOMETRIC AUGMENTATIONS ===

            # Random horizontal and vertical flips
            # Bacteria can appear in any orientation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            # Random 90° rotation (0, 90, 180, 270 degrees)
            # Simulates different microscope orientations
            image = tf.image.rot90(
                image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            )

            # Random zoom (0.85x to 1.15x) via resize + crop/pad
            # Simulates different magnification levels and focal planes
            zoom_factor = tf.random.uniform([], minval=0.85, maxval=1.15)
            image_shape = tf.shape(image)
            h, w = image_shape[0], image_shape[1]

            # Calculate new size after zoom
            new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)

            # Resize then crop or pad back to original size
            image = tf.image.resize(image, [new_h, new_w], method='bilinear')
            image = tf.image.resize_with_crop_or_pad(image, h, w)

            # Random translation (shift image up/down/left/right)
            # Simulates different cell positions in field of view
            max_shift = 10  # pixels
            shift_h = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
            shift_w = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
            image = tf.roll(image, shift=[shift_h, shift_w], axis=[0, 1])

            # === NOISE AUGMENTATION ===

            # Add Gaussian noise to simulate imaging artifacts
            # stddev=0.02 in [-1,1] space ≈ 2.5 intensity units in [0,255]
            # This preserves color while adding realistic microscopy noise
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
            image = image + noise

            # Clip to valid range [-1, 1]
            image = tf.clip_by_value(image, -1.0, 1.0)

            # NOTE: For continuous rotation (0-360°), install tensorflow-addons:
            # import tensorflow_addons as tfa
            # angle = tf.random.uniform([], minval=0, maxval=2*np.pi)
            # image = tfa.image.rotate(image, angle, interpolation='bilinear')

        return image, label

    def get_tf_dataset(self, batch_size: int = 32, shuffle: bool = True, balance_classes: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with optional class balancing.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            balance_classes: If True, ensures balanced class distribution in batches (for training)

        Returns:
            tf.data.Dataset yielding (images, labels)
        """
        if balance_classes and self.split == "train":
            # Create class-balanced dataset by interleaving class-specific datasets
            datasets_by_class = []

            # Get unique classes
            unique_labels = sorted(set(self.labels))

            for class_label in unique_labels:
                # Filter paths for this class
                class_indices = [i for i, label in enumerate(self.labels) if label == class_label]
                class_paths = [self.image_paths[i] for i in class_indices]
                class_labels_list = [self.labels[i] for i in class_indices]

                # Create dataset for this class
                class_ds = tf.data.Dataset.from_tensor_slices(
                    ([str(p) for p in class_paths], class_labels_list)
                )

                # Load and preprocess
                class_ds = class_ds.map(
                    self._load_and_preprocess_image,
                    num_parallel_calls=tf.data.AUTOTUNE
                )

                # Cache and shuffle
                class_ds = class_ds.cache()
                if shuffle:
                    class_ds = class_ds.shuffle(buffer_size=len(class_paths), reshuffle_each_iteration=True)

                # Repeat to ensure enough samples
                class_ds = class_ds.repeat()

                # Apply augmentation
                class_ds = class_ds.map(
                    self._augment_image,
                    num_parallel_calls=tf.data.AUTOTUNE
                )

                datasets_by_class.append(class_ds)

            # Interleave datasets to create balanced batches
            # Each batch will have equal samples from each class
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets_by_class,
                weights=[1.0 / len(datasets_by_class)] * len(datasets_by_class)
            )

            # Batch and prefetch
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        else:
            # Original unbalanced dataset creation
            # Create dataset from paths
            path_ds = tf.data.Dataset.from_tensor_slices(
                ([str(p) for p in self.image_paths], self.labels)
            )

            # Load and preprocess (Resize, Normalize)
            dataset = path_ds.map(
                self._load_and_preprocess_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Cache dataset in memory for faster training
            # This avoids reading/decoding images from disk every epoch
            dataset = dataset.cache()

            if shuffle:
                dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

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
            shuffle=(split == 'train'),
            balance_classes=(split == 'train')  # Enable class balancing for training
        )

    return datasets
