"""Dataset utilities for bacterial images using TensorFlow."""

from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf

from bacterial_gan.config import DataConfig
from bacterial_gan.constants import CLASS_LABELS


class GramStainDataset:
    """Dataset for Gram-stained bacterial images using TensorFlow."""

    def __init__(
        self,
        data_path: str,
        augment: bool = False,
        split: str = "train",
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data directory containing class subdirectories
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.augment = augment
        self.split = split

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

            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_idx)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")

        return image_paths, labels

    def _load_and_preprocess_image(self, image_path: str, label: int):
        """Load and preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image = (tf.cast(image, tf.float32) - 127.5) / 127.5

        return image, label

    def _augment_image(self, image: tf.Tensor, label: int):
        """
        Apply geometric augmentation for bacterial images.

        NOTE: Color augmentation is deliberately EXCLUDED because Gram stain colors are
        diagnostic (pink=gram-positive, blue=gram-negative). Changing colors would
        confuse the GAN about correct stain appearance.
        """
        if self.augment and self.split == "train":
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            image = tf.image.rot90(
                image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            )

            zoom_factor = tf.random.uniform([], minval=0.85, maxval=1.15)
            image_shape = tf.shape(image)
            h, w = image_shape[0], image_shape[1]

            new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)

            image = tf.image.resize(image, [new_h, new_w], method='bilinear')
            image = tf.image.resize_with_crop_or_pad(image, h, w)

            max_shift = 10
            shift_h = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
            shift_w = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
            image = tf.roll(image, shift=[shift_h, shift_w], axis=[0, 1])

            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
            image = image + noise

            image = tf.clip_by_value(image, -1.0, 1.0)

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
            datasets_by_class = []

            unique_labels = sorted(set(self.labels))

            for class_label in unique_labels:
                class_indices = [i for i, label in enumerate(self.labels) if label == class_label]
                class_paths = [self.image_paths[i] for i in class_indices]
                class_labels_list = [self.labels[i] for i in class_indices]

                class_ds = tf.data.Dataset.from_tensor_slices(
                    ([str(p) for p in class_paths], class_labels_list)
                )

                class_ds = class_ds.map(
                    self._load_and_preprocess_image,
                    num_parallel_calls=tf.data.AUTOTUNE
                )

                class_ds = class_ds.cache()
                if shuffle:
                    class_ds = class_ds.shuffle(buffer_size=len(class_paths), reshuffle_each_iteration=True)

                class_ds = class_ds.repeat()

                class_ds = class_ds.map(
                    self._augment_image,
                    num_parallel_calls=tf.data.AUTOTUNE
                )

                datasets_by_class.append(class_ds)

            dataset = tf.data.Dataset.sample_from_datasets(
                datasets_by_class,
                weights=[1.0 / len(datasets_by_class)] * len(datasets_by_class)
            )

            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        else:
            path_ds = tf.data.Dataset.from_tensor_slices(
                ([str(p) for p in self.image_paths], self.labels)
            )

            dataset = path_ds.map(
                self._load_and_preprocess_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )

            dataset = dataset.cache()

            if shuffle:
                dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )

            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def create_datasets(
    data_config: DataConfig,
    batch_size: int = 32,
) -> Dict[str, tf.data.Dataset]:
    """
    Create train, validation, and test datasets.

    Args:
        data_config: Data configuration
        batch_size: Batch size for all datasets

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
            augment=(split == 'train'),
            split=split,
        )

        datasets[split] = dataset.get_tf_dataset(
            batch_size=batch_size,
            shuffle=(split == 'train'),
            balance_classes=(split == 'train')
        )

    return datasets
