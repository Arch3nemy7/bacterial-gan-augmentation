"""
Generate synthetic bacterial images using trained StyleGAN2-ADA model.

Usage:
    poetry run python scripts/generate_samples.py --run-id <RUN_ID> --num-samples 8
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.models.stylegan2_wrapper import StyleGAN2ADA


def generate_from_checkpoint(
    checkpoint_path: str,
    num_samples_per_class: int = 8,
    latent_dim: int = 256,
    image_size: int = 256,
) -> tuple:
    """Generate images by loading checkpoint."""
    gan = StyleGAN2ADA(
        latent_dim=latent_dim,
        num_classes=2,
        image_size=image_size,
        use_simplified=True,
        use_ada=True,
        use_mixed_precision=False,
    )

    gan.load_checkpoint(checkpoint_path)

    total_samples = num_samples_per_class * 2
    class_labels = []
    for i in range(2):
        class_labels.extend([i] * num_samples_per_class)
    class_labels = tf.constant(class_labels, dtype=tf.int32)

    print(f"\nGenerating {total_samples} synthetic images...")
    images = gan.generate_samples(class_labels, total_samples)

    return images, class_labels.numpy()


def save_image_grid(images: np.ndarray, labels: np.ndarray, save_path: Path):
    """Save images in a grid layout."""
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    class_names = ["Gram-Positive", "Gram-Negative"]

    for i in range(num_images):
        img = (images[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Grid saved: {save_path}")
    plt.close()


def save_individual_images(images: np.ndarray, labels: np.ndarray, output_dir: Path, run_id: str):
    """Save individual synthetic images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["gram_positive", "gram_negative"]

    for i, (img, label) in enumerate(zip(images, labels)):
        img = (img + 1.0) / 2.0
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        class_name = class_names[label]
        filename = f"{class_name}_synthetic_{run_id[:8]}_{i:03d}.png"
        Image.fromarray(img).save(output_dir / filename)

    print(f"✅ {len(images)} images saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bacterial images")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--num-samples", type=int, default=8, help="Samples per class (default: 8)")
    parser.add_argument("--save-individual", action="store_true", help="Save individual images")
    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC BACTERIAL IMAGE GENERATION")
    print("=" * 80)

    settings = get_settings("configs/config.yaml")

    # Find checkpoint
    checkpoint_path = Path(f"models/{args.run_id}/checkpoint_epoch_0300.npy")
    if not checkpoint_path.exists():
        # Try to find latest checkpoint
        checkpoints = sorted(Path(f"models/{args.run_id}").glob("checkpoint_*.npy"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            print(f"❌ No checkpoints found in models/{args.run_id}/")
            return

    print(f"\nConfiguration:")
    print(f"  Run ID: {args.run_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples per class: {args.num_samples}")

    images, labels = generate_from_checkpoint(
        str(checkpoint_path),
        num_samples_per_class=args.num_samples,
        latent_dim=settings.training.latent_dim,
        image_size=settings.training.image_size,
    )

    print(f"\n✅ Generated {len(images)} images")
    print(f"   Gram-positive: {np.sum(labels == 0)}")
    print(f"   Gram-negative: {np.sum(labels == 1)}")

    grid_path = Path(f"samples/{args.run_id}/synthetic_grid.png")
    save_image_grid(images, labels, grid_path)

    if args.save_individual:
        save_individual_images(images, labels, Path("data/03_synthetic"), args.run_id)

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
