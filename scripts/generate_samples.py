"""
Generate synthetic bacterial images using trained GAN model.

This script:
1. Loads trained generator from MLflow run ID
2. Generates synthetic Gram-positive and Gram-negative images
3. Saves images in a grid layout for visual inspection
4. Saves individual images to data/03_synthetic/
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.models.gan_wrapper import ConditionalGAN
from bacterial_gan.models.architecture import SelfAttention, SpectralNormalization


def load_trained_generator(model_path: str) -> keras.Model:
    """Load trained generator model."""
    # Define custom objects for model loading
    custom_objects = {
        'SelfAttention': SelfAttention,
        'SpectralNormalization': SpectralNormalization
    }

    generator = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✅ Generator loaded from {model_path}")
    return generator


def generate_synthetic_images(
    generator: keras.Model,
    num_samples_per_class: int = 8,
    latent_dim: int = 100
) -> tuple:
    """
    Generate synthetic bacterial images.

    Args:
        generator: Trained generator model
        num_samples_per_class: Number of samples per class
        latent_dim: Latent dimension size

    Returns:
        Tuple of (generated_images, class_labels)
    """
    total_samples = num_samples_per_class * 2  # 2 classes

    # Generate class labels (balanced)
    class_labels = []
    for i in range(2):  # 0: Gram-positive, 1: Gram-negative
        class_labels.extend([i] * num_samples_per_class)
    class_labels = tf.constant(class_labels, dtype=tf.int32)

    # Generate random noise
    noise = tf.random.normal([total_samples, latent_dim])

    # Generate images
    print(f"\nGenerating {total_samples} synthetic bacterial images...")
    generated_images = generator([noise, class_labels], training=False)

    return generated_images.numpy(), class_labels.numpy()


def save_image_grid(images: np.ndarray, labels: np.ndarray, save_path: Path):
    """
    Save generated images in a grid layout.

    Args:
        images: Generated images [N, H, W, C]
        labels: Class labels [N]
        save_path: Path to save the grid image
    """
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    class_names = ["Gram-Positive", "Gram-Negative"]

    for i in range(num_images):
        # Convert from [-1, 1] to [0, 1]
        img = (images[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis('off')

    # Hide empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Image grid saved to {save_path}")
    plt.close()


def save_individual_images(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    run_id: str
):
    """
    Save individual synthetic images.

    Args:
        images: Generated images [N, H, W, C]
        labels: Class labels [N]
        output_dir: Output directory
        run_id: MLflow run ID
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["gram_positive", "gram_negative"]

    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert from [-1, 1] to [0, 255]
        img = (img + 1.0) / 2.0
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Save image
        class_name = class_names[label]
        filename = f"{class_name}_synthetic_{run_id[:8]}_{i:03d}.png"
        filepath = output_dir / filename

        # Use PIL to save
        from PIL import Image
        Image.fromarray(img).save(filepath)

    print(f"✅ {len(images)} individual images saved to {output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic bacterial images")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples per class (default: 8)"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual images to data/03_synthetic/"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC BACTERIAL IMAGE GENERATION")
    print("=" * 80)

    # Load configuration
    config_path = Path("configs/config.yaml")
    settings = get_settings(str(config_path))

    # Locate generator model
    model_path = Path(f"models/{args.run_id}/generator_final.keras")

    if not model_path.exists():
        print(f"\n❌ Error: Generator model not found at {model_path}")
        print(f"   Make sure the run ID is correct: {args.run_id}")
        return

    print(f"\nConfiguration:")
    print(f"  MLflow Run ID: {args.run_id}")
    print(f"  Generator model: {model_path}")
    print(f"  Samples per class: {args.num_samples}")

    # Load generator
    generator = load_trained_generator(str(model_path))

    # Generate synthetic images
    generated_images, labels = generate_synthetic_images(
        generator=generator,
        num_samples_per_class=args.num_samples,
        latent_dim=100
    )

    print(f"\n✅ Generated {len(generated_images)} synthetic images")
    print(f"   Gram-positive: {np.sum(labels == 0)}")
    print(f"   Gram-negative: {np.sum(labels == 1)}")

    # Save grid visualization
    grid_path = Path(f"samples/{args.run_id}/synthetic_samples_grid.png")
    save_image_grid(generated_images, labels, grid_path)

    # Save individual images if requested
    if args.save_individual:
        output_dir = Path("data/03_synthetic")
        save_individual_images(generated_images, labels, output_dir, args.run_id)

    print("\n" + "=" * 80)
    print("✅ SYNTHETIC IMAGE GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated images:")
    print(f"  Grid visualization: {grid_path}")
    if args.save_individual:
        print(f"  Individual images: data/03_synthetic/")

    print(f"\nTo generate more images:")
    print(f"  poetry run python scripts/generate_samples.py --run-id {args.run_id} --num-samples 16")


if __name__ == "__main__":
    main()
