"""Training pipeline with MLflow integration."""

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import time
from tqdm import tqdm

from bacterial_gan.config import Settings
from bacterial_gan.models.gan_wrapper import ConditionalGAN
from bacterial_gan.data.dataset import GramStainDataset


def save_sample_images(
    gan: ConditionalGAN,
    epoch: int,
    save_dir: Path,
    num_samples: int = 16
):
    """
    Generate and save sample images during training.

    Args:
        gan: ConditionalGAN instance
        epoch: Current epoch number
        save_dir: Directory to save images
        num_samples: Number of samples to generate
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples (8 from each class)
    samples_per_class = num_samples // gan.num_classes
    class_labels = []
    for i in range(gan.num_classes):
        class_labels.extend([i] * samples_per_class)
    class_labels = tf.constant(class_labels, dtype=tf.int32)

    # Generate images
    generated_images = gan.generate_samples(class_labels, num_samples)

    # Denormalize from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1.0) / 2.0
    generated_images = np.clip(generated_images, 0.0, 1.0)

    # Convert to float32 for matplotlib compatibility (mixed precision outputs float16)
    generated_images = generated_images.astype(np.float32)

    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    class_names = ["Gram-Positive", "Gram-Negative"]

    for i, (ax, img) in enumerate(zip(axes, generated_images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{class_names[class_labels[i]]}", fontsize=8)

    plt.tight_layout()
    filepath = save_dir / f"epoch_{epoch:04d}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    return str(filepath)


def run(settings: Settings, resume_from_checkpoint: Optional[str] = None):
    """
    Run complete training pipeline with MLflow tracking.

    Args:
        settings: Configuration settings
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
    """
    print("=" * 80)
    print("BACTERIAL GAN TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Set MLflow experiment
    mlflow.set_experiment("Bacterial GAN Augmentation")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print()

        # Log configuration parameters
        print("Logging configuration to MLflow...")
        mlflow.log_params({
            "image_size": settings.training.image_size,
            "batch_size": settings.training.batch_size,
            "epochs": settings.training.epochs,
            "learning_rate": settings.training.learning_rate,
            "beta1": settings.training.beta1,
            "latent_dim": settings.training.latent_dim,
            "loss_type": settings.training.loss_type,
            "n_critic": settings.training.n_critic,
            "lambda_gp": settings.training.lambda_gp,
        })
        mlflow.log_params({
            "data_dir": str(settings.data.processed_data_dir),
            "num_classes": 2,
        })

        # Initialize GAN
        print("Initializing Conditional GAN...")
        print()
        gan = ConditionalGAN(
            latent_dim=settings.training.latent_dim,
            num_classes=2,
            image_size=settings.training.image_size,
            channels=3,
            learning_rate=settings.training.learning_rate,
            beta1=settings.training.beta1,
            loss_type=settings.training.loss_type,
            lambda_gp=settings.training.lambda_gp,
            n_critic=settings.training.n_critic,
            use_mixed_precision=True
        )
        print()

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = gan.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            print()

        # Load dataset
        print("Loading dataset...")
        processed_data_path = Path(settings.data.processed_data_dir) / "train"

        # Try to load real dataset, fall back to dummy if not available
        train_dataset = None
        if processed_data_path.exists():
            try:
                dataset = GramStainDataset(
                    data_path=str(processed_data_path),
                    image_size=(settings.training.image_size, settings.training.image_size),
                    augment=True,
                    split="train"
                )
                train_dataset = dataset.get_tf_dataset(
                    batch_size=settings.training.batch_size,
                    shuffle=True
                )
                print(f"✅ Loaded dataset from {processed_data_path}")
            except (ValueError, FileNotFoundError) as e:
                print(f"⚠️  Could not load real dataset: {e}")
                train_dataset = None

        if train_dataset is None:
            print()
            print("⚠️  WARNING: No processed dataset found!")
            print(f"   Expected path: {processed_data_path}")
            print()
            print("   To use real data:")
            print("   1. Add raw images to data/01_raw/gram_positive/ and data/01_raw/gram_negative/")
            print("   2. Run: poetry run python scripts/prepare_data.py")
            print()
            print("   For now, creating DUMMY dataset for testing...")
            print()

            # Create dummy dataset for testing
            train_dataset = create_dummy_dataset(
                batch_size=settings.training.batch_size,
                image_size=settings.training.image_size
            )

        print()

        # Create directories for outputs
        models_dir = Path("models") / run_id
        samples_dir = Path("samples") / run_id
        models_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print()

        for epoch in range(start_epoch, settings.training.epochs):
            epoch_start_time = time.time()

            # Reset metrics
            gan.reset_metrics()

            # Training progress bar
            pbar = tqdm(
                enumerate(train_dataset),
                desc=f"Epoch {epoch+1}/{settings.training.epochs}",
                total=len(list(train_dataset)),
                ncols=100
            )

            for step, (images, labels) in pbar:
                # Train step
                metrics = gan.train_step(images, labels)

                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f"{metrics['gen_loss']:.4f}",
                    'D_loss': f"{metrics['disc_loss']:.4f}",
                    'GP': f"{metrics['gp']:.4f}"
                })

            # Epoch complete
            epoch_time = time.time() - epoch_start_time

            # Get final metrics for epoch
            gen_loss = float(gan.gen_loss_metric.result())
            disc_loss = float(gan.disc_loss_metric.result())
            gp = float(gan.gp_metric.result())

            # Log metrics to MLflow
            mlflow.log_metrics({
                "generator_loss": gen_loss,
                "discriminator_loss": disc_loss,
                "gradient_penalty": gp,
                "epoch_time_seconds": epoch_time
            }, step=epoch)

            # Print epoch summary
            print()
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Generator Loss: {gen_loss:.4f}")
            print(f"  Discriminator Loss: {disc_loss:.4f}")
            print(f"  Gradient Penalty: {gp:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print()

            # Save sample images every 10 epochs (4 samples for GPU memory constraints)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Generating sample images...")
                sample_path = save_sample_images(gan, epoch + 1, samples_dir, num_samples=4)
                mlflow.log_artifact(sample_path, "samples")
                print(f"✅ Samples saved to {sample_path}")
                print()

            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = models_dir / f"checkpoint_epoch_{epoch+1:04d}.npy"
                gan.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch,
                    metadata={
                        "gen_loss": gen_loss,
                        "disc_loss": disc_loss,
                        "run_id": run_id
                    }
                )
                mlflow.log_artifact(str(checkpoint_path), "checkpoints")
                print()

        # Training complete
        print("=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print()

        # Save final generator model
        print("Saving final generator model...")
        final_model_path = models_dir / "generator_final.keras"
        gan.save_generator(str(final_model_path))

        # Log final model to MLflow
        mlflow.tensorflow.log_model(
            model=gan.generator,
            artifact_path="generator_model",
            registered_model_name="bacterial-gan-generator"
        )

        # Generate final samples (8 samples for GPU memory)
        print("Generating final sample images...")
        final_sample_path = save_sample_images(gan, settings.training.epochs, samples_dir, num_samples=8)
        mlflow.log_artifact(final_sample_path, "final_samples")

        print()
        print("=" * 80)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("=" * 80)
        print()
        print(f"MLflow Run ID: {run_id}")
        print(f"Generator model: {final_model_path}")
        print(f"Samples directory: {samples_dir}")
        print()
        print("To view results:")
        print(f"  mlflow ui")
        print()
        print("To generate synthetic data:")
        print(f"  bacterial-gan generate-data --run-id {run_id} --num-images 1000")
        print()


def create_dummy_dataset(batch_size: int = 8, image_size: int = 128, num_batches: int = 10):
    """
    Create dummy dataset for testing when real data is not available.

    Args:
        batch_size: Batch size
        image_size: Image size
        num_batches: Number of batches to generate

    Returns:
        TensorFlow dataset
    """
    def generator():
        for _ in range(num_batches):
            # Random images
            images = tf.random.normal([batch_size, image_size, image_size, 3])
            # Random labels (0 or 1)
            labels = tf.random.uniform([batch_size], minval=0, maxval=2, dtype=tf.int32)
            yield images, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
        )
    )

    return dataset