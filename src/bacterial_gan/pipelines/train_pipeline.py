"""Training pipeline with MLflow integration."""

import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bacterial_gan.config import Settings
from bacterial_gan.data.dataset import GramStainDataset
from bacterial_gan.models.stylegan2_wrapper import StyleGAN2ADA
from bacterial_gan.utils.memory_optimization import (
    clear_session,
    configure_cpu_parallelism,
    configure_tensorflow_memory,
    enable_xla_compilation,
)


def save_sample_images(
    gan: StyleGAN2ADA,
    epoch: int,
    save_dir: Path,
    num_samples: int = 16,
    noise: Optional[tf.Tensor] = None,
    class_labels: Optional[tf.Tensor] = None,
) -> str:
    """Generate and save sample images during training."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if class_labels is None:
        samples_per_class = num_samples // gan.num_classes
        class_labels = []
        for i in range(gan.num_classes):
            class_labels.extend([i] * samples_per_class)
        class_labels = tf.constant(class_labels, dtype=tf.int32)

    generated_images = gan.generate_samples(class_labels, num_samples, noise=noise)
    generated_images = (generated_images + 1.0) / 2.0
    generated_images = np.clip(generated_images, 0.0, 1.0).astype(np.float32)

    grid_cols = int(np.ceil(np.sqrt(num_samples)))
    grid_rows = int(np.ceil(num_samples / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    class_names = ["Gram-Positive", "Gram-Negative"]

    for i, (ax, img) in enumerate(zip(axes, generated_images)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{class_names[class_labels[i]]}", fontsize=8)

    plt.tight_layout()
    filepath = save_dir / f"epoch_{epoch:04d}.png"
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()

    return str(filepath)


def run(settings: Settings, resume_from_checkpoint: Optional[str] = None):
    """Run training pipeline with MLflow tracking."""
    print("=" * 80)
    print("🧬 BACTERIAL GAN TRAINING PIPELINE (StyleGAN2-ADA)")
    print("=" * 80)
    print()

    print("🔧 Configuring TensorFlow...")
    configure_tensorflow_memory()
    configure_cpu_parallelism(num_threads=12)
    enable_xla_compilation(enable=settings.training.memory_optimization.enable_xla)
    clear_session()
    print()

    experiment_name = "Bacterial GAN Augmentation"
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted" in str(e).lower():
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                mlflow.set_experiment(experiment_name)
        else:
            raise

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"📊 MLflow Run ID: {run_id}")
        print()

        mlflow.log_params(
            {
                "image_size": settings.training.image_size,
                "batch_size": settings.training.batch_size,
                "epochs": settings.training.epochs,
                "learning_rate_g": settings.training.learning_rate_g,
                "learning_rate_d": settings.training.learning_rate_d,
                "latent_dim": settings.training.latent_dim,
                "loss_type": settings.training.loss_type,
                "r1_gamma": settings.training.r1_gamma,
                "r1_interval": settings.training.r1_interval,
                "use_ada": settings.training.use_ada,
                "ada_target": settings.training.ada_target,
                "use_simplified": settings.training.use_simplified,
                "data_dir": str(settings.data.processed_data_dir),
            }
        )

        gan = StyleGAN2ADA(
            latent_dim=settings.training.latent_dim,
            num_classes=2,
            image_size=settings.training.image_size,
            channels=3,
            learning_rate_g=settings.training.learning_rate_g,
            learning_rate_d=settings.training.learning_rate_d,
            beta1=settings.training.beta1,
            beta2=settings.training.beta2,
            loss_type=settings.training.loss_type,
            r1_gamma=settings.training.r1_gamma,
            r1_interval=settings.training.r1_interval,
            pl_weight=settings.training.pl_weight,
            pl_interval=settings.training.pl_interval,
            use_ada=settings.training.use_ada,
            ada_target=settings.training.ada_target,
            use_simplified=settings.training.use_simplified,
            n_critic=settings.training.n_critic,
            use_mixed_precision=settings.training.use_mixed_precision,
        )
        print()

        fixed_num_samples = settings.training.num_samples_during_training
        fixed_samples_per_class = fixed_num_samples // 2
        fixed_labels = []
        for i in range(2):
            fixed_labels.extend([i] * fixed_samples_per_class)
        if len(fixed_labels) < fixed_num_samples:
            fixed_labels.extend([0] * (fixed_num_samples - len(fixed_labels)))
        fixed_labels = tf.constant(fixed_labels, dtype=tf.int32)
        fixed_noise = tf.random.normal([fixed_num_samples, settings.training.latent_dim])

        start_epoch = 0
        if resume_from_checkpoint:
            print(f"⏸️  Resuming from: {resume_from_checkpoint}")
            checkpoint = gan.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            print()

        print("📂 Loading dataset...")
        processed_data_path = Path(settings.data.processed_data_dir) / "train"

        train_dataset = None
        num_batches = None
        if processed_data_path.exists():
            try:
                dataset = GramStainDataset(
                    data_path=str(processed_data_path),
                    image_size=(settings.training.image_size, settings.training.image_size),
                    augment=True,
                    split="train",
                )
                train_dataset = dataset.get_tf_dataset(
                    batch_size=settings.training.batch_size, shuffle=True
                )
                num_batches = dataset.num_samples // settings.training.batch_size
                print(f"✅ Loaded {dataset.num_samples} images")
            except (ValueError, FileNotFoundError) as e:
                print(f"⚠️  Could not load dataset: {e}")

        if train_dataset is None:
            print("⚠️  No dataset found, using dummy data for testing...")
            train_dataset = create_dummy_dataset(
                batch_size=settings.training.batch_size,
                image_size=settings.training.image_size,
            )
            num_batches = settings.training.dummy_num_batches

        print()

        print("=" * 80)
        print("🚀 TRAINING")
        print("=" * 80)
        print()

        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="stylegan2_"))
        samples_dir = temp_dir / "samples"
        checkpoints_dir = temp_dir / "checkpoints"
        samples_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, settings.training.epochs):
            epoch_start_time = time.time()
            gan.reset_metrics()

            pbar = tqdm(
                enumerate(train_dataset),
                desc=f"Epoch {epoch+1}/{settings.training.epochs}",
                total=num_batches,
                ncols=100,
            )

            for step, (images, labels) in pbar:
                metrics = gan.train_step(images, labels)

                postfix = {
                    "G": f"{metrics['gen_loss']:.3f}",
                    "D": f"{metrics['disc_loss']:.3f}",
                    "R1": f"{metrics['r1_penalty']:.3f}",
                }
                if metrics["ada_p"] > 0:
                    postfix["ADA"] = f"{metrics['ada_p']:.2f}"
                pbar.set_postfix(postfix)

            epoch_time = time.time() - epoch_start_time

            gen_loss = float(gan.gen_loss_metric.result())
            disc_loss = float(gan.disc_loss_metric.result())
            r1_penalty = float(gan.r1_metric.result())
            ada_p = gan.get_ada_probability()

            mlflow.log_metrics(
                {
                    "generator_loss": gen_loss,
                    "discriminator_loss": disc_loss,
                    "r1_penalty": r1_penalty,
                    "ada_probability": ada_p,
                    "epoch_time_seconds": epoch_time,
                },
                step=epoch,
            )

            print()
            print(
                f"📈 Epoch {epoch+1}: G={gen_loss:.4f} D={disc_loss:.4f} R1={r1_penalty:.4f} ADA={ada_p:.4f} ({epoch_time:.1f}s)"
            )

            if (epoch + 1) % settings.training.sample_interval == 0 or epoch == 0:
                sample_path = save_sample_images(
                    gan,
                    epoch + 1,
                    samples_dir,
                    num_samples=settings.training.num_samples_during_training,
                    noise=fixed_noise,
                    class_labels=fixed_labels,
                )
                mlflow.log_artifact(sample_path, "samples")

            if (epoch + 1) % settings.training.checkpoint_interval == 0:
                checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1:04d}.npy"
                gan.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch,
                    metadata={"gen_loss": gen_loss, "disc_loss": disc_loss, "run_id": run_id},
                )
                mlflow.log_artifact(str(checkpoint_path), "checkpoints")

        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)

        final_model_path = temp_dir / "generator_final.keras"
        gan.save_generator(str(final_model_path))

        mlflow.tensorflow.log_model(
            model=gan.generator,
            artifact_path="generator_model",
            registered_model_name="bacterial-gan-generator",
        )

        final_sample_path = save_sample_images(
            gan,
            settings.training.epochs,
            samples_dir,
            num_samples=settings.training.num_samples_final,
        )
        mlflow.log_artifact(final_sample_path, "final_samples")

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        print()
        print(f"📊 Run ID: {run_id}")
        print(f"📂 Artifacts: mlruns/<experiment_id>/{run_id}/artifacts/")
        print()
        print("To generate data:")
        print(f"   bacterial-gan generate-data --run-id {run_id} --num-images 1000")
        print()


def create_dummy_dataset(batch_size: int = 8, image_size: int = 256, num_batches: int = 10):
    """Create dummy dataset for testing."""

    def generator():
        for _ in range(num_batches):
            images = tf.random.normal([batch_size, image_size, image_size, 3])
            labels = tf.random.uniform([batch_size], minval=0, maxval=2, dtype=tf.int32)
            yield images, labels

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        ),
    )
