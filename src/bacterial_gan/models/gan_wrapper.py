"""
Wrapper class for combining generator and discriminator into a complete GAN system.

This file contains:
1. ConditionalGAN class that combines generator and discriminator
2. Training loop logic with proper loss computation
3. Gradient penalty implementation for WGAN-GP
4. Model checkpointing and loading utilities
5. Inference methods for generation
6. Integration with MLflow for experiment tracking
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from .architecture import build_generator, build_discriminator, gradient_penalty, get_loss_functions


class ConditionalGAN:
    """
    Conditional GAN for bacterial data augmentation.

    Features:
    1. Conditional generation based on class labels
    2. WGAN-GP training for stability
    3. Mixed precision training for efficiency (GTX 1650 optimized)
    4. Model checkpointing
    5. Sample generation during training
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 2,
        image_size: int = 128,
        channels: int = 3,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        loss_type: str = "wgan-gp",
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        use_mixed_precision: bool = True
    ):
        """
        Initialize Conditional GAN.

        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes (2 for Gram +/-)
            image_size: Size of generated images
            channels: Number of image channels (3 for RGB)
            learning_rate: Learning rate for both G and D
            beta1: Beta1 parameter for Adam optimizer
            loss_type: Type of GAN loss ("wgan-gp", "lsgan", "vanilla")
            lambda_gp: Gradient penalty coefficient for WGAN-GP
            n_critic: Number of discriminator updates per generator update
            use_mixed_precision: Enable mixed precision training for faster performance
        """
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.loss_type = loss_type
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        # Enable mixed precision for GTX 1650 (faster training)
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled (faster on GTX 1650)")

        # Build models
        print("🏗️  Building Generator...")
        self.generator = build_generator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            image_size=image_size,
            channels=channels
        )

        print("🏗️  Building Discriminator...")
        self.discriminator = build_discriminator(
            image_size=image_size,
            channels=channels,
            num_classes=num_classes,
            use_spectral_norm=(loss_type == "wgan-gp")
        )

        # Get loss functions
        self.gen_loss_fn, self.disc_loss_fn = get_loss_functions(loss_type)

        # Initialize optimizers with constant learning rate
        # Note: Removed aggressive exponential decay that was causing learning rate collapse
        # Previous decay (96% every 1000 steps) reduced LR from 0.0002 to 0.000038 after 130 epochs
        # Constant LR allows stable long-term training for WGAN-GP
        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1
        )

        # Metrics tracking
        self.gen_loss_metric = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.gp_metric = keras.metrics.Mean(name="gradient_penalty")

        print(f"✅ ConditionalGAN initialized with {loss_type.upper()} loss")
        print(f"   🧬 Generator params: {self.generator.count_params():,}")
        print(f"   🔍 Discriminator params: {self.discriminator.count_params():,}")

    @tf.function
    def train_discriminator_step(
        self,
        real_images: tf.Tensor,
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Train discriminator for one step.

        Args:
            real_images: Real images [batch_size, H, W, C]
            class_labels: Class labels [batch_size]

        Returns:
            Discriminator loss and gradient penalty
        """
        batch_size = tf.shape(real_images)[0]

        # Generate fake images
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_images = self.generator([noise, class_labels], training=True)

        with tf.GradientTape() as tape:
            # Get discriminator predictions
            real_predictions = self.discriminator([real_images, class_labels], training=True)
            fake_predictions = self.discriminator([fake_images, class_labels], training=True)

            # Calculate discriminator loss
            disc_loss = self.disc_loss_fn(real_predictions, fake_predictions)

            # Add gradient penalty for WGAN-GP
            gp = tf.constant(0.0, dtype=disc_loss.dtype)
            if self.loss_type == "wgan-gp":
                gp_raw = gradient_penalty(
                    self.discriminator,
                    real_images,
                    fake_images,
                    class_labels,
                    lambda_gp=self.lambda_gp
                )
                # Cast GP to same dtype as disc_loss for mixed precision training
                gp = tf.cast(gp_raw, disc_loss.dtype)

            # Add gradient penalty to discriminator loss (avoids += operator issues in graph mode)
            disc_loss = tf.add(disc_loss, gp)

        # Update discriminator with gradient clipping for additional stability
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # Relaxed from 1.0 to allow stronger gradients
        self.disc_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

        return disc_loss, gp

    @tf.function
    def train_generator_step(self, batch_size: int, class_labels: tf.Tensor) -> tf.Tensor:
        """
        Train generator for one step.

        Args:
            batch_size: Batch size
            class_labels: Class labels [batch_size]

        Returns:
            Generator loss
        """
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = self.generator([noise, class_labels], training=True)

            # Get discriminator predictions on fake images
            fake_predictions = self.discriminator([fake_images, class_labels], training=True)

            # Calculate generator loss
            gen_loss = self.gen_loss_fn(fake_predictions)

        # Update generator with gradient clipping for additional stability
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # Relaxed from 1.0 to allow stronger gradients
        self.gen_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return gen_loss

    def train_step(self, real_images: tf.Tensor, class_labels: tf.Tensor) -> Dict[str, float]:
        """
        Single training step (n_critic discriminator updates + 1 generator update).

        Args:
            real_images: Real images [batch_size, H, W, C]
            class_labels: Class labels [batch_size]

        Returns:
            Dictionary with loss metrics
        """
        batch_size = tf.shape(real_images)[0]

        # Train discriminator n_critic times
        for _ in range(self.n_critic):
            disc_loss, gp = self.train_discriminator_step(real_images, class_labels)
            self.disc_loss_metric.update_state(disc_loss)
            self.gp_metric.update_state(gp)

        # Train generator once
        gen_loss = self.train_generator_step(batch_size, class_labels)
        self.gen_loss_metric.update_state(gen_loss)

        return {
            "gen_loss": float(self.gen_loss_metric.result()),
            "disc_loss": float(self.disc_loss_metric.result()),
            "gp": float(self.gp_metric.result())
        }

    def generate_samples(
        self,
        class_labels: Optional[tf.Tensor] = None,
        num_samples: int = 16
    ) -> np.ndarray:
        """
        Generate synthetic samples.

        Args:
            class_labels: Class labels [num_samples]. If None, generates balanced samples
            num_samples: Number of samples to generate

        Returns:
            Generated images [num_samples, H, W, C] in range [-1, 1]
        """
        if class_labels is None:
            # Generate balanced samples (equal from each class)
            samples_per_class = num_samples // self.num_classes
            class_labels = []
            for i in range(self.num_classes):
                class_labels.extend([i] * samples_per_class)
            # Add remaining samples
            remaining = num_samples - len(class_labels)
            class_labels.extend([0] * remaining)
            class_labels = tf.constant(class_labels, dtype=tf.int32)

        noise = tf.random.normal([num_samples, self.latent_dim])
        generated_images = self.generator([noise, class_labels], training=False)

        return generated_images.numpy()

    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metadata: Optional metadata dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'generator_weights': self.generator.get_weights(),
            'discriminator_weights': self.discriminator.get_weights(),
            'config': {
                'latent_dim': self.latent_dim,
                'num_classes': self.num_classes,
                'image_size': self.image_size,
                'channels': self.channels,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'loss_type': self.loss_type,
                'lambda_gp': self.lambda_gp,
                'n_critic': self.n_critic
            }
        }

        # Save optimizer weights if available (only after optimizer has been built)
        # Optimizers are built after the first apply_gradients call
        try:
            # Build optimizers if not already built
            if len(self.gen_optimizer.variables) > 0:
                if hasattr(self.gen_optimizer, 'get_weights'):
                    checkpoint['gen_optimizer_weights'] = self.gen_optimizer.get_weights()
                else:
                    # Keras 3 / TF 2.16+ fallback
                    checkpoint['gen_optimizer_weights'] = [v.numpy() for v in self.gen_optimizer.variables]

            if len(self.disc_optimizer.variables) > 0:
                if hasattr(self.disc_optimizer, 'get_weights'):
                    checkpoint['disc_optimizer_weights'] = self.disc_optimizer.get_weights()
                else:
                    # Keras 3 / TF 2.16+ fallback
                    checkpoint['disc_optimizer_weights'] = [v.numpy() for v in self.disc_optimizer.variables]
        except (AttributeError, ValueError) as e:
            # Optimizer not built yet or incompatible - skip saving optimizer state
            print(f"⚠️  Skipping optimizer weights (error): {e}")

        if metadata:
            checkpoint['metadata'] = metadata

        np.save(filepath, checkpoint, allow_pickle=True)
        print(f"💾 Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = np.load(filepath, allow_pickle=True).item()

        # Load model weights first
        self.generator.set_weights(checkpoint['generator_weights'])
        self.discriminator.set_weights(checkpoint['discriminator_weights'])

        # Build optimizers if not already built (required before setting weights)
        if not self.gen_optimizer.built:
            self.gen_optimizer.build(self.generator.trainable_variables)
        if not self.disc_optimizer.built:
            self.disc_optimizer.build(self.discriminator.trainable_variables)

        # Load optimizer weights
        if 'gen_optimizer_weights' in checkpoint:
            try:
                if hasattr(self.gen_optimizer, 'set_weights'):
                    self.gen_optimizer.set_weights(checkpoint['gen_optimizer_weights'])
                else:
                    # Keras 3 / TF 2.16+ fallback
                    weights = checkpoint['gen_optimizer_weights']
                    if len(self.gen_optimizer.variables) == len(weights):
                        for var, weight in zip(self.gen_optimizer.variables, weights):
                            var.assign(weight)
            except Exception as e:
                print(f"⚠️  Could not load generator optimizer weights: {e}")
                print("   ℹ️  Training will continue with fresh optimizer state")

        if 'disc_optimizer_weights' in checkpoint:
            try:
                if hasattr(self.disc_optimizer, 'set_weights'):
                    self.disc_optimizer.set_weights(checkpoint['disc_optimizer_weights'])
                else:
                    # Keras 3 / TF 2.16+ fallback
                    weights = checkpoint['disc_optimizer_weights']
                    if len(self.disc_optimizer.variables) == len(weights):
                        for var, weight in zip(self.disc_optimizer.variables, weights):
                            var.assign(weight)
            except Exception as e:
                print(f"⚠️  Could not load discriminator optimizer weights: {e}")
                print("   ℹ️  Training will continue with fresh optimizer state")

        epoch = checkpoint['epoch']
        print(f"✅ Checkpoint loaded from {filepath} (epoch {epoch})")

        return checkpoint

    def reset_metrics(self):
        """Reset all metrics."""
        self.gen_loss_metric.reset_state()
        self.disc_loss_metric.reset_state()
        self.gp_metric.reset_state()

    def save_generator(self, filepath: str):
        """
        Save only the generator model (for deployment).

        Args:
            filepath: Path to save generator
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.generator.save(filepath)
        print(f"💾 Generator saved to {filepath}")

    def load_generator(self, filepath: str):
        """
        Load generator model.

        Args:
            filepath: Path to generator file
        """
        self.generator = keras.models.load_model(filepath)
        print(f"✅ Generator loaded from {filepath}")
