"""
StyleGAN2-ADA training wrapper for bacterial image augmentation.

Provides high-level training loop with ADA, R1 regularization, and checkpointing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .losses import StyleGAN2ADALoss, get_loss_functions, r1_regularization
from .stylegan2_ada import build_stylegan2_ada


class StyleGAN2ADA:
    """
    StyleGAN2-ADA for conditional bacterial image generation.

    Key features:
    - Class-conditional generation (Gram-positive/Gram-negative)
    - Adaptive Discriminator Augmentation (ADA) for limited data
    - R1 regularization with lazy computation
    - Mixed precision training support
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_classes: int = 2,
        image_size: int = 256,
        channels: int = 3,
        learning_rate_g: float = 0.0002,
        learning_rate_d: float = 0.0002,
        beta1: float = 0.0,
        beta2: float = 0.99,
        loss_type: str = "stylegan2",
        r1_gamma: float = 10.0,
        r1_interval: int = 16,
        pl_weight: float = 2.0,
        pl_interval: int = 4,
        use_ada: bool = True,
        ada_target: float = 0.6,
        use_simplified: bool = True,
        use_mixed_precision: bool = True,
        n_critic: int = 1,
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.loss_type = loss_type
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval
        self.use_ada = use_ada
        self.n_critic = n_critic

        if use_mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("✅ Mixed precision enabled")

        print("🏗️  Building StyleGAN2-ADA...")
        models = build_stylegan2_ada(
            latent_dim=latent_dim,
            num_classes=num_classes,
            image_size=image_size,
            channels=channels,
            use_simplified=use_simplified,
            use_ada=use_ada,
            ada_target=ada_target,
        )

        self.generator = models["generator"]
        self.discriminator = models["discriminator"]

        self.loss_module = StyleGAN2ADALoss(
            r1_gamma=r1_gamma,
            pl_weight=pl_weight,
            pl_interval=pl_interval,
            r1_interval=r1_interval,
        )

        self.gen_loss_fn, self.disc_loss_fn = get_loss_functions(loss_type)

        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_g, beta_1=beta1, beta_2=beta2
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_d, beta_1=beta1, beta_2=beta2
        )

        self.gen_loss_metric = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.r1_metric = keras.metrics.Mean(name="r1_penalty")
        self.ada_p_metric = keras.metrics.Mean(name="ada_p")

        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int64)

        self._build_models()

        print("✅ StyleGAN2-ADA initialized")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
        print(f"   Loss: {loss_type}, R1 gamma: {r1_gamma}")
        print(f"   ADA: {'Enabled' if use_ada else 'Disabled'}")

    def _build_models(self):
        """Build models with dummy forward pass."""
        dummy_noise = tf.zeros([1, self.latent_dim])
        dummy_labels = tf.zeros([1], dtype=tf.int32)
        dummy_images = tf.zeros([1, self.image_size, self.image_size, self.channels])

        _ = self.generator([dummy_noise, dummy_labels], training=False)
        _ = self.discriminator([dummy_images, dummy_labels], training=False)

    @tf.function
    def train_discriminator_step(
        self, real_images: tf.Tensor, class_labels: tf.Tensor, do_r1: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_images = self.generator([noise, class_labels], training=True)

        with tf.GradientTape() as tape:
            real_logits = self.discriminator([real_images, class_labels], training=True)
            fake_logits = self.discriminator([fake_images, class_labels], training=True)

            disc_loss = self.disc_loss_fn(real_logits, fake_logits)

            r1_penalty = tf.constant(0.0, dtype=disc_loss.dtype)
            if do_r1:
                r1_penalty_raw = r1_regularization(
                    self.discriminator, real_images, class_labels, gamma=self.r1_gamma
                )
                r1_penalty = tf.cast(r1_penalty_raw, disc_loss.dtype)
                disc_loss = disc_loss + r1_penalty * tf.cast(self.r1_interval, disc_loss.dtype)

        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        if self.use_ada:
            self.discriminator.update_ada(real_logits)
            ada_p = self.discriminator.ada.p
        else:
            ada_p = tf.constant(0.0)

        return disc_loss, r1_penalty, ada_p

    @tf.function
    def train_generator_step(self, batch_size: int, class_labels: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            fake_images = self.generator([noise, class_labels], training=True)
            fake_logits = self.discriminator([fake_images, class_labels], training=True)
            gen_loss = self.gen_loss_fn(fake_logits)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return gen_loss

    def train_step(self, real_images: tf.Tensor, class_labels: tf.Tensor) -> Dict[str, float]:
        batch_size = tf.shape(real_images)[0]
        current_iter = int(self.iteration.numpy())
        do_r1 = (current_iter > 0) and (current_iter % self.r1_interval == 0)

        if tf.reduce_any(tf.math.is_nan(real_images)):
            real_images = tf.where(
                tf.math.is_finite(real_images), real_images, tf.zeros_like(real_images)
            )

        for _ in range(self.n_critic):
            disc_loss, r1_penalty, ada_p = self.train_discriminator_step(
                real_images, class_labels, do_r1
            )
            if not tf.reduce_any(tf.math.is_nan(disc_loss)):
                self.disc_loss_metric.update_state(disc_loss)
                self.r1_metric.update_state(r1_penalty)
                self.ada_p_metric.update_state(ada_p)

        gen_loss = self.train_generator_step(batch_size, class_labels)
        if not tf.reduce_any(tf.math.is_nan(gen_loss)):
            self.gen_loss_metric.update_state(gen_loss)

        self.iteration.assign_add(1)

        return {
            "gen_loss": float(self.gen_loss_metric.result()),
            "disc_loss": float(self.disc_loss_metric.result()),
            "r1_penalty": float(self.r1_metric.result()),
            "ada_p": float(self.ada_p_metric.result()),
        }

    def generate_samples(
        self,
        class_labels: Optional[tf.Tensor] = None,
        num_samples: int = 16,
        noise: Optional[tf.Tensor] = None,
    ) -> np.ndarray:
        if class_labels is None:
            samples_per_class = num_samples // self.num_classes
            class_labels = []
            for i in range(self.num_classes):
                class_labels.extend([i] * samples_per_class)
            remaining = num_samples - len(class_labels)
            class_labels.extend([0] * remaining)
            class_labels = tf.constant(class_labels, dtype=tf.int32)

        if noise is None:
            noise = tf.random.normal([num_samples, self.latent_dim])

        return self.generator([noise, class_labels], training=False).numpy()

    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "iteration": int(self.iteration.numpy()),
            "generator_weights": self.generator.get_weights(),
            "discriminator_weights": self.discriminator.get_weights(),
            "config": {
                "latent_dim": self.latent_dim,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "channels": self.channels,
                "learning_rate_g": self.learning_rate_g,
                "learning_rate_d": self.learning_rate_d,
                "loss_type": self.loss_type,
                "r1_gamma": self.r1_gamma,
                "use_ada": self.use_ada,
            },
        }

        try:
            if len(self.gen_optimizer.variables) > 0:
                checkpoint["gen_optimizer_weights"] = [
                    v.numpy() for v in self.gen_optimizer.variables
                ]
            if len(self.disc_optimizer.variables) > 0:
                checkpoint["disc_optimizer_weights"] = [
                    v.numpy() for v in self.disc_optimizer.variables
                ]
        except Exception:
            pass

        if self.use_ada:
            checkpoint["ada_p"] = float(self.discriminator.ada.p.numpy())

        if metadata:
            checkpoint["metadata"] = metadata

        np.save(filepath, checkpoint, allow_pickle=True)
        print(f"💾 Saved checkpoint: {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict:
        checkpoint = np.load(filepath, allow_pickle=True).item()

        self.generator.set_weights(checkpoint["generator_weights"])
        self.discriminator.set_weights(checkpoint["discriminator_weights"])

        if "iteration" in checkpoint:
            self.iteration.assign(checkpoint["iteration"])

        if not self.gen_optimizer.built:
            self.gen_optimizer.build(self.generator.trainable_variables)
        if not self.disc_optimizer.built:
            self.disc_optimizer.build(self.discriminator.trainable_variables)

        if "gen_optimizer_weights" in checkpoint:
            try:
                weights = checkpoint["gen_optimizer_weights"]
                if len(self.gen_optimizer.variables) == len(weights):
                    for var, weight in zip(self.gen_optimizer.variables, weights):
                        var.assign(weight)
            except Exception:
                pass

        if "disc_optimizer_weights" in checkpoint:
            try:
                weights = checkpoint["disc_optimizer_weights"]
                if len(self.disc_optimizer.variables) == len(weights):
                    for var, weight in zip(self.disc_optimizer.variables, weights):
                        var.assign(weight)
            except Exception:
                pass

        if self.use_ada and "ada_p" in checkpoint:
            self.discriminator.ada.p.assign(checkpoint["ada_p"])

        print(f"✅ Loaded checkpoint: {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint

    def reset_metrics(self):
        self.gen_loss_metric.reset_state()
        self.disc_loss_metric.reset_state()
        self.r1_metric.reset_state()
        self.ada_p_metric.reset_state()

    def save_generator(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.generator.save(filepath)
        print(f"💾 Saved generator: {filepath}")

    def load_generator(self, filepath: str):
        self.generator = keras.models.load_model(filepath)
        print(f"✅ Loaded generator: {filepath}")

    def get_ada_probability(self) -> float:
        if self.use_ada:
            return float(self.discriminator.ada.p.numpy())
        return 0.0
