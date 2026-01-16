"""
StyleGAN2-ADA training wrapper for bacterial image augmentation.

Provides high-level training loop with:
- ADA (Adaptive Discriminator Augmentation)
- R1 and Path Length regularization with lazy computation
- EMA (Exponential Moving Average) generator for stable inference
- Mixed precision training with gradient scaling
- Truncation trick for controlled generation
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .losses import StyleGAN2ADALoss, get_loss_functions, path_length_regularization, r1_regularization
from .stylegan2_ada import ExponentialMovingAverage, build_stylegan2_ada


class StyleGAN2ADA:
    """
    StyleGAN2-ADA for conditional bacterial image generation.

    Key features:
    - Class-conditional generation (Gram-positive/Gram-negative)
    - Adaptive Discriminator Augmentation (ADA) for limited data
    - R1 regularization with lazy computation
    - Path Length regularization for smooth latent space
    - EMA generator for stable inference
    - Mixed precision training with gradient scaling
    - Truncation trick for quality/diversity control
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
        ema_decay: float = 0.999,
        use_ema: bool = True,
        use_pl_reg: bool = True,
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
        self.pl_weight = pl_weight
        self.pl_interval = pl_interval
        self.use_ada = use_ada
        self.n_critic = n_critic
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_pl_reg = use_pl_reg
        self.use_mixed_precision = use_mixed_precision

        if use_mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("✅ Mixed precision enabled (FP16)")

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

        # Optimizers (Keras 3 handles mixed precision automatically with policy)
        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_g, beta_1=beta1, beta_2=beta2
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_d, beta_1=beta1, beta_2=beta2
        )

        # Metrics
        self.gen_loss_metric = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.r1_metric = keras.metrics.Mean(name="r1_penalty")
        self.pl_metric = keras.metrics.Mean(name="pl_penalty")
        self.ada_p_metric = keras.metrics.Mean(name="ada_p")

        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Path length moving average
        self.pl_mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self._build_models()

        # EMA generator for stable inference
        if use_ema:
            self.generator_ema = ExponentialMovingAverage(self.generator, decay=ema_decay)
            print(f"✅ EMA enabled (decay={ema_decay})")

        print("✅ StyleGAN2-ADA initialized")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
        print(f"   Loss: {loss_type}, R1 gamma: {r1_gamma}")
        print(f"   Path Length: {'Enabled' if use_pl_reg else 'Disabled'} (weight={pl_weight})")
        print(f"   ADA: {'Enabled' if use_ada else 'Disabled'}")

    def _build_models(self):
        """Build models with dummy forward pass."""
        dummy_noise = tf.zeros([1, self.latent_dim])
        dummy_labels = tf.zeros([1], dtype=tf.int32)
        dummy_images = tf.zeros([1, self.image_size, self.image_size, self.channels])

        _ = self.generator([dummy_noise, dummy_labels], training=False)
        _ = self.discriminator([dummy_images, dummy_labels], training=False)

    def train_discriminator_step(
        self, real_images: tf.Tensor, class_labels: tf.Tensor, do_r1: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_images = self.generator([noise, class_labels], training=True)

        with tf.GradientTape() as tape:
            real_logits = self.discriminator([real_images, class_labels], training=True)
            fake_logits = self.discriminator([fake_images, class_labels], training=True)

            # Cast logits to float32 for stable loss computation
            real_logits_f32 = tf.cast(real_logits, tf.float32)
            fake_logits_f32 = tf.cast(fake_logits, tf.float32)

            disc_loss = self.disc_loss_fn(real_logits_f32, fake_logits_f32)

            r1_penalty = tf.constant(0.0, dtype=tf.float32)
            if do_r1:
                r1_penalty = r1_regularization(
                    self.discriminator, real_images, class_labels, gamma=self.r1_gamma
                )
                disc_loss = disc_loss + r1_penalty * tf.cast(self.r1_interval, tf.float32)

        # Compute gradients (Keras handles scaling automatically)
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Clip gradients and apply
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        if self.use_ada:
            self.discriminator.update_ada(real_logits_f32)
            ada_p = self.discriminator.ada.p
        else:
            ada_p = tf.constant(0.0)

        return disc_loss, r1_penalty, ada_p

    def train_generator_step(
        self, batch_size: int, class_labels: tf.Tensor, do_pl: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            fake_images = self.generator([noise, class_labels], training=True)
            fake_logits = self.discriminator([fake_images, class_labels], training=True)

            # Cast logits to float32 for stable loss computation
            fake_logits_f32 = tf.cast(fake_logits, tf.float32)
            gen_loss = self.gen_loss_fn(fake_logits_f32)

            # Path length regularization (lazy - every pl_interval steps)
            pl_penalty = tf.constant(0.0, dtype=tf.float32)
            if do_pl and self.use_pl_reg:
                pl_penalty, new_pl_mean = path_length_regularization(
                    self.generator,
                    noise,
                    class_labels,
                    pl_mean=self.pl_mean,
                    pl_weight=self.pl_weight,
                )
                # Scale by interval for lazy regularization
                gen_loss = gen_loss + pl_penalty * tf.cast(self.pl_interval, tf.float32)

        # Compute gradients (Keras handles scaling automatically)
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Clip gradients and apply
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # Update path length moving average
        if do_pl and self.use_pl_reg:
            self.pl_mean.assign(new_pl_mean)

        return gen_loss, pl_penalty

    def _core_train_step(
        self, real_images: tf.Tensor, class_labels: tf.Tensor, do_r1: bool, do_pl: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Core training step logic."""
        batch_size = tf.shape(real_images)[0]

        # Handle NaN values
        real_images = tf.where(
            tf.math.is_finite(real_images), real_images, tf.zeros_like(real_images)
        )

        # Train discriminator
        disc_loss, r1_penalty, ada_p = self.train_discriminator_step(
            real_images, class_labels, do_r1
        )

        # Train generator
        gen_loss, pl_penalty = self.train_generator_step(batch_size, class_labels, do_pl)

        return disc_loss, r1_penalty, ada_p, gen_loss, pl_penalty

    def train_step(self, real_images: tf.Tensor, class_labels: tf.Tensor) -> Dict[str, float]:
        current_iter = int(self.iteration.numpy())
        do_r1 = (current_iter > 0) and (current_iter % self.r1_interval == 0)
        do_pl = (current_iter > 0) and (current_iter % self.pl_interval == 0)

        # Execute training step
        disc_loss, r1_penalty, ada_p, gen_loss, pl_penalty = self._core_train_step(
            real_images, class_labels, do_r1, do_pl
        )

        # Update metrics
        self.disc_loss_metric.update_state(disc_loss)
        self.r1_metric.update_state(r1_penalty)
        self.ada_p_metric.update_state(ada_p)
        self.gen_loss_metric.update_state(gen_loss)
        self.pl_metric.update_state(pl_penalty)

        # Update EMA generator weights
        if self.use_ema:
            self.generator_ema.update()

        self.iteration.assign_add(1)

        return {
            "gen_loss": float(self.gen_loss_metric.result()),
            "disc_loss": float(self.disc_loss_metric.result()),
            "r1_penalty": float(self.r1_metric.result()),
            "pl_penalty": float(self.pl_metric.result()),
            "ada_p": float(self.ada_p_metric.result()),
        }

    def generate_samples(
        self,
        class_labels: Optional[tf.Tensor] = None,
        num_samples: int = 16,
        noise: Optional[tf.Tensor] = None,
        use_ema: bool = True,
        truncation_psi: float = 1.0,
    ) -> np.ndarray:
        """
        Generate samples using the generator.

        Args:
            class_labels: Optional class labels. If None, balanced across classes.
            num_samples: Number of samples to generate.
            noise: Optional latent vectors. If None, randomly sampled.
            use_ema: Whether to use EMA generator weights (recommended for quality).
            truncation_psi: Truncation strength (0=w_avg, 1=no truncation, 0.7 typical).

        Returns:
            Generated images as numpy array [N, H, W, C].
        """
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

        # Use EMA weights if available and requested
        if use_ema and self.use_ema:
            self.generator_ema.apply_shadow()

        try:
            # Use truncation if generator supports it and psi != 1.0
            if truncation_psi != 1.0 and hasattr(self.generator, 'generate_truncated'):
                images = self.generator.generate_truncated(
                    noise, class_labels, truncation_psi=truncation_psi
                ).numpy()
            else:
                images = self.generator([noise, class_labels], training=False).numpy()
        finally:
            # Restore original weights if EMA was applied
            if use_ema and self.use_ema:
                self.generator_ema.restore()

        return images

    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "iteration": int(self.iteration.numpy()),
            "generator_weights": self.generator.get_weights(),
            "discriminator_weights": self.discriminator.get_weights(),
            "pl_mean": float(self.pl_mean.numpy()),
            "config": {
                "latent_dim": self.latent_dim,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "channels": self.channels,
                "learning_rate_g": self.learning_rate_g,
                "learning_rate_d": self.learning_rate_d,
                "loss_type": self.loss_type,
                "r1_gamma": self.r1_gamma,
                "pl_weight": self.pl_weight,
                "use_ada": self.use_ada,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
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

        # Save EMA weights
        if self.use_ema:
            ema_shadow = self.generator_ema.get_shadow_variables()
            checkpoint["ema_weights"] = {
                name: var.numpy() for name, var in ema_shadow.items()
            }

        # Save w_avg if generator has it (for truncation trick)
        if hasattr(self.generator, 'w_avg'):
            checkpoint["w_avg"] = self.generator.w_avg.numpy()

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

        if "pl_mean" in checkpoint:
            self.pl_mean.assign(checkpoint["pl_mean"])

        # Build optimizers if needed
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

        # Load EMA weights
        if self.use_ema and "ema_weights" in checkpoint:
            try:
                self.generator_ema.set_shadow_variables(checkpoint["ema_weights"])
            except Exception:
                pass

        # Load w_avg for truncation trick
        if hasattr(self.generator, 'w_avg') and "w_avg" in checkpoint:
            self.generator.w_avg.assign(checkpoint["w_avg"])

        print(f"✅ Loaded checkpoint: {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint

    def reset_metrics(self):
        self.gen_loss_metric.reset_state()
        self.disc_loss_metric.reset_state()
        self.r1_metric.reset_state()
        self.pl_metric.reset_state()
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
