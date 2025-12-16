"""
Loss functions for StyleGAN2-ADA training.

Implements:
- Non-saturating logistic loss (standard GAN loss)
- R1 regularization (gradient penalty for discriminator)
- Path length regularization (smooth latent space)
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


def logistic_nonsaturating_generator_loss(fake_logits: tf.Tensor) -> tf.Tensor:
    """Generator loss: E[softplus(-D(G(z)))]"""
    return tf.reduce_mean(tf.nn.softplus(-fake_logits))


def logistic_nonsaturating_discriminator_loss(
    real_logits: tf.Tensor, fake_logits: tf.Tensor
) -> tf.Tensor:
    """Discriminator loss: E[softplus(-D(x))] + E[softplus(D(G(z)))]"""
    real_loss = tf.reduce_mean(tf.nn.softplus(-real_logits))
    fake_loss = tf.reduce_mean(tf.nn.softplus(fake_logits))
    return real_loss + fake_loss


def r1_regularization(
    discriminator: keras.Model,
    real_images: tf.Tensor,
    class_labels: Optional[tf.Tensor] = None,
    gamma: float = 10.0,
) -> tf.Tensor:
    """
    R1 gradient penalty: gamma/2 * E[||∇D(x)||²]

    Stabilizes training by penalizing discriminator gradients on real images.
    """
    real_images = tf.cast(real_images, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(real_images)
        if class_labels is not None:
            real_logits = discriminator([real_images, class_labels], training=True)
        else:
            real_logits = discriminator(real_images, training=True)

    gradients = tape.gradient(real_logits, real_images)
    if gradients is None:
        return tf.constant(0.0, dtype=tf.float32)

    gradients_squared = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    return (gamma / 2.0) * tf.reduce_mean(gradients_squared)


def path_length_regularization(
    generator: keras.Model,
    latent_w: tf.Tensor,
    class_labels: Optional[tf.Tensor] = None,
    pl_weight: float = 2.0,
    pl_decay: float = 0.01,
    pl_mean: tf.Variable = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Path length regularization for smooth latent space mapping.

    Encourages consistent magnitude of change when moving in latent space.
    """
    with tf.GradientTape() as tape:
        tape.watch(latent_w)
        if class_labels is not None:
            fake_images = generator([latent_w, class_labels], training=True)
        else:
            fake_images = generator(latent_w, training=True)

        noise = tf.random.normal(tf.shape(fake_images)) / np.sqrt(
            float(fake_images.shape[1] * fake_images.shape[2])
        )
        output = tf.reduce_sum(fake_images * noise)

    gradients = tape.gradient(output, latent_w)
    if gradients is None:
        if pl_mean is not None:
            return tf.constant(0.0, dtype=tf.float32), pl_mean
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    if len(gradients.shape) == 2:
        path_lengths = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
    else:
        path_lengths = tf.sqrt(
            tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=2), axis=1) + 1e-8
        )

    current_mean = tf.reduce_mean(path_lengths)
    if pl_mean is not None:
        new_pl_mean = pl_mean + pl_decay * (current_mean - pl_mean)
    else:
        new_pl_mean = current_mean

    pl_penalty = pl_weight * tf.reduce_mean((path_lengths - new_pl_mean) ** 2)
    return pl_penalty, new_pl_mean


class PerceptualLoss:
    """Perceptual loss using VGG19 features for high-level similarity."""

    def __init__(self, feature_layers: list = None):
        if feature_layers is None:
            feature_layers = ["block1_conv2", "block2_conv2", "block3_conv3"]

        vgg = keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in feature_layers]
        self.feature_extractor = keras.Model(inputs=vgg.input, outputs=outputs)
        self.feature_layers = feature_layers

    def __call__(
        self, real_images: tf.Tensor, fake_images: tf.Tensor, weights: list = None
    ) -> tf.Tensor:
        real_preprocessed = keras.applications.vgg19.preprocess_input((real_images + 1.0) * 127.5)
        fake_preprocessed = keras.applications.vgg19.preprocess_input((fake_images + 1.0) * 127.5)

        real_features = self.feature_extractor(real_preprocessed)
        fake_features = self.feature_extractor(fake_preprocessed)

        if weights is None:
            weights = [1.0] * len(self.feature_layers)

        total_loss = 0.0
        for real_feat, fake_feat, weight in zip(real_features, fake_features, weights):
            total_loss += weight * tf.reduce_mean(tf.abs(real_feat - fake_feat))

        return total_loss


def get_loss_functions(loss_type: str = "stylegan2") -> Tuple[Callable, Callable]:
    """Get generator and discriminator loss functions."""
    if loss_type in ("stylegan2", "logistic"):
        return (
            logistic_nonsaturating_generator_loss,
            logistic_nonsaturating_discriminator_loss,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'stylegan2'.")


def get_regularization_functions(loss_type: str = "stylegan2") -> Dict[str, Callable]:
    """Get regularization functions for training."""
    if loss_type in ("stylegan2", "logistic"):
        return {"r1": r1_regularization, "path_length": path_length_regularization}
    return {}


class StyleGAN2ADALoss:
    """Combined loss computation with lazy regularization."""

    def __init__(
        self,
        r1_gamma: float = 10.0,
        pl_weight: float = 2.0,
        pl_decay: float = 0.01,
        r1_interval: int = 16,
        pl_interval: int = 4,
    ):
        self.r1_gamma = r1_gamma
        self.pl_weight = pl_weight
        self.pl_decay = pl_decay
        self.r1_interval = r1_interval
        self.pl_interval = pl_interval
        self.pl_mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def generator_loss(
        self,
        fake_logits: tf.Tensor,
        generator: keras.Model = None,
        latent_w: tf.Tensor = None,
        class_labels: tf.Tensor = None,
        iteration: int = 0,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        gan_loss = logistic_nonsaturating_generator_loss(fake_logits)
        losses = {"g_gan_loss": gan_loss}
        total_loss = gan_loss

        if generator is not None and latent_w is not None:
            if iteration > 0 and iteration % self.pl_interval == 0:
                pl_penalty, new_pl_mean = path_length_regularization(
                    generator,
                    latent_w,
                    class_labels,
                    pl_weight=self.pl_weight,
                    pl_decay=self.pl_decay,
                    pl_mean=self.pl_mean,
                )
                self.pl_mean.assign(new_pl_mean)
                total_loss = total_loss + pl_penalty * self.pl_interval
                losses["g_pl_penalty"] = pl_penalty

        return total_loss, losses

    def discriminator_loss(
        self,
        discriminator: keras.Model,
        real_images: tf.Tensor,
        real_logits: tf.Tensor,
        fake_logits: tf.Tensor,
        class_labels: tf.Tensor = None,
        iteration: int = 0,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        gan_loss = logistic_nonsaturating_discriminator_loss(real_logits, fake_logits)
        losses = {"d_gan_loss": gan_loss}
        total_loss = gan_loss

        if iteration > 0 and iteration % self.r1_interval == 0:
            r1_penalty = r1_regularization(
                discriminator, real_images, class_labels, gamma=self.r1_gamma
            )
            total_loss = total_loss + r1_penalty * self.r1_interval
            losses["d_r1_penalty"] = r1_penalty

        return total_loss, losses
