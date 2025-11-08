"""GAN architecture optimized for bacterial image generation (GTX 1650)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class SpectralNormalization(keras.layers.Wrapper):
    """
    Spectral Normalization for stable GAN training.
    Constrains weight matrices to have spectral norm <= 1.

    Reference: Spectral Normalization for Generative Adversarial Networks
    """

    def __init__(self, layer, iteration=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        """Build wrapped layer and initialize singular vectors."""
        super().build(input_shape)

        # Get weight matrix
        self.w = self.layer.kernel
        w_shape = self.w.shape.as_list()

        # Initialize u vector for power iteration
        self.u = self.add_weight(
            shape=(1, w_shape[-1]),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )

    def call(self, inputs):
        """Apply spectral normalization to weights."""
        # Reshape weight matrix to 2D
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])

        # Power iteration to approximate spectral norm
        u_hat = self.u
        for _ in range(self.iteration):
            # v = W^T u
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
            # u = W v
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_reshaped))

        # Spectral norm: sigma = u^T W v
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))

        # Normalize weights by spectral norm
        self.layer.kernel.assign(self.w / sigma)

        # Update u for next iteration
        self.u.assign(u_hat)

        return self.layer(inputs)


class SelfAttention(layers.Layer):
    """
    Self-Attention layer for capturing long-range dependencies in images.

    Reference: Self-Attention Generative Adversarial Networks (SAGAN)
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        """Build attention components."""
        # Query, Key, Value convolutions
        self.query_conv = layers.Conv2D(self.channels // 8, 1)
        self.key_conv = layers.Conv2D(self.channels // 8, 1)
        self.value_conv = layers.Conv2D(self.channels, 1)

        # Learnable attention weight
        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        """Apply self-attention mechanism."""
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

        # Compute query, key, value
        query = self.query_conv(inputs)  # [B, H, W, C//8]
        key = self.key_conv(inputs)      # [B, H, W, C//8]
        value = self.value_conv(inputs)  # [B, H, W, C]

        # Reshape for matrix multiplication
        query = tf.reshape(query, [batch_size, -1, self.channels // 8])  # [B, H*W, C//8]
        key = tf.reshape(key, [batch_size, -1, self.channels // 8])      # [B, H*W, C//8]
        value = tf.reshape(value, [batch_size, -1, self.channels])       # [B, H*W, C]

        # Attention map: softmax(Q * K^T)
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))  # [B, H*W, H*W]

        # Apply attention to value
        out = tf.matmul(attention, value)  # [B, H*W, C]
        out = tf.reshape(out, [batch_size, height, width, self.channels])

        # Residual connection with learnable weight
        return self.gamma * out + inputs


# ============================================================================
# GENERATOR (U-Net for 128x128 images, optimized for GTX 1650)
# ============================================================================

def build_generator(
    latent_dim: int = 100,
    num_classes: int = 2,
    image_size: int = 128,
    channels: int = 3,
) -> keras.Model:
    """
    Build conditional U-Net generator optimized for GTX 1650 (4GB VRAM).

    Architecture:
    - Memory-efficient design with smaller filter counts
    - Skip connections for better gradient flow
    - Batch normalization for training stability
    - Self-attention at bottleneck for global coherence

    Args:
        latent_dim: Dimension of latent noise vector (default: 100)
        num_classes: Number of classes for conditioning (default: 2)
        image_size: Output image size (default: 128 for GTX 1650)
        channels: Number of output channels (default: 3 for RGB)

    Returns:
        Keras Model: Generator that takes [noise, class_label] and outputs images
    """
    # ========== INPUTS ==========
    noise_input = layers.Input(shape=(latent_dim,), name='noise')
    class_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # Embed class label to latent space
    class_embedding = layers.Embedding(num_classes, latent_dim)(class_input)
    class_embedding = layers.Flatten()(class_embedding)

    # Combine noise and class embedding
    combined = layers.Concatenate()([noise_input, class_embedding])

    # ========== INITIAL PROJECTION ==========
    # Project to 8x8x128 feature map
    x = layers.Dense(8 * 8 * 128, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 128))(x)

    # ========== UPSAMPLING PATH ==========
    # 8x8x128 -> 16x16x128
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 16x16x128 -> 32x32x128
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 32x32x128 -> 64x64x128
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # ========== SELF-ATTENTION at 64x64 ==========
    x = SelfAttention(128)(x)

    # ========== FINAL UPSAMPLING ==========
    # 64x64x128 -> 128x128x64
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # ========== OUTPUT ==========
    # 128x128x64 -> 128x128x3
    output = layers.Conv2D(channels, 7, padding='same', activation='tanh', name='output')(x)

    model = keras.Model(
        inputs=[noise_input, class_input],
        outputs=output,
        name='generator'
    )

    return model


# ============================================================================
# DISCRIMINATOR (PatchGAN optimized for GTX 1650)
# ============================================================================

def build_discriminator(
    image_size: int = 128,
    channels: int = 3,
    num_classes: int = 2,
    use_spectral_norm: bool = True,
) -> keras.Model:
    """
    Build conditional PatchGAN discriminator optimized for GTX 1650.

    Architecture:
    - Memory-efficient with smaller filter counts
    - Spectral normalization for training stability
    - Class conditioning via embedding concatenation
    - PatchGAN output for local texture discrimination

    Args:
        image_size: Input image size (default: 128)
        channels: Number of input channels (default: 3)
        num_classes: Number of classes (default: 2)
        use_spectral_norm: Whether to use spectral normalization (default: True)

    Returns:
        Keras Model: Discriminator that takes [image, class_label] and outputs validity map
    """
    # ========== INPUTS ==========
    image_input = layers.Input(shape=(image_size, image_size, channels), name='image')
    class_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # Embed class label and broadcast to image spatial dimensions
    class_embedding = layers.Embedding(num_classes, image_size * image_size)(class_input)
    class_embedding = layers.Reshape((image_size, image_size, 1))(class_embedding)

    # Concatenate image with class channel
    x = layers.Concatenate()([image_input, class_embedding])

    # ========== CONVOLUTIONAL LAYERS ==========
    def conv_block(x, filters, kernel_size=4, strides=2, use_sn=use_spectral_norm):
        """Convolutional block with optional spectral normalization."""
        conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')

        if use_sn:
            conv = SpectralNormalization(conv)

        x = conv(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        return x

    # 128x128x4 -> 64x64x64
    x = conv_block(x, 64, use_sn=use_spectral_norm)

    # 64x64x64 -> 32x32x128
    x = conv_block(x, 128, use_sn=use_spectral_norm)

    # 32x32x128 -> 16x16x256
    x = conv_block(x, 256, use_sn=use_spectral_norm)

    # 16x16x256 -> 8x8x512
    x = conv_block(x, 512, use_sn=use_spectral_norm)

    # ========== PATCHGAN OUTPUT ==========
    # Output: 8x8x1 patch predictions (no activation for WGAN)
    validity = layers.Conv2D(1, 4, padding='same', name='validity')(x)  # [B, 8, 8, 1]

    model = keras.Model(
        inputs=[image_input, class_input],
        outputs=validity,
        name='discriminator'
    )

    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def get_loss_functions(loss_type: str = "wgan-gp"):
    """
    Get loss functions for GAN training.

    Args:
        loss_type: Type of loss function
            - 'wgan-gp': Wasserstein GAN with Gradient Penalty (recommended)
            - 'lsgan': Least Squares GAN
            - 'vanilla': Original GAN with binary cross-entropy

    Returns:
        Tuple of (generator_loss_fn, discriminator_loss_fn)
    """
    if loss_type == "wgan-gp":
        # Wasserstein loss (no sigmoid, outputs are logits)
        def generator_loss(fake_output):
            """Generator tries to maximize discriminator output on fake images."""
            return -tf.reduce_mean(fake_output)

        def discriminator_loss(real_output, fake_output):
            """Discriminator tries to maximize gap between real and fake."""
            return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    elif loss_type == "lsgan":
        # Least Squares GAN - more stable training
        def generator_loss(fake_output):
            """Generator tries to make fake outputs close to 1."""
            return tf.reduce_mean((fake_output - 1) ** 2)

        def discriminator_loss(real_output, fake_output):
            """Discriminator tries to output 1 for real, 0 for fake."""
            real_loss = tf.reduce_mean((real_output - 1) ** 2)
            fake_loss = tf.reduce_mean(fake_output ** 2)
            return (real_loss + fake_loss) / 2

    else:  # vanilla
        # Binary cross-entropy loss (original GAN)
        cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

        def generator_loss(fake_output):
            """Generator tries to fool discriminator (label = 1)."""
            return cross_entropy(tf.ones_like(fake_output), fake_output)

        def discriminator_loss(real_output, fake_output):
            """Discriminator tries to classify real=1, fake=0."""
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            return real_loss + fake_loss

    return generator_loss, discriminator_loss


def gradient_penalty(discriminator, real_images, fake_images, class_labels, lambda_gp=10.0):
    """
    Calculate gradient penalty for WGAN-GP.

    Enforces 1-Lipschitz constraint by penalizing gradient norm != 1
    on interpolated samples between real and fake images.

    Args:
        discriminator: Discriminator model
        real_images: Batch of real images [B, H, W, C]
        fake_images: Batch of fake images [B, H, W, C]
        class_labels: Class labels [B, 1]
        lambda_gp: Gradient penalty coefficient (default: 10.0)

    Returns:
        Gradient penalty value (scalar)
    """
    # Cast to float32 for mixed precision training compatibility
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)

    batch_size = tf.shape(real_images)[0]

    # Random interpolation weight for each sample in batch
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)

    # Interpolated images: x_hat = alpha * real + (1 - alpha) * fake
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, class_labels], training=True)

    # Calculate gradients of discriminator output w.r.t. interpolated images
    gradients = tape.gradient(pred, interpolated)

    # Calculate L2 norm of gradients for each sample
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))

    # Gradient penalty: E[(||grad|| - 1)^2]
    gp = lambda_gp * tf.reduce_mean((gradients_norm - 1.0) ** 2)

    return gp


# ============================================================================
# MODEL BUILDING FUNCTIONS
# ============================================================================

def build_cgan(
    latent_dim: int = 100,
    num_classes: int = 2,
    image_size: int = 128,
    channels: int = 3,
    loss_type: str = "wgan-gp",
):
    """
    Build complete conditional GAN with generator and discriminator.

    Args:
        latent_dim: Latent noise dimension
        num_classes: Number of classes for conditioning
        image_size: Image size (height = width)
        channels: Number of image channels
        loss_type: Loss function type ('wgan-gp', 'lsgan', 'vanilla')

    Returns:
        Dictionary containing:
            - 'generator': Generator model
            - 'discriminator': Discriminator model
            - 'g_loss_fn': Generator loss function
            - 'd_loss_fn': Discriminator loss function
    """
    generator = build_generator(latent_dim, num_classes, image_size, channels)
    discriminator = build_discriminator(image_size, channels, num_classes)

    g_loss_fn, d_loss_fn = get_loss_functions(loss_type)

    return {
        'generator': generator,
        'discriminator': discriminator,
        'g_loss_fn': g_loss_fn,
        'd_loss_fn': d_loss_fn,
    }
