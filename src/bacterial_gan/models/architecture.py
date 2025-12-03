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
# GENERATOR (U-Net for 128x128 or 256x256 images)
# ============================================================================

def build_generator(
    latent_dim: int = 100,
    num_classes: int = 2,
    image_size: int = 128,
    channels: int = 3,
) -> keras.Model:
    """
    Build conditional U-Net generator with dynamic image size support.

    Architecture:
    - Memory-efficient design with smaller filter counts
    - Skip connections for better gradient flow
    - Batch normalization for training stability
    - Self-attention at bottleneck for global coherence
    - Dynamic upsampling based on target image_size

    Args:
        latent_dim: Dimension of latent noise vector (default: 100)
        num_classes: Number of classes for conditioning (default: 2)
        image_size: Output image size - 128 or 256 (default: 128)
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
    # Project to 8x8x192 feature map (increased capacity)
    x = layers.Dense(8 * 8 * 192, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 192))(x)

    # ========== SELF-ATTENTION at 8x8 (memory-efficient) ==========
    # Self-attention at 8x8 creates only 64x64 attention matrix
    # Much more memory-friendly than 4096x4096 at 64x64 resolution
    # Provides global coherence while fitting in GTX 1650 memory
    x = SelfAttention(192)(x)

    skip_8x8 = x  # Save for potential skip connection

    # ========== UPSAMPLING PATH WITH SKIP CONNECTIONS (U-Net style) ==========
    # 8x8x192 -> 16x16x192 (UpSampling + Conv2D to avoid checkerboard artifacts)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = layers.Conv2D(192, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    skip_16x16 = x  # Save for skip connection

    # 16x16x192 -> 32x32x192 (UpSampling + Conv2D)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = layers.Conv2D(192, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    skip_32x32 = x  # Save for skip connection

    # 32x32x192 -> 64x64x192 (UpSampling + Conv2D)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = layers.Conv2D(192, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Add skip connection from 32x32 (upsampled to 64x64)
    skip_32x32_up = layers.UpSampling2D(size=2, interpolation='bilinear')(skip_32x32)
    skip_32x32_up = layers.Conv2D(96, 3, padding='same', use_bias=False)(skip_32x32_up)
    skip_32x32_up = layers.BatchNormalization()(skip_32x32_up)
    skip_32x32_up = layers.LeakyReLU(0.2)(skip_32x32_up)
    x = layers.Concatenate()([x, skip_32x32_up])  # 64x64x(192+96)=288

    # Process concatenated features
    x = layers.Conv2D(192, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # ========== SELF-ATTENTION at 64x64 (DISABLED due to OOM on GTX 1650) ==========
    # Self-attention creates 4096x4096 attention matrix (64x64 spatial positions)
    # With batch_size=16, this requires ~1GB memory - too much for GTX 1650
    # if image_size == 128:
    #     x = SelfAttention(192)(x)

    # ========== UPSAMPLING TO 128x128 WITH SKIP CONNECTION ==========
    # 64x64x192 -> 128x128x64 (UpSampling + Conv2D)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Add skip connection from 16x16 (upsampled 3 times to 128x128: 16→32→64→128)
    skip_16x16_up = layers.UpSampling2D(size=2, interpolation='bilinear')(skip_16x16)  # 16→32
    skip_16x16_up = layers.Conv2D(96, 3, padding='same', use_bias=False)(skip_16x16_up)
    skip_16x16_up = layers.BatchNormalization()(skip_16x16_up)
    skip_16x16_up = layers.LeakyReLU(0.2)(skip_16x16_up)
    skip_16x16_up = layers.UpSampling2D(size=2, interpolation='bilinear')(skip_16x16_up)  # 32→64
    skip_16x16_up = layers.Conv2D(48, 3, padding='same', use_bias=False)(skip_16x16_up)
    skip_16x16_up = layers.BatchNormalization()(skip_16x16_up)
    skip_16x16_up = layers.LeakyReLU(0.2)(skip_16x16_up)
    skip_16x16_up = layers.UpSampling2D(size=2, interpolation='bilinear')(skip_16x16_up)  # 64→128
    skip_16x16_up = layers.Conv2D(32, 3, padding='same', use_bias=False)(skip_16x16_up)
    skip_16x16_up = layers.BatchNormalization()(skip_16x16_up)
    skip_16x16_up = layers.LeakyReLU(0.2)(skip_16x16_up)
    x = layers.Concatenate()([x, skip_16x16_up])  # 128x128x(64+32)=96

    # Process concatenated features
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # ========== CONDITIONAL UPSAMPLING FOR 256x256 ==========
    if image_size == 256:
        # 128x128x64 -> 256x256x64 (UpSampling + Conv2D)
        x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    # ========== OUTPUT ==========
    output = layers.Conv2D(channels, 7, padding='same', activation='tanh', name='output')(x)

    model = keras.Model(
        inputs=[noise_input, class_input],
        outputs=output,
        name='generator'
    )

    return model


# ============================================================================
# DISCRIMINATOR (PatchGAN for 128x128 or 256x256 images)
# ============================================================================

def build_discriminator(
    image_size: int = 128,
    channels: int = 3,
    num_classes: int = 2,
    use_spectral_norm: bool = True,
) -> keras.Model:
    """
    Build conditional PatchGAN discriminator with dynamic image size support.

    Architecture:
    - Memory-efficient with smaller filter counts
    - Spectral normalization for training stability
    - Class conditioning via embedding concatenation
    - PatchGAN output for local texture discrimination

    Args:
        image_size: Input image size - 128 or 256 (default: 128)
        channels: Number of input channels (default: 3)
        num_classes: Number of classes (default: 2)
        use_spectral_norm: Whether to use spectral normalization (default: True)

    Returns:
        Keras Model: Discriminator that takes [image, class_label] and outputs validity map
    """
    # ========== INPUTS ==========
    image_input = layers.Input(shape=(image_size, image_size, channels), name='image')
    class_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # Embed class label to a reasonable size (100 dims instead of image_size^2)
    class_embedding = layers.Embedding(num_classes, 100)(class_input)
    class_embedding = layers.Flatten()(class_embedding)  # [B, 100]

    # Expand and tile to spatial dimensions [B, 100] -> [B, H, W, 1]
    class_embedding = layers.Dense(image_size * image_size)(class_embedding)
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
        # Dropout removed - Spectral Normalization + Gradient Penalty provide sufficient regularization
        return x

    # 128x128x4 -> 64x64x64
    x = conv_block(x, 64, use_sn=use_spectral_norm)

    # 64x64x64 -> 32x32x128
    x = conv_block(x, 128, use_sn=use_spectral_norm)

    # 32x32x128 -> 16x16x256
    x = conv_block(x, 256, use_sn=use_spectral_norm)

    # 16x16x256 -> 8x8x384 (increased to better balance with generator capacity)
    x = conv_block(x, 384, use_sn=use_spectral_norm)

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
