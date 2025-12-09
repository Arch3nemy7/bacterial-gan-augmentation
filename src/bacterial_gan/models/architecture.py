"""GAN architecture optimized for bacterial image generation (RTX 4090 - 24GB VRAM)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# CUSTOM LAYERS
# ============================================================================


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


class MinibatchDiscrimination(layers.Layer):
    """
    Minibatch Discrimination layer to detect mode collapse.
    
    Compares each sample to all other samples in the batch to detect
    if the generator is producing similar images (mode collapse).
    
    Reference: Salimans et al. "Improved Techniques for Training GANs"
    """
    
    def __init__(self, num_kernels=50, kernel_dim=5, **kwargs):
        super().__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
    
    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        # Tensor T with shape [input_dim, num_kernels * kernel_dim]
        self.T = self.add_weight(
            name='T',
            shape=(self.input_dim, self.num_kernels * self.kernel_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        # x shape: [batch, features]
        # M shape: [batch, num_kernels, kernel_dim]
        M = tf.reshape(
            tf.matmul(x, self.T),
            [-1, self.num_kernels, self.kernel_dim]
        )
        
        # Compute L1 distance between all pairs in batch
        # M_i - M_j for all pairs
        M_expanded = tf.expand_dims(M, 0)  # [1, batch, num_kernels, kernel_dim]
        M_transposed = tf.expand_dims(M, 1)  # [batch, 1, num_kernels, kernel_dim]
        
        # L1 distance
        diffs = tf.reduce_sum(tf.abs(M_expanded - M_transposed), axis=3)  # [batch, batch, num_kernels]
        
        # Apply negative exponential
        c = tf.exp(-diffs)  # [batch, batch, num_kernels]
        
        # Sum over batch dimension (excluding self)
        # o(x_i) = sum_{j != i} c(x_i, x_j)
        o = tf.reduce_sum(c, axis=1) - 1  # Subtract 1 to exclude self-comparison
        
        # Concatenate with original features
        return tf.concat([x, o], axis=-1)


class GaussianNoise(layers.Layer):
    """
    Gaussian noise layer for discriminator input.
    Helps prevent discriminator from memorizing training data.
    """
    
    def __init__(self, stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.normal(tf.shape(inputs), stddev=self.stddev)
        return inputs


# ============================================================================
# GENERATOR (ResNet for 128x128 or 256x256 images)
# ============================================================================

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Residual block with skip connection.
    
    Structure:
    Input -> [Conv-BN-ReLU] -> [Conv-BN] -> Add(Input) -> ReLU
    """
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if dimensions change
    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.2)(x)
    return x


def build_generator(
    latent_dim: int = 256,
    num_classes: int = 2,
    image_size: int = 256,
    channels: int = 3,
) -> keras.Model:
    """
    Build conditional ResNet generator optimized for RTX 4090 (24GB VRAM, Balanced Capacity).

    Architecture Improvements for RTX 4090:
    - 4x increased capacity: 512 base filters (vs 256 on GTX 1650)
    - Deeper network: 2 residual blocks per resolution (vs 1)
    - Dual self-attention: at 32x32 and 64x64 resolutions (memory-optimized)
    - Larger latent space: 256 dimensions (vs 100)
    - Optimized for 256x256 images with batch size 12-20
    - Designed for mixed precision (FP16) training

    Args:
        latent_dim: Dimension of latent noise vector (default: 256, increased from 100)
        num_classes: Number of classes for conditioning (default: 2)
        image_size: Output image size - must be 256 (default: 256)
        channels: Number of output channels (default: 3 for RGB)

    Returns:
        Keras Model: Generator that takes [noise, class_label] and outputs images
    """
    if image_size != 256:
        raise ValueError(f"This generator is optimized for 256x256 images, got {image_size}")

    # ========== INPUTS ==========
    noise_input = layers.Input(shape=(latent_dim,), name='noise')
    class_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # Embed class label to latent space (larger embedding)
    class_embedding = layers.Embedding(num_classes, latent_dim)(class_input)
    class_embedding = layers.Flatten()(class_embedding)

    # Combine noise and class embedding
    combined = layers.Concatenate()([noise_input, class_embedding])

    # ========== INITIAL PROJECTION (HIGH CAPACITY) ==========
    # Project to 8x8x512 feature map (4x capacity increase from GTX 1650)
    x = layers.Dense(8 * 8 * 512, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 512))(x)

    # Initial processing with 2 residual blocks
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # ========== UPSAMPLING PATH WITH DEEP RESIDUAL BLOCKS ==========

    # 8x8 -> 16x16 (Filters: 512, 2x deeper)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # 16x16 -> 32x32 (Filters: 384)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 384)
    x = residual_block(x, 384)

    # ========== SELF-ATTENTION at 32x32 ==========
    x = SelfAttention(384)(x)

    # 32x32 -> 64x64 (Filters: 256)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # ========== SELF-ATTENTION at 64x64 ==========
    x = SelfAttention(256)(x)

    # 64x64 -> 128x128 (Filters: 128)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # 128x128 -> 256x256 (Filters: 64, final high-resolution layer)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # ========== OUTPUT ==========
    # Larger kernel for better texture synthesis at high resolution
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
    image_size: int = 256,
    channels: int = 3,
    num_classes: int = 2,
) -> keras.Model:
    """
    Build conditional PatchGAN discriminator optimized for RTX 4090 (24GB VRAM, Balanced Capacity).

    Architecture Improvements for RTX 4090:
    - 3x increased capacity: 96→192→384→512 filter progression (memory-optimized)
    - Dual self-attention: at 64x64 and 32x32 resolutions
    - Balanced depth: 4 conv layers for efficiency
    - Strong regularization: Dropout + GaussianNoise + MinibatchDiscrimination
    - Optimized for 256x256 images with batch size 12-20
    - Designed for mixed precision (FP16) training

    Args:
        image_size: Input image size - must be 256 (default: 256)
        channels: Number of input channels (default: 3)
        num_classes: Number of classes (default: 2)

    Returns:
        Keras Model: Discriminator that takes [image, class_label] and outputs validity score
    """
    if image_size != 256:
        raise ValueError(f"This discriminator is optimized for 256x256 images, got {image_size}")

    # ========== INPUTS ==========
    image_input = layers.Input(shape=(image_size, image_size, channels), name='image')
    class_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # ========== INPUT NOISE (prevents memorization) ==========
    x = GaussianNoise(stddev=0.05)(image_input)  # Reduced noise for high-capacity model

    # Embed class label
    class_embedding = layers.Embedding(num_classes, 1, name='class_emb')(class_input)
    class_embedding = layers.Flatten()(class_embedding)
    class_embedding = layers.Reshape((1, 1, 1))(class_embedding)

    # Tile to match image spatial dimensions
    class_embedding = layers.Lambda(lambda x: tf.tile(x, [1, image_size, image_size, 1]))(class_embedding)

    # Concatenate image with class channel
    x = layers.Concatenate()([x, class_embedding])

    # ========== CONVOLUTIONAL LAYERS (MEMORY-OPTIMIZED) ==========
    def conv_block(x, filters, kernel_size=4, strides=2, dropout_rate=0.3):
        """Convolutional block with LayerNorm and dropout."""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    # 256x256x4 -> 128x128x96 (1.5x increase from 64, memory-efficient)
    x = conv_block(x, 96, dropout_rate=0.2)

    # 128x128x96 -> 64x64x192 (2x increase)
    x = conv_block(x, 192, dropout_rate=0.25)

    # ========== SELF-ATTENTION at 64x64 ==========
    x = SelfAttention(192)(x)

    # 64x64x192 -> 32x32x384 (2x increase)
    x = conv_block(x, 384, dropout_rate=0.3)

    # ========== SELF-ATTENTION at 32x32 ==========
    x = SelfAttention(384)(x)

    # 32x32x384 -> 16x16x512 (capped at 512 to prevent OOM)
    x = conv_block(x, 512, dropout_rate=0.35)

    # ========== MINIBATCH DISCRIMINATION ==========
    # Global pooling to reduce spatial dimensions
    pooled = layers.GlobalAveragePooling2D()(x)

    # Minibatch discrimination with balanced capacity
    mb_features = MinibatchDiscrimination(num_kernels=100, kernel_dim=8)(pooled)

    # Dense layers for final classification
    dense = layers.Dense(384)(mb_features)
    dense = layers.LeakyReLU(0.2)(dense)
    dense = layers.Dropout(0.4)(dense)

    dense = layers.Dense(192)(dense)
    dense = layers.LeakyReLU(0.2)(dense)
    dense = layers.Dropout(0.3)(dense)

    # ========== OUTPUT ==========
    # Single validity score (for WGAN, no activation)
    validity = layers.Dense(1, name='validity')(dense)

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

    # Handle case where gradients might be None (shouldn't happen, but safety check)
    if gradients is None:
        return tf.constant(0.0, dtype=tf.float32)

    # Calculate L2 norm of gradients for each sample
    # Add epsilon for numerical stability to prevent sqrt(0) issues
    gradients_squared = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    gradients_norm = tf.sqrt(gradients_squared + 1e-8)

    # Gradient penalty: E[(||grad|| - 1)^2]
    gp = lambda_gp * tf.reduce_mean((gradients_norm - 1.0) ** 2)

    return gp


# ============================================================================
# MODEL BUILDING FUNCTIONS
# ============================================================================

def build_cgan(
    latent_dim: int = 256,
    num_classes: int = 2,
    image_size: int = 256,
    channels: int = 3,
    loss_type: str = "wgan-gp",
):
    """
    Build complete conditional GAN optimized for RTX 4090 (24GB VRAM).

    Args:
        latent_dim: Latent noise dimension (default: 256, increased from 100)
        num_classes: Number of classes for conditioning (default: 2)
        image_size: Image size (height = width, must be 256)
        channels: Number of image channels (default: 3)
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
