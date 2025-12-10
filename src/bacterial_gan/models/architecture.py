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
    Build conditional ResNet generator optimized for RTX 4070 Ti (12GB VRAM).

    Architecture Optimizations for RTX 4070 Ti:
    - Balanced capacity: 384 base filters (optimized for 12GB VRAM)
    - Deep network: 2 residual blocks per resolution for quality
    - Single self-attention: at 32x32 resolution (optimal for 256x256 images)
    - Latent space: 256 dimensions for rich representations
    - Optimized for 256x256 images with batch size 10-14
    - Designed for mixed precision (FP16) training
    - Memory efficient while maintaining high quality output

    Args:
        latent_dim: Dimension of latent noise vector (default: 256)
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

    # ========== INITIAL PROJECTION (BALANCED CAPACITY) ==========
    # Project to 8x8x384 feature map (optimized for RTX 4070 Ti)
    x = layers.Dense(8 * 8 * 384, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 384))(x)

    # Initial processing with 2 residual blocks
    x = residual_block(x, 384)
    x = residual_block(x, 384)

    # ========== UPSAMPLING PATH WITH DEEP RESIDUAL BLOCKS ==========

    # 8x8 -> 16x16 (Filters: 384)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 384)
    x = residual_block(x, 384)

    # 16x16 -> 32x32 (Filters: 256)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # ========== SELF-ATTENTION at 32x32 (optimal for 256x256 images) ==========
    x = SelfAttention(256)(x)

    # 32x32 -> 64x64 (Filters: 192)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 192)
    x = residual_block(x, 192)

    # 64x64 -> 128x128 (Filters: 96)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = residual_block(x, 96)
    x = residual_block(x, 96)

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

def residual_block_down(x, filters, kernel_size=3, downsample=True):
    """
    Residual block with downsampling for Discriminator (ResNet style).
    
    Structure:
    Input -> [Conv-LN-LeakyReLU] -> [Conv-LN-LeakyReLU] -> Add(Input_Shortcut)
    
    Note: Uses LayerNormalization (LN) instead of BatchNorm for WGAN-GP.
    """
    shortcut = x
    stride = 2 if downsample else 1
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)  # Dropout for discriminator regularization
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.LayerNormalization()(x)
    
    # Adjust shortcut if dimensions change
    if downsample or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        # No normalization on shortcut path usually, but can add if needed for stability
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.2)(x)
    return x


def build_discriminator(
    image_size: int = 256,
    channels: int = 3,
    num_classes: int = 2,
) -> keras.Model:
    """
    Build conditional PatchGAN discriminator optimized for RTX 4070 Ti (12GB VRAM).

    Architecture Optimizations for RTX 4070 Ti:
    - Reduced capacity to prevent discriminator dominance
    - Balanced capacity: 64->64->128->256 filter progression
    - Single self-attention: at 32x32 resolution
    - Strong regularization: Dropout + GaussianNoise + MinibatchDiscrimination
    - Optimized for 256x256 images
    
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
    x = GaussianNoise(stddev=0.05)(image_input)
    
    # Initial Conv (No residual usually for the very first layer mapping from RGB)
    x = layers.Conv2D(96, 4, strides=2, padding='same')(x) # 256 -> 128
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)

    # ========== RESIDUAL DOWN-SAMPLING BLOCKS ==========
    
    # 128x128 -> 64x64 (Filters: 96) - Increased from 64 for better capacity
    x = residual_block_down(x, 96)

    # 64x64 -> 32x32 (Filters: 160) - Increased from 128 for better capacity
    x = residual_block_down(x, 160)

    # ========== SELF-ATTENTION at 32x32 (optimal resolution) ==========
    x = SelfAttention(160)(x)

    # 32x32 -> 16x16 (Filters: 320) - Increased from 256 for better capacity
    x = residual_block_down(x, 320)

    # ========== MINIBATCH DISCRIMINATION ==========
    # Global pooling to reduce spatial dimensions
    pooled = layers.GlobalAveragePooling2D()(x)

    # Dense layers for final classification (feature extraction)
    dense = layers.Dense(320)(pooled) # Increased from 256 for better capacity
    dense = layers.LeakyReLU(0.2)(dense)
    dense = layers.Dropout(0.4)(dense)

    # Final feature vector (160 dimensions) - Increased from 128 for better capacity
    feature_vector = layers.Dense(160)(dense)
    feature_vector = layers.LeakyReLU(0.2)(feature_vector)
    feature_vector = layers.Dropout(0.3)(feature_vector)

    # ========== PROJECTION DISCRIMINATOR OUTPUT ==========
    # 1. Linear validity score from features (f(x))
    validity_score = layers.Dense(1, name='validity_score')(feature_vector)

    # 2. Class projection (y^T * V * x) -> implemented as dot product of embeddings
    # Embed class label to same dimension as feature vector
    class_embedding = layers.Embedding(num_classes, 160, name='projection_embedding')(class_input)
    class_embedding = layers.Flatten()(class_embedding)

    # Inner product (Projection)
    projection = layers.Dot(axes=1)([feature_vector, class_embedding])

    # Final output = f(x) + projection
    final_output = layers.Add(name='validity')([validity_score, projection])

    model = keras.Model(
        inputs=[image_input, class_input],
        outputs=final_output,
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
    Build complete conditional GAN optimized for RTX 4070 Ti (12GB VRAM).

    Args:
        latent_dim: Latent noise dimension (default: 256)
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
