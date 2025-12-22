"""
StyleGAN2-ADA Architecture for conditional bacterial image generation.

Optimized for limited data scenarios with Adaptive Discriminator Augmentation.
Supports class-conditional generation (Gram-positive/Gram-negative).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for generator weights.

    Maintains a smoothed copy of model weights for stable inference and evaluation.
    This is a critical component for high-quality GAN training, as the raw generator
    weights can be noisy during training.

    Usage:
        ema = ExponentialMovingAverage(generator, decay=0.999)
        # During training:
        ema.update()
        # For inference/evaluation:
        ema.apply_shadow()  # Use EMA weights
        ... generate samples ...
        ema.restore()  # Restore original weights
    """

    def __init__(self, model: keras.Model, decay: float = 0.999):
        """
        Initialize EMA tracker.

        Args:
            model: The Keras model to track.
            decay: EMA decay rate (0.999-0.9999 typical). Higher = smoother.
        """
        self.model = model
        self.decay = decay
        self.shadow_variables = {}
        self.backup_variables = {}
        self._initialized = False
        self._var_count = 0

    def _initialize_shadows(self):
        """Lazily initialize shadow variables from model weights."""
        current_var_count = len(self.model.trainable_variables)

        # Re-initialize if model has new variables (e.g., after lazy building)
        if self._initialized and current_var_count == self._var_count:
            return

        # Clear old shadows and create new ones
        self.shadow_variables = {}
        for var in self.model.trainable_variables:
            # Create a copy with the same shape
            self.shadow_variables[var.name] = tf.Variable(
                tf.identity(var), trainable=False, name=f"ema_{var.name.replace('/', '_').replace(':', '_')}"
            )
        self._initialized = True
        self._var_count = current_var_count

    def update(self):
        """Update shadow variables with current model weights."""
        self._initialize_shadows()

        for var in self.model.trainable_variables:
            if var.name in self.shadow_variables:
                shadow_var = self.shadow_variables[var.name]
                # Check shapes match (they should after _initialize_shadows)
                if shadow_var.shape == var.shape:
                    # EMA update: shadow = decay * shadow + (1 - decay) * current
                    shadow_var.assign(
                        self.decay * shadow_var + (1.0 - self.decay) * var
                    )

    def apply_shadow(self):
        """Apply shadow (EMA) weights to the model. Call restore() to undo."""
        self._initialize_shadows()
        self.backup_variables = {}
        for var in self.model.trainable_variables:
            if var.name in self.shadow_variables:
                shadow_var = self.shadow_variables[var.name]
                if shadow_var.shape == var.shape:
                    # Store a numpy copy for backup
                    self.backup_variables[var.name] = var.numpy().copy()
                    var.assign(shadow_var)

    def restore(self):
        """Restore original weights after apply_shadow()."""
        for var in self.model.trainable_variables:
            if var.name in self.backup_variables:
                backup_value = self.backup_variables[var.name]
                if var.shape == backup_value.shape:
                    var.assign(backup_value)
        self.backup_variables = {}

    def get_shadow_variables(self) -> Dict[str, tf.Variable]:
        """Get dictionary of shadow variables for checkpointing."""
        self._initialize_shadows()
        return self.shadow_variables

    def set_shadow_variables(self, shadow_dict: Dict[str, np.ndarray]):
        """Load shadow variables from checkpoint data."""
        self._initialize_shadows()
        for name, value in shadow_dict.items():
            if name in self.shadow_variables:
                if self.shadow_variables[name].shape == value.shape:
                    self.shadow_variables[name].assign(value)


class PixelNorm(layers.Layer):
    """Pixel-wise feature normalization."""

    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon
        )


class LearnedConstant(layers.Layer):
    """Learned 4x4 constant input for synthesis network."""

    def __init__(self, channels: int, size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.size = size

    def build(self, input_shape):
        self.const = self.add_weight(
            name="const",
            shape=(1, self.size, self.size, self.channels),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, batch_size: int) -> tf.Tensor:
        # Tile const and let TensorFlow handle dtype via compute_dtype
        const = self.const
        # For mixed precision, cast to compute dtype
        if hasattr(self, '_compute_dtype') and self._compute_dtype:
            const = tf.cast(const, self._compute_dtype)
        return tf.tile(const, [batch_size, 1, 1, 1])


class ModulatedConv2D(layers.Layer):
    """Style-modulated convolution with weight demodulation."""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        demodulate: bool = True,
        gain: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.gain = gain

    def build(self, input_shape):
        feature_shape, style_shape = input_shape
        in_channels = feature_shape[-1]

        fan_in = in_channels * self.kernel_size * self.kernel_size
        self.wscale = float(self.gain / np.sqrt(fan_in))

        self.weight = self.add_weight(
            name="weight",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=tf.initializers.RandomNormal(stddev=1.0),
            trainable=True,
        )

        self.style_dense = layers.Dense(
            in_channels,
            kernel_initializer=tf.initializers.RandomNormal(stddev=1.0),
            bias_initializer="ones",
        )
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        x, style = inputs
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        in_channels = self.weight.shape[2]

        # Style modulation: compute per-sample modulation factors
        s = self.style_dense(style)  # [B, C_in]

        # Scale base weight and apply style modulation
        w = self.weight * self.wscale  # [k, k, C_in, C_out]
        # Modulate: w_mod[b, k1, k2, c_in, c_out] = w[k1, k2, c_in, c_out] * s[b, c_in]
        w = w[tf.newaxis, ...] * s[:, tf.newaxis, tf.newaxis, :, tf.newaxis]
        # w shape: [B, k, k, C_in, C_out]

        if self.demodulate:
            # Demodulation: normalize by L2 norm across spatial and input channels
            sigma = tf.sqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3]) + 1e-8)  # [B, C_out]
            w = w / sigma[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        # Vectorized convolution using im2col + einsum approach
        # Extract image patches: [B, H, W, k*k*C_in]
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape patches for einsum: [B, H, W, k*k, C_in]
        patches = tf.reshape(
            patches,
            [batch_size, height, width, self.kernel_size * self.kernel_size, in_channels],
        )

        # Reshape weights for einsum: [B, k*k, C_in, C_out]
        w_reshaped = tf.reshape(
            w,
            [batch_size, self.kernel_size * self.kernel_size, in_channels, self.filters],
        )

        # Efficient batched convolution via einsum
        # patches: [B, H, W, k*k, C_in], w_reshaped: [B, k*k, C_in, C_out]
        # output: [B, H, W, C_out]
        output = tf.einsum("bhwkc,bkco->bhwo", patches, w_reshaped)

        return output


class NoiseInjection(layers.Layer):
    """Per-pixel noise injection for stochastic variation."""

    def build(self, input_shape):
        self.noise_strength = self.add_weight(
            name="noise_strength",
            shape=(1, 1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        if noise is None:
            # Generate noise with same dtype as input for mixed precision compatibility
            noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1], dtype=x.dtype)
        else:
            # Cast provided noise to match input dtype
            noise = tf.cast(noise, x.dtype)
        # Cast noise_strength to input dtype for mixed precision
        noise_strength = tf.cast(self.noise_strength, x.dtype)
        return x + noise_strength * noise


class StyleBlock(layers.Layer):
    """Synthesis block: upsample -> modulated conv -> noise -> activation."""

    def __init__(self, filters: int, upsample: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.upsample = upsample

    def build(self, input_shape):
        self.mod_conv = ModulatedConv2D(self.filters, kernel_size=3)
        self.noise = NoiseInjection()
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.filters),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor], noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        x, style = inputs
        if self.upsample:
            x = tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[2] * 2], method="bilinear")
        x = self.mod_conv([x, style])
        x = self.noise(x, noise)
        # Cast bias to match input dtype for mixed precision compatibility
        x = x + tf.cast(self.bias, x.dtype)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x


class ToRGB(layers.Layer):
    """Convert features to RGB using 1x1 modulated convolution."""

    def build(self, input_shape):
        self.mod_conv = ModulatedConv2D(3, kernel_size=1, demodulate=False)
        self.bias = self.add_weight(
            name="bias", shape=(1, 1, 1, 3), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        x, style = inputs
        x = self.mod_conv([x, style])
        # Cast bias to match input dtype for mixed precision compatibility
        return x + tf.cast(self.bias, x.dtype)


class MappingNetwork(keras.Model):
    """Maps z -> w latent space with class conditioning."""

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_classes: int = 2,
        label_dim: int = 0,
        normalize_latent: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_dim = label_dim if label_dim > 0 else latent_dim
        self.normalize_latent = normalize_latent

        if normalize_latent:
            self.pixel_norm = PixelNorm()

        if num_classes > 0:
            self.class_embedding = layers.Embedding(
                num_classes,
                self.label_dim,
                embeddings_initializer=tf.initializers.RandomNormal(stddev=1.0),
            )

        self.mapping_layers = []
        for i in range(num_layers):
            in_dim = latent_dim + (self.label_dim if num_classes > 0 and i == 0 else 0)
            self.mapping_layers.append(
                layers.Dense(
                    latent_dim,
                    kernel_initializer=tf.initializers.RandomNormal(stddev=1.0 / np.sqrt(in_dim)),
                    bias_initializer="zeros",
                )
            )

    def call(self, z: tf.Tensor, class_labels: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self.normalize_latent:
            w = self.pixel_norm(z)
        else:
            w = z

        if class_labels is not None and self.num_classes > 0:
            if len(class_labels.shape) == 2:
                class_labels = tf.squeeze(class_labels, axis=-1)
            class_embed = self.class_embedding(class_labels)
            # Cast divisor to match dtype for mixed precision compatibility
            class_embed = class_embed / tf.sqrt(tf.cast(self.label_dim, class_embed.dtype))
            w = tf.concat([w, class_embed], axis=-1)

        for i, layer in enumerate(self.mapping_layers):
            w = layer(w)
            if i < len(self.mapping_layers) - 1:
                w = tf.nn.leaky_relu(w, alpha=0.2)

        return w


class SynthesisNetwork(keras.Model):
    """Generates images from w latent codes with progressive resolution."""

    def __init__(
        self,
        image_size: int = 256,
        latent_dim: int = 512,
        base_channels: int = 512,
        max_channels: int = 512,
        channels: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.max_channels = max_channels

        self.log2_size = int(np.log2(image_size))
        self.num_layers = (self.log2_size - 2) * 2 + 1

        channels_4x4 = self._get_channels(4)
        self.const_input = LearnedConstant(channels_4x4, size=4)
        self.first_block = StyleBlock(channels_4x4, upsample=False)
        self.first_to_rgb = ToRGB()

        self.blocks = []
        self.to_rgbs = []

        for res_log2 in range(3, self.log2_size + 1):
            res = 2**res_log2
            ch = self._get_channels(res)
            self.blocks.append(StyleBlock(ch, upsample=True))
            self.blocks.append(StyleBlock(ch, upsample=False))
            self.to_rgbs.append(ToRGB())

    def _get_channels(self, resolution: int) -> int:
        channels = self.base_channels * (4 / resolution)
        return min(max(int(channels), 64), self.max_channels)

    def call(self, w: tf.Tensor, noise_inputs: Optional[List[tf.Tensor]] = None) -> tf.Tensor:
        batch_size = tf.shape(w)[0]

        if len(w.shape) == 2:
            w_all = [w] * self.num_layers
        else:
            w_all = [w[:, i] for i in range(w.shape[1])]

        x = self.const_input(batch_size)
        noise = noise_inputs[0] if noise_inputs else None
        x = self.first_block([x, w_all[0]], noise=noise)
        rgb = self.first_to_rgb([x, w_all[0]])

        layer_idx = 1
        for i, to_rgb in enumerate(self.to_rgbs):
            # Block 1 (Upsample)
            block1 = self.blocks[i * 2]
            noise = (
                noise_inputs[layer_idx]
                if noise_inputs and layer_idx < len(noise_inputs)
                else None
            )
            x = block1([x, w_all[min(layer_idx, len(w_all) - 1)]], noise=noise)
            layer_idx += 1

            # Block 2 (Conv)
            block2 = self.blocks[i * 2 + 1]
            noise = (
                noise_inputs[layer_idx]
                if noise_inputs and layer_idx < len(noise_inputs)
                else None
            )
            x = block2([x, w_all[min(layer_idx, len(w_all) - 1)]], noise=noise)
            layer_idx += 1

            # RGB Update (cast for mixed precision compatibility)
            rgb = tf.image.resize(rgb, [tf.shape(x)[1], tf.shape(x)[2]], method="bilinear")
            new_rgb = to_rgb([x, w_all[min(layer_idx - 1, len(w_all) - 1)]])
            rgb = tf.cast(rgb, new_rgb.dtype) + new_rgb

        return rgb


class StyleGAN2Generator(keras.Model):
    """
    Complete StyleGAN2 generator with mapping and synthesis networks.

    Supports:
    - Style mixing regularization during training
    - Truncation trick for controlled generation
    - Per-layer w vectors for fine-grained control
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 256,
        num_classes: int = 2,
        mapping_layers: int = 8,
        channels: int = 3,
        base_channels: int = 512,
        style_mixing_prob: float = 0.9,
        w_avg_beta: float = 0.995,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_classes = num_classes
        self.style_mixing_prob = style_mixing_prob
        self.w_avg_beta = w_avg_beta

        self.mapping = MappingNetwork(
            latent_dim=latent_dim, num_layers=mapping_layers, num_classes=num_classes
        )
        self.synthesis = SynthesisNetwork(
            image_size=image_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channels=channels,
        )

        # Number of style layers in synthesis network
        self.num_ws = self.synthesis.num_layers

        # Running average of w for truncation trick
        self.w_avg = tf.Variable(
            tf.zeros([latent_dim]), trainable=False, name="w_avg"
        )

    def _update_w_avg(self, w: tf.Tensor):
        """Update running average of w vectors (in float32 for stability)."""
        # Cast to float32 for stable accumulation (avoid float16 precision issues)
        batch_avg = tf.cast(tf.reduce_mean(w, axis=0), tf.float32)
        self.w_avg.assign(
            self.w_avg_beta * self.w_avg + (1.0 - self.w_avg_beta) * batch_avg
        )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = True,
        style_mixing: bool = True,
    ) -> tf.Tensor:
        """
        Forward pass with optional style mixing.

        Args:
            inputs: [z, class_labels] where z is [B, latent_dim]
            training: Whether in training mode
            style_mixing: Whether to apply style mixing (only during training)
        """
        z, class_labels = inputs
        batch_size = tf.shape(z)[0]

        # Map z to w
        w = self.mapping(z, class_labels)

        # Update w_avg for truncation trick
        if training:
            self._update_w_avg(w)

        # Style mixing regularization
        if training and style_mixing and self.style_mixing_prob > 0:
            do_mixing = tf.random.uniform([]) < self.style_mixing_prob

            def apply_mixing():
                # Generate second latent code
                z2 = tf.random.normal([batch_size, self.latent_dim])
                w2 = self.mapping(z2, class_labels)

                # Random crossover point (1 to num_ws-1)
                crossover = tf.random.uniform(
                    [], 1, self.num_ws, dtype=tf.int32
                )

                # Create per-layer w vectors: [B, num_ws, latent_dim]
                w_broadcast = tf.tile(w[:, tf.newaxis, :], [1, self.num_ws, 1])
                w2_broadcast = tf.tile(w2[:, tf.newaxis, :], [1, self.num_ws, 1])

                # Create mixing mask (use same dtype as w for mixed precision compatibility)
                layer_idx = tf.range(self.num_ws)
                mask = tf.cast(layer_idx < crossover, w.dtype)
                mask = mask[tf.newaxis, :, tf.newaxis]  # [1, num_ws, 1]

                # Mix w and w2
                w_mixed = w_broadcast * mask + w2_broadcast * (tf.constant(1.0, dtype=w.dtype) - mask)
                return w_mixed

            def no_mixing():
                # Broadcast w to all layers: [B, num_ws, latent_dim]
                return tf.tile(w[:, tf.newaxis, :], [1, self.num_ws, 1])

            w_all = tf.cond(do_mixing, apply_mixing, no_mixing)
        else:
            # No mixing - broadcast w to all layers
            w_all = tf.tile(w[:, tf.newaxis, :], [1, self.num_ws, 1])

        # Generate images
        images = self.synthesis(w_all)
        return tf.clip_by_value(
            images,
            tf.constant(-1.0, dtype=images.dtype),
            tf.constant(1.0, dtype=images.dtype),
        )

    def generate_truncated(
        self,
        z: tf.Tensor,
        class_labels: Optional[tf.Tensor] = None,
        truncation_psi: float = 0.7,
        truncation_cutoff: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Generate images with truncation trick for higher quality.

        Args:
            z: Latent vectors [B, latent_dim]
            class_labels: Optional class labels [B] or [B, 1]
            truncation_psi: Truncation strength (0=w_avg, 1=no truncation)
            truncation_cutoff: Apply truncation only to first N layers (None=all)

        Returns:
            Generated images [B, H, W, C]
        """
        batch_size = tf.shape(z)[0]

        # Map z to w
        w = self.mapping(z, class_labels)

        # Create per-layer w: [B, num_ws, latent_dim]
        w_broadcast = tf.tile(w[:, tf.newaxis, :], [1, self.num_ws, 1])

        # Apply truncation
        if truncation_psi != 1.0:
            w_avg_broadcast = self.w_avg[tf.newaxis, tf.newaxis, :]

            if truncation_cutoff is None:
                # Apply to all layers
                w_truncated = w_avg_broadcast + truncation_psi * (w_broadcast - w_avg_broadcast)
            else:
                # Apply only to first truncation_cutoff layers (mixed precision compatible)
                layer_idx = tf.range(self.num_ws)
                mask = tf.cast(layer_idx < truncation_cutoff, w_broadcast.dtype)
                mask = mask[tf.newaxis, :, tf.newaxis]

                w_truncated_part = w_avg_broadcast + truncation_psi * (w_broadcast - w_avg_broadcast)
                w_truncated = w_truncated_part * mask + w_broadcast * (tf.constant(1.0, dtype=w_broadcast.dtype) - mask)

            w_broadcast = w_truncated

        # Generate images
        images = self.synthesis(w_broadcast)
        return tf.clip_by_value(
            images,
            tf.constant(-1.0, dtype=images.dtype),
            tf.constant(1.0, dtype=images.dtype),
        )


class MinibatchStdDev(layers.Layer):
    """
    Minibatch Standard Deviation layer for discriminator.

    Computes the standard deviation of features across the minibatch and
    appends it as an additional feature channel. This helps the discriminator
    detect mode collapse by making it aware of batch-level statistics.

    Reference: Progressive GAN, StyleGAN, StyleGAN2
    """

    def __init__(self, group_size: int = 4, num_features: int = 1, **kwargs):
        """
        Initialize MinibatchStdDev layer.

        Args:
            group_size: Number of samples in each group for stddev computation.
                        None or 0 means use the entire minibatch.
            num_features: Number of stddev features to output (usually 1).
        """
        super().__init__(**kwargs)
        self.group_size = group_size
        self.num_features = num_features

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x shape: [B, H, W, C]
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = x.shape[-1]

        # Determine group size (use full batch if group_size is 0 or None)
        group_size = self.group_size if self.group_size and self.group_size > 0 else batch_size
        group_size = tf.minimum(group_size, batch_size)

        # Ensure batch_size is divisible by group_size
        num_groups = batch_size // group_size

        # Number of channels per feature
        channels_per_feature = channels // self.num_features

        # Reshape: [G, M, H, W, F, C/F] where G=num_groups, M=group_size, F=num_features
        y = tf.reshape(
            x[:num_groups * group_size],
            [num_groups, group_size, height, width, self.num_features, channels_per_feature],
        )

        # Compute variance across the group (dim=1)
        y = y - tf.reduce_mean(y, axis=1, keepdims=True)  # Center
        y = tf.reduce_mean(tf.square(y), axis=1)  # Variance: [G, H, W, F, C/F]
        y = tf.sqrt(y + 1e-8)  # Stddev

        # Average over H, W, and channels per feature
        y = tf.reduce_mean(y, axis=[1, 2, 4])  # [G, F]

        # Tile to match spatial dimensions and batch size
        y = tf.reshape(y, [num_groups, 1, 1, self.num_features])
        y = tf.tile(y, [group_size, height, width, 1])

        # Handle case where batch_size isn't perfectly divisible
        y = y[:batch_size]

        # Concatenate with input
        return tf.concat([x, y], axis=-1)


class ResidualBlockDown(layers.Layer):
    """Residual block with downsampling for discriminator."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv2 = layers.Conv2D(self.filters, 3, padding="same")
        self.skip = layers.Conv2D(self.filters, 1)
        self.downsample = layers.AveragePooling2D(2)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = tf.nn.leaky_relu(x, alpha=0.2)
        h = self.conv1(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        h = self.conv2(h)
        h = self.downsample(h)

        skip = self.skip(x)
        skip = self.downsample(skip)

        return (h + skip) / np.sqrt(2.0)


class AdaptiveAugmentation(layers.Layer):
    """
    Adaptive Discriminator Augmentation (ADA).

    Dynamically adjusts augmentation probability based on discriminator overfitting.
    Critical for training with limited data.

    Augmentation pipeline (matching official StyleGAN2-ADA 'bgcfnc'):
    - b: Pixel blitting (brightness, contrast)
    - g: Geometric (translation, rotation, scaling)
    - c: Color (saturation, hue)
    - f: Filtering (blur)
    - n: Noise
    - c: Cutout
    """

    def __init__(
        self,
        p: float = 0.0,
        target: float = 0.6,
        speed: float = 0.001,
        enable_blitting: bool = True,
        enable_geometric: bool = True,
        enable_color: bool = True,
        enable_filtering: bool = True,
        enable_noise: bool = True,
        enable_cutout: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = tf.Variable(p, trainable=False, dtype=tf.float32)
        self.target = target
        self.speed = speed

        # Augmentation toggles
        self.enable_blitting = enable_blitting
        self.enable_geometric = enable_geometric
        self.enable_color = enable_color
        self.enable_filtering = enable_filtering
        self.enable_noise = enable_noise
        self.enable_cutout = enable_cutout

    def update_p(self, real_sign: tf.Tensor):
        """Adjust augmentation probability based on D(x) sign."""
        adjust = tf.sign(real_sign - self.target) * self.speed
        new_p = tf.clip_by_value(self.p + adjust, 0.0, 1.0)
        self.p.assign(new_p)

    def _apply_blitting(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Apply brightness and contrast adjustments (mixed precision compatible)."""
        dtype = images.dtype

        # Brightness
        brightness = tf.cast(tf.random.uniform([batch_size, 1, 1, 1], -0.2, 0.2), dtype)
        images = images + brightness

        # Contrast
        contrast = tf.cast(tf.random.uniform([batch_size, 1, 1, 1], 0.8, 1.2), dtype)
        mean = tf.reduce_mean(images, axis=[1, 2], keepdims=True)
        images = mean + contrast * (images - mean)

        return images

    def _apply_geometric(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Apply translation, rotation, and scaling (mixed precision compatible)."""
        # Save original dtype for later restoration
        original_dtype = images.dtype

        # ImageProjectiveTransformV3 requires float32 images
        images_f32 = tf.cast(images, tf.float32)

        h = tf.shape(images_f32)[1]
        w = tf.shape(images_f32)[2]
        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)

        # Random rotation (0, 90, 180, 270 degrees)
        # Generate rotation choice per image
        rot_choice = tf.random.uniform([batch_size], 0, 4, dtype=tf.int32)
        angles = tf.cast(rot_choice, tf.float32) * (np.pi / 2.0)

        # Translation (±12.5% of image size)
        tx = tf.random.uniform([batch_size], -0.125, 0.125) * w_f
        ty = tf.random.uniform([batch_size], -0.125, 0.125) * h_f

        # Scaling (85% to 115%)
        scale = tf.random.uniform([batch_size], 0.85, 1.15)

        # Build affine transformation matrix
        cos_a = tf.cos(angles)
        sin_a = tf.sin(angles)

        # Center coordinates
        cx = w_f / 2.0
        cy = h_f / 2.0

        # Affine matrix components (scale + rotate around center + translate)
        a0 = scale * cos_a
        a1 = -scale * sin_a
        a2 = tx + cx - a0 * cx - a1 * cy
        b0 = scale * sin_a
        b1 = scale * cos_a
        b2 = ty + cy - b0 * cx - b1 * cy

        transforms = tf.stack([a0, a1, a2, b0, b1, b2, tf.zeros(batch_size), tf.zeros(batch_size)], axis=1)

        images_transformed = tf.raw_ops.ImageProjectiveTransformV3(
            images=images_f32,
            transforms=transforms,
            output_shape=tf.shape(images_f32)[1:3],
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )

        # Cast back to original dtype
        return tf.cast(images_transformed, original_dtype)

    def _apply_color(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Apply saturation and hue adjustments (mixed precision compatible)."""
        dtype = images.dtype

        # Saturation
        gray = tf.reduce_mean(images, axis=-1, keepdims=True)
        sat = tf.cast(tf.random.uniform([batch_size, 1, 1, 1], 0.8, 1.2), dtype)
        images = gray + sat * (images - gray)

        # Hue rotation (simplified - rotate color channels)
        def apply_hue():
            # Slight hue shift using channel mixing
            hue_angle = tf.random.uniform([], -0.1, 0.1) * np.pi
            cos_h = tf.cast(tf.cos(hue_angle), dtype)
            sin_h = tf.cast(tf.sin(hue_angle), dtype)

            # Apply hue rotation matrix (simplified)
            r, g, b = images[..., 0:1], images[..., 1:2], images[..., 2:3]
            new_r = r * cos_h + g * sin_h
            new_g = g * cos_h - r * sin_h
            return tf.concat([new_r, new_g, b], axis=-1)

        # 30% chance to apply hue shift
        do_hue = tf.random.uniform([]) < 0.3
        images = tf.cond(do_hue, apply_hue, lambda: images)

        return images

    def _apply_filtering(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Apply Gaussian blur filtering (mixed precision compatible)."""
        dtype = images.dtype
        # Probabilistic blur (50% chance)
        do_blur = tf.random.uniform([]) < 0.5

        def apply_blur():
            # Simple 3x3 Gaussian kernel - cast to images dtype
            kernel = tf.constant(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype
            ) / tf.constant(16.0, dtype=dtype)
            kernel = tf.reshape(kernel, [3, 3, 1, 1])
            kernel = tf.tile(kernel, [1, 1, 3, 1])  # For each channel

            # Apply depthwise convolution for blur
            blurred = tf.nn.depthwise_conv2d(
                images, kernel, strides=[1, 1, 1, 1], padding="SAME"
            )
            return blurred

        return tf.cond(do_blur, apply_blur, lambda: images)

    def _apply_noise(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Add random noise to images (mixed precision compatible)."""
        # Generate noise in float32 (TF random doesn't support float16 stddev)
        noise_std = tf.random.uniform([], 0.0, 0.05)
        noise = tf.random.normal(tf.shape(images), stddev=noise_std, dtype=tf.float32)
        # Cast to images dtype
        noise = tf.cast(noise, images.dtype)
        return images + noise

    def _apply_cutout(self, images: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Apply random rectangular cutout."""
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)

        # Cutout size (10-30% of image)
        cutout_h = tf.maximum(tf.cast(tf.random.uniform([], 0.1, 0.3) * h_f, tf.int32), 1)
        cutout_w = tf.maximum(tf.cast(tf.random.uniform([], 0.1, 0.3) * w_f, tf.int32), 1)

        # Random position - ensure valid range
        max_y = tf.maximum(h - cutout_h, 1)
        max_x = tf.maximum(w - cutout_w, 1)
        y_start = tf.random.uniform([batch_size], 0, max_y, dtype=tf.int32)
        x_start = tf.random.uniform([batch_size], 0, max_x, dtype=tf.int32)

        # Create a single cutout mask (same for all batch items for simplicity)
        # This is more graph-friendly
        y_center = tf.random.uniform([], cutout_h // 2, h - cutout_h // 2, dtype=tf.int32)
        x_center = tf.random.uniform([], cutout_w // 2, w - cutout_w // 2, dtype=tf.int32)

        y_range = tf.range(h)
        x_range = tf.range(w)

        y_mask = tf.logical_and(
            y_range >= y_center - cutout_h // 2,
            y_range < y_center + cutout_h // 2,
        )  # [H]

        x_mask = tf.logical_and(
            x_range >= x_center - cutout_w // 2,
            x_range < x_center + cutout_w // 2,
        )  # [W]

        # Combine into 2D mask [H, W, 1] - cast to images dtype for mixed precision
        mask = tf.cast(
            tf.logical_and(y_mask[:, tf.newaxis], x_mask[tf.newaxis, :]),
            images.dtype,
        )[:, :, tf.newaxis]

        # Apply cutout (set masked region to 0)
        images = images * (tf.constant(1.0, dtype=images.dtype) - mask)

        return images

    def call(self, images: tf.Tensor, training: bool = True) -> tf.Tensor:
        if not training:
            return images

        batch_size = tf.shape(images)[0]
        dtype = images.dtype

        def apply_augmentation():
            aug_images = images

            # Apply enabled augmentations
            if self.enable_blitting:
                aug_images = self._apply_blitting(aug_images, batch_size)

            if self.enable_geometric:
                aug_images = self._apply_geometric(aug_images, batch_size)

            if self.enable_color:
                aug_images = self._apply_color(aug_images, batch_size)

            if self.enable_filtering:
                aug_images = self._apply_filtering(aug_images, batch_size)

            if self.enable_noise:
                aug_images = self._apply_noise(aug_images, batch_size)

            if self.enable_cutout:
                aug_images = self._apply_cutout(aug_images, batch_size)

            # Use dtype-aware clip values for mixed precision compatibility
            return tf.clip_by_value(
                aug_images,
                tf.constant(-1.0, dtype=dtype),
                tf.constant(1.0, dtype=dtype),
            )

        def no_augmentation():
            return images

        do_aug = tf.random.uniform([], 0.0, 1.0) < self.p
        return tf.cond(do_aug, apply_augmentation, no_augmentation)


class StyleGAN2Discriminator(keras.Model):
    """StyleGAN2 discriminator with ADA, minibatch stddev, and class conditioning."""

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 2,
        channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        use_ada: bool = True,
        ada_target: float = 0.6,
        mbstd_group_size: int = 4,
        mbstd_num_features: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_ada = use_ada

        if use_ada:
            self.ada = AdaptiveAugmentation(p=0.0, target=ada_target)

        log2_size = int(np.log2(image_size))
        self.from_rgb = layers.Conv2D(base_channels, 1)

        self.blocks = []
        in_channels = base_channels
        for i in range(log2_size - 2):
            out_channels = min(in_channels * 2, max_channels)
            self.blocks.append(ResidualBlockDown(out_channels))
            in_channels = out_channels

        # Minibatch standard deviation layer (helps prevent mode collapse)
        self.mbstd = MinibatchStdDev(
            group_size=mbstd_group_size, num_features=mbstd_num_features
        )

        # Final conv accounts for extra channel from mbstd
        self.final_conv = layers.Conv2D(in_channels, 3, padding="same")
        self.flatten = layers.Flatten()

        if num_classes > 0:
            self.class_embedding = layers.Embedding(
                num_classes,
                in_channels * 4 * 4,
                embeddings_initializer=tf.initializers.RandomNormal(stddev=1.0),
            )

        self.dense = layers.Dense(1)

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> tf.Tensor:
        images, class_labels = inputs

        if self.use_ada and training:
            images = self.ada(images, training=training)

        x = self.from_rgb(images)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        for block in self.blocks:
            x = block(x)

        # Apply minibatch stddev before final layers
        x = self.mbstd(x)

        x = self.final_conv(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.flatten(x)

        output = self.dense(x)

        if self.num_classes > 0 and class_labels is not None:
            if len(class_labels.shape) == 2:
                class_labels = tf.squeeze(class_labels, axis=-1)
            class_embed = self.class_embedding(class_labels)
            output = output + tf.reduce_sum(x * class_embed, axis=1, keepdims=True)

        return output

    def update_ada(self, real_logits: tf.Tensor):
        if self.use_ada:
            real_sign = tf.reduce_mean(tf.sign(real_logits))
            self.ada.update_p(real_sign)


class SimplifiedStyleGAN2Generator(keras.Model):
    """Memory-efficient StyleGAN2 generator for resource-constrained training."""

    def __init__(
        self,
        latent_dim: int = 256,
        image_size: int = 256,
        num_classes: int = 2,
        channels: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_classes = num_classes

        self.mapping = MappingNetwork(latent_dim=latent_dim, num_layers=4, num_classes=num_classes)

        self.num_blocks = int(np.log2(image_size)) - 2

        self.start_size = 4
        self.start_channels = 256

        self.dense = layers.Dense(
            self.start_size * self.start_size * self.start_channels, use_bias=False
        )
        self.reshape = layers.Reshape((self.start_size, self.start_size, self.start_channels))

        self.blocks = []

        all_channels = [256, 192, 128, 96, 64, 32]
        channel_progression = all_channels[: self.num_blocks]

        for out_channels in channel_progression:
            self.blocks.append(
                {
                    "upsample": layers.UpSampling2D(2, interpolation="bilinear"),
                    "conv1": layers.Conv2D(out_channels, 3, padding="same", use_bias=False),
                    "bn1": layers.BatchNormalization(),
                    "style1": layers.Dense(out_channels, use_bias=True),
                    "conv2": layers.Conv2D(out_channels, 3, padding="same", use_bias=False),
                    "bn2": layers.BatchNormalization(),
                }
            )

        self.to_rgb = layers.Conv2D(channels, 7, padding="same", activation="tanh")

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> tf.Tensor:
        z, class_labels = inputs
        w = self.mapping(z, class_labels)

        x = self.dense(w)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.reshape(x)

        for block in self.blocks:
            x = block["upsample"](x)
            x = block["conv1"](x)
            x = block["bn1"](x, training=training)

            style = block["style1"](w)
            style = tf.reshape(style, [-1, 1, 1, style.shape[-1]])
            x = x * (1 + style)

            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = block["conv2"](x)
            x = block["bn2"](x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)

        return self.to_rgb(x)


class SimplifiedStyleGAN2Discriminator(keras.Model):
    """Memory-efficient StyleGAN2 discriminator with ADA and minibatch stddev."""

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 2,
        channels: int = 3,
        use_ada: bool = True,
        ada_target: float = 0.6,
        mbstd_group_size: int = 4,
        mbstd_num_features: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_ada = use_ada

        if use_ada:
            self.ada = AdaptiveAugmentation(p=0.0, target=ada_target)

        self.num_blocks = int(np.log2(image_size)) - 2

        self.blocks = []
        all_channels = [64, 96, 128, 192, 256, 256]
        channel_progression = all_channels[: self.num_blocks]

        for out_channels in channel_progression:
            self.blocks.append(
                {
                    "conv1": layers.Conv2D(out_channels, 4, strides=2, padding="same"),
                    "ln": layers.LayerNormalization(),
                    "dropout": layers.Dropout(0.2),
                }
            )

        final_channels = channel_progression[-1] if channel_progression else 64
        self.final_feature_size = 4 * 4 * final_channels

        # Minibatch standard deviation layer (helps prevent mode collapse)
        self.mbstd = MinibatchStdDev(
            group_size=mbstd_group_size, num_features=mbstd_num_features
        )

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256)
        self.dense2 = layers.Dense(1)

        if num_classes > 0:
            self.class_embedding = layers.Embedding(num_classes, 256)

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> tf.Tensor:
        images, class_labels = inputs

        if self.use_ada and training:
            images = self.ada(images, training=training)

        x = images
        for block in self.blocks:
            x = block["conv1"](x)
            x = block["ln"](x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = block["dropout"](x, training=training)

        # Apply minibatch stddev before final layers
        x = self.mbstd(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        output = self.dense2(x)

        if self.num_classes > 0 and class_labels is not None:
            if len(class_labels.shape) == 2:
                class_labels = tf.squeeze(class_labels, axis=-1)
            class_embed = self.class_embedding(class_labels)
            projection = tf.reduce_sum(x[:, :256] * class_embed, axis=1, keepdims=True)
            output = output + projection

        return output

    def update_ada(self, real_logits: tf.Tensor):
        if self.use_ada:
            real_sign = tf.reduce_mean(tf.sign(real_logits))
            self.ada.update_p(real_sign)


def build_stylegan2_ada(
    latent_dim: int = 256,
    num_classes: int = 2,
    image_size: int = 256,
    channels: int = 3,
    use_simplified: bool = True,
    use_ada: bool = True,
    ada_target: float = 0.6,
) -> Dict[str, keras.Model]:
    """Build StyleGAN2-ADA generator and discriminator."""
    if use_simplified:
        generator = SimplifiedStyleGAN2Generator(
            latent_dim=latent_dim,
            image_size=image_size,
            num_classes=num_classes,
            channels=channels,
        )
        discriminator = SimplifiedStyleGAN2Discriminator(
            image_size=image_size,
            num_classes=num_classes,
            channels=channels,
            use_ada=use_ada,
            ada_target=ada_target,
        )
    else:
        generator = StyleGAN2Generator(
            latent_dim=latent_dim,
            image_size=image_size,
            num_classes=num_classes,
            channels=channels,
        )
        discriminator = StyleGAN2Discriminator(
            image_size=image_size,
            num_classes=num_classes,
            channels=channels,
            use_ada=use_ada,
            ada_target=ada_target,
        )

    return {"generator": generator, "discriminator": discriminator}
