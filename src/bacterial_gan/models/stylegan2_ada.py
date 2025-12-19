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
        return tf.tile(self.const, [batch_size, 1, 1, 1])


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
        self.wscale = tf.cast(self.gain / np.sqrt(fan_in), tf.float32)

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
        in_channels = x.shape[-1]

        s = self.style_dense(style)
        w = self.weight * self.wscale
        w = w[tf.newaxis, ...] * s[:, tf.newaxis, tf.newaxis, :, tf.newaxis]

        if self.demodulate:
            sigma = tf.sqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3]) + 1e-8)
            w = w / sigma[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        x = tf.reshape(x, [1, tf.shape(x)[1], tf.shape(x)[2], batch_size * in_channels])
        w = tf.transpose(w, [1, 2, 3, 0, 4])
        w = tf.reshape(
            w, [self.kernel_size, self.kernel_size, in_channels * batch_size, self.filters]
        )

        x = tf.nn.conv2d(x, w, strides=1, padding="SAME")
        x = tf.reshape(x, [batch_size, tf.shape(x)[1], tf.shape(x)[2], self.filters])
        return x


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
            noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
        return x + self.noise_strength * noise


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
        x = x + self.bias
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
        return x + self.bias


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
            class_embed = class_embed / tf.sqrt(tf.cast(self.label_dim, tf.float32))
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
        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgbs)):
            noise = (
                noise_inputs[layer_idx] if noise_inputs and layer_idx < len(noise_inputs) else None
            )
            x = block([x, w_all[min(layer_idx, len(w_all) - 1)]], noise=noise)

            if i % 2 == 1:
                rgb = tf.image.resize(rgb, [tf.shape(x)[1], tf.shape(x)[2]], method="bilinear")
                rgb = rgb + to_rgb([x, w_all[min(layer_idx, len(w_all) - 1)]])

            layer_idx += 1

        return rgb


class StyleGAN2Generator(keras.Model):
    """Complete StyleGAN2 generator with mapping and synthesis networks."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 256,
        num_classes: int = 2,
        mapping_layers: int = 8,
        channels: int = 3,
        base_channels: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_classes = num_classes

        self.mapping = MappingNetwork(
            latent_dim=latent_dim, num_layers=mapping_layers, num_classes=num_classes
        )
        self.synthesis = SynthesisNetwork(
            image_size=image_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channels=channels,
        )

    def call(self, inputs: List[tf.Tensor], training: bool = True) -> tf.Tensor:
        z, class_labels = inputs
        w = self.mapping(z, class_labels)
        images = self.synthesis(w)
        return tf.clip_by_value(images, -1.0, 1.0)


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
    """

    def __init__(self, p: float = 0.0, target: float = 0.6, speed: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.p = tf.Variable(p, trainable=False, dtype=tf.float32)
        self.target = target
        self.speed = speed

    def update_p(self, real_sign: tf.Tensor):
        """Adjust augmentation probability based on D(x) sign."""
        adjust = tf.sign(real_sign - self.target) * self.speed
        new_p = tf.clip_by_value(self.p + adjust, 0.0, 1.0)
        self.p.assign(new_p)

    def call(self, images: tf.Tensor, training: bool = True) -> tf.Tensor:
        if not training:
            return images

        batch_size = tf.shape(images)[0]

        def apply_augmentation():
            aug_images = images

            aug_images = aug_images + tf.random.uniform([batch_size, 1, 1, 1], -0.2, 0.2)
            contrast = tf.random.uniform([batch_size, 1, 1, 1], 0.8, 1.2)
            aug_images = aug_images * contrast

            gray = tf.reduce_mean(aug_images, axis=-1, keepdims=True)
            sat = tf.random.uniform([batch_size, 1, 1, 1], 0.8, 1.2)
            aug_images = gray + sat * (aug_images - gray)

            tx = tf.random.uniform([batch_size], -0.125, 0.125)
            ty = tf.random.uniform([batch_size], -0.125, 0.125)

            h, w = tf.shape(aug_images)[1], tf.shape(aug_images)[2]
            transforms = tf.stack(
                [
                    tf.ones(batch_size),
                    tf.zeros(batch_size),
                    tx * tf.cast(w, tf.float32),
                    tf.zeros(batch_size),
                    tf.ones(batch_size),
                    ty * tf.cast(h, tf.float32),
                    tf.zeros(batch_size),
                    tf.zeros(batch_size),
                ],
                axis=1,
            )

            aug_images = tf.raw_ops.ImageProjectiveTransformV3(
                images=aug_images,
                transforms=transforms,
                output_shape=tf.shape(aug_images)[1:3],
                interpolation="BILINEAR",
                fill_mode="REFLECT",
                fill_value=0.0,
            )

            return tf.clip_by_value(aug_images, -1.0, 1.0)

        def no_augmentation():
            return images

        do_aug = tf.random.uniform([], 0.0, 1.0) < self.p
        return tf.cond(do_aug, apply_augmentation, no_augmentation)


class StyleGAN2Discriminator(keras.Model):
    """StyleGAN2 discriminator with ADA and class conditioning."""

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 2,
        channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        use_ada: bool = True,
        ada_target: float = 0.6,
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
    """Memory-efficient StyleGAN2 discriminator with ADA."""

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 2,
        channels: int = 3,
        use_ada: bool = True,
        ada_target: float = 0.6,
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
