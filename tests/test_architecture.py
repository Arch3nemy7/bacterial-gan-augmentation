"""Test StyleGAN2-ADA architecture builds correctly."""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from bacterial_gan.models import StyleGAN2ADA, build_stylegan2_ada


def test_stylegan2_ada():
    """Test StyleGAN2-ADA builds and generates correctly."""
    print("=" * 60)
    print("STYLEGAN2-ADA ARCHITECTURE TEST")
    print("=" * 60)

    print("\n1. Testing build_stylegan2_ada...")
    models = build_stylegan2_ada(
        latent_dim=256,
        num_classes=2,
        image_size=256,
        use_simplified=True,
        use_ada=True,
    )

    generator = models["generator"]
    discriminator = models["discriminator"]
    print(f"   ✅ Generator: {generator.name}")
    print(f"   ✅ Discriminator: {discriminator.name}")

    print("\n2. Testing forward pass...")
    noise = tf.random.normal([2, 256])
    labels = tf.constant([0, 1], dtype=tf.int32)

    fake_images = generator([noise, labels], training=False)
    print(f"   ✅ Generated: {fake_images.shape}")

    real_images = tf.random.normal([2, 256, 256, 3])
    real_logits = discriminator([real_images, labels], training=False)
    fake_logits = discriminator([fake_images, labels], training=False)
    print(f"   ✅ Real logits: {real_logits.shape}")
    print(f"   ✅ Fake logits: {fake_logits.shape}")

    print("\n3. Testing StyleGAN2ADA wrapper...")
    gan = StyleGAN2ADA(
        latent_dim=256,
        num_classes=2,
        image_size=256,
        use_simplified=True,
        use_ada=True,
        use_mixed_precision=False,
    )
    print(f"   ✅ Generator params: {gan.generator.count_params():,}")
    print(f"   ✅ Discriminator params: {gan.discriminator.count_params():,}")

    print("\n4. Testing sample generation...")
    samples = gan.generate_samples(num_samples=4)
    print(f"   ✅ Samples: {samples.shape}")
    print(f"   ✅ Range: [{samples.min():.2f}, {samples.max():.2f}]")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_stylegan2_ada()
