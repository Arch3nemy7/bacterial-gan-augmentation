"""Test GAN architecture and memory usage."""

import tensorflow as tf
from bacterial_gan.models.architecture import build_generator, build_discriminator, build_cgan


def main():
    """Test model architectures."""
    print("=" * 80)
    print("BACTERIAL GAN ARCHITECTURE TEST")
    print("=" * 80)

    # Check GPU availability
    print("\n" + "-" * 80)
    print("GPU Information:")
    print("-" * 80)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"  Found GPU: {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Enabled memory growth for {gpu}")
            except RuntimeError as e:
                print(f"  Error setting memory growth: {e}")
    else:
        print("  No GPU found. Training will use CPU (very slow!).")
        print("  If you have a GPU, make sure TensorFlow can detect it.")

    # Build Generator
    print("\n" + "-" * 80)
    print("Building Generator (U-Net)...")
    print("-" * 80)

    generator = build_generator(
        latent_dim=100,
        num_classes=2,
        image_size=128,
        channels=3
    )

    print(f"\nGenerator Architecture:")
    generator.summary()

    # Test Generator forward pass
    print("\n" + "-" * 80)
    print("Testing Generator Forward Pass...")
    print("-" * 80)

    noise = tf.random.normal([4, 100])
    labels = tf.constant([0, 1, 0, 1], dtype=tf.int32)

    fake_images = generator([noise, labels], training=False)
    print(f"  Input noise shape: {noise.shape}")
    print(f"  Input labels: {labels.numpy()}")
    print(f"  Output image shape: {fake_images.shape}")
    print(f"  Output range: [{fake_images.numpy().min():.2f}, {fake_images.numpy().max():.2f}]")
    print(f"  Expected range: [-1.0, 1.0] (tanh activation)")

    # Build Discriminator
    print("\n" + "-" * 80)
    print("Building Discriminator (PatchGAN)...")
    print("-" * 80)

    discriminator = build_discriminator(
        image_size=128,
        channels=3,
        num_classes=2,
        use_spectral_norm=True
    )

    print(f"\nDiscriminator Architecture:")
    discriminator.summary()

    # Test Discriminator forward pass
    print("\n" + "-" * 80)
    print("Testing Discriminator Forward Pass...")
    print("-" * 80)

    predictions = discriminator([fake_images, labels], training=False)
    print(f"  Input image shape: {fake_images.shape}")
    print(f"  Input labels: {labels.numpy()}")
    print(f"  Output predictions shape: {predictions.shape}")
    print(f"  Output is PatchGAN: 8x8 patch predictions")
    print(f"  Prediction range: [{predictions.numpy().min():.2f}, {predictions.numpy().max():.2f}]")

    # Calculate model parameters
    print("\n" + "-" * 80)
    print("Model Parameters & Memory Estimation:")
    print("-" * 80)

    generator_params = generator.count_params()
    discriminator_params = discriminator.count_params()
    total_params = generator_params + discriminator_params

    print(f"  Generator parameters: {generator_params:,}")
    print(f"  Discriminator parameters: {discriminator_params:,}")
    print(f"  Total parameters: {total_params:,}")

    # Rough VRAM estimation
    # 4 bytes per param (float32) * 3 (weights + gradients + optimizer states)
    vram_estimate_gb = (total_params * 4 * 3) / (1024**3)
    print(f"\n  Estimated VRAM usage: ~{vram_estimate_gb:.2f} GB")
    print(f"  Your GTX 1650 Max Q: 4 GB VRAM")

    if vram_estimate_gb < 3.5:
        print(f"  ✅ Should fit comfortably in your GPU!")
    elif vram_estimate_gb < 4.0:
        print(f"  ⚠️  Tight fit. Reduce batch size if you get OOM errors.")
    else:
        print(f"  ❌ May not fit. Consider reducing model size.")

    # Test complete cGAN build
    print("\n" + "-" * 80)
    print("Testing Complete cGAN Build...")
    print("-" * 80)

    cgan = build_cgan(
        latent_dim=100,
        num_classes=2,
        image_size=128,
        channels=3,
        loss_type="wgan-gp"
    )

    print(f"  Generator: {cgan['generator'].name}")
    print(f"  Discriminator: {cgan['discriminator'].name}")
    print(f"  Loss type: WGAN-GP (Wasserstein GAN with Gradient Penalty)")

    # Test loss functions
    real_output = tf.random.normal([4, 8, 8, 1])
    fake_output = tf.random.normal([4, 8, 8, 1])

    g_loss = cgan['g_loss_fn'](fake_output)
    d_loss = cgan['d_loss_fn'](real_output, fake_output)

    print(f"\n  Test Generator Loss: {g_loss.numpy():.4f}")
    print(f"  Test Discriminator Loss: {d_loss.numpy():.4f}")

    print("\n" + "=" * 80)
    print("✅ ARCHITECTURE TEST COMPLETE!")
    print("=" * 80)
    print("\nYour GAN architecture is ready for training!")
    print("\nNext steps:")
    print("  1. Add your dataset to data/01_raw/")
    print("  2. Test data pipeline: poetry run python scripts/test_data_pipeline.py")
    print("  3. Start training: bacterial-gan train")
    print("\nRecommended settings for GTX 1650 Max Q:")
    print("  - Image size: 128x128 (already configured)")
    print("  - Batch size: 8 (reduce to 4 if OOM)")
    print("  - Enable mixed precision training for faster training")


if __name__ == "__main__":
    main()
