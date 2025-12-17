#!/usr/bin/env python3
"""
Test training pipeline with dummy data.

This script tests the complete training pipeline without requiring real dataset.
It runs a few epochs with synthetic data to verify everything works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from bacterial_gan.config import get_settings
from bacterial_gan.pipelines import train_pipeline


def main():
    print("=" * 80)
    print("TRAINING PIPELINE TEST")
    print("=" * 80)
    print()
    print("This will test the training pipeline with DUMMY data (10 batches).")
    print("No real bacterial images are required.")
    print()

    config_path = project_root / "configs" / "config.yaml"
    settings = get_settings(str(config_path))

    original_epochs = settings.training.epochs
    settings.training.epochs = 3

    print(f"Configuration:")
    print(f"  Image size: {settings.training.image_size}")
    print(f"  Batch size: {settings.training.batch_size}")
    print(f"  Epochs: {settings.training.epochs} (reduced from {original_epochs} for testing)")
    print(f"  Learning rate: {settings.training.learning_rate}")
    print(f"  Loss type: {settings.training.loss_type}")
    print()

    try:
        if sys.stdin.isatty():
            input("Press Enter to start training test...")
        else:
            print("Running in non-interactive mode, starting training...")
    except:
        print("Starting training...")
    print()

    try:
        train_pipeline.run(settings)
        print()
        print("=" * 80)
        print("✅ TRAINING TEST SUCCESSFUL!")
        print("=" * 80)
        print()
        print("The training pipeline is working correctly.")
        print("When you have real bacterial images, you can:")
        print("  1. Add them to data/01_raw/gram_positive/ and data/01_raw/gram_negative/")
        print("  2. Run full training: bacterial-gan train")
        print()
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TRAINING TEST FAILED!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
