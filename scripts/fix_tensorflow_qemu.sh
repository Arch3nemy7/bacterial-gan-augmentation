#!/bin/bash
# Fix TensorFlow for QEMU virtual CPUs
# The pre-built TensorFlow wheels use AVX2/FMA instructions that QEMU doesn't support
# This script reinstalls TensorFlow with CPU-only support

set -e

echo "============================================="
echo "TensorFlow QEMU Compatibility Fix"
echo "============================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Problem: TensorFlow was built with AVX2/FMA CPU instructions"
echo "         that QEMU virtual CPUs don't support."
echo ""
echo "Solution: Reinstall TensorFlow CPU-only version"
echo ""

if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

echo "1. Uninstalling incompatible TensorFlow..."
.venv/bin/pip uninstall -y tensorflow tensorflow-intel || true

echo ""
echo "2. Installing CPU-only TensorFlow..."
echo "   Note: This removes CUDA support but works on virtual CPUs"

# Install the base TensorFlow without CUDA
# Use an older version if needed for better compatibility
.venv/bin/pip install tensorflow-cpu==2.15.0

echo ""
echo "3. Verifying TensorFlow installation..."
if .venv/bin/python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>&1; then
    echo "✅ TensorFlow installed successfully!"
else
    echo "❌ TensorFlow still failing. Trying alternative approach..."
    echo ""
    echo "   Installing older TensorFlow version with better CPU compatibility..."
    .venv/bin/pip uninstall -y tensorflow tensorflow-cpu || true
    .venv/bin/pip install tensorflow-cpu==2.13.0

    if .venv/bin/python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>&1; then
        echo "✅ TensorFlow 2.13.0 installed successfully!"
    else
        echo "❌ TensorFlow installation failed."
        echo ""
        echo "Your QEMU CPU may be too limited for pre-built TensorFlow."
        echo "Options:"
        echo "  1. Use a different server with real CPUs"
        echo "  2. Build TensorFlow from source (very slow, ~8+ hours)"
        echo "  3. Use TensorFlow 2.11 (last version with broad CPU support):"
        echo "     .venv/bin/pip install tensorflow-cpu==2.11.0"
        exit 1
    fi
fi

echo ""
echo "4. Testing CLI..."
if .venv/bin/bacterial-gan --help > /dev/null 2>&1; then
    echo "✅ CLI works!"
else
    echo "⚠️  CLI still has issues. Checking error..."
    .venv/bin/bacterial-gan --help || true
fi

echo ""
echo "============================================="
echo "Fix Complete!"
echo "============================================="
echo ""
echo "⚠️  IMPORTANT: You are now using tensorflow-cpu (no GPU support)"
echo "    Training will be MUCH slower on CPU-only."
echo ""
echo "Next steps:"
echo "  Test CLI: .venv/bin/bacterial-gan --help"
echo "  Start training: .venv/bin/bacterial-gan train"
echo ""
