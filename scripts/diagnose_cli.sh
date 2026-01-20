#!/bin/bash
# Diagnostic script to check why CLI verification fails

set -e

echo "============================================="
echo "CLI Diagnostic Script"
echo "============================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "1. Checking Python installation..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python --version
    echo "✅ Python found"
else
    echo "❌ No .venv/bin/python found"
    exit 1
fi

echo ""
echo "2. Checking if bacterial-gan executable exists..."
if [ -f ".venv/bin/bacterial-gan" ]; then
    echo "✅ .venv/bin/bacterial-gan exists"
    ls -lh .venv/bin/bacterial-gan
else
    echo "❌ .venv/bin/bacterial-gan not found"
    exit 1
fi

echo ""
echo "3. Testing basic CLI without TensorFlow import..."
echo "   (Setting TF_CPP_MIN_LOG_LEVEL=3 to suppress warnings)"
export TF_CPP_MIN_LOG_LEVEL=3
timeout 60s .venv/bin/bacterial-gan --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CLI responds within 60 seconds"
else
    echo "❌ CLI timed out or failed"
    exit 1
fi

echo ""
echo "4. Measuring TensorFlow import time..."
echo "   (This might take a while on virtual CPUs...)"
START=$(date +%s)
.venv/bin/python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>&1 | grep -v "^2026"
END=$(date +%s)
DIFF=$((END - START))
echo "   Import time: ${DIFF} seconds"

if [ $DIFF -gt 30 ]; then
    echo "⚠️  TensorFlow import is very slow (${DIFF}s). This is expected on virtual CPUs."
fi

echo ""
echo "5. Checking available memory..."
free -h

echo ""
echo "6. Checking CPU info..."
echo "CPU model: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "CPU cores: $(nproc)"
echo "CPU flags: $(grep flags /proc/cpuinfo | head -1 | cut -d: -f2 | grep -o 'avx\|avx2\|fma\|sse4_1\|sse4_2' | sort -u | tr '\n' ' ')"

echo ""
echo "============================================="
echo "Diagnostic Complete!"
echo "============================================="
