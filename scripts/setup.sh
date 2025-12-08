#!/bin/bash
# Quick setup script for bacterial-gan project
# Platform: Linux/macOS (requires bash)
# Installs Poetry and all dependencies

set -e  # Exit on error

echo "============================================="
echo "Bacterial GAN Project Setup"
echo "============================================="
echo ""
echo "⚠️  WARNING: This script is designed for Linux/macOS"
echo "    For Windows, use WSL or Git Bash"
echo ""

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    POETRY_PATH="$HOME/.local/bin"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    POETRY_PATH="$HOME/.local/bin"
else
    OS="Unknown"
    POETRY_PATH="$HOME/.local/bin"
    echo "⚠️  Unknown OS detected. Proceeding anyway..."
fi

echo "Detected OS: $OS"
echo ""

# Check for python3-venv (Common issue on Ubuntu VPS)
if [[ "$OS" == "Linux" ]] && command -v apt-get &> /dev/null; then
    if ! dpkg -s python3-venv &> /dev/null && ! python3 -c "import venv" &> /dev/null; then
        echo "⚠️  'python3-venv' might be missing."
        echo "    If Poetry fails, try running: sudo apt-get install -y python3-venv"
        echo ""
    fi
fi

# Check if Poetry is installed
POETRY_CMD=""
if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
    echo "✅ Poetry already installed ($(poetry --version))"
elif [ -f "$POETRY_PATH/poetry" ]; then
    POETRY_CMD="$POETRY_PATH/poetry"
    export PATH="$POETRY_PATH:$PATH"
    echo "✅ Poetry found at $POETRY_PATH"
else
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -

    # Add Poetry to PATH
    export PATH="$POETRY_PATH:$PATH"
    POETRY_CMD="poetry"

    echo "✅ Poetry installed to $POETRY_PATH"
    echo ""
    echo "⚠️  IMPORTANT: Add Poetry to your PATH permanently:"
    echo "    For bash: echo 'export PATH=\"$POETRY_PATH:\$PATH\"' >> ~/.bashrc"
    echo "    For zsh:  echo 'export PATH=\"$POETRY_PATH:\$PATH\"' >> ~/.zshrc"
    echo "    For fish: fish_add_path $POETRY_PATH"
    echo "              (or: set -Ua fish_user_paths $POETRY_PATH)"
    echo ""
    # Only pause if we actually installed it, so automated scripts don't hang
    # read -p "Press Enter to continue..."
fi

echo ""
echo "📦 Installing Python dependencies..."
echo "   (This includes TensorFlow with CUDA support as defined in pyproject.toml)"
$POETRY_CMD install

echo ""
echo "✅ Verifying GPU detection..."
if $POETRY_CMD run python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU available:', len(gpus) > 0); print('GPU devices:', gpus); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
    echo "✅ GPU detected successfully!"
else
    echo "⚠️  No GPU detected. Training will use CPU (much slower)."
    echo "    If you have an NVIDIA GPU, ensure drivers are installed:"
    echo "    nvidia-smi should show your GPU"
fi

echo ""
echo "✅ Verifying CLI..."
if $POETRY_CMD run bacterial-gan --help &> /dev/null; then
    echo "✅ 'bacterial-gan' CLI is ready."
else
    echo "❌ CLI check failed. Something went wrong."
    exit 1
fi

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "Installed:"
echo "  ✅ Poetry dependency manager"
echo "  ✅ Python packages (TensorFlow, etc.)"
echo ""
echo "Next steps:"
echo "  1. Add dataset to: data/01_raw/"
echo "  2. Start training: $POETRY_CMD run bacterial-gan train"
echo ""
echo "To clean up everything later:"
echo "  ./scripts/cleanup.sh"
echo ""
