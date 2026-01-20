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

# ============================================
# Python 3.11 Installation via pyenv
# ============================================
REQUIRED_PYTHON_VERSION="3.11.9"
PYTHON_MAJOR_MINOR="3.11"

echo "============================================="
echo "Checking Python 3.11 Installation"
echo "============================================="
echo ""

# Check if pyenv directory exists or command is available
if command -v pyenv &> /dev/null; then
    echo "✅ pyenv already installed ($(pyenv --version))"
elif [ -d "$HOME/.pyenv" ]; then
    echo "✅ pyenv directory found at $HOME/.pyenv"
    echo "   Setting up pyenv in current shell..."
else
    echo "📦 Installing pyenv..."

    # Install pyenv dependencies for Ubuntu/Debian
    if [[ "$OS" == "Linux" ]] && command -v apt-get &> /dev/null; then
        echo "   Installing pyenv build dependencies..."
        echo "   (You may be prompted for sudo password)"
        sudo apt-get update
        sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
            libffi-dev liblzma-dev git
    fi

    # Install pyenv
    curl https://pyenv.run | bash

    echo "✅ pyenv installed"
fi

# Always show the shell configuration instructions if pyenv command is not available
if ! command -v pyenv &> /dev/null; then
    echo ""
    echo "⚠️  IMPORTANT: Add pyenv to your shell configuration:"
    echo "    For bash, add to ~/.bashrc:"
    echo "      export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "      command -v pyenv >/dev/null || export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "      eval \"\$(pyenv init -)\""
    echo ""
    echo "    For zsh, add to ~/.zshrc:"
    echo "      export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "      command -v pyenv >/dev/null || export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "      eval \"\$(pyenv init -)\""
    echo ""
    echo "    Then run: source ~/.bashrc (or source ~/.zshrc)"
    echo ""
fi

# Ensure pyenv is in PATH for this script
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
fi

# Check if Python 3.11.9 is installed via pyenv
if pyenv versions | grep -q "$REQUIRED_PYTHON_VERSION"; then
    echo "✅ Python $REQUIRED_PYTHON_VERSION already installed via pyenv"
else
    echo "📦 Installing Python $REQUIRED_PYTHON_VERSION via pyenv..."
    echo "   (This may take several minutes...)"
    pyenv install "$REQUIRED_PYTHON_VERSION"
    echo "✅ Python $REQUIRED_PYTHON_VERSION installed"
fi

# Set Python 3.11.9 as local version for this project
echo "📌 Setting Python $REQUIRED_PYTHON_VERSION as local version for this project..."
pyenv local "$REQUIRED_PYTHON_VERSION"

# Verify Python version
CURRENT_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Current Python version: $CURRENT_PYTHON_VERSION"

# Verify that we're actually using Python 3.11
if [[ ! "$CURRENT_PYTHON_VERSION" =~ ^3\.11\. ]]; then
    echo "❌ ERROR: Expected Python 3.11.x but got $CURRENT_PYTHON_VERSION"
    echo "   pyenv may not be properly initialized in this shell."
    echo "   Try running these commands manually:"
    echo "   export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "   export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "   eval \"\$(pyenv init -)\""
    echo "   pyenv local $REQUIRED_PYTHON_VERSION"
    exit 1
fi
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
POETRY_NEEDS_REINSTALL=false

if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
    echo "✅ Poetry already installed ($(poetry --version))"
elif [ -f "$POETRY_PATH/poetry" ]; then
    POETRY_CMD="$POETRY_PATH/poetry"
    export PATH="$POETRY_PATH:$PATH"
    echo "✅ Poetry found at $POETRY_PATH"
    echo "   Checking Poetry version: $($POETRY_CMD --version)"
else
    POETRY_NEEDS_REINSTALL=true
fi

# Check if Poetry was installed with the correct Python
# Poetry stores its installation info in a predictable location
if [ -n "$POETRY_CMD" ] && [ -d "$HOME/.local/share/pypoetry/venv" ]; then
    POETRY_PYTHON_VERSION=$($HOME/.local/share/pypoetry/venv/bin/python --version 2>&1 | awk '{print $2}')
    echo "   Poetry's Python version: $POETRY_PYTHON_VERSION"
    if [[ ! "$POETRY_PYTHON_VERSION" =~ ^3\.11\. ]]; then
        echo "⚠️  Poetry was installed with Python $POETRY_PYTHON_VERSION"
        echo "   Need to reinstall with Python 3.11..."
        POETRY_NEEDS_REINSTALL=true
    fi
fi

if [ "$POETRY_NEEDS_REINSTALL" = true ]; then
    echo "📦 Installing Poetry with Python $PYTHON_MAJOR_MINOR..."

    # Remove old Poetry installation completely
    if [ -d "$HOME/.local/share/pypoetry" ]; then
        echo "   Removing old Poetry installation..."
        rm -rf "$HOME/.local/share/pypoetry"
    fi
    if [ -f "$POETRY_PATH/poetry" ]; then
        rm -f "$POETRY_PATH/poetry"
    fi
    # Also remove Poetry cache
    rm -rf "$HOME/.cache/pypoetry" 2>/dev/null || true

    # Use the pyenv Python to install Poetry
    curl -sSL https://install.python-poetry.org | python -

    # Add Poetry to PATH
    export PATH="$POETRY_PATH:$PATH"
    POETRY_CMD="poetry"

    echo "✅ Poetry installed to $POETRY_PATH using Python 3.11"
    echo ""
    echo "⚠️  IMPORTANT: Add Poetry to your PATH permanently:"
    echo "    For bash: echo 'export PATH=\"$POETRY_PATH:\$PATH\"' >> ~/.bashrc"
    echo "    For zsh:  echo 'export PATH=\"$POETRY_PATH:\$PATH\"' >> ~/.zshrc"
    echo "    For fish: fish_add_path $POETRY_PATH"
    echo "              (or: set -Ua fish_user_paths $POETRY_PATH)"
    echo ""
fi

echo ""
echo "📦 Configuring Poetry to use Python $PYTHON_MAJOR_MINOR..."

# Remove any existing Poetry virtual environments for this project
echo "   Removing any existing Poetry virtual environments..."
$POETRY_CMD env remove --all 2>/dev/null || true
rm -rf .venv 2>/dev/null || true

# Configure Poetry to use in-project virtualenvs
echo "   Configuring Poetry to use in-project virtualenvs..."
$POETRY_CMD config virtualenvs.in-project true --local

# Get the exact path to the pyenv Python
PYENV_PYTHON_PATH=$(pyenv which python)
echo "   Using Python from: $PYENV_PYTHON_PATH"

# Tell Poetry to use this specific Python version
echo "   Setting Poetry to use Python 3.11..."
$POETRY_CMD env use "$PYENV_PYTHON_PATH" || {
    echo "   Failed to use poetry env use, creating venv manually..."
    $PYENV_PYTHON_PATH -m venv .venv
}

echo ""
echo "📦 Installing Python dependencies..."
echo "   (This includes TensorFlow with CUDA support as defined in pyproject.toml)"

# Verify the venv was created with the correct Python
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
    echo "   Virtual environment Python version: $VENV_PYTHON_VERSION"

    if [[ "$VENV_PYTHON_VERSION" =~ ^3\.11\. ]]; then
        # Activate the venv and install with pip as a fallback
        echo "   Installing dependencies using pip in venv..."
        .venv/bin/pip install --upgrade pip setuptools wheel
        .venv/bin/pip install -e .
    else
        echo "❌ ERROR: Virtual environment has wrong Python version: $VENV_PYTHON_VERSION"
        exit 1
    fi
else
    # Try Poetry install
    $POETRY_CMD install
fi

echo ""
echo "✅ Verifying GPU detection..."
echo "   (This may take 15-30 seconds on virtual CPUs...)"
if [ -f ".venv/bin/python" ]; then
    # Use venv Python directly with timeout
    if timeout 60s .venv/bin/python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU available:', len(gpus) > 0); print('GPU devices:', gpus); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
        echo "✅ GPU detected successfully!"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "⚠️  GPU detection timed out (>60s). This is common on virtual CPUs."
            echo "    Skipping GPU check. Training will use available hardware."
        else
            echo "⚠️  No GPU detected. Training will use CPU (much slower)."
            echo "    If you have an NVIDIA GPU, ensure drivers are installed:"
            echo "    nvidia-smi should show your GPU"
        fi
    fi
else
    # Use Poetry with timeout
    if timeout 60s $POETRY_CMD run python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU available:', len(gpus) > 0); print('GPU devices:', gpus); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
        echo "✅ GPU detected successfully!"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "⚠️  GPU detection timed out (>60s). This is common on virtual CPUs."
            echo "    Skipping GPU check. Training will use available hardware."
        else
            echo "⚠️  No GPU detected. Training will use CPU (much slower)."
            echo "    If you have an NVIDIA GPU, ensure drivers are installed:"
            echo "    nvidia-smi should show your GPU"
        fi
    fi
fi

echo ""
echo "✅ Verifying CLI..."
echo "   (This may take 15-30 seconds on virtual CPUs due to TensorFlow import...)"
if [ -f ".venv/bin/bacterial-gan" ]; then
    # Use venv CLI directly with increased timeout
    if timeout 90s .venv/bin/bacterial-gan --help &> /dev/null; then
        echo "✅ 'bacterial-gan' CLI is ready."
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "⚠️  CLI verification timed out (>90s)."
            echo "    This is common on virtual CPUs. The CLI should still work, just slowly."
            echo "    Try running manually: .venv/bin/bacterial-gan --help"
        else
            echo "❌ CLI check failed with exit code $EXIT_CODE."
            echo "    Try running manually to see the error:"
            echo "    .venv/bin/bacterial-gan --help"
            exit 1
        fi
    fi
else
    # Use Poetry with increased timeout
    if timeout 90s $POETRY_CMD run bacterial-gan --help &> /dev/null; then
        echo "✅ 'bacterial-gan' CLI is ready."
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "⚠️  CLI verification timed out (>90s)."
            echo "    This is common on virtual CPUs. The CLI should still work, just slowly."
            echo "    Try running manually: $POETRY_CMD run bacterial-gan --help"
        else
            echo "❌ CLI check failed with exit code $EXIT_CODE."
            echo "    Try running manually to see the error:"
            echo "    $POETRY_CMD run bacterial-gan --help"
            exit 1
        fi
    fi
fi

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "Installed:"
echo "  ✅ pyenv (Python version manager)"
echo "  ✅ Python $REQUIRED_PYTHON_VERSION (via pyenv)"
echo "  ✅ Poetry dependency manager"
echo "  ✅ Python packages (TensorFlow, etc.)"
echo ""
echo "⚠️  IMPORTANT: If this is your first time installing pyenv:"
echo "   Add pyenv to your shell config (~/.bashrc or ~/.zshrc):"
echo "   export PYENV_ROOT=\"\$HOME/.pyenv\""
echo "   command -v pyenv >/dev/null || export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
echo "   eval \"\$(pyenv init -)\""
echo ""
echo "   Then reload your shell: source ~/.bashrc (or source ~/.zshrc)"
echo ""
echo "Next steps:"
echo "  1. Add dataset to: data/01_raw/"
if [ -f ".venv/bin/bacterial-gan" ]; then
    echo "  2. Start training: .venv/bin/bacterial-gan train"
    echo "     Or activate venv: source .venv/bin/activate && bacterial-gan train"
else
    echo "  2. Start training: $POETRY_CMD run bacterial-gan train"
fi
echo ""
echo "To clean up everything later:"
echo "  ./scripts/cleanup.sh"
echo ""
