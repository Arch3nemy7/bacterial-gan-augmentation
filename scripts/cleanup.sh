#!/bin/bash
# Cleanup script for bacterial-gan project
# Platform: Linux/macOS (requires bash)
# This removes all installed dependencies and virtual environments

set -e  # Exit on error

echo "============================================="
echo "Bacterial GAN Project Cleanup"
echo "============================================="
echo ""
echo "⚠️  WARNING: This script is designed for Linux/macOS"
echo "    For Windows, use WSL or Git Bash"
echo ""

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check if Poetry is available
POETRY_CMD=""
if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
elif [ -f "$HOME/.local/bin/poetry" ]; then
    POETRY_CMD="$HOME/.local/bin/poetry"
else
    echo "⚠️  Poetry not found. Skipping virtual environment removal."
fi

# Remove virtual environment
if [ -n "$POETRY_CMD" ]; then
    echo "🗑️  Removing Poetry virtual environment..."

    # Get virtualenv info before removal
    VENV_PATH=$($POETRY_CMD env info --path 2>/dev/null || echo "")

    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        VENV_SIZE=$(du -sh "$VENV_PATH" 2>/dev/null | cut -f1 || echo "unknown size")
        echo "    Found virtualenv: $VENV_PATH ($VENV_SIZE)"
        $POETRY_CMD env remove --all
        echo "    ✅ Removed"
    else
        echo "    ⚠️  No virtual environment found for this project"
    fi

    # Also remove .venv if it exists in project
    if [ -d ".venv" ]; then
        VENV_SIZE=$(du -sh ".venv" 2>/dev/null | cut -f1 || echo "unknown size")
        echo "    Found .venv in project directory ($VENV_SIZE)"
        rm -rf ".venv"
        echo "    ✅ Removed"
    fi
fi

echo ""
echo "🗑️  Cleaning Poetry and pip caches (where CUDA libraries are stored)..."
TOTAL_CACHE_SIZE=0

# Poetry cache (this is where the ~2.7GB CUDA wheels are cached!)
if [ -d "$HOME/.cache/pypoetry" ]; then
    CACHE_SIZE=$(du -sh "$HOME/.cache/pypoetry" 2>/dev/null | cut -f1 || echo "0")
    echo "    Poetry cache: ~/.cache/pypoetry ($CACHE_SIZE)"
    read -p "    Remove Poetry cache? This includes cached CUDA libraries (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$HOME/.cache/pypoetry"
        echo "    ✅ Poetry cache removed"
    else
        echo "    ⏭️  Keeping Poetry cache"
    fi
fi

# pip cache
if [ -d "$HOME/.cache/pip" ]; then
    CACHE_SIZE=$(du -sh "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo "0")
    echo "    pip cache: ~/.cache/pip ($CACHE_SIZE)"
    read -p "    Remove pip cache? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$HOME/.cache/pip"
        echo "    ✅ pip cache removed"
    else
        echo "    ⏭️  Keeping pip cache"
    fi
fi

echo ""
echo "🗑️  Cleaning Python cache files in project..."
if command -v make &> /dev/null && [ -f "Makefile" ]; then
    make clean 2>/dev/null
else
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
fi
echo "✅ Python cache cleaned"

echo ""
echo "🗑️  Removing MLflow artifacts..."
if [ -d "mlruns" ] || [ -d "mlartifacts" ]; then
    MLFLOW_SIZE=$(du -sh mlruns mlartifacts 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")
    rm -rf mlruns/ mlartifacts/
    echo "✅ MLflow artifacts removed"
else
    echo "⏭️  No MLflow artifacts found"
fi

echo ""
echo "🗑️  Removing generated models (optional)..."
if ls models/*.h5 2>/dev/null || ls models/*.keras 2>/dev/null; then
    read -p "Do you want to remove trained models in models/? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf models/*.h5 models/*.keras 2>/dev/null || true
        echo "✅ Models removed"
    else
        echo "⏭️  Keeping trained models"
    fi
else
    echo "⏭️  No trained models found"
fi

echo ""
echo "============================================="
echo "Cleanup Complete!"
echo "============================================="
echo ""
echo "What was removed:"
echo "  ✅ Poetry virtual environment (if existed)"
echo "  ✅ Python cache files in project"
echo "  ✅ MLflow run artifacts (if existed)"
echo "  ✅ Poetry/pip caches (if you confirmed)"
echo ""
echo "What remains:"
echo "  📁 Source code (src/, app/, configs/, etc.)"
echo "  📁 Poetry tool itself (~/.local/bin/poetry)"
echo "  📁 System packages (if any)"
echo ""
echo "💡 To completely remove Poetry:"
echo "    curl -sSL https://install.python-poetry.org | python3 - --uninstall"
echo ""
echo "💡 To reinstall everything:"
echo "    ./scripts/setup.sh"
echo ""
echo "💡 To check for any remaining residue:"
echo "    du -sh ~/.cache/pypoetry ~/.cache/pip 2>/dev/null"
echo ""
