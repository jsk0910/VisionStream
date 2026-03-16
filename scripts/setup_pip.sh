#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  VisionStream — pip + venv 자동 설정 스크립트
# ═══════════════════════════════════════════════════════════
#  Usage:
#    bash scripts/setup_pip.sh [CUDA_VERSION]
#
#  Examples:
#    bash scripts/setup_pip.sh 11.8
#    bash scripts/setup_pip.sh 12.1    (default)
#    bash scripts/setup_pip.sh 12.8
# ═══════════════════════════════════════════════════════════

set -e

# Default CUDA version
CUDA_VER="${1:-12.1}"

# Map version to file suffix (e.g., 12.1 -> 121)
CUDA_SUFFIX=$(echo "$CUDA_VER" | tr -d '.')

# Map version to PyTorch index URL suffix (e.g., 12.1 -> cu121)
TORCH_SUFFIX="cu${CUDA_SUFFIX}"

# Project root (relative to scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

REQ_FILE="$PROJECT_ROOT/envs/requirements/requirements_cuda${CUDA_SUFFIX}.txt"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "═══════════════════════════════════════════════════════"
echo " VisionStream pip + venv Setup"
echo " CUDA Version: $CUDA_VER"
echo " Requirements: $REQ_FILE"
echo " venv Path:    $VENV_DIR"
echo "═══════════════════════════════════════════════════════"

# Check if requirements file exists
if [ ! -f "$REQ_FILE" ]; then
    echo "[ERROR] Requirements file not found: $REQ_FILE"
    echo "Available CUDA versions: 11.8, 12.1, 12.8"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "[INFO] Using existing virtual environment at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with correct CUDA version
echo "[INFO] Installing PyTorch with CUDA $CUDA_VER..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_SUFFIX}"

# Install remaining requirements
echo "[INFO] Installing requirements from $REQ_FILE..."
pip install -r "$REQ_FILE"

# Install VisionStream in editable mode
echo "[INFO] Installing VisionStream in editable mode..."
pip install -e "$PROJECT_ROOT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Setup Complete!"
echo " Activate with: source .venv/bin/activate"
echo "═══════════════════════════════════════════════════════"
