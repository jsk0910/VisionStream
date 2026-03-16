#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  VisionStream — Conda 환경 자동 설정 스크립트
# ═══════════════════════════════════════════════════════════
#  Usage:
#    bash scripts/setup_conda.sh [CUDA_VERSION]
#
#  Examples:
#    bash scripts/setup_conda.sh 11.8
#    bash scripts/setup_conda.sh 12.1    (default)
#    bash scripts/setup_conda.sh 12.8
# ═══════════════════════════════════════════════════════════

set -e

# Default CUDA version
CUDA_VER="${1:-12.1}"

# Map version to file suffix (e.g., 12.1 -> 121)
CUDA_SUFFIX=$(echo "$CUDA_VER" | tr -d '.')

# Project root (relative to scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ENV_FILE="$PROJECT_ROOT/envs/conda/environment_cuda${CUDA_SUFFIX}.yml"

echo "═══════════════════════════════════════════════════════"
echo " VisionStream Conda Setup"
echo " CUDA Version: $CUDA_VER"
echo " Environment File: $ENV_FILE"
echo "═══════════════════════════════════════════════════════"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "[ERROR] Environment file not found: $ENV_FILE"
    echo "Available CUDA versions: 11.8, 12.1, 12.8"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^visionstream "; then
    echo "[INFO] Removing existing 'visionstream' environment..."
    conda env remove -n visionstream -y
fi

# Create conda environment
echo "[INFO] Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

# Activate and install editable package
echo "[INFO] Installing VisionStream in editable mode..."
eval "$(conda shell.bash hook)"
conda activate visionstream
pip install -e "$PROJECT_ROOT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Setup Complete!"
echo " Activate with: conda activate visionstream"
echo "═══════════════════════════════════════════════════════"
