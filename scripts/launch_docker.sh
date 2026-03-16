#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  VisionStream — Docker 컨테이너 실행 스크립트
# ═══════════════════════════════════════════════════════════
#  Usage:
#    bash scripts/launch_docker.sh [CUDA_VERSION]
#
#  Examples:
#    bash scripts/launch_docker.sh 11.8
#    bash scripts/launch_docker.sh 12.1    (default)
#    bash scripts/launch_docker.sh 12.8
# ═══════════════════════════════════════════════════════════

set -e

# Default CUDA version
CUDA_VER="${1:-12.1}"

# Map version to file suffix
CUDA_SUFFIX=$(echo "$CUDA_VER" | tr -d '.')

# Project root (relative to scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DOCKERFILE="$PROJECT_ROOT/envs/docker/Dockerfile.cuda${CUDA_SUFFIX}"
COMPOSE_FILE="$PROJECT_ROOT/envs/docker/docker-compose.yml"

echo "═══════════════════════════════════════════════════════"
echo " VisionStream Docker Launcher"
echo " CUDA Version: $CUDA_VER"
echo " Dockerfile:   $DOCKERFILE"
echo "═══════════════════════════════════════════════════════"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "[ERROR] Dockerfile not found: $DOCKERFILE"
    echo "Available CUDA versions: 11.8, 12.1, 12.8"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found. Please install Docker first."
    exit 1
fi

# Use docker compose if compose file exists
if [ -f "$COMPOSE_FILE" ]; then
    echo "[INFO] Starting via docker compose (CUDA $CUDA_VER)..."
    export CUDA_VER="$CUDA_SUFFIX"
    cd "$PROJECT_ROOT/envs/docker"
    docker compose up --build
else
    # Fallback: build and run directly
    IMAGE_NAME="visionstream:cuda${CUDA_SUFFIX}"
    echo "[INFO] Building Docker image: $IMAGE_NAME..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$PROJECT_ROOT"
    echo "[INFO] Running container..."
    docker run --gpus all -it \
        -v "$PROJECT_ROOT:/app" \
        --network host \
        "$IMAGE_NAME"
fi
