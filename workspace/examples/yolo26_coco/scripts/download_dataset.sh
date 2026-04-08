#!/bin/bash
# download_yolo26_dataset.sh
# Downloads COCO val2017 dataset for YOLO26 evaluation.
# Usage: ./download_yolo26_dataset.sh [--mini]

set -e

# Default directories
DATA_DIR="${PWD}/data"
COCO_DIR="${DATA_DIR}/coco"
IMG_DIR="${COCO_DIR}/val2017"
ANN_DIR="${COCO_DIR}/annotations"

MINI=0
if [[ "$1" == "--mini" ]]; then
    MINI=1
    IMG_DIR="${COCO_DIR}/mini_val2017"
    echo "[!] Mini subset mode enabled."
fi

mkdir -p "$DATA_DIR"
mkdir -p "$COCO_DIR"
mkdir -p "$ANN_DIR"

echo "=========================================================="
echo " downloading COCO val2017 subset for YOLO26 / VisionStream"
echo "=========================================================="

if [[ $MINI -eq 1 ]]; then
    mkdir -p "$IMG_DIR"
    echo "Creating mini subset (fake dummy data for testing)..."
    # We create 10 dummy images to test the pipeline quickly
    for i in {1..10}; do
        # Use a python snippet to generate a valid RGB JPEG
        python3 -c "from PIL import Image; import numpy as np; img=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8); Image.fromarray(img).save('${IMG_DIR}/0000000000${i}.jpg')"
    done
    
    # Create a dummy annotations file
    echo '{"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}' > "${ANN_DIR}/instances_val2017.json"
    
    echo "Mini Dataset created: ${IMG_DIR}"
    exit 0
fi

if [[ -d "${IMG_DIR}" && "$(ls -A ${IMG_DIR} 2>/dev/null)" ]]; then
    echo "[OK] COCO val2017 images already exist."
else
    echo "Downloading COCO val2017 images (~1 GB)..."
    wget -c http://images.cocodataset.org/zips/val2017.zip -O "${COCO_DIR}/val2017.zip"
    unzip -q "${COCO_DIR}/val2017.zip" -d "${COCO_DIR}/"
    rm "${COCO_DIR}/val2017.zip"
fi

ANN_FILE="${ANN_DIR}/instances_val2017.json"
if [[ -f "${ANN_FILE}" ]]; then
    echo "[OK] COCO val2017 annotations already exist."
else
    echo "Downloading COCO val2017 annotations (~241 MB)..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "${COCO_DIR}/annotations_trainval2017.zip"
    unzip -q "${COCO_DIR}/annotations_trainval2017.zip" -d "${COCO_DIR}/"
    rm "${COCO_DIR}/annotations_trainval2017.zip"
fi

echo "=========================================================="
echo " Download complete! "
echo " Dataset is ready at ${COCO_DIR}"
echo "=========================================================="
