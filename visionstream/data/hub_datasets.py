"""
Hub Datasets — COCO and DIV2K with automatic download helpers.

These datasets require additional dependencies:
  - COCO: pycocotools, torchvision
  - DIV2K: downloaded from official mirror via HTTP

Both datasets conform to BaseDataset, returning:
    {"image": Tensor [C, H, W] float [0, 1], "filename": str}
"""
import os
import glob
import shutil
import urllib.request
import zipfile
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional

from modules.registry import BaseDataset, register_dataset


def _download_file(url: str, dest_path: str, desc: str = "") -> None:
    """Download a file with progress indication."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[Download] {desc or url}")
    print(f"  → {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[Download] Complete: {os.path.getsize(dest_path) / 1e6:.1f} MB")


def _extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip archive."""
    print(f"[Extract] {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("[Extract] Done.")


# ═══════════════════════════════════════════════════════════════
#  COCO 2017 Validation Dataset
# ═══════════════════════════════════════════════════════════════

COCO_VAL_IMAGES_URL = (
    "http://images.cocodataset.org/zips/val2017.zip"
)
COCO_VAL_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


@register_dataset("coco_val2017")
class COCOValDataset(BaseDataset):
    """COCO 2017 Validation Dataset (5,000 images).

    For image-quality evaluation (codec → vision task), this dataset
    returns images only. Annotations are loaded separately if available
    (for mAP evaluation with the detection pipeline).

    Args:
        root: Path to the COCO data directory.
              Expected structure: root/val2017/*.jpg
        auto_download: If True, download COCO val2017 images (~1 GB).
        with_annotations: If True, also load annotations for detection/segmentation.
    """

    def __init__(self, root: str = "./data/coco", auto_download: bool = False,
                 with_annotations: bool = False, **kwargs):
        self.root = os.path.abspath(root)
        self.img_dir = os.path.join(self.root, "val2017")
        self.with_annotations = with_annotations

        # Auto-download if requested and images don't exist
        if auto_download and not os.path.isdir(self.img_dir):
            self._download()

        # Discover image files
        if os.path.isdir(self.img_dir):
            self.files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        else:
            self.files = []
            print(
                f"[COCOValDataset] Images not found at {self.img_dir}.\n"
                f"  To download, set auto_download=True or run:\n"
                f"    wget {COCO_VAL_IMAGES_URL} -P {self.root} && "
                f"unzip {self.root}/val2017.zip -d {self.root}"
            )

        # Load annotations if requested
        self.annotations = None
        if with_annotations:
            ann_file = os.path.join(
                self.root, "annotations", "instances_val2017.json"
            )
            if os.path.exists(ann_file):
                try:
                    from pycocotools.coco import COCO
                    self.annotations = COCO(ann_file)
                except ImportError:
                    print("[COCOValDataset] pycocotools not installed. "
                          "Annotations will not be loaded.")

    def _download(self):
        """Download COCO val2017 images."""
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "val2017.zip")
        _download_file(COCO_VAL_IMAGES_URL, zip_path, "COCO val2017 images (~1 GB)")
        _extract_zip(zip_path, self.root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        result = {"image": tensor, "filename": os.path.basename(path)}

        # Optionally include annotations
        if self.annotations is not None:
            img_id = int(os.path.splitext(os.path.basename(path))[0])
            ann_ids = self.annotations.getAnnIds(imgIds=img_id)
            result["annotations"] = self.annotations.loadAnns(ann_ids)

        return result


# ═══════════════════════════════════════════════════════════════
#  DIV2K Validation Dataset
# ═══════════════════════════════════════════════════════════════

DIV2K_VAL_HR_URL = (
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
)
DIV2K_VAL_LR_X4_URL = (
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
)


@register_dataset("div2k_val")
class DIV2KValDataset(BaseDataset):
    """DIV2K Validation Dataset (100 high-resolution images, 2K resolution).

    Commonly used for super-resolution evaluation. Returns HR images by default;
    optionally also returns LR (low-resolution) downscaled versions.

    Args:
        root: Path to DIV2K data directory.
              Expected structure: root/DIV2K_valid_HR/*.png
        scale: Downscale factor for LR images (2, 3, or 4). Default: 4.
        auto_download: If True, download DIV2K validation set (~500 MB).
        return_lr: If True, also return LR image in __getitem__.
    """

    def __init__(self, root: str = "./data/div2k", scale: int = 4,
                 auto_download: bool = False, return_lr: bool = False, **kwargs):
        self.root = os.path.abspath(root)
        self.scale = scale
        self.return_lr = return_lr

        self.hr_dir = os.path.join(self.root, "DIV2K_valid_HR")
        self.lr_dir = os.path.join(
            self.root, f"DIV2K_valid_LR_bicubic", f"X{scale}"
        )

        # Auto-download if requested and HR images don't exist
        if auto_download and not os.path.isdir(self.hr_dir):
            self._download()

        # Discover HR image files
        if os.path.isdir(self.hr_dir):
            self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        else:
            self.hr_files = []
            print(
                f"[DIV2KValDataset] HR images not found at {self.hr_dir}.\n"
                f"  To download, set auto_download=True or run:\n"
                f"    wget {DIV2K_VAL_HR_URL} -P {self.root} && "
                f"unzip {self.root}/DIV2K_valid_HR.zip -d {self.root}"
            )

    def _download(self):
        """Download DIV2K validation HR (and optionally LR) images."""
        os.makedirs(self.root, exist_ok=True)

        # Download HR
        hr_zip = os.path.join(self.root, "DIV2K_valid_HR.zip")
        _download_file(DIV2K_VAL_HR_URL, hr_zip, "DIV2K validation HR (~500 MB)")
        _extract_zip(hr_zip, self.root)

        # Download LR if return_lr is requested
        if self.return_lr and self.scale == 4:
            lr_zip = os.path.join(self.root, "DIV2K_valid_LR_bicubic_X4.zip")
            _download_file(DIV2K_VAL_LR_X4_URL, lr_zip, "DIV2K validation LR x4")
            _extract_zip(lr_zip, self.root)

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        hr_path = self.hr_files[index]
        hr_img = Image.open(hr_path).convert("RGB")
        hr_arr = np.array(hr_img).astype(np.float32) / 255.0
        hr_tensor = torch.from_numpy(hr_arr).permute(2, 0, 1)

        result = {
            "image": hr_tensor,
            "filename": os.path.basename(hr_path),
        }

        # Optionally return LR image
        if self.return_lr:
            basename = os.path.splitext(os.path.basename(hr_path))[0]
            lr_name = f"{basename}x{self.scale}.png"
            lr_path = os.path.join(self.lr_dir, lr_name)
            if os.path.exists(lr_path):
                lr_img = Image.open(lr_path).convert("RGB")
                lr_arr = np.array(lr_img).astype(np.float32) / 255.0
                result["lr_image"] = torch.from_numpy(lr_arr).permute(2, 0, 1)

        return result
