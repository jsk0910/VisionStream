"""
Built-in Dataset Implementations — Kodak, ImageFolder, DIV2K stubs.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List

from modules.registry import BaseDataset, register_dataset


@register_dataset("kodak")
class KodakDataset(BaseDataset):
    """Kodak Lossless True Color Image Suite (24 images, 768×512)."""

    def __init__(self, root=None, **kwargs):
        self.root = root or os.path.join(
            os.path.dirname(__file__), "../../data/kodak"
        )
        self.root = os.path.abspath(self.root)
        self.files = sorted(glob.glob(os.path.join(self.root, "*.png")))
        if not self.files:
            raise FileNotFoundError(f"No PNG images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        return {"image": tensor, "filename": os.path.basename(path)}


@register_dataset("image_folder")
class ImageFolderDataset(BaseDataset):
    """Generic dataset from any folder of images."""

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(self, root: str, **kwargs):
        self.root = os.path.abspath(root)
        self.files = sorted([
            os.path.join(self.root, f) for f in os.listdir(self.root)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {"image": tensor, "filename": os.path.basename(path)}
