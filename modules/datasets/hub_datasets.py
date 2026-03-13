"""
Hub Datasets — COCO and DIV2K auto-download helpers.
"""
import os
import torch
from typing import Any, Dict, List, Optional
from modules.registry import register_dataset

@register_dataset("coco_val2017")
class COCODataset:
    """Helper to load COCO 2017 validation set."""
    def __init__(self, root: str = "./data/coco", transform=None, **kwargs):
        self.root = root
        self.transform = transform
        # In a real implementation, this would use torchvision.datasets.CocoDetection
        # and handle automatic downloading if needed.
        if not os.path.exists(root):
            print(f"Dataset not found at {root}. In production, we would trigger download.")

    def __getitem__(self, index):
        # Placeholder
        return torch.randn(3, 224, 224), {"image_id": index}

    def __len__(self):
        return 5000  # COCO val size


@register_dataset("div2k_val")
class DIV2KDataset:
    """Helper to load DIV2K validation set (for SR)."""
    def __init__(self, root: str = "./data/div2k", scale: int = 4, **kwargs):
        self.root = root
        self.scale = scale
        if not os.path.exists(root):
            print(f"Dataset not found at {root}. Triggering placeholder.")

    def __getitem__(self, index):
        # Placeholder
        return torch.randn(3, 256, 256), torch.randn(3, 1024, 1024)

    def __len__(self):
        return 100
