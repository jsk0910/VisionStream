"""
YOLO26 DataLoader Loader
Provides a standalone PyTorch DataLoader compatible with YOLO26 inference
and evaluation, wrapping COCO datasets with letterbox resizing.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from modules.registry import get_dataset


class LetterboxTransform:
    """YOLO standard letterbox resize keeping aspect ratio."""
    def __init__(self, target_size=(640, 640), fill_color=(114, 114, 114)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize image tensor [C, H, W] to target_size with padding.
        Input is expected to be [0, 1] float.
        """
        import torch.nn.functional as F
        
        c, h, w = img_tensor.shape
        th, tw = self.target_size
        
        # Calculate scale factor
        scale = min(th / h, tw / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        if new_h != h or new_w != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=(new_h, new_w), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
            
        # Pad differences
        pad_h = th - new_h
        pad_w = tw - new_w
        
        # Padding: (left, right, top, bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Default YOLO pad color is 114 (grey), which is 114/255.0 = 0.447
        fill_val = self.fill_color[0] / 255.0
        
        img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=fill_val)
        return img_tensor


class YOLO26DatasetWrapper(torch.utils.data.Dataset):
    """Wraps a registered BaseDataset (like COCO) with YOLO26 preprocessing."""
    def __init__(self, dataset_name: str, target_size=(640, 640), **kwargs):
        cls = get_dataset(dataset_name)
        self.dataset = cls(**kwargs)
        self.transform = LetterboxTransform(target_size=target_size)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]  # Dict with "image", "filename", "annotations"
        
        original_img = item["image"]
        original_shape = original_img.shape[1:] # (H, W)
        
        # Preprocess for YOLO
        img = self.transform(original_img)
        item["image"] = img
        item["original_shape"] = original_shape
        
        return item


def collate_yolo_batch(batch):
    """Custom collate function for YOLO data."""
    images = torch.stack([item["image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    original_shapes = [item["original_shape"] for item in batch]
    
    result = {
        "images": images,
        "filenames": filenames,
        "original_shapes": original_shapes
    }
    
    # Collate annotations if they exist
    if "annotations" in batch[0] and batch[0]["annotations"] is not None:
        result["annotations"] = [item["annotations"] for item in batch]
        
    return result


def create_yolo26_dataloader(
    dataset_name="coco_val2017",
    batch_size=16,
    img_size=640,
    shuffle=False,
    num_workers=4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create a YOLO26-compatible standard PyTorch DataLoader.
    """
    dataset = YOLO26DatasetWrapper(
        dataset_name=dataset_name, 
        target_size=(img_size, img_size), 
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_yolo_batch,
        pin_memory=True
    )
