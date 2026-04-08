"""
Built-in Transforms — Composable preprocessing implementing BaseTransform.
"""
import torch
import torchvision.transforms.functional as TF
from modules.registry import BaseTransform, register_transform


@register_transform("resize")
class ResizeTransform(BaseTransform):
    """Resize image to target size."""
    def __init__(self, size=640, **kwargs):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.resize(x, self.size, antialias=True)

    def __repr__(self) -> str:
        return f"ResizeTransform(size={self.size})"


@register_transform("center_crop")
class CenterCropTransform(BaseTransform):
    """Center crop to target size."""
    def __init__(self, size=256, **kwargs):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.center_crop(x, self.size)

    def __repr__(self) -> str:
        return f"CenterCropTransform(size={self.size})"


@register_transform("random_crop")
class RandomCropTransform(BaseTransform):
    """Random crop for training augmentation."""
    def __init__(self, size=256, **kwargs):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.crop(x, *torch.randint(0, max(1, x.shape[-2] - self.size[0]), (1,)).item().__int__(),
                       *torch.randint(0, max(1, x.shape[-1] - self.size[1]), (1,)).item().__int__(),
                       *self.size) if x.shape[-2] >= self.size[0] and x.shape[-1] >= self.size[1] else TF.resize(x, self.size, antialias=True)

    def __repr__(self) -> str:
        return f"RandomCropTransform(size={self.size})"


@register_transform("normalize")
class NormalizeTransform(BaseTransform):
    """Ensure tensor is float32 in [0, 1] range."""
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return x.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return "NormalizeTransform()"


@register_transform("horizontal_flip")
class HorizontalFlipTransform(BaseTransform):
    """Random horizontal flip (50% probability) for training augmentation."""
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return TF.hflip(x)
        return x

    def __repr__(self) -> str:
        return f"HorizontalFlipTransform(p={self.p})"


@register_transform("to_bchw")
class ToBCHWTransform(BaseTransform):
    """Convert HWC to BCHW format."""
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x

    def __repr__(self) -> str:
        return "ToBCHWTransform()"
