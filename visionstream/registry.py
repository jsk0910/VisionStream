"""
VisionStream Registry System
─────────────────────────────
Provides Abstract Base Classes and a decorator-based registry pattern so that
researchers can add new codecs, vision models, datasets, metrics, and
preprocessing transforms WITHOUT modifying any core framework code.

Usage:
    from modules.registry import BaseCodec, register_codec, get_codec

    @register_codec("my_codec")
    class MyCodec(BaseCodec):
        ...

    codec = get_codec("my_codec")(**kwargs)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

# ═══════════════════════════════════════════════════════════════
#  Generic Registry
# ═══════════════════════════════════════════════════════════════

_REGISTRIES: Dict[str, Dict[str, Type]] = {
    "codec": {},
    "vision_model": {},
    "dataset": {},
    "metric": {},
    "transform": {},
}


def _register(category: str, name: str):
    """Decorator factory for registering a class under *category*."""
    def decorator(cls):
        if name in _REGISTRIES[category]:
            raise ValueError(
                f"[Registry] '{name}' is already registered in '{category}'. "
                f"Existing: {_REGISTRIES[category][name]}, New: {cls}"
            )
        _REGISTRIES[category][name] = cls
        cls._registry_name = name          # tag for introspection
        return cls
    return decorator


def _get(category: str, name: str) -> Type:
    """Retrieve a registered class by *name* from *category*."""
    if name not in _REGISTRIES[category]:
        available = list(_REGISTRIES[category].keys())
        raise KeyError(
            f"[Registry] '{name}' not found in '{category}'. "
            f"Available: {available}"
        )
    return _REGISTRIES[category][name]


def _list(category: str) -> List[str]:
    """List all registered names in *category*."""
    return list(_REGISTRIES[category].keys())


# ═══════════════════════════════════════════════════════════════
#  Base Classes (ABCs)
# ═══════════════════════════════════════════════════════════════

class BaseCodec(ABC):
    """Abstract base for all image/video codecs."""

    @abstractmethod
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compress an image tensor.
        Args:
            x: [B, C, H, W] float tensor in [0, 1].
        Returns:
            dict with at least:
                "bitstream": bytes — the compressed payload
                "bpp": float — bits per pixel
                "encode_ms": float — encoding latency in ms
        """
        ...

    @abstractmethod
    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decompress a payload back to image tensor.
        Args:
            payload: dict produced by compress()
            shape: original (B, C, H, W)
        Returns:
            Reconstructed [B, C, H, W] float tensor in [0, 1].
        """
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convenience: compress + decompress, returning (x_hat, metrics)."""
        info = self.compress(x)
        x_hat = self.decompress(info, x.shape)
        return x_hat, info


class BaseVisionModel(ABC):
    """Abstract base for downstream vision tasks (detection, segmentation, etc.)."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> Any:
        """
        Run inference on image tensor.
        Args:
            x: [B, C, H, W] float tensor in [0, 1].
        Returns:
            Model-specific results (bboxes, masks, etc.)
        """
        ...

    @abstractmethod
    def get_task_type(self) -> str:
        """Return task type string: 'detection', 'segmentation', 'classification'."""
        ...


class BaseDataset(Dataset, ABC):
    """Abstract base for all evaluation/training datasets."""

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns a dict with at least:
            "image": torch.Tensor [C, H, W] float [0, 1]
            "filename": str
        Optionally:
            "annotations": Any (bboxes, masks, etc.)
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def get_name(self) -> str:
        return getattr(self, '_registry_name', self.__class__.__name__)


class BaseMetric(ABC):
    """Abstract base for quality / task-accuracy metrics."""

    @abstractmethod
    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        """
        Compute metric between original and reconstructed images.
        Args:
            original:      [B, C, H, W] float [0, 1]
            reconstructed: [B, C, H, W] float [0, 1]
        Returns:
            Scalar metric value.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name, e.g. 'PSNR (dB)'."""
        ...

    def higher_is_better(self) -> bool:
        """Override to False for metrics where lower is better (e.g. BPP)."""
        return True


class BaseTransform(ABC):
    """Abstract base for composable image preprocessing transforms."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to tensor.
        Args:
            x: [C, H, W] or [B, C, H, W] tensor.
        Returns:
            Transformed tensor.
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════
#  Public registration decorators & lookup functions
# ═══════════════════════════════════════════════════════════════

def register_codec(name: str):
    """Register a codec class: @register_codec("jpeg")"""
    return _register("codec", name)

def register_vision_model(name: str):
    """Register a vision model: @register_vision_model("yolov8n")"""
    return _register("vision_model", name)

def register_dataset(name: str):
    """Register a dataset: @register_dataset("kodak")"""
    return _register("dataset", name)

def register_metric(name: str):
    """Register a metric: @register_metric("psnr")"""
    return _register("metric", name)

def register_transform(name: str):
    """Register a transform: @register_transform("resize")"""
    return _register("transform", name)


def get_codec(name: str) -> Type[BaseCodec]:
    return _get("codec", name)

def get_vision_model(name: str) -> Type[BaseVisionModel]:
    return _get("vision_model", name)

def get_dataset(name: str) -> Type[BaseDataset]:
    return _get("dataset", name)

def get_metric(name: str) -> Type[BaseMetric]:
    return _get("metric", name)

def get_transform(name: str) -> Type[BaseTransform]:
    return _get("transform", name)


def list_codecs() -> List[str]:
    return _list("codec")

def list_vision_models() -> List[str]:
    return _list("vision_model")

def list_datasets() -> List[str]:
    return _list("dataset")

def list_metrics() -> List[str]:
    return _list("metric")

def list_transforms() -> List[str]:
    return _list("transform")


def list_all() -> Dict[str, List[str]]:
    """Return all registered components grouped by category."""
    return {cat: list(reg.keys()) for cat, reg in _REGISTRIES.items()}
