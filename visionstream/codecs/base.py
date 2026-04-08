"""
VisionStream Codec Base Classes
================================
Defines the Abstract Base Classes for all codecs in VisionStream.

Intra Codecs (BaseIntraCodec):
    Compress/decompress still images or feature maps as a single unit.
    Used for: JPEG, WebP, Neural LIC (ELIC, GMM-based), Feature Codecs.

Inter Codecs (BaseInterCodec):
    Compress/decompress sequences of images/feature maps using temporal redundancy.
    Used for: H.264, H.265, Neural Video Codecs (DCVC, etc.).

To implement a custom codec:
    from visionstream.codecs.base import BaseIntraCodec
    from visionstream.registry import register_codec

    @register_codec("my_codec")
    class MyCodec(BaseIntraCodec):
        def compress(self, x): ...
        def decompress(self, payload, shape): ...
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import torch
import numpy as np


class BaseIntraCodec(ABC):
    """Abstract base for all still-image / feature-map codecs (Intra).

    All intra codecs must implement:
        compress(x)    → payload dict
        decompress(payload, shape) → reconstructed tensor

    The payload dict must include at minimum:
        "bitstream": bytes  — the compressed payload
        "bpp":       float  — bits per pixel
        "encode_ms": float  — encoding latency in ms
    """

    @abstractmethod
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compress an image or feature map tensor.

        Args:
            x: [B, C, H, W] float tensor in [0, 1] (images)
               or arbitrary float tensor (feature maps).
        Returns:
            dict with at least:
                "bitstream": bytes — compressed payload
                "bpp": float      — bits per pixel
                "encode_ms": float — encoding latency ms
        """
        ...

    @abstractmethod
    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        """Decompress a payload back to tensor.

        Args:
            payload: dict produced by compress()
            shape:   original (B, C, H, W) shape
        Returns:
            Reconstructed tensor (same shape as input to compress).
        """
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convenience: compress + decompress, returning (x_hat, metrics)."""
        info = self.compress(x)
        x_hat = self.decompress(info, x.shape)
        return x_hat, info


class BaseInterCodec(ABC):
    """Abstract base for video / temporal sequence codecs (Inter).

    All inter codecs must implement:
        encode_sequence(frames)         → payload dict
        decode_sequence(payload, shape) → reconstructed frames

    Designed for Phase 12+ Neural Video Codec (DCVC, etc.) integration.
    """

    @abstractmethod
    def encode_sequence(self, frames: List[torch.Tensor]) -> Dict[str, Any]:
        """Encode a sequence of frames.

        Args:
            frames: List of [C, H, W] float tensors in [0, 1].
        Returns:
            dict with at least:
                "bitstream": bytes — compressed payload
                "bpp": float      — bits per pixel (averaged)
                "encode_ms": float — encoding latency ms
        """
        ...

    @abstractmethod
    def decode_sequence(self, payload: Dict[str, Any],
                         shape: Tuple[int, int, int]) -> List[torch.Tensor]:
        """Decode a payload back to a sequence of frames.

        Args:
            payload: dict produced by encode_sequence()
            shape:   (C, H, W) shape of individual frames
        Returns:
            List of reconstructed [C, H, W] float tensors.
        """
        ...
