"""
JPEG / WebP Codec — Traditional baseline codec using Pillow.
"""
import io
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

from modules.registry import BaseCodec, register_codec


@register_codec("jpeg")
class JPEGCodec(BaseCodec):
    """Standard JPEG compression via Pillow."""

    def __init__(self, quality: int = 50, **kwargs):
        self.quality = quality

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        B, C, H, W = x.shape
        t0 = time.time()
        bitstreams = []
        for i in range(B):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=self.quality)
            bitstreams.append(buf.getvalue())
        encode_ms = (time.time() - t0) * 1000
        total_bytes = sum(len(b) for b in bitstreams)
        bpp = (total_bytes * 8) / (H * W * B)
        return {
            "bitstream": bitstreams,
            "bpp": bpp,
            "bytes": total_bytes,
            "encode_ms": encode_ms,
            "shape": (B, C, H, W),
        }

    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        t0 = time.time()
        tensors = []
        for bs in payload["bitstream"]:
            pil_img = Image.open(io.BytesIO(bs)).convert("RGB")
            arr = np.array(pil_img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        decode_ms = (time.time() - t0) * 1000
        payload["decode_ms"] = decode_ms
        return torch.stack(tensors)


@register_codec("webp")
class WebPCodec(BaseCodec):
    """WebP compression via Pillow."""

    def __init__(self, quality: int = 50, **kwargs):
        self.quality = quality

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        B, C, H, W = x.shape
        t0 = time.time()
        bitstreams = []
        for i in range(B):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="WEBP", quality=self.quality)
            bitstreams.append(buf.getvalue())
        encode_ms = (time.time() - t0) * 1000
        total_bytes = sum(len(b) for b in bitstreams)
        bpp = (total_bytes * 8) / (H * W * B)
        return {
            "bitstream": bitstreams, "bpp": bpp, "bytes": total_bytes,
            "encode_ms": encode_ms, "shape": (B, C, H, W),
        }

    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        t0 = time.time()
        tensors = []
        for bs in payload["bitstream"]:
            pil_img = Image.open(io.BytesIO(bs)).convert("RGB")
            arr = np.array(pil_img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        payload["decode_ms"] = (time.time() - t0) * 1000
        return torch.stack(tensors)
