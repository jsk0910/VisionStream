"""
V2 ELIC+GMM+Attention Neural Codec — BaseCodec wrapper for the Phase 6 model.
"""
import os
import sys
import time
import torch
from typing import Dict, Any, Tuple

# Ensure learned_compression package is importable
_lc_dir = os.path.join(os.path.dirname(__file__), "../../user_workspace/custom_codecs/learned_compression")
sys.path.insert(0, os.path.abspath(_lc_dir))

from modules.registry import BaseCodec, register_codec

try:
    import visionstream as vs
    HAS_VS = True
except ImportError:
    HAS_VS = False


@register_codec("v2_elic")
class V2ELICCodec(BaseCodec):
    """Phase 6 Next-Gen Neural Codec with C++ Arithmetic Coder."""

    def __init__(self, device="cuda:0", checkpoint=None, **kwargs):
        from model_v2 import HybridCompressionModelV2
        self.device = device
        self.model = HybridCompressionModelV2(device=device).to(device)
        self.model.eval()

        ckpt = checkpoint or os.path.join(_lc_dir, "checkpoint_v2.pth")
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=device))

        self._cdf = self._build_cdf()

    def _build_cdf(self):
        total = 1 << 16
        cdf = [0]
        for i in range(255):
            pm = max(1, int(total * (0.5 ** (i + 1))))
            cdf.append(min(total, cdf[-1] + pm))
        cdf[-1] = total
        for i in range(len(cdf) - 1):
            if cdf[i + 1] <= cdf[i]:
                cdf[i + 1] = cdf[i] + 1
        if cdf[-1] > total:
            cdf = [int((v / cdf[-1]) * total) for v in cdf]
            cdf[-1] = total
        return cdf

    @staticmethod
    def _pad(x, p=64):
        h, w = x.shape[2], x.shape[3]
        hp, wp = (p - h % p) % p, (p - w % p) % p
        return torch.nn.functional.pad(x, (0, wp, 0, hp)), hp, wp

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        x = x.to(self.device)
        oh, ow = x.shape[2], x.shape[3]
        xp, _, _ = self._pad(x)

        with torch.no_grad():
            t0 = time.time()
            y = self.model.encoder(xp * 255.0)
            y_hat = self.model.quantize(y, is_training=False)
            z = self.model.hyper_encoder(y)
            z_hat = self.model.quantize(z, is_training=False)

            y_sym = torch.clamp(y_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            z_sym = torch.clamp(z_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            syms = y_sym + z_sym
            idxs = [0] * len(syms)

            if HAS_VS:
                bs = vs.ArithmeticCoder.encode(syms, idxs, self._cdf, [256], [0], 16)
            else:
                bs = bytes(syms)  # fallback

            encode_ms = (time.time() - t0) * 1000

        bpp = (len(bs) * 8) / (oh * ow)
        return {
            "bitstream": bs, "bpp": bpp, "bytes": len(bs), "encode_ms": encode_ms,
            "y_hat": y_hat, "z_hat": z_hat, "orig_hw": (oh, ow),
        }

    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        with torch.no_grad():
            t0 = time.time()
            y_hat = payload["y_hat"]
            oh, ow = payload["orig_hw"]
            x_hat = self.model.decoder(y_hat) / 255.0
            recon = torch.clamp(x_hat[:, :, :oh, :ow], 0.0, 1.0)
            payload["decode_ms"] = (time.time() - t0) * 1000
        return recon
