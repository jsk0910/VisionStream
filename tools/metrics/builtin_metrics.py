"""
Built-in Metrics — PSNR, MS-SSIM, and metric registry implementations.
"""
import torch
from typing import Optional
from modules.registry import BaseMetric, register_metric


@register_metric("psnr")
class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio (dB)."""

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        mse = torch.nn.functional.mse_loss(reconstructed, original)
        if mse == 0:
            return 100.0
        return (10 * torch.log10(1.0 / mse)).item()

    def name(self) -> str:
        return "PSNR (dB)"

    def higher_is_better(self) -> bool:
        return True


@register_metric("msssim")
class MSSSIMMetric(BaseMetric):
    """Multi-Scale Structural Similarity Index."""

    def __init__(self, **kwargs):
        try:
            from pytorch_msssim import ms_ssim
            self._ms_ssim = ms_ssim
        except ImportError:
            raise ImportError("Install pytorch-msssim: pip install pytorch-msssim")

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        # Ensure 4D BCHW
        if original.ndim == 3:
            original = original.unsqueeze(0)
        if reconstructed.ndim == 3:
            reconstructed = reconstructed.unsqueeze(0)
        return self._ms_ssim(
            reconstructed, original, data_range=1.0, size_average=True
        ).item()

    def name(self) -> str:
        return "MS-SSIM"

    def higher_is_better(self) -> bool:
        return True


@register_metric("mse")
class MSEMetric(BaseMetric):
    """Mean Squared Error."""

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        return torch.nn.functional.mse_loss(reconstructed, original).item()

    def name(self) -> str:
        return "MSE"

    def higher_is_better(self) -> bool:
        return False


@register_metric("bpp")
class BPPMetric(BaseMetric):
    """Bits Per Pixel — reads from codec output rather than computing on tensors."""

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                bpp: Optional[float] = None, **kwargs) -> float:
        if bpp is not None:
            return bpp
        return 0.0

    def name(self) -> str:
        return "BPP"

    def higher_is_better(self) -> bool:
        return False
