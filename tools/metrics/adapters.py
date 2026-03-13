"""
Metric Adapters — LPIPS, mAP, and FID registration.
"""
from typing import Any, Dict
from modules.registry import register_metric

@register_metric("lpips")
class LPIPSMetric:
    def __init__(self, device: str = "cuda:0", **kwargs):
        try:
            import lpips
            self.fn = lpips.LPIPS(net='alex').to(device)
        except ImportError:
            self.fn = None
            print("LPIPS not installed. Run 'pip install lpips'")

    def compute(self, x: Any, y: Any) -> float:
        if self.fn is None: return 0.0
        return self.fn(x, y).item()


@register_metric("map")
class COCOmAPMetric:
    def __init__(self, **kwargs):
        pass
    def compute(self, results: Any, targets: Any) -> float:
        return 0.0 # Placeholder for pycocotools logic


@register_metric("fid")
class FIDMetric:
    def __init__(self, **kwargs):
        pass
    def compute(self, real_images: Any, fake_images: Any) -> float:
        return 0.0 # Placeholder for pytorch-fid logic
