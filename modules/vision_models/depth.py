"""
Depth Estimation Adapter — transformers wrapper for BaseVisionModel.
"""
import torch
from typing import Any, Dict, Optional
from modules.registry import BaseVisionModel, register_vision_model

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@register_vision_model("depth_anything")
class DepthAnythingModel(BaseVisionModel):
    """Wrapper for Depth-Anything-V2 from HuggingFace."""

    def __init__(self, model_variant: str = "depth-anything/Depth-Anything-V2-Small-hf", 
                 device: str = "cuda:0", **kwargs):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_variant)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_variant).to(device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> Any:
        """
        x: Input tensor (B, C, H, W)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
            
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth  # (B, H, W)
            
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=x.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze(1)

    def get_task_type(self) -> str:
        return "depth_estimation"
