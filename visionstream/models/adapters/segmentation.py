"""
Segmentation Adapter — transformers wrapper for BaseVisionModel.
"""
import torch
from typing import Any, Dict, Optional
from modules.registry import BaseVisionModel, register_vision_model

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@register_vision_model("segformer")
class SegFormerModel(BaseVisionModel):
    """Wrapper for SegFormer (semantic segmentation) from HuggingFace."""

    def __init__(self, model_variant: str = "nvidia/segformer-b0-finetuned-ade-512-512", 
                 device: str = "cuda:0", **kwargs):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(model_variant)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_variant).to(device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> Any:
        """
        x: Input tensor (B, C, H, W) or (C, H, W)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        # x is expected to be [0, 1] float tensor
        # Processor handles resizing and normalization usually, but it takes PIL/numpy
        # To avoid CPU roundtrip, we can manually normalize or use processor's internal logic
        # For simplicity in this adapter, we let processor handle it if possible, 
        # but to keep it on GPU, we'd need a more custom path.
        
        # Fallback to processor (might involve CPU)
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (B, num_labels, H/4, W/4)

            # Upscale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            labels = upsampled_logits.argmax(dim=1)
            
        return labels

    def get_task_type(self) -> str:
        return "segmentation"


@register_vision_model("segformer_b0")
class SegFormerB0(SegFormerModel):
    def __init__(self, **kwargs):
        super().__init__(model_variant="nvidia/segformer-b0-finetuned-ade-512-512", **kwargs)
