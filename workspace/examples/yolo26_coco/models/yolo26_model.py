"""
YOLO26 Vision Model — Detection model implementing BaseVisionModel.
Supports both end-to-end (NMS-free) and standard prediction modes.
"""
import torch
import torch.nn as nn
from typing import Any, List
from modules.registry import BaseVisionModel, register_vision_model


class BaseYOLO26Model(BaseVisionModel):
    """Base class for YOLO26 variants."""
    
    def __init__(self, model_name: str, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        from ultralytics import YOLO
        self.device = device
        self.conf = conf_threshold
        self.end2end = end2end
        
        # Load the pretrained model. Ultralytics will auto-download if missing.
        self._yolo = YOLO(f"{model_name}.pt")
        self.model = self._yolo.model  # Expose the underlying nn.Module
        self.model.to(self.device)

    def predict(self, x: torch.Tensor) -> Any:
        """
        Run inference.
        Args:
            x: [B, C, H, W] float tensor in [0, 1]
        Returns:
            Ultralytics Results object containing boxes, scores, etc.
        """
        # YOLO expects float [0-1] or [0-255]
        # We ensure it's on the right device and in the right format
        x = x.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        
        # The ultralytics predict API handles scaling [0,1] -> [0,255] if necessary internally,
        # but to be safe and consistent with older YOLO versions in standard pipelines:
        if x.max() <= 1.0:
            x = x * 255.0
            
        return self._yolo(x, conf=self.conf, verbose=False, end2end=self.end2end)

    def get_task_type(self) -> str:
        return "detection"

    def get_split_points(self) -> List[str]:
        """
        Returns a list of known valid split points for YOLO26.
        These correspond to layer names inside `self.model`.
        """
        return [
            "model.4",   # Backbone P3
            "model.6",   # Backbone P4
            "model.9",   # Backbone P5
            "model.22",  # Neck output
        ]


@register_vision_model("yolo26n")
class YOLO26NModel(BaseYOLO26Model):
    """Ultralytics YOLO26-Nano object detector."""
    def __init__(self, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        super().__init__("yolo26n", device, conf_threshold, end2end, **kwargs)


@register_vision_model("yolo26s")
class YOLO26SModel(BaseYOLO26Model):
    """Ultralytics YOLO26-Small object detector."""
    def __init__(self, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        super().__init__("yolo26s", device, conf_threshold, end2end, **kwargs)


@register_vision_model("yolo26m")
class YOLO26MModel(BaseYOLO26Model):
    """Ultralytics YOLO26-Medium object detector."""
    def __init__(self, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        super().__init__("yolo26m", device, conf_threshold, end2end, **kwargs)


@register_vision_model("yolo26l")
class YOLO26LModel(BaseYOLO26Model):
    """Ultralytics YOLO26-Large object detector."""
    def __init__(self, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        super().__init__("yolo26l", device, conf_threshold, end2end, **kwargs)


@register_vision_model("yolo26x")
class YOLO26XModel(BaseYOLO26Model):
    """Ultralytics YOLO26-ExtraLarge object detector."""
    def __init__(self, device="cuda:0", conf_threshold=0.25, end2end=True, **kwargs):
        super().__init__("yolo26x", device, conf_threshold, end2end, **kwargs)
