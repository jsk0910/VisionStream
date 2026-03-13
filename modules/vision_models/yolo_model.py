"""
YOLOv8 Vision Model — Detection model implementing BaseVisionModel.
"""
import torch
from typing import Any
from modules.registry import BaseVisionModel, register_vision_model


@register_vision_model("yolov8n")
class YOLOv8Model(BaseVisionModel):
    """Ultralytics YOLOv8-Nano object detector."""

    def __init__(self, device="cuda:0", conf_threshold=0.25, **kwargs):
        from ultralytics import YOLO
        self.device = device
        self.conf = conf_threshold
        self.model = YOLO("yolov8n.pt")

    def predict(self, x: torch.Tensor) -> Any:
        # YOLO expects float32 [0-255] or float [0-1]
        x = x.float()
        if x.max() <= 1.0:
            x = x * 255.0
        results = self.model(x, conf=self.conf, verbose=False)
        return results

    def get_task_type(self) -> str:
        return "detection"


@register_vision_model("yolov8s")
class YOLOv8SModel(BaseVisionModel):
    """Ultralytics YOLOv8-Small object detector."""

    def __init__(self, device="cuda:0", conf_threshold=0.25, **kwargs):
        from ultralytics import YOLO
        self.device = device
        self.conf = conf_threshold
        self.model = YOLO("yolov8s.pt")

    def predict(self, x: torch.Tensor) -> Any:
        if x.max() <= 1.0:
            x = (x * 255).to(torch.uint8)
        results = self.model(x, conf=self.conf, verbose=False)
        return results

    def get_task_type(self) -> str:
        return "detection"
