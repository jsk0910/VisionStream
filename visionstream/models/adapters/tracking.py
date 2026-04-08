"""
Tracking Adapter — ByteTrack wrapper via supervision.
"""
import torch
from typing import Any, Dict
from modules.registry import BaseVisionModel, register_vision_model

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False

@register_vision_model("bytetrack")
class ByteTrackModel(BaseVisionModel):
    """Wrapper for ByteTrack (multi-object tracking) via supervision."""

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 30, minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30, device: str = "cpu", **kwargs):
        if not HAS_SUPERVISION:
            raise ImportError("Please install supervision: pip install supervision")
            
        self.device = device
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

    def predict(self, x: torch.Tensor, detection_results: Any = None) -> Any:
        """
        ByteTrack is usually a post-processing step on top of detections.
        
        Args:
            x: Raw input tensor (not directly used by ByteTrack, but kept for interface).
            detection_results: supervision Detections object from a previous model (e.g. YOLO).
            
        Returns:
            Updated Detections object with tracking IDs.
        """
        if detection_results is None:
            raise ValueError("ByteTrack requires detection_results (supervision Detections).")
            
        # Update tracker with current detections
        tracked_detections = self.tracker.update_with_detections(detections=detection_results)
        
        return tracked_detections

    def get_task_type(self) -> str:
        return "tracking"
