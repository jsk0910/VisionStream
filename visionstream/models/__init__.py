"""
VisionStream Models
===================
Vision model adapters and split inference machinery.

Split Inference (models/split/):
    SplitModelWrapper  — hook-based layer interception
    AutoSplitter       — torch.fx based automatic split point discovery

Built-in Adapters (models/adapters/):
    classification, detection, segmentation, depth, tracking, super_resolution

User models: subclass BaseVisionModel and register via:
    from visionstream.registry import register_vision_model
"""

from visionstream.models.split.split_model import SplitModelWrapper, SplitVisionModel
from visionstream.models.split.auto_split import AutoSplitter

__all__ = ["SplitModelWrapper", "SplitVisionModel", "AutoSplitter"]
