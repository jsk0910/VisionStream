import os
import sys
from .registry import *

# Pre-load to ensure models register themselves
try:
    from .vision_models.resnet import ResNetAdapter
    from .vision_models.yolo import YOLOv8Adapter
    from .vision_models.dinov2 import DinoV2Adapter
    from .vision_models.split_model import SplitVisionModel
except ImportError:
    pass

try:
    from .codecs.jpeg_codec import JPEGCodec
    from .codecs.ffmpeg_codec import FFmpegH264Codec, FFmpegH265Codec
except ImportError:
    pass
