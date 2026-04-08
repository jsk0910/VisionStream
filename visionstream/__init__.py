"""
VisionStream — Hybrid Vision Research Framework
================================================
Layer P: Python package root.

Usage:
    from visionstream import registry
    from visionstream.codecs.intra.jpeg import JPEGCodec
    from visionstream.models.split import SplitModelWrapper
    from visionstream.pipeline.split_inference import SplitInferencePipeline
"""

from visionstream import registry

# Pre-load built-in codecs
try:
    from visionstream.codecs.intra.jpeg import JPEGCodec, WebPCodec
except ImportError:
    pass

try:
    from visionstream.codecs.inter.ffmpeg_video import H264Codec, H265Codec
except ImportError:
    pass

__version__ = "3.1.0"
__all__ = ["registry"]

# Expose C++ bindings
try:
    # 패키지 내부 (.so가 visionstream 폴더 내에 빌드된 경우)
    from ._C import Node, ArithmeticCoder, GraphManager, VisionBuffer, DataType, DeviceType
    __all__.extend(["Node", "ArithmeticCoder", "GraphManager", "VisionBuffer", "DataType", "DeviceType"])
except ImportError:
    try:
        # 루트 빌드 폴더(또는 sys.path)에 있는 경우
        from _C import Node, ArithmeticCoder, GraphManager, VisionBuffer, DataType, DeviceType
        __all__.extend(["Node", "ArithmeticCoder", "GraphManager", "VisionBuffer", "DataType", "DeviceType"])
    except ImportError:
        pass
