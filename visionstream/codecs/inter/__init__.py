"""
Inter Codecs — Built-in video / temporal sequence compression adapters.

Phase 12: H.264 and H.265 via FFmpeg.
Phase 12+: Neural Inter Codecs (DCVC, etc.) — user implementations in workspace/.

Available (Phase 12):
    H264Codec  → libx264 via FFmpeg
    H265Codec  → libx265 via FFmpeg
"""

try:
    from visionstream.codecs.inter.ffmpeg_video import H264Codec, H265Codec
    __all__ = ["H264Codec", "H265Codec"]
except ImportError:
    # ffmpeg-python not installed
    __all__ = []
