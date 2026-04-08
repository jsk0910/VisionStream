"""
Intra Codecs — Built-in still-image compression adapters.

Available:
    jpeg    → JPEGCodec, WebPCodec
    ffmpeg  → (PNG via ffmpeg, future)
    neural  → NeuralCodecNode (C++ ArithmeticCoder bridge)
    v2_elic → V2ELICCodec (Phase 6 ELIC+GMM neural codec)
"""

from visionstream.codecs.intra.jpeg import JPEGCodec, WebPCodec

__all__ = ["JPEGCodec", "WebPCodec"]
