"""
VisionStream Codecs
===================
Intra (still image) and Inter (video sequence) codec adapters.

BaseIntraCodec and BaseInterCodec ABCs are defined here.
Built-in implementations live in intra/ and inter/ subpackages.

User implementations should subclass these ABCs and register via:
    from visionstream.registry import register_codec
"""

from visionstream.codecs.base import BaseIntraCodec, BaseInterCodec

__all__ = ["BaseIntraCodec", "BaseInterCodec"]
