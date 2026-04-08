import pytest
import torch

from visionstream.codecs.intra.jpeg import JPEGCodec, WebPCodec

@pytest.fixture
def dummy_image():
    # Create a batch of 2 dummy rgb images (2, 3, 64, 64) with values in [0, 1]
    return torch.rand(2, 3, 64, 64)

@pytest.mark.parametrize("codec_cls", [JPEGCodec, WebPCodec])
def test_codec_roundtrip(codec_cls, dummy_image):
    codec = codec_cls(quality=80)
    
    # Compress
    payload = codec.compress(dummy_image)
    
    # Output assertions
    assert "bitstream" in payload
    assert isinstance(payload["bitstream"], list)
    assert len(payload["bitstream"]) == dummy_image.shape[0]
    
    assert "bpp" in payload
    assert payload["bpp"] > 0.0
    
    assert "encode_ms" in payload
    assert "shape" in payload
    assert payload["shape"] == (2, 3, 64, 64)
    
    # Decompress
    reconstructed = codec.decompress(payload, dummy_image.shape)
    
    # Output assertions
    assert reconstructed.shape == dummy_image.shape
    assert reconstructed.dtype == torch.float32
    assert reconstructed.max() <= 1.0 + 1e-5
    assert reconstructed.min() >= 0.0 - 1e-5
    assert "decode_ms" in payload

def test_codec_forward_method(dummy_image):
    codec = JPEGCodec(quality=50)
    x_hat, info = codec.forward(dummy_image)
    
    assert x_hat.shape == dummy_image.shape
    assert "bpp" in info
    assert "bitstream" in info
