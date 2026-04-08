import pytest
import torch
from typing import Dict, Any, Tuple

from visionstream.pipeline.split_inference import SplitInferencePipeline

class DummyCodec:
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        return {
            "bitstream": b"101010",
            "bpp": 0.5,
            "encode_ms": 1.2
        }
    
    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        # Just return zeros of the expected shape to simulate decompression
        return torch.zeros(shape)

class DummySplitModel:
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate returning a smaller feature map
        return torch.rand(x.shape[0], 16, x.shape[2]//2, x.shape[3]//2)
    
    def resume_inference(self, original_input: torch.Tensor, features: torch.Tensor) -> str:
        # Dummy prediction
        return "predicted_class"

def test_pipeline_without_codec():
    model = DummySplitModel()
    pipeline = SplitInferencePipeline(
        model=model,
        split_point="layer1",
        feature_codec=None,
        transmit_fn=None,
        resume_device="cpu"
    )
    
    x = torch.rand(1, 3, 64, 64)
    result = pipeline.run(x)
    
    assert result == "predicted_class"
    metrics = pipeline.get_metrics()
    assert metrics["split_point"] == "layer1"
    assert "feature_shape" in metrics
    assert metrics["bpp"] == 0.0
    assert metrics["transmitted"] is False

def test_pipeline_with_codec_and_transmit():
    model = DummySplitModel()
    codec = DummyCodec()
    
    # Simple transmit function that flips bytes to see if it was called
    def dummy_transmit(bits: bytes) -> bytes:
        return b"received_" + bits
    
    pipeline = SplitInferencePipeline(
        model=model,
        split_point="layer2",
        feature_codec=codec,
        transmit_fn=dummy_transmit,
        resume_device="cpu"
    )
    
    x = torch.rand(1, 3, 64, 64)
    result = pipeline.run(x)
    
    assert result == "predicted_class"
    metrics = pipeline.get_metrics()
    
    assert metrics["split_point"] == "layer2"
    assert metrics["bpp"] == 0.5
    assert metrics["transmitted"] is True
    assert metrics["encode_ms"] == 1.2
