import pytest
import torch
from typing import Dict, Any, Tuple

from visionstream.registry import (
    BaseCodec, BaseVisionModel,
    register_codec, get_codec, list_codecs,
    register_vision_model, get_vision_model,
    _REGISTRIES
)

def setup_module(module):
    """Clear registries before tests if needed."""
    for cat in _REGISTRIES:
        _REGISTRIES[cat].clear()

def teardown_module(module):
    """Clear registries after tests."""
    for cat in _REGISTRIES:
        _REGISTRIES[cat].clear()

class DummyCodec(BaseCodec):
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        return {"bitstream": b"dummy", "bpp": 1.0, "encode_ms": 1.0}
    
    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape)

class DummyVisionModel(BaseVisionModel):
    def predict(self, x: torch.Tensor) -> Any:
        return "dummy_prediction"
    
    def get_task_type(self) -> str:
        return "dummy_task"

def test_successful_registration():
    @register_codec("test_dummy_codec")
    class TestCodec(DummyCodec):
        pass

    assert "test_dummy_codec" in list_codecs()
    cls = get_codec("test_dummy_codec")
    assert cls == TestCodec
    assert getattr(cls, "_registry_name") == "test_dummy_codec"

def test_duplicate_registration_fails():
    @register_vision_model("test_duplicate_model")
    class TestModel1(DummyVisionModel):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_vision_model("test_duplicate_model")
        class TestModel2(DummyVisionModel):
            pass

def test_get_nonexistent_key_fails():
    with pytest.raises(KeyError, match="not found in 'codec'"):
        get_codec("non_existent_super_codec")

def test_abc_enforcement():
    # Attempting to instantiate a class that misses abstract methods should raise TypeError
    @register_codec("incomplete_codec")
    class IncompleteCodec(BaseCodec):
        def compress(self, x):
            return {}
        # decompress is missing
    
    cls = get_codec("incomplete_codec")
    with pytest.raises(TypeError):
        instance = cls()
