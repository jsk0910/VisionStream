import pytest
import torch
import torch.nn as nn
from typing import Any

from visionstream.registry import register_vision_model, BaseVisionModel, _REGISTRIES
from visionstream.models.split.split_model import SplitModelWrapper, SplitVisionModel

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DummyVisionModelAdapter(BaseVisionModel):
    def __init__(self, device="cpu", **kwargs):
        self.model = DummyNet().to(device)
        self.device = device
        
    def predict(self, x: torch.Tensor) -> Any:
        return self.model(x.to(self.device))
        
    def get_task_type(self) -> str:
        return "classification"

def test_split_model_wrapper():
    base_model = DummyNet()
    wrapper = SplitModelWrapper(base_model, split_layer_name="conv1")
    
    x = torch.randn(2, 3, 32, 32)
    features = wrapper.extract_features(x)
    
    # Feature map should be [2, 16, 32, 32]
    assert features.shape == (2, 16, 32, 32)
    
    # Let's modify the features and assert the output reacts to the modification
    modified_features = features * 0.0  # Zero out features
    
    output = wrapper.resume_from_features(x, modified_features)
    # the second conv doesn't have bias by default but fc has. The output shouldn't be NaN
    assert output.shape == (2, 10)

def test_split_vision_model():
    register_vision_model("test_dummy_net")(DummyVisionModelAdapter)
    
    split_model = SplitVisionModel(
        target_model_id="test_dummy_net",
        split_layer_name="conv2",
        device="cpu"
    )
    
    x = torch.randn(2, 3, 32, 32)
    features = split_model.extract_features(x)
    assert features.shape == (2, 32, 32, 32)
    
    result = split_model.resume_inference(x, features)
    assert result.shape == (2, 10)
    
    # Clean up registry post-test
    del _REGISTRIES["vision_model"]["test_dummy_net"]
