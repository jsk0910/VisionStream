import torch
import torch.nn as nn
import pytest

# Ensure project modules can be loaded
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visionstream.models.split.auto_split import AutoSplitter


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 16 * 16, 10)  # Assuming 64x64 input

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TestAutoSplitter:
    @pytest.fixture
    def cnn_model(self):
        return SimpleCNN()

    def test_ui_graph_structure(self, cnn_model):
        splitter = AutoSplitter(cnn_model)
        nodes = splitter.get_ui_graph_structure()
        
        # Verify it returns a list of dictionaries
        assert isinstance(nodes, list)
        assert len(nodes) > 0
        assert "id" in nodes[0]
        assert "op_type" in nodes[0]
        
        # Extract names to verify expected layers got traced
        node_names = [n["id"] for n in nodes]
        assert "conv1" in node_names
        assert "relu1" in node_names
        assert "flatten" in node_names
        assert "fc" in node_names

    def test_create_split_modules_two_parts(self, cnn_model):
        splitter = AutoSplitter(cnn_model)
        
        # Split after pool1
        split_nodes = ["pool1"]
        parts = splitter.create_split_modules(split_nodes)
        
        assert len(parts) == 2
        
        # Test correctness with dummy data
        x = torch.rand(1, 3, 64, 64)
        
        # Original forward
        cnn_model.eval()
        with torch.no_grad():
            expected_out = cnn_model(x)
            
        # Split forward
        with torch.no_grad():
            out_part_0 = parts[0](x)
            # FX split_module often packages outputs/inputs as tuples if there are multiple.
            # But for a simple sequential chain, it should be a single tensor or 1-element tuple.
            # Let's handle tuple unpacking gracefully if necessary.
            
            # Since our module has 1 input and 1 output between these layers, 
            # PyTorch FX split_module usually returns a tuple of outputs or a single tensor.
            if isinstance(out_part_0, tuple):
                out_part_0 = out_part_0[0]
                
            out_part_1 = parts[1](out_part_0)
            if isinstance(out_part_1, tuple):
                out_part_1 = out_part_1[0]
                
        assert torch.allclose(expected_out, out_part_1, atol=1e-5), "Split inference did not match original."

    def test_create_split_modules_three_parts(self, cnn_model):
        splitter = AutoSplitter(cnn_model)
        
        # Split after relu1 and pool2
        split_nodes = ["relu1", "pool2"]
        parts = splitter.create_split_modules(split_nodes)
        
        assert len(parts) == 3
        
        x = torch.rand(1, 3, 64, 64)
        
        cnn_model.eval()
        with torch.no_grad():
            expected_out = cnn_model(x)
            
        with torch.no_grad():
            out = x
            for part in parts:
                out = part(out)
                if isinstance(out, tuple):
                    out = out[0]
                    
        assert torch.allclose(expected_out, out, atol=1e-5), "Split inference did not match original."
