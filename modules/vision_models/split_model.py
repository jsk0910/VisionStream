"""
Split Model Adapter — PyTorch forward hooks for Video Coding for Machines (VCM).
Allows intercepting feature maps at a specific layer, and resuming inference
from that layer onwards.
"""
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional
from modules.registry import BaseVisionModel, get_vision_model, _list

class SplitModelWrapper(nn.Module):
    """
    Wraps an existing PyTorch model to intercept forward passes at a specific layer.
    """
    def __init__(self, base_model: nn.Module, split_layer_name: str):
        super().__init__()
        self.base_model = base_model
        self.split_layer_name = split_layer_name
        self._feature_map = None
        
        # Register forward hook
        self._register_hook()

    def _register_hook(self):
        target_module = None
        for name, module in self.base_model.named_modules():
            if name == self.split_layer_name:
                target_module = module
                break
                
        if target_module is None:
            raise ValueError(f"Layer '{self.split_layer_name}' not found in the model.")
            
        def hook(module, input, output):
            self._feature_map = output
            
        target_module.register_forward_hook(hook)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the model up to the split layer and returns the intercepted feature map."""
        self._feature_map = None
        # We run the full forward pass, but we just want the intercepted feature.
        # In a real VCM deployment, we might want to *halt* execution here, 
        # but standard PyTorch hooks don't halt easily without raising exceptions.
        # For simplicity in this research framework, we execute the pass 
        # and capture the intermediate tensor.
        with torch.no_grad():
            _ = self.base_model(x)
        features = self._feature_map
        self._feature_map = None
        return features
        
    def resume_from_features(self, x: torch.Tensor, features: torch.Tensor) -> Any:
        """
        Resumes the model inference, injecting the reconstructed features at the 
        split layer.
        
        To achieve this cleanly without heavy graph modification, we can use a 
        pre-forward hook on the layer *after* the split, or a forward hook on the 
        split layer itself that aggressively replaces the output.
        """
        # A simpler approach: register a forward hook that replaces the output 
        # of the split layer with our reconstructed `features`.
        target_module = dict(self.base_model.named_modules())[self.split_layer_name]
        
        def replace_hook(module, input, output):
            return features
            
        handle = target_module.register_forward_hook(replace_hook)
        
        # Run forward pass (the hook will replace the intermediate activation)
        try:
            with torch.no_grad():
                result = self.base_model(x)
        finally:
            handle.remove()
            
        return result


class SplitVisionModel(BaseVisionModel):
    """
    Adapter bridging a registered VisionStream model with the SplitModelWrapper.
    """
    def __init__(self, target_model_id: str, split_layer_name: str, device: str = "cuda:0", **kwargs):
        self.device = device
        self.target_model_id = target_model_id
        
        # Instantiate the underlying vision model
        model_cls = get_vision_model(target_model_id)
        self.vision_model = model_cls(device=device, **kwargs)
        
        # We need raw access to the underlying nn.Module
        if not hasattr(self.vision_model, "model") or not isinstance(self.vision_model.model, nn.Module):
            raise ValueError(f"Model ID '{target_model_id}' does not expose a standard PyTorch 'model' attribute.")
            
        self.split_wrapper = SplitModelWrapper(self.vision_model.model, split_layer_name)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.split_wrapper.extract_features(x)
        
    def resume_inference(self, original_x: torch.Tensor, reconstructed_features: torch.Tensor) -> Any:
        """
        Resume inference. 
        Note: The wrapper might just return raw logits. We might need to map it back 
        through the vision_model's predict post-processing if it does more than just 
        a forward pass. For now, we return the raw output of the wrapper.
        """
        original_x = original_x.to(self.device)
        reconstructed_features = reconstructed_features.to(self.device)
        return self.split_wrapper.resume_from_features(original_x, reconstructed_features)

    def predict(self, x: torch.Tensor) -> Any:
        # Standard predict (un-split)
        return self.vision_model.predict(x)

    def get_task_type(self) -> str:
        return self.vision_model.get_task_type()
