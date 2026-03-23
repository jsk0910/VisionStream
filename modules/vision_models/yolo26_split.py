"""
YOLO26 Split Computing Adapter
Provides an interface to extract feature maps from YOLO26 at specific
split points (Backbone P3, P4, P5, or Neck output) and resume inference.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union
from modules.registry import get_vision_model


# Known optimal split points for YOLO models (layer indices)
YOLO26_SPLIT_POINTS = {
    "backbone_p3": "model.4",   # P3 feature map (stride 8)
    "backbone_p4": "model.6",   # P4 feature map (stride 16)
    "backbone_p5": "model.9",   # P5 feature map (stride 32)
    "neck_out":    "model.22",  # FPN/Neck combined output
}


class YOLO26SplitModel:
    """
    Dedicated SplitModel Wrapper for YOLO26.
    Unlike standard sequential models, YOLO's FPN neck requires multiple
    features from the backbone. This wrapper isolates execution up to a
    specific layer, returning the required features for resumption.
    """
    def __init__(self, variant: str = "yolo26n", split_point: str = "backbone_p4", device: str = "cuda:0"):
        """
        Args:
            variant: "yolo26n", "yolo26s", etc.
            split_point: One of YOLO26_SPLIT_POINTS keys, or a direct layer name
            device: e.g., "cuda:0"
        """
        self.device = device
        self.variant = variant
        self.split_point_name = YOLO26_SPLIT_POINTS.get(split_point, split_point)
        
        # Instantiate BaseVisionModel adapter
        model_cls = get_vision_model(variant)
        self.vision_wrapper = model_cls(device=device)
        self.model = self.vision_wrapper.model  # The underlying nn.Module
        
        # Verify the split point exists
        self._verify_layer(self.split_point_name)
        
        self._feature_capsule = None

    def _verify_layer(self, layer_name: str):
        found = False
        for name, _ in self.model.named_modules():
            if name == layer_name:
                found = True
                break
        if not found:
            raise ValueError(f"Split layer '{layer_name}' not found in {self.variant} architecture.")

    def list_split_points(self) -> Dict[str, str]:
        """Returns the available predefined split points."""
        return YOLO26_SPLIT_POINTS.copy()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the model up to the split layer, intercepting the intermediate 
        feature map.
        
        Returns:
            The intercepted feature tensor.
        """
        x = x.to(self.device)
        self._feature_capsule = None
        
        target_module = dict(self.model.named_modules())[self.split_point_name]
        
        def hook(module, input, output):
            self._feature_capsule = output
            # In a true VCM system, we would raise an Exception here to halt 
            # execution and save compute. For this research framework, we let
            # it finish the forward pass but grab the tensor.
            
        handle = target_module.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                _ = self.vision_wrapper.predict(x)
        finally:
            handle.remove()
            
        features = self._feature_capsule
        self._feature_capsule = None
        return features

    def resume_inference(self, x_dummy: torch.Tensor, features: torch.Tensor) -> Any:
        """
        Resumes inference from the split point by overriding the target layer's
        output with our reconstructed features.
        
        Args:
            x_dummy: Original input image shape (needed to drive the forward pass graph).
                     (In a fully decoupled system, this is bypassed. PyTorch needs it 
                      for execution flow without heavy graph surgery).
            features: The feature map previously extracted and potentially decompressed.
        """
        x_dummy = x_dummy.to(self.device)
        features = features.to(self.device)
        
        target_module = dict(self.model.named_modules())[self.split_point_name]
        
        def replace_hook(module, input, output):
            return features
            
        handle = target_module.register_forward_hook(replace_hook)
        
        try:
            with torch.no_grad():
                result = self.vision_wrapper.predict(x_dummy)
        finally:
            handle.remove()
            
        return result

    def get_feature_shape(self, input_size: Tuple[int, int] = (640, 640)) -> Tuple[int, ...]:
        """Helper to determine what the feature shape is for a given input size."""
        dummy = torch.zeros((1, 3, *input_size), device=self.device)
        features = self.extract_features(dummy)
        # Handle cases where output is a list/tuple (like Neck outputs in YOLO)
        if isinstance(features, (list, tuple)):
            return [f.shape for f in features]
        return features.shape
