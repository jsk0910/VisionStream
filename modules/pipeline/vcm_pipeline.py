"""
VCM Pipeline Runner
String together a SplitModel and a Codec to compress intermediate features.
"""
import torch
from typing import Any, Dict

class VCMPipeline:
    """
    Runner for Video Coding for Machines.
    """
    def __init__(self, split_model: Any, codec: Any):
        """
        Args:
            split_model: Instance of SplitVisionModel.
            codec: Instance of BaseCodec (v2_elic, jpeg, etc.).
        """
        self.split_model = split_model
        self.codec = codec
        self.last_metrics = {}

    def run(self, original_input: torch.Tensor) -> Any:
        """
        Runs the full VCM pipeline on an input tensor.
        1. Base part of vision model -> Feature Map.
        2. Encode feature map.
        3. Decode feature map.
        4. Tail part of vision model -> Predictions.
        """
        # 1. Extract feature map at the split layer
        feature_map = self.split_model.extract_features(original_input)
        
        # 2-3. Compress and Decompress the feature map
        # Note: A standard codec like JPEG handles 3-channel images. 
        # Feature maps are usually N-dimensional floats. We need a neural codec 
        # capable of handling arbitrary channels (like our V2 ELIC or custom tensor codecs).
        
        reconstructed_features, metrics = self.codec.forward(feature_map)
        self.last_metrics = metrics
        
        # 4. Resume downstream task
        output = self.split_model.resume_inference(original_input, reconstructed_features)
        
        return output

    def get_metrics(self) -> Dict[str, Any]:
        """Returns BPP, latency, etc., from the last codec pass."""
        return self.last_metrics
