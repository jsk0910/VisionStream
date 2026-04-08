import torch
import torchvision.transforms as T
import sys
import os

# Assume visionstream c++ module is accessible in the context path.
try:
    import visionstream as vs
except ImportError:
    # Dummy mock for pure python local testing outside the graph
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.is_bypassed = False
    vs = type('vs', (), {'Node': MockNode})()

class PreprocessingNode(vs.Node):
    def __init__(self, name, target_size=(640, 640), normalize=True):
        super().__init__(name)
        self.target_size = target_size
        self.normalize = normalize
        
        transforms = [T.Resize(target_size)]
        if self.normalize:
            # Standard ImageNet normalization or YOLO specific (usually / 255.0)
            # Depending on model requirements. YOLOv8 typically expects RGB 0-1 range.
            # DALI outputs 0-255 uint8, thus we need ToTensor and Normalize if required.
            pass # Keep it simple for the wrapper, handled in forward PyTorch logic
            
    def process(self, input_tensor):
        """
        Input could be a VisionBuffer or PyTorch tensor.
        For Phase 2 Prototype, we assume input_tensor is a PyTorch GPU Tensor.
        """
        if self.is_bypassed:
            return input_tensor
            
        print(f"[{self.name}] Preprocessing input shape: {input_tensor.shape}")
        
        # Expecting NCHW or NHWC tensor. DALI decode generally gives HWC.
        # Transpose to NCHW for vision models
        if input_tensor.ndim == 3: # HWC
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # NCHW
        elif input_tensor.ndim == 4 and input_tensor.shape[-1] == 3: # NHWC
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            
        # Convert uint8 to float32 and scale to [0, 1]
        if input_tensor.dtype == torch.uint8:
            input_tensor = input_tensor.float() / 255.0

        # Resize
        input_tensor = T.functional.resize(input_tensor, self.target_size, antialias=True)
        
        return input_tensor
