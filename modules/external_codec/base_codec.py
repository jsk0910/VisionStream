import sys

try:
    import visionstream as vs
except ImportError:
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.is_bypassed = False
    vs = type('vs', (), {'Node': MockNode})()

class CodecNode(vs.Node):
    """
    A unified base class for Compression Codecs in the pipeline.
    This serves as a placeholder node that can be toggled to evaluate
    the R-D / R-A impact of compression.
    """
    def __init__(self, name, codec_type="pass_through"):
        super().__init__(name)
        self.codec_type = codec_type
        # Metrics to track
        self.last_bpp = 0.0
        
    def process(self, input_tensor):
        if self.is_bypassed:
            self.last_bpp = 0.0
            return input_tensor
            
        print(f"[{self.name}] Compressing using {self.codec_type}...")
        
        # For Phase 2 Baseline, we implement a dummy pass-through
        # or simple quantize-dequantize to simulate minimum loss.
        # This will later integrate NVENC or neural encoders.
        
        # Dummy "lossy" simulation: Add tiny noise or round
        # noisy_tensor = input_tensor + torch.randn_like(input_tensor) * 0.01
        
        # Simulate BPP (Bits per pixel) calculation
        if input_tensor is not None:
            # shape typically [N, C, H, W]
            pixels = input_tensor.shape[-1] * input_tensor.shape[-2]
            self.last_bpp = 24.0 # Dummy constant bpp
            
        return input_tensor
