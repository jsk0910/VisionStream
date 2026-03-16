import torch
import time
import os
import sys

# Attempt to load the compiled visionstream module from the build directory
try:
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build'))
    sys.path.insert(0, build_dir)
    import visionstream as vs
    HAS_VS = True
except ImportError:
    HAS_VS = False
    
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.is_bypassed = False
            
    vs = type('vs', (), {'Node': MockNode})()

class NeuralCodecNode(vs.Node):
    def __init__(self, name, precision=16):
        super().__init__(name)
        self.precision = precision
        # For simulation, we define a fixed CDF map
        # In reality, a neural network (Hyperprior / GMM) provides these per spatial location
        self.num_cdfs = 1
        self.max_cdf_length = 256
        self.dummy_cdfs = self._build_dummy_cdf()
        
    def _build_dummy_cdf(self):
        """Creates a dummy exponential CDF for compression."""
        total = 1 << self.precision
        cdf = [0]
        prob_mass = []
        for i in range(self.max_cdf_length - 1):
            pm = max(1, int(total * (0.5 ** (i+1))))
            prob_mass.append(pm)
            
        # Normalize strictly to total
        sum_pm = sum(prob_mass)
        for i in range(len(prob_mass)):
            prob_mass[i] = int((prob_mass[i] / sum_pm) * (total - self.max_cdf_length)) + 1
            
        for pm in prob_mass:
            next_val = min(total, cdf[-1] + pm)
            cdf.append(next_val)
            
        cdf[-1] = total
        return cdf

    def process(self, input_tensor):
        """
        Simulate a Neural Compression forward pass using the C++ Arithmetic Coder.
        input_tensor: PyTorch tensor [B, C, H, W] representing an image or feature map.
        """
        if self.is_bypassed or not HAS_VS:
            return input_tensor
            
        print(f"[{self.name}] Running Neural Compression...")
        
        # 1. Neural Encoder Simulation (quantize to symbols 0~255)
        # We simulate feature maps by downsampling and cast to uint8
        batch, channels, height, width = input_tensor.shape
        downsampled = torch.nn.functional.interpolate(input_tensor.float(), scale_factor=0.125)
        symbols_tensor = downsampled.to(torch.uint8)
        
        # Flatten for Arithmetic Coder
        symbols_flat = symbols_tensor.flatten().tolist()
        indexes_flat = [0] * len(symbols_flat)  # Using CDF 0 for all
        
        cdf_sizes = [self.max_cdf_length]
        offsets = [0]
        
        # 2. Entropy Encoding (C++/CUDA Backend)
        start_ae = time.time()
        bitstream = vs.ArithmeticCoder.encode(
            symbols_flat, indexes_flat, self.dummy_cdfs, cdf_sizes, offsets, self.precision
        )
        encode_time = (time.time() - start_ae) * 1000
        
        # Simulated bits per pixel (BPP) tracking
        original_pixels = height * width
        bpp = (len(bitstream) * 8) / original_pixels
        self.last_bpp = bpp
        print(f"[{self.name}] Encoded to {len(bitstream)} bytes. AE Time: {encode_time:.2f} ms. Approx BPP: {bpp:.4f}")
        
        # 3. Entropy Decoding (C++/CUDA Backend)
        start_ad = time.time()
        decoded_flat = vs.ArithmeticCoder.decode(
            bitstream, indexes_flat, self.dummy_cdfs, cdf_sizes, offsets, self.precision
        )
        decode_time = (time.time() - start_ad) * 1000
        print(f"[{self.name}] Decoded {len(decoded_flat)} symbols. AD Time: {decode_time:.2f} ms.")
        
        # 4. Neural Decoder Simulation (reconstruct)
        # In reality, pass decoded_flat to neural decoder. We just return original for test pipeline continuity.
        return input_tensor
