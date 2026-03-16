import os
import sys
import glob
import time
import torch
import torchvision.io as io
from pytorch_msssim import ms_ssim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import visionstream as vs
from modules.preprocessing.basic_transforms import PreprocessingNode
from modules.vision_models.yolo_node import YoloInferenceNode
from user_workspace.custom_codecs.learned_compression.model_v2 import HybridCompressionModelV2

class V2NeuralCodecNode(vs.Node):
    """
    Wraps the Phase 6 HybridCompressionModelV2 for pipeline integration.
    Performs C++ Encode/Decode over the generated V2 latents.
    """
    def __init__(self, name, device="cuda"):
        super().__init__(name)
        self.device = device
        self.model = HybridCompressionModelV2(device=device).to(device)
        self.model.eval()
        
        # Load Phase 6 Weights
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../user_workspace/custom_codecs/learned_compression/checkpoint_v2.pth'))
        if os.path.exists(ckpt_path):
            print(f"[{self.name}] Loaded V2 checkpoint: {ckpt_path}")
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"[{self.name}] WARNING: Checkpoint not found at {ckpt_path}")

        # Dummy CDF for C++ Arithmetic Coder simulation
        self.dummy_cdf = self._get_dummy_cdf()
        self.cdf_sizes = [256]
        self.offsets = [0]
        
        self.last_bpp = 0.0
        self.last_psnr = 0.0
        self.last_msssim = 0.0
        self.encoding_latency = 0.0
        self.decoding_latency = 0.0
        
    def _get_dummy_cdf(self):
        total = 1 << 16
        cdf = [0]
        for i in range(256 - 1):
            pm = max(1, int(total * (0.5 ** (i+1))))
            cdf.append(min(total, cdf[-1] + pm))
        cdf[-1] = total
        for i in range(len(cdf)-1):
            if cdf[i+1] <= cdf[i]:
                cdf[i+1] = cdf[i] + 1
        if cdf[-1] > total:
            cdf = [int((val / cdf[-1]) * total) for val in cdf]
            cdf[-1] = total
        return cdf
        
    def _pad_image(self, x, p=64):
        h, w = x.shape[2], x.shape[3]
        h_pad = (p - h % p) % p
        w_pad = (p - w % p) % p
        padding = (0, w_pad, 0, h_pad)
        return torch.nn.functional.pad(x, padding, mode='constant', value=0.0), h_pad, w_pad

    def process(self, input_tensor):
        if self.is_bypassed:
            return input_tensor
            
        print(f"[{self.name}] Input Shape: {input_tensor.shape}")
        
        # Pre-process for V2 Model Input
        if input_tensor.max() > 1.0:
            img_tensor = input_tensor.float() / 255.0
        else:
            img_tensor = input_tensor.float()
            
        img_padded, h_pad, w_pad = self._pad_image(img_tensor, p=64)
        orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]
        
        with torch.no_grad():
            # 1. ENCODE Neural
            y = self.model.encoder(img_padded * 255.0)
            y_hat = self.model.quantize(y, is_training=False)
            z = self.model.hyper_encoder(y)
            z_hat = self.model.quantize(z, is_training=False)
            
            # 2. ENCODE C++ Arithmetic (Simulate Bandwidth Constrained Transmission)
            y_sym = torch.clamp(y_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            z_sym = torch.clamp(z_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            all_syms = y_sym + z_sym
            all_indexes = [0] * len(all_syms)

            t0 = time.time()
            bitstream = vs.ArithmeticCoder.encode(all_syms, all_indexes, self.dummy_cdf, self.cdf_sizes, self.offsets, 16)
            t1 = time.time()
            self.encoding_latency = (t1 - t0) * 1000
            
            # Record Compression Ratio
            self.last_bpp = (len(bitstream) * 8) / (orig_h * orig_w)
            print(f"[{self.name}] Compressed to {len(bitstream)} Bytes ({self.last_bpp:.4f} bpp) in {self.encoding_latency:.1f} ms")
            
            # 3. DECODE C++ Arithmetic
            t2 = time.time()
            _ = vs.ArithmeticCoder.decode(bitstream, all_indexes, self.dummy_cdf, self.cdf_sizes, self.offsets, 16)
            t3 = time.time()
            self.decoding_latency = (t3 - t2) * 1000
            
            # 4. DECODE Neural
            # psi = self.model.hyper_decoder(z_hat)
            # params = self.model.entropy(y_hat, psi) -> Not needed strictly for inference reconstruction
            x_hat_padded = self.model.decoder(y_hat) / 255.0
            x_hat = x_hat_padded[:, :, :orig_h, :orig_w]
            
            # Convert back to standard distribution for downstream YOLO node [0, 1] range BCHW
            output_tensor = torch.clamp(x_hat, 0.0, 1.0)
            target_tensor = img_padded[:, :, :orig_h, :orig_w]
            
            mse = torch.nn.functional.mse_loss(output_tensor, target_tensor)
            self.last_psnr = (10 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0
            self.last_msssim = ms_ssim(output_tensor, target_tensor, data_range=1.0, size_average=True).item()
            
            print(f"[{self.name}] Reconstruction complete. Output Shape: {output_tensor.shape}")
            return output_tensor

def main():
    print("=== VisionStream Phase 7: End-to-End System Integration Pipeline ===\n")
    
    # Initialize Core Pipeline Nodes
    prep_node = PreprocessingNode("Preprocessor", target_size=(640, 640))
    codec_node = V2NeuralCodecNode("V2_Codec_Node", device="cuda:0")
    yolo_node = YoloInferenceNode("YOLO_Detector", model_name="yolov8n.pt", device="cuda:0")
    
    # Load Kodak Test Image
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/kodak'))
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if not image_paths:
        print("Required Kodak images missing. Exiting.")
        return
        
    test_img_path = image_paths[0] # kodim01.png
    print(f"Loading Base Image: {test_img_path}")
    
    # Raw Image [1, C, H, W] uint8
    raw_img = io.read_image(test_img_path).unsqueeze(0).to("cuda:0")
    print(f"Raw Image Shape: {raw_img.shape}")
    
    print("\n--- [START PIPELINE EXECUTION] ---")
    t_start = time.time()
    
    # Step 1: Pre-process
    t0 = time.time()
    prep_out = prep_node.process(raw_img)
    prep_time = (time.time() - t0) * 1000
    
    # Step 2: V2 Neural Codec (Encode -> Transmit -> Decode)
    t1 = time.time()
    codec_out = codec_node.process(prep_out)
    nn_time = (time.time() - t1) * 1000
    
    # Step 3: YOLO Inference on Reconstructed Output
    t2 = time.time()
    yolo_out = yolo_node.process(codec_out)
    yolo_time = (time.time() - t2) * 1000
    
    t_end = time.time()
    total_time = (t_end - t_start) * 1000
    print("--- [END PIPELINE EXECUTION] ---\n")
    
    # Extract YOLO Data to prove Vision model received valid image bytes
    boxes = yolo_out[0].boxes
    print(f"YOLO Detected Objects: {len(boxes)}")
    for i, box in enumerate(boxes):
        cls_name = yolo_out[0].names[int(box.cls)]
        conf = float(box.conf)
        print(f" - {cls_name} ({conf:.2f})")
        if i >= 4:
            print(" - ... (truncated)")
            break
            
    # Print Final Profiling
    print("\n=== End-to-End Latency Profiling ===")
    print(f"{'Phase':<20} | {'Latency (ms)':<10}")
    print("-" * 35)
    print(f"{'[Node] Preprocessing':<20} | {prep_time:.2f}")
    print(f"{'[Node] V2 Neural Codec':<20} | {nn_time:.2f}")
    print(f"{'  -> [C++] AE Encode':<20} | {codec_node.encoding_latency:.2f}")
    print(f"{'  -> [C++] AE Decode':<20} | {codec_node.decoding_latency:.2f}")
    print(f"{'[Node] YOLO Object Det':<20} | {yolo_time:.2f}")
    print("-" * 35)
    print(f"{'Total Pipeline Time':<20} | {total_time:.2f}")
    print(f"{'Transmission Payload':<20} | {codec_node.last_bpp:.4f} bpp")
    print(f"{'Reconstruction PSNR':<20} | {codec_node.last_psnr:.2f} dB")
    print(f"{'Recon. MS-SSIM':<20} | {codec_node.last_msssim:.4f}")
    print("====================================================")


if __name__ == "__main__":
    main()
