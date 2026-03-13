import os
import sys
import glob
import time
import torch
import torchvision.io as io
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
from user_workspace.custom_codecs.learned_compression.model import HybridCompressionModel
import visionstream as vs  # compiled C++ core

def pad_image(x, p=64):
    """Pad image to be divisible by p."""
    h, w = x.shape[2], x.shape[3]
    h_pad = (p - h % p) % p
    w_pad = (p - w % p) % p
    padding = (0, w_pad, 0, h_pad)
    return torch.nn.functional.pad(x, padding, mode='constant', value=0.0), h_pad, w_pad

def get_dummy_cdf(max_cdf_length=256, precision=16):
    """
    In the paper (and properly implemented neural codecs), the network provides 
    the parameters (sigma, mu) for each latent. Here we simulate the C++ batched 
    Arithmetic Coder interface with a mock probability distribution derived
    from the symbols, or just use a fixed dummy CDF for testing the pipe.
    For true operation, standard deviation & mu generate dynamic CDFs via SciPy/PyTorch.
    For this benchmark, we simulate the bitstream write payload length.
    """
    total = 1 << precision
    cdf = [0]
    for i in range(max_cdf_length - 1):
        pm = max(1, int(total * (0.5 ** (i+1))))
        cdf.append(min(total, cdf[-1] + pm))
    cdf[-1] = total
    # Ensure strict monotonicity
    for i in range(len(cdf)-1):
        if cdf[i+1] <= cdf[i]: cdf[i+1] = cdf[i] + 1
    if cdf[-1] > total:
        cdf = [int((val / cdf[-1]) * total) for val in cdf]
        cdf[-1] = total
    return cdf

def evaluate():
    print("=== VisionStream Phase 5: Custom Neural Codec Evaluation on Kodak ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = HybridCompressionModel(device=device).to(device)
    model.eval()
    
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../user_workspace/custom_codecs/learned_compression/checkpoint.pth'))
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[Warning] No checkpoint found at {ckpt_path}. Using uninitialized weights.")
        
    # 2. Dataset Setup
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/kodak'))
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if not image_paths:
        print("No Kodak images found. Exiting.")
        return
        
    print(f"Evaluating on {len(image_paths)} Kodak images.\n")
    
    dummy_cdf = get_dummy_cdf()
    cdf_sizes = [256]
    offsets = [0]
    
    total_bpp = 0.0
    total_psnr = 0.0
    total_ae_time = 0.0
    total_ad_time = 0.0
    total_nn_time = 0.0

    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        img_tensor = io.read_image(path).float() / 255.0  # [0, 1]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # We need padding since the model expects divisibility by 64 (due to down/up sampling)
        img_padded, h_pad, w_pad = pad_image(img_tensor, p=64)
        
        b, c, h, w = img_padded.shape
        orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            # Neural Encoder
            y = model.encoder(img_padded * 255.0)
            y_hat = model.quantize(y, is_training=False)
            
            z = model.hyper_encoder(y)
            z_hat = model.quantize(z, is_training=False)
            
            # --- Entropy / C++ AE Simulation ---
            # Symbols are shifted to positive integers for AC
            # y_hat has shape [1, 192, H/16, W/16]
            # z_hat has shape [1, 192, H/64, W/64]
            # We shift them arbitrarily to simulate a [0, 255] discrete range for AE
            y_sym = torch.clamp(y_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            z_sym = torch.clamp(z_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
            
            # Combine symbols for a single AC stream
            all_syms = y_sym + z_sym
            all_indexes = [0] * len(all_syms)

            t1 = time.time()
            # Actual C++ batched encoding
            bitstream = vs.ArithmeticCoder.encode(all_syms, all_indexes, dummy_cdf, cdf_sizes, offsets, 16)
            t2 = time.time()
            
            # Decode
            decoded_syms = vs.ArithmeticCoder.decode(bitstream, all_indexes, dummy_cdf, cdf_sizes, offsets, 16)
            t3 = time.time()
            # -----------------------------------
            
            # Neural Decoder (using original quantized spatial representations)
            # (In reality, decompressed symbols are reshaped and shifted back here)
            phi = model.context(y_hat)
            psi = model.hyper_decoder(z_hat)
            sigma_mu = model.entropy(torch.cat([phi, psi], dim=1))
            sigma, mu = torch.split(sigma_mu, y_hat.shape[1], dim=1)
            
            x_hat_padded = model.decoder(y_hat) / 255.0
            
            # Crop padding
            x_hat = x_hat_padded[:, :, :orig_h, :orig_w]
            
            # Calculate metrics
            torch.cuda.synchronize()
            t4 = time.time()
            
            mse = torch.nn.functional.mse_loss(x_hat, img_tensor)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else 100.0
            
            bpp = (len(bitstream) * 8) / (orig_h * orig_w)
            
            total_bpp += bpp
            total_psnr += psnr.item()
            total_nn_time += (t1 - t0) + (t4 - t3)
            total_ae_time += (t2 - t1)
            total_ad_time += (t3 - t2)
            
            print(f"[{filename}] PSNR: {psnr:.2f} dB | BPP: {bpp:.4f} | AE: {(t2-t1)*1000:.1f}ms | AD: {(t3-t2)*1000:.1f}ms | NN: {((t1-t0)+(t4-t3))*1000:.1f}ms | Bytes: {len(bitstream)}")
            
    print("\n=== Summary (Kodak 24 images) ===")
    print(f"Average PSNR: {total_psnr / len(image_paths):.2f} dB")
    print(f"Average BPP : {total_bpp / len(image_paths):.4f} bpp")
    print(f"Average NN inference latency: {(total_nn_time / len(image_paths))*1000:.2f} ms")
    print(f"Average C++ AE encode latency : {(total_ae_time / len(image_paths))*1000:.2f} ms")
    print(f"Average C++ AE decode latency : {(total_ad_time / len(image_paths))*1000:.2f} ms")

if __name__ == "__main__":
    evaluate()
