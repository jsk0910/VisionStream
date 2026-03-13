import sys
import os
import glob
import time
import torch
import torchvision.io as io

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.preprocessing.basic_transforms import PreprocessingNode
from modules.external_codec.neural_codec import NeuralCodecNode

def test_kodak():
    print("=== VisionStream Phase 4: Kodak Dataset Evaluation ===")
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/kodak'))
    
    if not os.path.exists(data_dir):
        print(f"[Error] Kodak directory not found at {data_dir}")
        return
        
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    if len(image_paths) == 0:
        print(f"[Error] No PNG images found in {data_dir}")
        return
        
    print(f"Found {len(image_paths)} images in Kodak dataset.")
    
    # 1. Initialize Nodes
    # Kodak images are typically 768x512 or 512x768. We will resize them to 512x512 for uniform batching and YOLO testing if needed.
    # We set normalize=False because NeuralCodec currently expects 0-255 uint8 range for its dummy symbols.
    prep_node = PreprocessingNode("Preprocessor", target_size=(512, 512), normalize=False)
    codec_node = NeuralCodecNode("Neural_Codec_AE", precision=16)
    
    total_bpp = 0.0
    total_ae_time = 0.0
    total_prep_time = 0.0
    
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        
        # Load image via torchvision (Outputs C, H, W in uint8)
        try:
            img_tensor = io.read_image(path).to("cuda:0" if torch.cuda.is_available() else "cpu")
            # Convert to NCHW
            img_tensor = img_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue
            
        # 1. Preprocessing
        t0 = time.time()
        prep_out = prep_node.process(img_tensor)
        t_prep = (time.time() - t0) * 1000
        
        # 2. Neural Compression
        t1 = time.time()
        # NeuralCodec process() handles the AE/AD internally and prints stats.
        codec_out = codec_node.process(prep_out)
        t_codec = (time.time() - t1) * 1000
        
        # Retrieve stats
        bpp = getattr(codec_node, 'last_bpp', 0.0)
        
        print(f"Image {i+1:02d} ({filename}): Prep {t_prep:.2f}ms | Codec {t_codec:.2f}ms | est. BPP {bpp:.4f}")
        
        total_bpp += bpp
        total_prep_time += t_prep
        total_ae_time += t_codec
        
    avg_bpp = total_bpp / len(image_paths)
    avg_prep = total_prep_time / len(image_paths)
    avg_ae = total_ae_time / len(image_paths)
    
    print("\n=== Kodak Evaluation Summary ===")
    print(f"Total Images: {len(image_paths)}")
    print(f"Average Preprocessing Latency: {avg_prep:.2f} ms")
    print(f"Average Neural Codec (AE+AD) Latency: {avg_ae:.2f} ms")
    print(f"Average Bitrate: {avg_bpp:.4f} bpp")
    print("=========================================\n")

if __name__ == "__main__":
    test_kodak()
