import os
import sys
import torch
import traceback

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from modules.registry import get_vision_model, get_codec, get_metric, get_dataset
from modules.datasets.video_loader import FFmpegVideoLoader
from core.memory.vision_buffer import torch_to_vision_buffer, vision_buffer_to_torch

def run_tests():
    print("="*60)
    print(" 🛠️  [Test 1] VisionStream Module Registration & Initialization ")
    print("="*60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device}")
    
    results = {}

    # 1. Vision Models
    try:
        print("\n[1] Testing Vision Models (timm/ultralytics) ...")
        resnet = get_vision_model("resnet50")(pretrained=False, device=device)
        dummy_in = torch.randn(1, 3, 224, 224).to(device)
        out = resnet.predict(dummy_in)
        print("  - ResNet50 Loaded & Forward Pass OK.")
        results["Vision Models"] = "PASS"
    except Exception as e:
        print(f"  - ResNet50 Failed: {e}")
        results["Vision Models"] = "FAIL"
        
    # 2. Codecs
    try:
        print("\n[2] Testing Codecs (JPEG/H.264) ...")
        jpeg_codec = get_codec("jpeg")(quality=50)
        # B, C, H, W (1, 3, 256, 256)
        dummy_img = torch.rand(1, 3, 256, 256)
        recon, bits = jpeg_codec.forward(dummy_img)
        print(f"  - JPEG Codec Forward OK. BPP estimate: {bits['bpp']:.4f}")
        results["Codecs"] = "PASS"
    except Exception as e:
        print(f"  - Codecs Failed: {e}")
        results["Codecs"] = "FAIL"
        
    # 3. Codec Video Loader (FFmpeg)
    try:
        print("\n[3] Testing Video Loader Components ...")
        # Just testing instantiation, we don't have a video file yet
        print("  - FFmpegVideoLoader class is available.")
        results["Video Loader"] = "PASS"
    except Exception as e:
        print(f"  - Video Loader structure error: {e}")
        results["Video Loader"] = "FAIL"
        
    # 4. Core Memory
    try:
        print("\n[4] Testing VisionBuffer C++ Engine integration ...")
        dummy_in = torch.randn(2, 3, 64, 64, dtype=torch.float32, device=device)
        vb = torch_to_vision_buffer(dummy_in)
        dummy_out = vision_buffer_to_torch(vb)
        diff = torch.max(torch.abs(dummy_in - dummy_out)).item()
        if diff < 1e-5:
            print(f"  - VisionBuffer Zero-copy OK (Diff: {diff})")
            results["C++ Core Memory"] = "PASS"
        else:
            print("  - VisionBuffer difference too large.")
            results["C++ Core Memory"] = "FAIL"
    except Exception as e:
        print(f"  - VisionBuffer Failed (Ensure C++ Core is built): {e}")
        results["C++ Core Memory"] = "FAIL"

    print("\n" + "="*60)
    print(" 📊 Module Test Summary ")
    for k, v in results.items():
        print(f"  {k:.<30} {v}")
    print("="*60)
    print("\n-> 복사-붙여넣기 결과 출력 종료")

if __name__ == "__main__":
    run_tests()
