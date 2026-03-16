import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from modules.vision_models.split_model import SplitVisionModel
from modules.pipeline.vcm_pipeline import VCMPipeline
from tools.metrics.task_loss import TaskDistillationLoss, TaskAwareRD_Loss
from tools.mlops.versioning import ExperimentTracker
from tools.mlops.monitor import get_logger
from modules.preprocessing.channel_sim import ChannelSimulatorBase
from modules.registry import get_codec

def run_tests():
    print("="*60)
    print(" 🚀  [Test 2] Integration, VCM, & MLOps Pipeline ")
    print("="*60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}

    # 1. VCM Pipeline (Split Model)
    try:
        print("\n[1] Testing VCM Pipeline (`SplitModel`) ...")
        # We split ResNet50 at layer1
        split_model = SplitVisionModel(target_model_id="resnet50", split_layer_name="layer1", device=device)
        codec = get_codec("jpeg")(quality=50) # Fallback to JPEG for testing dimension handling
        
        # JPEG expects 3 channels. Let's just test extraction & resumption in VCM
        dummy_in = torch.randn(1, 3, 224, 224).to(device)
        fmap = split_model.extract_features(dummy_in)
        print(f"  - Extracted Feature Map Shape: {fmap.shape}")
        
        # Resume without compression
        out = split_model.resume_inference(dummy_in, fmap)
        print(f"  - Resumed Output Shape: {out.shape}")
        results["VCM Pipeline"] = "PASS"
    except Exception as e:
        print(f"  - VCM Pipeline Failed: {e}")
        results["VCM Pipeline"] = "FAIL"
        
    # 2. Task Loss
    try:
        print("\n[2] Testing Task-Aware Loss ...")
        fmap_recon = fmap + 0.1 * torch.randn_like(fmap)
        bpp = torch.tensor([0.5], requires_grad=True).to(device)
        rd_loss = TaskAwareRD_Loss(lmbda=0.1, mode="mse").to(device)
        loss = rd_loss(fmap, fmap_recon, bpp)
        print(f"  - Task-Aware RD Loss: {loss.item():.4f}")
        results["Task Loss"] = "PASS"
    except Exception as e:
        print(f"  - Task Loss Failed: {e}")
        results["Task Loss"] = "FAIL"
        
    # 3. MLOps
    try:
        print("\n[3] Testing MLOps (Versioning & Simulator) ...")
        tracker = ExperimentTracker("test_integration")
        tracker.save_config({"test_bpp": 0.5})
        print(f"  - Experiment tracked at: {tracker.get_run_dir()}")
        
        sim = ChannelSimulatorBase(mode="awgn", snr_db=10.0)
        noisy_fmap = sim.forward(fmap)
        print(f"  - AWGN Simulator generated diff: {torch.abs(noisy_fmap - fmap).mean().item():.4f}")
        results["MLOps"] = "PASS"
    except Exception as e:
        print(f"  - MLOps Failed: {e}")
        results["MLOps"] = "FAIL"

    print("\n" + "="*60)
    print(" 📊 Integration Test Summary ")
    for k, v in results.items():
        print(f"  {k:.<30} {v}")
    print("="*60)
    print("\n-> 복사-붙여넣기 결과 출력 종료")

if __name__ == "__main__":
    run_tests()
