import os
import sys
import torch
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from user_workspace.custom_codecs.learned_compression.model_v2 import (
    HybridCompressionModelV2,
)
from user_workspace.custom_codecs.learned_compression.entropy_gmm import (
    RateDistortionLossV2,
)


def run_tests():
    print("=" * 60)
    print(" 🧠  [Test 3] LIC (Learned Image Compression) V2 Training ")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}

    try:
        print("\n[1] Initializing V2 Model and Optimizers ...")
        # 1. Model init
        model = HybridCompressionModelV2(device=device).to(device)
        criterion = RateDistortionLossV2(lmbda=0.01).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        print(
            f"  - Model initialized. # params: {sum(p.numel() for p in model.parameters())}"
        )

        # 2. Dummy Data
        print("\n[2] Forward Pass & RD Loss Calculation ...")
        img = torch.rand(2, 3, 256, 256).to(device)

        output = model(img)
        x_hat, params, y_hat, z_hat = output

        loss, distortion, bpp = criterion(img, x_hat, params, y_hat, z_hat)
        print(
            f"  - Loss Total: {loss.item():.4f} | BPP: {bpp.item():.4f} | MSE: {distortion.item():.4f}"
        )

        # 3. Backward Pass
        print("\n[3] Backward Pass & Optimization Step ...")
        optimizer.zero_grad()
        loss.backward()

        # Checks if grads flow to Encoder and Decoder
        enc_grad = next(model.encoder.parameters()).grad
        dec_grad = next(model.decoder.parameters()).grad

        has_grads = enc_grad is not None and dec_grad is not None
        if has_grads:
            print(
                f"  - Gradients calculated successfully. (Encoder norm: {enc_grad.norm().item():.4f})"
            )
            optimizer.step()
            print("  - Optimizer stepped.")
            results["LIC Training"] = "PASS"
        else:
            print("  - Sub-modules did not receive gradients!")
            results["LIC Training"] = "FAIL"

    except Exception as e:
        print(f"  - LIC Training Failed: {e}")
        import traceback

        traceback.print_exc()
        results["LIC Training"] = "FAIL"

    print("\n" + "=" * 60)
    print(" 📊 LIC V2 Training Summary ")
    for k, v in results.items():
        print(f"  {k:.<30} {v}")
    print("=" * 60)
    print("\n-> 복사-붙여넣기 결과 출력 종료")


if __name__ == "__main__":
    run_tests()
