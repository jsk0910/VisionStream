import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from user_workspace.custom_codecs.learned_compression.model_v2 import (
    HybridCompressionModelV2,
)
from user_workspace.custom_codecs.learned_compression.entropy_gmm import (
    RateDistortionLossV2,
)


def main():
    print("=== VisionStream Phase 6: Next-Gen Hybrid Codec Training ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    epochs = 100
    batch_size = 4  # RTX 3050 constraint
    lr = 1e-4
    lmbda = 0.01

    # Dataset Preparation (Kodak local valid test)
    data_dir_parent = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data")
    )
    os.makedirs(data_dir_parent, exist_ok=True)

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.RandomCrop(256), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(root=data_dir_parent, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Model & Loss Initialization
    model = HybridCompressionModelV2(device=device).to(device)
    rd_loss = RateDistortionLossV2(lmbda=lmbda, K=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(
        f"Total V2 model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print("Starting Phase 6 Training Loop...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bpp = 0.0
        total_mse = 0.0

        start_time = time.time()

        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= 100:
                break

            x = images.to(device) * 255.0

            optimizer.zero_grad()

            # Forward pass
            x_hat, params, y_hat, z_hat = model(x)

            # Compute R-D loss
            loss, mse, bpp = rd_loss(x, x_hat, params, y_hat, z_hat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_bpp += bpp.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/100] | Loss: {loss.item():.4f} | MSE: {mse.item():.4f} | BPP: {bpp.item():.4f}"
                )

        avg_loss = total_loss / max(1, len(train_loader))
        avg_mse = total_mse / max(1, len(train_loader))
        avg_bpp = total_bpp / max(1, len(train_loader))
        epoch_time = time.time() - start_time

        print(f"--- Epoch {epoch + 1} Summary ({epoch_time:.2f}s) ---")
        print(
            f"Avg Loss: {avg_loss:.4f} | Avg MSE: {avg_mse:.4f} | Avg BPP: {avg_bpp:.4f}\n"
        )

    ckpt_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_v2.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Phase 6 Training Complete. Checkpoint saved to -> {ckpt_path}")


if __name__ == "__main__":
    main()
