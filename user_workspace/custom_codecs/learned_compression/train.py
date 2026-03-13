import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from user_workspace.custom_codecs.learned_compression.model import HybridCompressionModel
from user_workspace.custom_codecs.learned_compression.entropy import RateDistortionLoss

def main():
    print("=== VisionStream Phase 5: Custom Neural Codec Training ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    epochs = 5
    batch_size = 4  # Small batch size for 4GB/8GB VRAM (RTX 3050 constraint)
    lr = 1e-4
    lmbda = 0.01  # Rate-distortion tradeoff
    
    # 1. Dataset Preparation
    # We will use STL10 if COCO is too large to download quickly, or a tiny custom subset.
    # To test the pipeline swiftly, torchvision's datasets are easy to auto-download.
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading/Loading Dataset...")
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor() # Converts to [0.0, 1.0]
    ])
    
    # Using the local Kodak dataset for rapid pipeline validation
    # ImageFolder requires a subfolder for class labels, so we point it to the parent directory of 'kodak'
    from torchvision.datasets import ImageFolder
    data_dir_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))
    
    train_dataset = ImageFolder(root=data_dir_parent, transform=transform)
    # Since Kodak only has 24 images, we repeat it virtually to have more batches per epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    
    # 2. Model & Loss Initialization
    model = HybridCompressionModel(device=device).to(device)
    # The original Loss implementation operates on [0, 255] for MSE.
    # We adjust the lambda factor in RD loss internally, but we'll scale x to [0, 255] when feeding to RDLoss.
    rd_loss = RateDistortionLoss(lmbda=lmbda).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Starting Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bpp = 0.0
        total_mse = 0.0
        
        start_time = time.time()
        
        # We only use a max of 100 batches per epoch to test the pipeline rapidly
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= 100:
                break
                
            # Input images are [0.0, 1.0]
            # Convert to [0, 255] range as the model expects values without internal normalisation
            # or we feed [0, 1] but usually MSE is on [0, 255] scale in compression literature
            x = images.to(device) * 255.0
            
            optimizer.zero_grad()
            
            # Forward pass
            x_hat, sigma, mu, y_hat, z_hat = model(x)
            
            # Compute R-D loss
            loss, mse, bpp = rd_loss(x, x_hat, mu, sigma, y_hat, z_hat)
            
            loss.backward()
            
            # Gradient clipping to prevent explosion in early steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_bpp += bpp.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/100] | Loss: {loss.item():.4f} | MSE: {mse.item():.4f} | BPP: {bpp.item():.4f}")
                
        avg_loss = total_loss / 100
        avg_mse = total_mse / 100
        avg_bpp = total_bpp / 100
        epoch_time = time.time() - start_time
        
        print(f"--- Epoch {epoch+1} Summary ({epoch_time:.2f}s) ---")
        print(f"Avg Loss: {avg_loss:.4f} | Avg MSE: {avg_mse:.4f} | Avg BPP: {avg_bpp:.4f}\n")
        
    # Save checkpoint
    ckpt_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Training Complete. Target checkpoint saved to -> {ckpt_path}")

if __name__ == "__main__":
    main()
