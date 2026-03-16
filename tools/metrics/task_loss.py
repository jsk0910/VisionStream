"""
Task-Aware Loss Adapters
Optimizes neural codecs to preserve task-critical features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskDistillationLoss(nn.Module):
    """
    Distills knowledge from the uncompressed 'teacher' feature map 
    into the compressed 'student' feature map.
    """
    def __init__(self, mode: str = "mse"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ["mse", "cosine", "l1"]:
            raise ValueError(f"Unsupported loss mode: {mode}")

    def forward(self, original_features: torch.Tensor, reconstructed_features: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss between unmodified Intermediate Feature Map and Decoded Feature Map.
        """
        if self.mode == "mse":
            return F.mse_loss(reconstructed_features, original_features)
        elif self.mode == "l1":
            return F.l1_loss(reconstructed_features, original_features)
        elif self.mode == "cosine":
            # Flatten spatial dimensions
            b, c, h, w = original_features.shape
            orig_flat = original_features.view(b, c, -1)
            recon_flat = reconstructed_features.view(b, c, -1)
            # Cosine similarity between feature channels
            sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
            return 1.0 - sim.mean()


class TaskAwareRD_Loss(nn.Module):
    """
    Rate-Distortion loss tailored for Feature Map Compression. 
    L = R(bits) + \lambda * D_task(reconstructed, original)
    """
    def __init__(self, lmbda: float = 0.01, mode: str = "mse"):
        super().__init__()
        self.lmbda = lmbda
        self.task_loss = TaskDistillationLoss(mode=mode)

    def forward(self, original_features: torch.Tensor, reconstructed_features: torch.Tensor, bits_bpp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original_features: Feature map before compression.
            reconstructed_features: Feature map after decompression.
            bits_bpp: Rate estimation (bits per pixel) from the entropy model.
        Returns:
            Scalar loss value.
        """
        d_loss = self.task_loss(original_features, reconstructed_features)
        total_loss = bits_bpp.mean() + self.lmbda * d_loss
        return total_loss
