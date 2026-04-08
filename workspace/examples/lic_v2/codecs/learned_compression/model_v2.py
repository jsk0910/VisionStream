import torch
import torch.nn as nn
from .network_v2 import EncoderV2, DecoderV2, HyperEncoderV2, HyperDecoderV2
from .entropy_gmm import ELICContextModel

class HybridCompressionModelV2(nn.Module):
    def __init__(self, device="cuda"):
        super(HybridCompressionModelV2, self).__init__()
        self.device = device
        self.encoder = EncoderV2(dim_in=3, channels=192)
        self.decoder = DecoderV2(dim_in=192, out_channels=3)
        self.hyper_encoder = HyperEncoderV2(dim_in=192, channels=192)
        self.hyper_decoder = HyperDecoderV2(dim_in=192, hidden_channels=288, out_channels=192) # output 192 for psi
        
        # ELIC Context Model takes latents and hyper-prior and predicts GMM params
        self.entropy = ELICContextModel(dim_in=192, groups=[16, 16, 32, 128], K=3)
        
    def quantize(self, x, is_training=True):
        """
        During training, use proxy uniform noise U(-1/2, 1/2).
        During inference, use round().
        """
        if is_training:
            uniform = -1 * torch.rand_like(x) + 0.5
            return x + uniform
        else:
            return torch.round(x)

    def forward(self, x):
        """
        Forward pass for Training.
        """
        # Encode image to latents
        y = self.encoder(x)
        y_hat = self.quantize(y, is_training=self.training)
        
        # Encode latents to hyperlatents
        z = self.hyper_encoder(y)
        z_hat = self.quantize(z, is_training=self.training)
        
        # Decode hyperlatents
        psi = self.hyper_decoder(z_hat)
        
        # ELIC Context Model (grouped channel prediction for GMM params)
        params = self.entropy(y_hat, psi)

        # Reconstruct image
        x_hat = self.decoder(y_hat)
        
        return x_hat, params, y_hat, z_hat
