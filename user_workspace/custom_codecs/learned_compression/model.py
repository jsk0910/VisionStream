import torch
import torch.nn as nn
from .network import Encoder, Decoder, HyperEncoder, HyperDecoder
from .entropy import ContextPrediction, EntropyParameters

class HybridCompressionModel(nn.Module):
    def __init__(self, device="cuda"):
        super(HybridCompressionModel, self).__init__()
        self.device = device
        self.encoder = Encoder(dim_in=3)
        self.decoder = Decoder(dim_in=192)
        self.hyper_encoder = HyperEncoder(dim_in=192)
        self.hyper_decoder = HyperDecoder(dim_in=192)
        
        # Context model takes y_hat (quantized latents)
        self.context = ContextPrediction(dim_in=192)
        # Entropy parameters takes concatenated [context(y_hat), hyper_decoder(z_hat)]
        self.entropy = EntropyParameters(dim_in=384 * 2)
        
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
        Forward pass for Training and Validation.
        Computes quantizations, latents, and spatial entropy parameters.
        """
        # Encode image to latents
        y = self.encoder(x)
        y_hat = self.quantize(y, is_training=self.training)
        
        # Encode latents to hyperlatents
        z = self.hyper_encoder(y)
        z_hat = self.quantize(z, is_training=self.training)
        
        # Decode hyperlatents
        psi = self.hyper_decoder(z_hat)
        
        # Context prediction from quantized latents
        phi = self.context(y_hat)
        
        # Predict Gaussian parameters (sigma, mu) for the latent space
        phi_psi = torch.cat([phi, psi], dim=1)
        sigma_mu = self.entropy(phi_psi)
        
        # Split into scale and mean
        sigma, mu = torch.split(sigma_mu, y_hat.shape[1], dim=1)
        sigma = torch.clamp(sigma, min=1e-5)

        # Reconstruct image
        x_hat = self.decoder(y_hat)
        
        return x_hat, sigma, mu, y_hat, z_hat
