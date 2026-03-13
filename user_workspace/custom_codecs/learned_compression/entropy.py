import torch
import torch.nn as nn
import math

class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for Context Prediction (Autoregressive model).
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ContextPrediction(nn.Module):
    def __init__(self, dim_in=192):
        super(ContextPrediction, self).__init__()
        self.masked = MaskedConv2d("A", in_channels=dim_in, out_channels=384, kernel_size=5, stride=1, padding=2)
    
    def forward(self, x):
        return self.masked(x)

class EntropyParameters(nn.Module):
    def __init__(self, dim_in=768):
        super(EntropyParameters, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 640, 1, 1)
        self.conv2 = nn.Conv2d(640, 512, 1, 1)
        self.conv3 = nn.Conv2d(512, 384, 1, 1)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=0.01):
        super(RateDistortionLoss, self).__init__()
        self.lmbda = lmbda
        
    def _standardized_cumulative(self, inputs):
        """
        Numerically stable Gaussian CDF.
        Uses torch.erfc for precision on small tails.
        """
        half = 0.5
        const = -(2 ** -0.5)
        # 0.5 * erfc(-x / sqrt(2)) is more stable than 0.5 * (1 + erf(x / sqrt(2)))
        return half * torch.erfc(const * inputs)

    def latent_rate(self, y, mu, sigma):
        """
        Stable computation of -log2(CDF(y+0.5) - CDF(y-0.5))
        """
        sigma = torch.clamp(sigma, min=1e-5)
        
        # Standardize the targets
        upper = (y + 0.5 - mu) / sigma
        lower = (y - 0.5 - mu) / sigma
        
        # Calculate PMF by difference of CDFs
        c_upper = self._standardized_cumulative(upper)
        c_lower = self._standardized_cumulative(lower)
        
        # Add epsilon to prevent log(0)
        pmf = torch.clamp(c_upper - c_lower, min=1e-6)
        return -torch.log2(pmf)

    def hyperlatent_rate(self, z):
        """
        Simplified hyperprior rate (assumes standard normal distribution for z).
        """
        upper = z + 0.5
        lower = z - 0.5
        c_upper = self._standardized_cumulative(upper)
        c_lower = self._standardized_cumulative(lower)
        pmf = torch.clamp(c_upper - c_lower, min=1e-6)
        return -torch.log2(pmf)

    def forward(self, x, x_hat, mu, sigma, y_hat, z_hat):
        # Mean Squared Error for Distortion
        distortion = nn.functional.mse_loss(x_hat, x)
        
        # Calculate Bits (Rates)
        # Sum over channels, height, width, then mean over batch
        rate_y = self.latent_rate(y_hat, mu, sigma).sum(dim=(1,2,3)).mean()
        rate_z = self.hyperlatent_rate(z_hat).sum(dim=(1,2,3)).mean()
        
        total_rate = rate_y + rate_z
        
        # R-D Loss
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp_loss = total_rate / num_pixels
        
        loss = self.lmbda * 255**2 * distortion + bpp_loss
        
        return loss, distortion, bpp_loss
