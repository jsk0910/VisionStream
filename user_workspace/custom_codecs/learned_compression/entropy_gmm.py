import torch
import torch.nn as nn
from .entropy import MaskedConv2d

class ELICContextModel(nn.Module):
    """
    Uneven Channel-Conditioned Context Model (similar to ELIC).
    Splits the 192 channels into 4 groups: [16, 16, 32, 128].
    Each group relies on the spatial context of itself (MaskedConv2d) 
    and the channel context from previous groups + hyper-prior psi.
    """
    def __init__(self, dim_in=192, groups=[16, 16, 32, 128], K=3):
        super(ELICContextModel, self).__init__()
        self.groups = groups
        self.K = K
        
        # Spatial Contexts (Masked Conv for each group)
        self.spatial_convs = nn.ModuleList([
            MaskedConv2d('A', in_channels=g, out_channels=g * 2, kernel_size=5, stride=1, padding=2)
            for g in groups
        ])
        
        # Channel-Conditioned Parameter Predictors
        # Output sizes: K weights + K means + K scales (3 * K channels per latent channel)
        self.param_predictors = nn.ModuleList()
        accum_channels = 0
        for g in groups:
            # Inputs to predictor: Spatial Context (g*2) + Prev Groups (accum_channels) + Hyper (192, usually same as dim_in)
            in_ch = (g * 2) + accum_channels + dim_in
            out_ch = g * 3 * K
            
            predictor = nn.Sequential(
                nn.Conv2d(in_ch, 256, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 256, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, out_ch, 1, 1)
            )
            self.param_predictors.append(predictor)
            accum_channels += g

    def forward(self, y_hat, psi):
        """
        y_hat: B x 192 x H x W
        psi: B x 192 x H x W (from HyperDecoder)
        """
        y_splits = torch.split(y_hat, self.groups, dim=1)
        
        params_list = []
        prev_y = []
        
        for i, y_i in enumerate(y_splits):
            # Spatial context
            phi_i = self.spatial_convs[i](y_i)
            
            # Combine inputs
            if len(prev_y) > 0:
                y_prev_cat = torch.cat(prev_y, dim=1)
                ctx = torch.cat([phi_i, y_prev_cat, psi], dim=1)
            else:
                ctx = torch.cat([phi_i, psi], dim=1)
                
            # Predict parameters for this group
            params_i = self.param_predictors[i](ctx)
            params_list.append(params_i)
            
            # Add current y_i to previous (simulating availability after decoding)
            prev_y.append(y_i)
            
        # Concat along channel dim
        return torch.cat(params_list, dim=1)


class RateDistortionLossV2(nn.Module):
    def __init__(self, lmbda=0.01, K=3):
        super(RateDistortionLossV2, self).__init__()
        self.lmbda = lmbda
        self.K = K
        
    def _standardized_cumulative(self, inputs):
        """
        Numerically stable Gaussian CDF using erfc.
        """
        half = 0.5
        const = -(2 ** -0.5)
        return half * torch.erfc(const * inputs)
        
    def latent_rate_gmm(self, y, params):
        """
        Compute rate using Gaussian Mixture Model with logsumexp for stability.
        params shape: [B, C * 3 * K, H, W]
        y shape: [B, C, H, W]
        """
        B, C, H, W = y.shape
        params = params.view(B, C, 3, self.K, H, W)
        
        # Split params into weights, means, scales
        weight_logits = params[:, :, 0, :, :, :]  # [B, C, K, H, W]
        means = params[:, :, 1, :, :, :]          # [B, C, K, H, W]
        scales = params[:, :, 2, :, :, :]         # [B, C, K, H, W]
        
        # Softmax over K for weights (done safely via log-space later in logsumexp)
        log_weights = torch.log_softmax(weight_logits, dim=2)
        
        scales = torch.clamp(torch.exp(scales), min=1e-5) # Ensure strictly positive scale
        
        # Expand y to match K mixtures
        y_exp = y.unsqueeze(2).expand(B, C, self.K, H, W)
        
        # Gaussian PMF calculation
        upper = (y_exp + 0.5 - means) / scales
        lower = (y_exp - 0.5 - means) / scales
        
        c_upper = self._standardized_cumulative(upper)
        c_lower = self._standardized_cumulative(lower)
        
        # Probability for each mixture component
        pmf_k = torch.clamp(c_upper - c_lower, min=1e-6)
        log_pmf_k = torch.log(pmf_k)
        
        # Total log probability: log( sum_k (weights_k * pmf_k) ) -> logsumexp(log_weights + log_pmf_k)
        log_pmf_total = torch.logsumexp(log_weights + log_pmf_k, dim=2)
        
        # Rate is -log2(P)
        rate = -log_pmf_total / torch.log(torch.tensor(2.0, device=y.device))
        return rate

    def hyperlatent_rate(self, z):
        # Simplified standard normal for hyperlatents
        upper = z + 0.5
        lower = z - 0.5
        c_upper = self._standardized_cumulative(upper)
        c_lower = self._standardized_cumulative(lower)
        pmf = torch.clamp(c_upper - c_lower, min=1e-6)
        return -torch.log2(pmf)

    def forward(self, x, x_hat, params, y_hat, z_hat):
        # MSE Distortion
        distortion = nn.functional.mse_loss(x_hat, x)
        
        # Rates
        rate_y = self.latent_rate_gmm(y_hat, params).sum(dim=(1,2,3)).mean()
        rate_z = self.hyperlatent_rate(z_hat).sum(dim=(1,2,3)).mean()
        
        total_rate = rate_y + rate_z
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp_loss = total_rate / num_pixels
        
        # lmbda balances D and R. MSE operates on [0, 255]
        loss = self.lmbda * 255**2 * distortion + bpp_loss
        
        return loss, distortion, bpp_loss
