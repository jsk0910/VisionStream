import torch
import torch.nn as nn
from torch.autograd import Function

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class GDN(nn.Module):
    """Generalized divisive normalization layer."""
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.build(ch)
    
    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset
        
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)
        
        eye = torch.eye(ch)
        g = self.gamma_init * eye + self.pedestal
        self.gamma = nn.Parameter(torch.sqrt(g))
    
    def forward(self, inputs):
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal
        
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(inputs.size(1), inputs.size(1), 1, 1)
        
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)
        
        if self.inverse:
            return inputs * norm_
        else:
            return inputs / norm_

class Encoder(nn.Module):
    def __init__(self, dim_in=3):
        super(Encoder, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.gdn1 = GDN(192)
        self.gdn2 = GDN(192)
        self.gdn3 = GDN(192)
    
    def forward(self, x):
        x = self.gdn1(self.first_conv(x))
        x = self.gdn2(self.conv1(x))
        x = self.gdn3(self.conv2(x))
        return self.conv3(x)

class Decoder(nn.Module):
    def __init__(self, dim_in=192):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.last_deconv = nn.ConvTranspose2d(in_channels=192, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(192, inverse=True)
        self.igdn2 = GDN(192, inverse=True)
        self.igdn3 = GDN(192, inverse=True)
    
    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        return self.last_deconv(x)

class HyperEncoder(nn.Module):
    def __init__(self, dim_in=192):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=2)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

class HyperDecoder(nn.Module):
    def __init__(self, dim_in=192):
        super(HyperDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=192, out_channels=288, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=288, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        return self.deconv3(x)
