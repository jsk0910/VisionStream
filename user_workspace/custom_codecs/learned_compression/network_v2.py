import torch
import torch.nn as nn
from .network import GDN  # Reuse the GDN from Phase 5

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with LeakyReLU.
    Maintains spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 1x1 conv for channel matching if necessary
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class ChannelAttention(nn.Module):
    """Simple Channel Attention."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Simple Spatial Attention."""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y

class CBAMBlock(nn.Module):
    """Channel and Spatial Attention Block."""
    def __init__(self, channels):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class EncoderV2(nn.Module):
    """
    Enhanced Encoder with Residual Blocks and CBAM Attention.
    Downsamples by a factor of 16 (2^4) via 4 stride=2 convolutions.
    """
    def __init__(self, dim_in=3, channels=192):
        super(EncoderV2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(dim_in, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels)
        )
        self.res1 = ResidualBlock(channels, channels)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels)
        )
        self.res2 = ResidualBlock(channels, channels)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            GDN(channels),
            CBAMBlock(channels)
        )
        self.res3 = ResidualBlock(channels, channels)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            CBAMBlock(channels)
        )

    def forward(self, x):
        x = self.res1(self.layer1(x))
        x = self.res2(self.layer2(x))
        x = self.res3(self.layer3(x))
        return self.layer4(x)

class DecoderV2(nn.Module):
    """
    Enhanced Decoder with Residual Blocks and CBAM Attention.
    Upsamples by a factor of 16 via 4 ConvTranspose2d layers.
    """
    def __init__(self, dim_in=192, out_channels=3):
        super(DecoderV2, self).__init__()
        self.layer1 = nn.Sequential(
            CBAMBlock(dim_in),
            nn.ConvTranspose2d(dim_in, dim_in, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(dim_in, inverse=True)
        )
        self.res1 = ResidualBlock(dim_in, dim_in)
        
        self.layer2 = nn.Sequential(
            CBAMBlock(dim_in),
            nn.ConvTranspose2d(dim_in, dim_in, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(dim_in, inverse=True)
        )
        self.res2 = ResidualBlock(dim_in, dim_in)
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_in, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(dim_in, inverse=True)
        )
        self.res3 = ResidualBlock(dim_in, dim_in)
        
        self.layer4 = nn.ConvTranspose2d(dim_in, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.res1(self.layer1(x))
        x = self.res2(self.layer2(x))
        x = self.res3(self.layer3(x))
        return self.layer4(x)

class HyperEncoderV2(nn.Module):
    def __init__(self, dim_in=192, channels=192):
        super(HyperEncoderV2, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.res1 = ResidualBlock(channels, channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.res2 = ResidualBlock(channels, channels)
        
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        
    def forward(self, x):
        x = self.res1(self.relu(self.conv1(x)))
        x = self.res2(self.relu(self.conv2(x)))
        return self.conv3(x)

class HyperDecoderV2(nn.Module):
    def __init__(self, dim_in=192, hidden_channels=288, out_channels=384):
        super(HyperDecoderV2, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.res1 = ResidualBlock(dim_in, dim_in)
        
        self.deconv2 = nn.ConvTranspose2d(dim_in, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.res1(self.relu(self.deconv1(x)))
        x = self.res2(self.relu(self.deconv2(x)))
        return self.deconv3(x)
