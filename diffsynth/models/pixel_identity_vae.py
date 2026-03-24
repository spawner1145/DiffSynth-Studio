import torch
import torch.nn as nn
import math


class PixelIdentityVAE(nn.Module):
    """直接像素空间传"""

    def __init__(self, image_channels: int = 3):
        super().__init__()
        self.image_channels = int(image_channels)
        self.z_dim = self.image_channels
        self.downsample_factor = 1
        self.patch_size = 1

    def encode(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(f"PixelIdentityVAE.encode expects BCHW tensor, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelIdentityVAE.encode expects {self.image_channels} channels, got {int(x.shape[1])}."
            )
        return x

    def decode(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(f"PixelIdentityVAE.decode expects BCHW tensor, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelIdentityVAE.decode expects {self.image_channels} channels, got {int(x.shape[1])}."
            )
        return x


class PixelLogitVAE(nn.Module):
    """映射[-1,1]像素到近似高斯分布(0, 1)

    编码:  x → p = (x+1)/2 → clamp(eps, 1-eps) → logit(p) / scale
    解码:  z → sigmoid(z * scale) * 2 - 1
    """

    def __init__(self, image_channels: int = 3, logit_scale: float = 4.0, clamp_eps: float = 1e-5):
        super().__init__()
        self.image_channels = int(image_channels)
        self.z_dim = self.image_channels
        self.downsample_factor = 1
        self.patch_size = 1
        self.logit_scale = float(logit_scale)
        self.clamp_eps = float(clamp_eps)

    def encode(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(f"PixelLogitVAE.encode expects BCHW tensor, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelLogitVAE.encode expects {self.image_channels} channels, got {int(x.shape[1])}."
            )
        p = (x + 1.0) * 0.5
        p = p.clamp(self.clamp_eps, 1.0 - self.clamp_eps)
        z = torch.log(p) - torch.log1p(-p)
        return z / self.logit_scale

    def decode(self, z, **kwargs):
        if z.ndim != 4:
            raise ValueError(f"PixelLogitVAE.decode expects BCHW tensor, got shape={tuple(z.shape)}")
        if int(z.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelLogitVAE.decode expects {self.image_channels} channels, got {int(z.shape[1])}."
            )
        p = torch.sigmoid(z * self.logit_scale)
        return p * 2.0 - 1.0
