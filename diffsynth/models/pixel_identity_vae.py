import torch
import torch.nn as nn
import math


class PixelIdentityVAE(nn.Module):
    """直接像素空间透传，encode/decode 均为恒等映射"""

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
    """Logit 变换：将 [-1,1] 像素映射到近似高斯的无界空间

    编码:  x ∈ [-1,1] → p = (x+1)/2 → clamp(eps, 1-eps) → logit(p) / scale
    解码:  z → sigmoid(z * scale) * 2 - 1

    来自 normalizing flow 文献（RealNVP, Glow）
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

IMAGENET_MEAN_11 = [-0.030, -0.088, -0.188]
IMAGENET_STD_11 = [0.458, 0.448, 0.450]

class PixelNormalizedVAE(nn.Module):
    """逐通道标准化的像素空间 VAE：将 [-1,1] 像素标准化到近似 N(0,1)
    """

    def __init__(
        self,
        image_channels: int = 3,
        pixel_mean=None,
        pixel_std=None,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.z_dim = self.image_channels
        self.downsample_factor = 1
        self.patch_size = 1

        if pixel_mean is None:
            if self.image_channels == 3:
                pixel_mean = IMAGENET_MEAN_11
            elif self.image_channels == 4:
                pixel_mean = IMAGENET_MEAN_11 + [0.0]  # alpha 通道 mean=0
            else:
                pixel_mean = [0.0] * self.image_channels
        if pixel_std is None:
            if self.image_channels == 3:
                pixel_std = IMAGENET_STD_11
            elif self.image_channels == 4:
                pixel_std = IMAGENET_STD_11 + [0.5]  # alpha 通道 std=0.5
            else:
                pixel_std = [0.45] * self.image_channels

        if len(pixel_mean) != self.image_channels:
            raise ValueError(
                f"PixelNormalizedVAE pixel_mean length ({len(pixel_mean)}) must match image_channels ({self.image_channels})."
            )
        if len(pixel_std) != self.image_channels:
            raise ValueError(
                f"PixelNormalizedVAE pixel_std length ({len(pixel_std)}) must match image_channels ({self.image_channels})."
            )
        if any(float(v) <= 0.0 for v in pixel_std):
            raise ValueError("PixelNormalizedVAE pixel_std must be strictly positive for every channel.")

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(1, -1, 1, 1))

    def encode(self, x, **kwargs):
        if x.ndim != 4:
            raise ValueError(f"PixelNormalizedVAE.encode expects BCHW tensor, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelNormalizedVAE.encode expects {self.image_channels} channels, got {int(x.shape[1])}."
            )
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def decode(self, z, **kwargs):
        if z.ndim != 4:
            raise ValueError(f"PixelNormalizedVAE.decode expects BCHW tensor, got shape={tuple(z.shape)}")
        if int(z.shape[1]) != self.image_channels:
            raise ValueError(
                f"PixelNormalizedVAE.decode expects {self.image_channels} channels, got {int(z.shape[1])}."
            )
        mean = self.pixel_mean.to(device=z.device, dtype=z.dtype)
        std = self.pixel_std.to(device=z.device, dtype=z.dtype)
        return z * std + mean
