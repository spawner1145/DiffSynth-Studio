import torch
import torch.nn as nn


class PixelIdentityVAE(nn.Module):
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
