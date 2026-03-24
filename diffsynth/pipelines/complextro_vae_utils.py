from __future__ import annotations

from typing import Any

from ..models.flux2_vae import Flux2VAE
from ..models.pixel_identity_vae import PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE
from ..models.qwen_image_vae import QwenImageVAE


DEFAULT_PIXEL_PATCH_SIZE = 32


def _parse_pixel_vae_type(vae_type: str | None) -> tuple[str, int | None]:
    value = "flux2" if vae_type is None else str(vae_type).strip().lower()
    if value.startswith("pixel:"):
        _, raw_patch_size = value.split(":", 1)
        if raw_patch_size == "":
            raise ValueError("Pixel-space Complextro VAE type must be 'pixel' or 'pixel:<patch_size>'.")
        patch_size = int(raw_patch_size)
        if patch_size <= 0:
            raise ValueError(f"Pixel-space patch size must be positive, got {patch_size}.")
        return "pixel", patch_size
    if value.startswith("pixel_logit:"):
        _, raw_patch_size = value.split(":", 1)
        if raw_patch_size == "":
            raise ValueError("Pixel-logit VAE type must be 'pixel_logit' or 'pixel_logit:<patch_size>'.")
        patch_size = int(raw_patch_size)
        if patch_size <= 0:
            raise ValueError(f"Pixel-logit patch size must be positive, got {patch_size}.")
        return "pixel_logit", patch_size
    if value.startswith("pixel_norm:"):
        _, raw_patch_size = value.split(":", 1)
        if raw_patch_size == "":
            raise ValueError("Pixel-norm VAE type must be 'pixel_norm' or 'pixel_norm:<patch_size>'.")
        patch_size = int(raw_patch_size)
        if patch_size <= 0:
            raise ValueError(f"Pixel-norm patch size must be positive, got {patch_size}.")
        return "pixel_norm", patch_size
    return value, None


def normalize_complextro_vae_type(vae_type: str | None) -> str:
    value, _ = _parse_pixel_vae_type(vae_type)
    aliases = {
        "flux": "flux2",
        "flux2": "flux2",
        "qwen": "qwen_image",
        "qwen-image": "qwen_image",
        "qwen_image": "qwen_image",
        "pixel": "pixel",
        "pixel_space": "pixel",
        "pixel-space": "pixel",
        "pixel_logit": "pixel_logit",
        "pixel-logit": "pixel_logit",
        "logit": "pixel_logit",
        "pixel_norm": "pixel_norm",
        "pixel-norm": "pixel_norm",
        "pixel_normalized": "pixel_norm",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported Complextro VAE type: {vae_type!r}. Expected 'flux2', 'qwen_image', 'pixel', 'pixel_logit', 'pixel_norm', or 'pixel:<patch_size>' / 'pixel_norm:<patch_size>'."
        )
    return aliases[value]


def get_complextro_vae_spec(
    *,
    vae_type: str | None,
    vae_file: str | None,
    use_alpha_layer_vae: bool,
) -> dict[str, Any]:
    raw_type, pixel_patch_size = _parse_pixel_vae_type(vae_type)
    resolved_type = normalize_complextro_vae_type(raw_type)
    if resolved_type not in ("pixel", "pixel_logit", "pixel_norm") and vae_file in (None, ""):
        raise ValueError("Complextro VAE requires vae_file to be set explicitly.")
    if resolved_type == "flux2":
        return {
            "vae_type": "flux2",
            "model_class": Flux2VAE,
            "model_file": vae_file,
            "config": {"use_alpha_layer": use_alpha_layer_vae},
            "latent_channels": 128,
            "latent_downsample_factor": 16,
            "latent_patch_size": 1,
        }
    if resolved_type == "pixel":
        image_channels = 4 if use_alpha_layer_vae else 3
        patch_size = DEFAULT_PIXEL_PATCH_SIZE if pixel_patch_size is None else int(pixel_patch_size)
        return {
            "vae_type": "pixel",
            "model_class": PixelIdentityVAE,
            "model_file": None,
            "config": {"image_channels": image_channels},
            "latent_channels": image_channels,
            "latent_downsample_factor": 1,
            "latent_patch_size": patch_size,
        }
    if resolved_type == "pixel_logit":
        image_channels = 4 if use_alpha_layer_vae else 3
        patch_size = DEFAULT_PIXEL_PATCH_SIZE if pixel_patch_size is None else int(pixel_patch_size)
        return {
            "vae_type": "pixel_logit",
            "model_class": PixelLogitVAE,
            "model_file": None,
            "config": {"image_channels": image_channels},
            "latent_channels": image_channels,
            "latent_downsample_factor": 1,
            "latent_patch_size": patch_size,
        }
    if resolved_type == "pixel_norm":
        image_channels = 4 if use_alpha_layer_vae else 3
        patch_size = DEFAULT_PIXEL_PATCH_SIZE if pixel_patch_size is None else int(pixel_patch_size)
        return {
            "vae_type": "pixel_norm",
            "model_class": PixelNormalizedVAE,
            "model_file": None,
            "config": {"image_channels": image_channels},
            "latent_channels": image_channels,
            "latent_downsample_factor": 1,
            "latent_patch_size": patch_size,
        }
    return {
        "vae_type": "qwen_image",
        "model_class": QwenImageVAE,
        "model_file": vae_file,
        "config": {"image_channels": 4 if use_alpha_layer_vae else 3},
        "latent_channels": 16,
        "latent_downsample_factor": 8,
        "latent_patch_size": 2,
    }


def apply_complextro_vae_config(complextro_model_config: dict[str, Any], latent_channels: int) -> None:
    configured_in_channels = complextro_model_config.get("in_channels", None)
    if configured_in_channels is None:
        complextro_model_config["in_channels"] = latent_channels
        return
    if int(configured_in_channels) != int(latent_channels):
        raise ValueError(
            f"complextro_model_config['in_channels'] ({configured_in_channels}) must match the selected VAE latent "
            f"channels ({latent_channels})."
        )


def apply_complextro_vae_shape_config(
    complextro_model_config: dict[str, Any],
    *,
    latent_channels: int,
    latent_downsample_factor: int,
    latent_patch_size: int,
) -> None:
    apply_complextro_vae_config(complextro_model_config, latent_channels)
    configured_downsample = complextro_model_config.get("latent_downsample_factor", None)
    if configured_downsample is None:
        complextro_model_config["latent_downsample_factor"] = int(latent_downsample_factor)
    elif int(configured_downsample) != int(latent_downsample_factor):
        raise ValueError(
            f"complextro_model_config['latent_downsample_factor'] ({configured_downsample}) must match the selected "
            f"VAE downsample factor ({latent_downsample_factor})."
        )
    configured_patch_size = complextro_model_config.get("latent_patch_size", None)
    if configured_patch_size is None:
        complextro_model_config["latent_patch_size"] = int(latent_patch_size)
        return
    if int(configured_patch_size) != int(latent_patch_size):
        raise ValueError(
            f"complextro_model_config['latent_patch_size'] ({configured_patch_size}) must match the selected "
            f"VAE patch size ({latent_patch_size})."
        )


def infer_complextro_vae_latent_channels(vae) -> int | None:
    if vae is None:
        return None
    if isinstance(vae, PixelLogitVAE):
        return int(vae.image_channels)
    if isinstance(vae, PixelNormalizedVAE):
        return int(vae.image_channels)
    if isinstance(vae, PixelIdentityVAE):
        return int(vae.image_channels)
    if isinstance(vae, QwenImageVAE):
        return 16
    if isinstance(vae, Flux2VAE):
        patch_hw = 4
        if hasattr(vae, "bn") and hasattr(vae.bn, "num_features"):
            return int(vae.bn.num_features)
        return patch_hw * int(getattr(vae.encoder, "out_channels", 32))
    if hasattr(vae, "bn") and hasattr(vae.bn, "num_features"):
        return int(vae.bn.num_features)
    if hasattr(vae, "z_dim"):
        return int(getattr(vae, "z_dim"))
    return None


def infer_complextro_vae_downsample_factor(vae) -> int:
    if isinstance(vae, PixelLogitVAE):
        return 1
    if isinstance(vae, PixelNormalizedVAE):
        return 1
    if isinstance(vae, PixelIdentityVAE):
        return 1
    if isinstance(vae, QwenImageVAE):
        return 8
    if isinstance(vae, Flux2VAE):
        return 16
    if hasattr(vae, "z_dim"):
        return 8
    return 16
