from __future__ import annotations

from typing import Any

from ..models.flux2_vae import Flux2VAE
from ..models.qwen_image_vae import QwenImageVAE


def normalize_complextro_vae_type(vae_type: str | None) -> str:
    value = "flux2" if vae_type is None else str(vae_type).strip().lower()
    aliases = {
        "flux": "flux2",
        "flux2": "flux2",
        "qwen": "qwen_image",
        "qwen-image": "qwen_image",
        "qwen_image": "qwen_image",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported Complextro VAE type: {vae_type!r}. Expected 'flux2' or 'qwen_image'.")
    return aliases[value]


def get_complextro_vae_spec(
    *,
    vae_type: str | None,
    vae_file: str | None,
    use_alpha_layer_vae: bool,
) -> dict[str, Any]:
    resolved_type = normalize_complextro_vae_type(vae_type)
    if vae_file in (None, ""):
        raise ValueError("Complextro VAE requires vae_file to be set explicitly.")
    if resolved_type == "flux2":
        return {
            "vae_type": "flux2",
            "model_class": Flux2VAE,
            "model_file": vae_file,
            "config": {"use_alpha_layer": use_alpha_layer_vae},
            "latent_channels": 128,
        }
    return {
        "vae_type": "qwen_image",
        "model_class": QwenImageVAE,
        "model_file": vae_file,
        "config": {"image_channels": 4 if use_alpha_layer_vae else 3},
        "latent_channels": 16,
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


def infer_complextro_vae_latent_channels(vae) -> int | None:
    if vae is None:
        return None
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
    if isinstance(vae, QwenImageVAE):
        return 8
    if isinstance(vae, Flux2VAE):
        return 16
    if hasattr(vae, "z_dim"):
        return 8
    return 16
