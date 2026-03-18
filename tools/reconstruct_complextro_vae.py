"""
flux2:

python tools/reconstruct_complextro_vae.py ^
--vae_type flux2 ^
--vae_file /path/to/flux2_vae.safetensors ^
--image /path/to/test.png ^
--output_dir /path/to/out

qwen image:

python tools/reconstruct_complextro_vae.py ^
--vae_type qwen_image ^
--vae_file /path/to/qwen_image_vae.safetensors ^
--image /path/to/test.png ^
--output_dir /path/to/out

透明通道版本：

python tools/reconstruct_complextro_vae.py ^
--vae_type qwen_image ^
--vae_file /path/to/qwen_image_layered_vae.safetensors ^
--use_alpha_layer_vae ^
--image /path/to/test.png ^
--output_dir /path/to/out
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from diffsynth.core import load_model
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.qwen_image_vae import QwenImageVAE
from diffsynth.pipelines.complextro_vae_utils import get_complextro_vae_spec


def load_pil_image(path: str, expected_channels: int) -> Image.Image:
    image = Image.open(path)
    target_mode = "RGBA" if expected_channels == 4 else "RGB"
    if image.mode != target_mode:
        image = image.convert(target_mode)
    return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, -1)
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
    image = image * 2.0 - 1.0
    return image


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    image = image.permute(1, 2, 0).contiguous().numpy()
    mode = "RGBA" if image.shape[2] == 4 else "RGB"
    return Image.fromarray(image, mode=mode)


def build_vae(vae_type: str, vae_file: str, use_alpha_layer_vae: bool, device: str, torch_dtype: torch.dtype):
    vae_spec = get_complextro_vae_spec(
        vae_type=vae_type,
        vae_file=vae_file,
        use_alpha_layer_vae=use_alpha_layer_vae,
    )
    vae = load_model(
        vae_spec["model_class"],
        vae_spec["model_file"],
        config=vae_spec["config"],
        torch_dtype=torch_dtype,
        device=device,
    )
    return vae, vae_spec


@torch.inference_mode()
def reconstruct(vae, image: Image.Image, device: str, torch_dtype: torch.dtype) -> Image.Image:
    image_tensor = preprocess_image(image).to(device=device, dtype=torch_dtype)
    latents = vae.encode(image_tensor)
    recon = vae.decode(latents)
    return tensor_to_pil(recon[0])


def main():
    parser = argparse.ArgumentParser(description="Encode/decode images with Complextro-compatible VAE and save reconstructions.")
    parser.add_argument("--vae_type", type=str, default="flux2", help="flux2 or qwen_image")
    parser.add_argument("--vae_file", type=str, required=True, help="VAE model file")
    parser.add_argument("--use_alpha_layer_vae", action="store_true", help="Use RGBA VAE variant")
    parser.add_argument("--image", type=str, nargs="+", required=True, help="Input image path(s)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reconstructed images")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae, vae_spec = build_vae(
        vae_type=args.vae_type,
        vae_file=args.vae_file,
        use_alpha_layer_vae=args.use_alpha_layer_vae,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    expected_channels = 4 if args.use_alpha_layer_vae else 3
    print(f"Loaded VAE type={vae_spec['vae_type']} latent_channels={vae_spec['latent_channels']} image_channels={expected_channels}")

    for image_path in args.image:
        input_image = load_pil_image(image_path, expected_channels=expected_channels)
        recon = reconstruct(vae, input_image, device=args.device, torch_dtype=torch_dtype)

        stem = Path(image_path).stem
        output_ext = ".png" if expected_channels == 4 else ".png"
        output_path = output_dir / f"{stem}_{vae_spec['vae_type']}_recon{output_ext}"
        recon.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
