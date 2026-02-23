import os
import torch
from PIL import Image

from transformers import AutoTokenizer
from diffsynth.core import load_model
from diffsynth.models.z_image_text_encoder import ZImageTextEncoder
from diffsynth.utils.state_dict_converters.z_image_text_encoder import ZImageTextEncoderStateDictConverter
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M
from diffsynth.pipelines.complextro import ComplextroPipeline


def build_complextro_pipe(
    device="cuda",
    torch_dtype=torch.bfloat16,
    qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3/model.safetensors",
    flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
    complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e3-s10059.safetensors",
    qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3",
    siglip_model_file=None,
    complextro_model_config=None,
):
    pipe = ComplextroPipeline(device=device, torch_dtype=torch_dtype)
    if complextro_model_config is None:
        complextro_model_config = {}

    pipe.text_encoder = load_model(
        ZImageTextEncoder,
        qwen_model_file,
        config={"model_size": "0.6B"},
        torch_dtype=torch_dtype,
        device=device,
        state_dict_converter=ZImageTextEncoderStateDictConverter,
    )
    pipe.vae = load_model(
        Flux2VAE,
        flux2_vae_file,
        torch_dtype=torch_dtype,
        device=device,
    )
    pipe.tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir)
    pipe.dit = load_model(
        ComplextroImageDiT,
        complextro_dit_file,
        config=complextro_model_config,
        torch_dtype=torch_dtype,
        device=device,
    )

    text_hidden_size = int(pipe.text_encoder.model.config.hidden_size)
    dit_text_dim = int(pipe.dit.txt_in.in_features)
    if text_hidden_size != dit_text_dim:
        raise ValueError(
            f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
            f"Please align ZImageTextEncoder model_size and ComplextroImageDiT(text_embed_dim=...)."
        )

    if siglip_model_file is not None and os.path.exists(siglip_model_file):
        pipe.image_encoder = load_model(
            Siglip2ImageEncoder428M,
            siglip_model_file,
            torch_dtype=torch_dtype,
            device=device,
        )

    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.bfloat16
    # 需要和训练时候配置一样
    complextro_model_config = {
        "num_layers": 12,
        "num_refiner_layers": 1,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "rope_axes_dim": [16, 56, 56],
    }

    pipe = build_complextro_pipe(
        device=device,
        torch_dtype=dtype,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3/model.safetensors",
        flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e36-s120708.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3",
        siglip_model_file=None,
        complextro_model_config=complextro_model_config,
    )

    prompts = [
        "green, lizard, plant, Grass, Poison, seed on back, red eyes, smiling expression, short stout limbs, sharp claws",
        "orange, cream, lizard, Fire, flame on tail tip, large eyes, smiling expression, cream-colored belly patch, sharp claws",
        "蓝色，米色，棕色，乌龟，水系，龟壳，大眼睛，短四肢，卷曲尾巴",
    ]

    for seed, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=30,
            cfg_scale=7.0,
            seed=seed,
            height=256,
            width=256,
            omni_mode=False,
        )
        image.save(f"complextro_image_{seed}.jpg")

    # Omni 示例
    # 使用前先加载 SigLIP 模型（build_complextro_pipe 里传 siglip_model_file），并准备条件图。
    # cond_images = [Image.open("cond1.jpg").convert("RGB"), Image.open("cond2.jpg").convert("RGB")]
    # omni_out = pipe(
    #     prompt="keep subject identity, turn into anime style",
    #     negative_prompt="",
    #     num_inference_steps=30,
    #     cfg_scale=7.0,
    #     seed=123,
    #     height=256,
    #     width=256,
    #     omni_mode=True,
    #     edit_image=cond_images,
    #     image_noise_mask=[0, 0, 1],
    # )
    # omni_out.save("complextro_omni.jpg")
