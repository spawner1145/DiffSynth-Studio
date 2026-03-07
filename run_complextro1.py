import os
import torch
from PIL import Image

from transformers import AutoProcessor
from diffsynth.core import load_model
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M
from diffsynth.pipelines.complextro import ComplextroPipeline


def build_complextro_pipe(
    device="cuda",
    torch_dtype=torch.bfloat16,
    qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw/model.safetensors",
    flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
    complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e3-s10059.safetensors",
    qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw",
    qwen_model_size: str = "2B",
    siglip_model_file="",
    use_alpha_layer_vae: bool = False,
    complextro_model_config=None,
):
    pipe = ComplextroPipeline(device=device, torch_dtype=torch_dtype)
    if complextro_model_config is None:
        complextro_model_config = {}
    else:
        complextro_model_config = dict(complextro_model_config)

    siglip_model_file = "" if siglip_model_file is None else str(siglip_model_file).strip()
    siglip_enabled = bool(siglip_model_file)
    if siglip_enabled:
        if not os.path.exists(siglip_model_file):
            raise FileNotFoundError(f"SigLIP model file not found: {siglip_model_file}")
        expected_siglip_feat_dim = 1152
        configured_siglip_feat_dim = complextro_model_config.get("siglip_feat_dim", None)
        if configured_siglip_feat_dim is None:
            complextro_model_config["siglip_feat_dim"] = expected_siglip_feat_dim
        elif int(configured_siglip_feat_dim) != expected_siglip_feat_dim:
            raise ValueError(
                f"siglip_feat_dim ({configured_siglip_feat_dim}) must match Siglip2ImageEncoder428M output dim ({expected_siglip_feat_dim})."
            )

    pipe.text_encoder = load_model(
        QwenImageTextEncoder,
        qwen_model_file,
        config={"model_type": "qwen3_5", "model_size": qwen_model_size},
        torch_dtype=torch_dtype,
        device=device,
        state_dict_converter=QwenImageTextEncoderStateDictConverter,
    )
    pipe.vae = load_model(
        Flux2VAE,
        flux2_vae_file,
        config={"use_alpha_layer": use_alpha_layer_vae},
        torch_dtype=torch_dtype,
        device=device,
    )
    pipe.processor = AutoProcessor.from_pretrained(qwen_tokenizer_dir)
    pipe.tokenizer = pipe.processor.tokenizer

    text_config = getattr(pipe.text_encoder.model.config, "text_config", pipe.text_encoder.model.config)
    text_hidden_size = int(text_config.hidden_size)
    configured_text_dim = complextro_model_config.get("text_embed_dim", None)
    if configured_text_dim is None:
        complextro_model_config["text_embed_dim"] = text_hidden_size
    elif int(configured_text_dim) != text_hidden_size:
        raise ValueError(
            f"complextro_model_config['text_embed_dim'] ({configured_text_dim}) must match text encoder hidden_size ({text_hidden_size})."
        )

    pipe.dit = load_model(
        ComplextroImageDiT,
        complextro_dit_file,
        config=complextro_model_config,
        torch_dtype=torch_dtype,
        device=device,
    )

    dit_text_dim = int(pipe.dit.txt_in.in_features)
    if text_hidden_size != dit_text_dim:
        raise ValueError(
            f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
            f"Please align QwenImageTextEncoder(model_type='qwen3_5', model_size='{qwen_model_size}') and ComplextroImageDiT(text_embed_dim=...)."
        )

    if siglip_enabled:
        pipe.image_encoder = load_model(
            Siglip2ImageEncoder428M,
            siglip_model_file,
            torch_dtype=torch_dtype,
            device=device,
        )

    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


if __name__ == "__main__":
    """
    Complextro + Qwen3.5 使用说明（推理）

    1) omni_mode=False 也可以读取参考图：
        - 现在文本编码阶段会走 Qwen3.5 的多模态聊天模板。
        - 只要传入 edit_image，TE 就会把图像作为视觉上下文参与 prompt 编码。

    2) omni_mode=False 的含义：
        - 仅“文本编码器读图”（TE multimodal）。
        - 不启用 Complextro 的 omni latent 拼接路径。

    3) omni_mode=True 的含义：
        - 在 TE 读图之外，还会启用 omni 路径（edit latents / image_noise_mask）。
        - `edit_image` 用于 TE / SigLIP 视觉参考。
        - `edit_latent` 用于 condition latent，可与 `edit_image` 分开传。
        - `edit_latent="0"` 表示该槽位插入等长 pad token。
        - `edit_latent="1"` 表示该槽位直接复用同位 `edit_image`。
        - `edit_latent` 也可以写独立图片路径，此时会单独读取那张图做 VAE 编码。

        4) Prompt 扩展语法：
                - `<prompt start>` 前为 system prompt，后为 user 正文。
                    例："你是风格控制助手<prompt start>生成可爱像素风角色"
                - `<break>` 会把正文拆成多个段，分别编码后再拼接。
                    例："系统提示<prompt start>主体描述<break>风格描述<break>背景描述"
    """
    device = "cuda"
    dtype = torch.bfloat16
    # 需要和训练时候配置一样
    complextro_model_config = {
        "num_layers": 8,
        "num_refiner_layers": 0,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "rope_axes_dim": [16, 56, 56],
    }

    pipe = build_complextro_pipe(
        device=device,
        torch_dtype=dtype,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw/model.safetensors-00001-of-00001.safetensors",
        flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v0/model-e4-s67044.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw",
        qwen_model_size="2B",
        siglip_model_file="",
        use_alpha_layer_vae=False,
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

    # 非 Omni 但读取参考图（仅 TE 多模态）示例：
    # ref_images = [Image.open("ref1.jpg").convert("RGB"), Image.open("ref2.jpg").convert("RGB")]
    # image = pipe(
    #     prompt="保持参考图主体特征，转换为手办渲染风格",
    #     negative_prompt="",
    #     num_inference_steps=30,
    #     cfg_scale=7.0,
    #     seed=2026,
    #     height=256,
    #     width=256,
    #     omni_mode=False,
    #     edit_image=ref_images,
    # )
    # image.save("complextro_text_encoder_multimodal.jpg")

    # Omni 示例
    # 使用前先加载 SigLIP 模型（build_complextro_pipe 里传 siglip_model_file），并准备条件图。
    # 这里 edit_image / edit_latent 都支持直接写路径。
    # cond_images = ["cond1.jpg", "cond2.jpg", "cond3.jpg"]
    # cond_latents = ["1", "0", "latent_ref_3.jpg"]
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
    #     edit_latent=cond_latents,
    #     image_noise_mask=[0, 0, 1],
    # )
    # omni_out.save("complextro_omni.jpg")
