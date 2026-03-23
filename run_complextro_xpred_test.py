import os
import importlib
import torch
from PIL import Image
import torchvision.utils as vutils

from transformers import AutoProcessor
from diffsynth.core import load_model
from diffsynth.core.vram import AutoWrappedModule
from diffsynth.configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.models.pixel_identity_vae import PixelIdentityVAE
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M
from diffsynth.pipelines.complextro import ComplextroPipeline
from diffsynth.pipelines.complextro_vae_utils import (
    apply_complextro_vae_shape_config,
    get_complextro_vae_spec,
    infer_complextro_vae_latent_channels,
)


def build_complextro_pipe(
    device="cuda",
    torch_dtype=torch.bfloat16,
    qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw/model.safetensors",
    vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
    complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e3-s10059.safetensors",
    qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw",
    qwen_model_size: str = "2B",
    siglip_model_file="",
    vae_type: str = "flux2",
    use_alpha_layer_vae: bool = False,
    prediction_type: str = "flow",
    jit_p_mean: float = -0.8,
    jit_p_std: float = 0.8,
    jit_noise_scale: float = 1.0,
    jit_t_eps: float = 5e-2,
    jit_sampling_method: str = "heun",
    jit_cfg_interval_min: float = 0.0,
    jit_cfg_interval_max: float = 1.0,
    complextro_model_config=None,
    enable_vram_offload: bool = False,
    vram_config: dict | None = None,
    vram_limit: float | None = None,
):
    pipe = ComplextroPipeline(device=device, torch_dtype=torch_dtype)
    pipe.prediction_type = str(prediction_type)
    pipe.jit_p_mean = float(jit_p_mean)
    pipe.jit_p_std = float(jit_p_std)
    pipe.jit_noise_scale = float(jit_noise_scale)
    pipe.jit_t_eps = float(jit_t_eps)
    pipe.jit_sampling_method = str(jit_sampling_method)
    pipe.jit_cfg_interval_min = float(jit_cfg_interval_min)
    pipe.jit_cfg_interval_max = float(jit_cfg_interval_max)
    if complextro_model_config is None:
        complextro_model_config = {}
    else:
        complextro_model_config = dict(complextro_model_config)
    vae_spec = get_complextro_vae_spec(
        vae_type=vae_type,
        vae_file=vae_file,
        use_alpha_layer_vae=use_alpha_layer_vae,
    )
    apply_complextro_vae_shape_config(
        complextro_model_config,
        latent_channels=vae_spec["latent_channels"],
        latent_downsample_factor=vae_spec["latent_downsample_factor"],
        latent_patch_size=vae_spec["latent_patch_size"],
    )

    if enable_vram_offload:
        if vram_config is None:
            vram_config = {
                "offload_dtype": torch_dtype,
                "offload_device": "cpu",
                "onload_dtype": torch_dtype,
                "onload_device": device,
                "preparing_dtype": torch_dtype,
                "preparing_device": device,
                "computation_dtype": torch_dtype,
                "computation_device": device,
            }
        else:
            vram_config = dict(vram_config)

    def resolve_module_map(model_class):
        def import_class(class_path: str):
            split = class_path.rfind(".")
            module_name, class_name = class_path[:split], class_path[split + 1:]
            return getattr(importlib.import_module(module_name), class_name)

        model_class_path = f"{model_class.__module__}.{model_class.__name__}"
        if model_class_path == "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder":
            return {model_class: AutoWrappedModule}
        if model_class_path in VERSION_CHECKER_MAPS:
            raw_map = VERSION_CHECKER_MAPS[model_class_path]()
            return {import_class(source): import_class(target) for source, target in raw_map.items()}
        if model_class_path not in VRAM_MANAGEMENT_MODULE_MAPS:
            raise KeyError(f"No VRAM management module map registered for {model_class_path}.")
        raw_map = VRAM_MANAGEMENT_MODULE_MAPS[model_class_path]
        return {import_class(source): import_class(target) for source, target in raw_map.items()}

    def load_model_with_optional_offload(model_class, model_file, *, config=None, state_dict_converter=None):
        load_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "device": device,
            "state_dict_converter": state_dict_converter,
        }
        if enable_vram_offload:
            load_kwargs["module_map"] = resolve_module_map(model_class)
            load_kwargs["vram_config"] = vram_config
            load_kwargs["vram_limit"] = vram_limit
        return load_model(model_class, model_file, **load_kwargs)

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

    pipe.text_encoder = load_model_with_optional_offload(
        QwenImageTextEncoder,
        qwen_model_file,
        config={"model_type": "qwen3_5", "model_size": qwen_model_size},
        state_dict_converter=QwenImageTextEncoderStateDictConverter,
    )
    if vae_spec["model_file"] is None and vae_spec["model_class"] is PixelIdentityVAE:
        pipe.vae = PixelIdentityVAE(**vae_spec["config"]).to(device=device, dtype=torch_dtype)
    else:
        pipe.vae = load_model_with_optional_offload(
            vae_spec["model_class"],
            vae_spec["model_file"],
            config=vae_spec["config"],
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

    pipe.dit = load_model_with_optional_offload(
        ComplextroImageDiT,
        complextro_dit_file,
        config=complextro_model_config,
    )
    vae_latent_channels = infer_complextro_vae_latent_channels(pipe.vae)
    dit_latent_channels = int(pipe.dit.latent_channels)
    if vae_latent_channels is not None and vae_latent_channels != dit_latent_channels:
        raise ValueError(
            f"Selected VAE latent channels ({vae_latent_channels}) do not match ComplextroImageDiT in_channels "
            f"({dit_latent_channels})."
        )

    dit_text_dim = int(pipe.dit.txt_in.in_features)
    if text_hidden_size != dit_text_dim:
        raise ValueError(
            f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
            f"Please align QwenImageTextEncoder(model_type='qwen3_5', model_size='{qwen_model_size}') and ComplextroImageDiT(text_embed_dim=...)."
        )

    if siglip_enabled:
        pipe.image_encoder = load_model_with_optional_offload(
            Siglip2ImageEncoder428M,
            siglip_model_file,
        )

    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


@torch.no_grad()
def debug_xpred_once(
    pipe,
    image_path: str,
    prompt: str,
    out_dir: str = "debug_xpred",
    t_value: float = 0.5,
    seed: int = 0,
):
    if pipe.prediction_type not in ("jit_xpred", "bridge_xpred"):
        raise ValueError("debug_xpred_once requires prediction_type='jit_xpred' or 'bridge_xpred'.")
    if not isinstance(pipe.vae, PixelIdentityVAE):
        raise ValueError("debug_xpred_once requires pixel-space Complextro (PixelIdentityVAE).")

    os.makedirs(out_dir, exist_ok=True)
    pipe.load_models_to_device(["vae", "text_encoder", "dit"])

    image_pil = Image.open(image_path)
    image_pil = pipe._normalize_image_mode_for_vae(image_pil)
    x = pipe.preprocess_image(image_pil, torch_dtype=torch.float32, device=pipe.device)
    x = pipe.vae.encode(x).float()

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=pipe.device, dtype=torch.float32)
    noise = noise * float(pipe.jit_noise_scale)

    t = torch.tensor([float(t_value)], device=pipe.device, dtype=torch.float32).view(1, 1, 1, 1)
    z = t * x + (1.0 - t) * noise

    inputs_posi = {"prompt": [prompt]}
    inputs_nega = {"negative_prompt": [""]}
    inputs_shared = {
        "cfg_scale": 1.0,
        "input_image": None,
        "denoising_strength": 1.0,
        "edit_image": None,
        "edit_latent": None,
        "edit_image_auto_resize": True,
        "omni_mode": False,
        "image_noise_mask": None,
        "height": image_pil.height,
        "width": image_pil.width,
        "seed": seed,
        "rand_device": "cuda",
        "batch_size": 1,
        "num_inference_steps": 30,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "latents": z.to(dtype=pipe.torch_dtype),
    }

    skip_units = {
        "ComplextroUnit_NoiseInitializer",
        "ComplextroUnit_InputImageEmbedder",
    }
    for unit in pipe.units:
        if unit.__class__.__name__ in skip_units:
            continue
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_pred = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t.flatten()).float()
    if pipe.prediction_type == "bridge_xpred":
        lambda2 = t.pow(2) + (1.0 - t).pow(2)
        mean_t = (t / lambda2.clamp_min(float(pipe.jit_t_eps))) * z
        scale = ((1.0 - t) / lambda2.clamp_min(float(pipe.jit_t_eps)).sqrt()).clamp_min(float(pipe.jit_t_eps))
        x_pred = mean_t + scale * model_pred
    else:
        x_pred = model_pred

    def save_tensor_image(tensor, path):
        tensor = tensor.detach().cpu().clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0
        vutils.save_image(tensor, path)

    save_tensor_image(x, os.path.join(out_dir, "x.png"))
    save_tensor_image(noise.clamp(-1, 1), os.path.join(out_dir, "noise_vis.png"))
    save_tensor_image(z, os.path.join(out_dir, f"z_t{t_value:.2f}.png"))
    save_tensor_image(x_pred, os.path.join(out_dir, f"xpred_t{t_value:.2f}.png"))

    pipe.load_models_to_device([])


@torch.no_grad()
def debug_xpred_multi_t(
    pipe,
    image_path: str,
    prompt: str,
    out_dir: str = "debug_xpred_multi",
    t_values=None,
    seed: int = 0,
):
    if t_values is None:
        t_values = [0.05, 0.1, 0.2, 0.5, 0.8]
    for t_value in t_values:
        local_out_dir = os.path.join(out_dir, f"t_{str(t_value).replace('.', '_')}")
        debug_xpred_once(
            pipe,
            image_path=image_path,
            prompt=prompt,
            out_dir=local_out_dir,
            t_value=float(t_value),
            seed=seed,
        )


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
    enable_vram_offload = False
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    # 需要和训练时候配置一样
    """
    complextro_model_config = {
        "num_layers": 8,
        "num_refiner_layers": 0,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "rope_axes_dim": [16, 56, 56],
    }
    """
    # 2.25B
    complextro_model_config = {
        "num_layers": 10,
        "num_refiner_layers": 0,
        "hidden_size": 2304,
        "num_attention_heads": 24,
        "attention_head_dim": 96,
        "rope_axes_dim": [32, 32, 32],
        "use_text_modulation": True,
    }

    inference_num_inference_steps = 50
    inference_cfg_scale = 1.0
    inference_denoising_strengths = [0.8, 0.85, 0.9, 1.0]
    inference_omni_mode = False
    inference_output_root = "complextro_xpred_sweep"
    inference_test_input_image = True
    inference_input_image_path = "/root/autodl-tmp/DiffSynth-Studio/data/images/001Bulbasaur.png"
    inference_input_image_strengths = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    inference_input_output_root = "complextro_xpred_i2i_sweep"

    pipe = build_complextro_pipe(
        device=device,
        torch_dtype=dtype,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/Qwen3_5_2b_claude_heretic/model.safetensors",
        vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/edit/model-e3-s10059.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/Qwen3_5_2b_claude_heretic",
        qwen_model_size="2B",
        siglip_model_file="",
        vae_type="pixel:16",
        use_alpha_layer_vae=False,
        prediction_type="jit_xpred",
        jit_sampling_method="heun",
        jit_cfg_interval_min=0.0,
        jit_cfg_interval_max=1.0,
        complextro_model_config=complextro_model_config,
        enable_vram_offload=enable_vram_offload,
        vram_config=vram_config,
    )

    enable_debug_xpred = True
    debug_image_path = "/root/autodl-tmp/DiffSynth-Studio/data/images/001Bulbasaur.png"
    debug_prompt = "green, lizard, plant, Grass, Poison, seed on back, red eyes, smiling expression, short stout limbs, sharp claws"
    debug_out_dir = "debug_xpred"
    debug_t_value = 0.5
    debug_seed = 0
    debug_t_values = [0.05, 0.1, 0.2, 0.5, 0.8]
    debug_multi_t = True

    if enable_debug_xpred:
        if debug_multi_t:
            debug_xpred_multi_t(
                pipe,
                image_path=debug_image_path,
                prompt=debug_prompt,
                out_dir=debug_out_dir,
                t_values=debug_t_values,
                seed=debug_seed,
            )
        else:
            debug_xpred_once(
                pipe,
                image_path=debug_image_path,
                prompt=debug_prompt,
                out_dir=debug_out_dir,
                t_value=debug_t_value,
                seed=debug_seed,
            )

    prompts = [
        "green, lizard, plant, Grass, Poison, seed on back, red eyes, smiling expression, short stout limbs, sharp claws",
        "orange, cream, lizard, Fire, flame on tail tip, large eyes, smiling expression, cream-colored belly patch, sharp claws",
        "蓝色，米色，棕色，乌龟，水系，龟壳，大眼睛，短四肢，卷曲尾巴",
    ]

    os.makedirs(inference_output_root, exist_ok=True)
    for denoising_strength in inference_denoising_strengths:
        run_dir = os.path.join(
            inference_output_root,
            f"ds_{str(denoising_strength).replace('.', '_')}_steps_{inference_num_inference_steps}_cfg_{str(inference_cfg_scale).replace('.', '_')}",
        )
        os.makedirs(run_dir, exist_ok=True)
        for seed, prompt in enumerate(prompts):
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=inference_num_inference_steps,
                cfg_scale=inference_cfg_scale,
                denoising_strength=denoising_strength,
                seed=seed,
                height=256,
                width=256,
                omni_mode=inference_omni_mode,
            )
            image.save(os.path.join(run_dir, f"complextro_image_{seed}.png"))

    if inference_test_input_image:
        input_image = Image.open(inference_input_image_path).convert("RGB").resize((256, 256))
        os.makedirs(inference_input_output_root, exist_ok=True)
        for denoising_strength in inference_input_image_strengths:
            run_dir = os.path.join(
                inference_input_output_root,
                f"ds_{str(denoising_strength).replace('.', '_')}_steps_{inference_num_inference_steps}_cfg_{str(inference_cfg_scale).replace('.', '_')}",
            )
            os.makedirs(run_dir, exist_ok=True)
            for seed, prompt in enumerate(prompts):
                image = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    input_image=input_image,
                    num_inference_steps=inference_num_inference_steps,
                    cfg_scale=inference_cfg_scale,
                    denoising_strength=denoising_strength,
                    seed=seed,
                    height=256,
                    width=256,
                    omni_mode=inference_omni_mode,
                )
                image.save(os.path.join(run_dir, f"complextro_image_{seed}.png"))

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
    # image.save("complextro_text_encoder_multimodal.png")

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
    # )
    # omni_out.save("complextro_omni.png")
