import os, ast, argparse, importlib
import torch, accelerate
from accelerate import DistributedDataParallelKwargs
from typing import List, Optional, Any

from transformers import AutoProcessor
from diffsynth.core import UnifiedDataset, ImageTextPairDataset, load_model
from diffsynth.core.vram import AutoWrappedModule
from diffsynth.configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
from diffsynth.core.data.operators import ImageCropAndResize, LoadImage, ToAbsolutePath
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    JiTXPredLoss,
    BridgeXPredLoss,
    ModelLogger,
    launch_training_task,
)
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.models.pixel_identity_vae import PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M
from diffsynth.pipelines.complextro import ComplextroPipeline
from diffsynth.pipelines.complextro_vae_utils import (
    apply_complextro_vae_shape_config,
    get_complextro_vae_spec,
    infer_complextro_vae_latent_channels,
)


class ComplextroTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        device,
        qwen_model_file="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b/model.safetensors",
        vae_file="/mnt/raid0/linux-train/diffusion-model-v1/flux2-vae/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b",
        qwen_model_size: str = "2B",
        siglip_model_file="",
        #complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e43-s19221.safetensors",
        complextro_dit_file="",
        train_omni: bool = False,
        vae_type: str = "flux2",
        use_alpha_layer_vae: bool = False,
        complextro_model_config: Optional[dict] = None,
        prediction_type: str = "flow",
        condition_drop_prob: float = 0.0,
        jit_p_mean: float = -0.8,
        jit_p_std: float = 0.8,
        jit_noise_scale: float = 1.0,
        jit_t_eps: float = 5e-2,
        jit_sampling_method: str = "heun",
        jit_cfg_interval_min: float = 0.0,
        jit_cfg_interval_max: float = 1.0,
        jit_loss_weighting: str = "balanced",
        freq_loss_enabled: bool = False,  # DeCo 原始推荐：开启；这里默认关闭以保持原行为
        freq_loss_weight: float = 0.0,  # DeCo 原值：1.0
        freq_loss_mode: str = "dct",  # DeCo 原值：DCT block spectral loss
        freq_loss_block_size: int = 8,  # DeCo 原值：8
        freq_loss_profile: str = "jpeg",  # DeCo 风格：JPEG-inspired weighting
        freq_loss_quality: int = 85,  # DeCo 原值：85
        freq_loss_jpeg_mode: str = "inv_gamma",  # DeCo 原值：inv_gamma
        freq_loss_gamma: float = 1.0,  # DeCo 原值：1.0
        freq_loss_color_space: str = "rgb",  # DeCo 原实现等价于先转 YCbCr；推荐实验时改成 ycbcr
        freq_loss_weight_floor: float = 0.1,
        freq_loss_hf_scale: float = 0.25,
        freq_loss_lf_scale: float = 1.0,
        freq_loss_t_adaptive: bool = False,
        freq_loss_t_min_hf_scale: float = 0.25,
        freq_loss_t_max_hf_scale: float = 1.0,
        freq_loss_t_gamma: float = 1.0,
        enable_vram_offload: bool = False,
        vram_config: Optional[dict] = None,
        vram_limit: Optional[float] = None,
    ):
        super().__init__()
        self.train_omni = train_omni
        self.condition_drop_prob = float(condition_drop_prob)
        self.complextro_model_config = {} if complextro_model_config is None else dict(complextro_model_config)
        self.enable_vram_offload = enable_vram_offload
        self.vae_spec = get_complextro_vae_spec(
            vae_type=vae_type,
            vae_file=vae_file,
            use_alpha_layer_vae=use_alpha_layer_vae,
        )
        apply_complextro_vae_shape_config(
            self.complextro_model_config,
            latent_channels=self.vae_spec["latent_channels"],
            latent_downsample_factor=self.vae_spec["latent_downsample_factor"],
            latent_patch_size=self.vae_spec["latent_patch_size"],
        )
        siglip_enabled = bool(siglip_model_file) and os.path.exists(siglip_model_file)
        if siglip_enabled:
            expected_siglip_feat_dim = 1152
            configured_siglip_feat_dim = self.complextro_model_config.get("siglip_feat_dim", None)
            if configured_siglip_feat_dim is None:
                self.complextro_model_config["siglip_feat_dim"] = expected_siglip_feat_dim
            elif int(configured_siglip_feat_dim) != expected_siglip_feat_dim:
                raise ValueError(
                    f"siglip_feat_dim ({configured_siglip_feat_dim}) must match Siglip2ImageEncoder428M output dim ({expected_siglip_feat_dim})."
                )

        self.pipe = ComplextroPipeline(device=device, torch_dtype=torch.bfloat16)
        self.pipe.prediction_type = str(prediction_type)
        self.pipe.jit_p_mean = float(jit_p_mean)
        self.pipe.jit_p_std = float(jit_p_std)
        self.pipe.jit_noise_scale = float(jit_noise_scale)
        self.pipe.jit_t_eps = float(jit_t_eps)
        self.pipe.jit_sampling_method = str(jit_sampling_method)
        self.pipe.jit_cfg_interval_min = float(jit_cfg_interval_min)
        self.pipe.jit_cfg_interval_max = float(jit_cfg_interval_max)
        self.pipe.jit_loss_weighting = str(jit_loss_weighting)
        self.pipe.freq_loss_enabled = bool(freq_loss_enabled)
        self.pipe.freq_loss_weight = float(freq_loss_weight)
        self.pipe.freq_loss_mode = str(freq_loss_mode)
        self.pipe.freq_loss_block_size = int(freq_loss_block_size)
        self.pipe.freq_loss_profile = str(freq_loss_profile)
        self.pipe.freq_loss_quality = int(freq_loss_quality)
        self.pipe.freq_loss_jpeg_mode = str(freq_loss_jpeg_mode)
        self.pipe.freq_loss_gamma = float(freq_loss_gamma)
        self.pipe.freq_loss_color_space = str(freq_loss_color_space)
        self.pipe.freq_loss_weight_floor = float(freq_loss_weight_floor)
        self.pipe.freq_loss_hf_scale = float(freq_loss_hf_scale)
        self.pipe.freq_loss_lf_scale = float(freq_loss_lf_scale)
        self.pipe.freq_loss_t_adaptive = bool(freq_loss_t_adaptive)
        self.pipe.freq_loss_t_min_hf_scale = float(freq_loss_t_min_hf_scale)
        self.pipe.freq_loss_t_max_hf_scale = float(freq_loss_t_max_hf_scale)
        self.pipe.freq_loss_t_gamma = float(freq_loss_t_gamma)
        if enable_vram_offload:
            if vram_config is None:
                vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": device,
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": device,
                    "computation_dtype": torch.bfloat16,
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

        def load_aux_model(model_class, model_file, *, config=None, state_dict_converter=None):
            load_kwargs = {
                "config": config,
                "torch_dtype": torch.bfloat16,
                "device": device,
                "state_dict_converter": state_dict_converter,
            }
            if enable_vram_offload:
                load_kwargs["module_map"] = resolve_module_map(model_class)
                load_kwargs["vram_config"] = vram_config
                load_kwargs["vram_limit"] = vram_limit
            return load_model(model_class, model_file, **load_kwargs)

        self.pipe.text_encoder = load_aux_model(
            QwenImageTextEncoder,
            qwen_model_file,
            config={"model_type": "qwen3_5", "model_size": qwen_model_size},
            state_dict_converter=QwenImageTextEncoderStateDictConverter,
        )
        if self.vae_spec["model_file"] is None and self.vae_spec["model_class"] in (
            PixelIdentityVAE,
            PixelLogitVAE,
            PixelNormalizedVAE,
        ):
            self.pipe.vae = self.vae_spec["model_class"](**self.vae_spec["config"]).to(device=device, dtype=torch.float32)
        else:
            self.pipe.vae = load_aux_model(
                self.vae_spec["model_class"],
                self.vae_spec["model_file"],
                config=self.vae_spec["config"],
            )
        self.pipe.processor = AutoProcessor.from_pretrained(qwen_tokenizer_dir)
        self.pipe.tokenizer = self.pipe.processor.tokenizer

        if siglip_enabled:
            self.pipe.image_encoder = load_aux_model(
                Siglip2ImageEncoder428M,
                siglip_model_file,
            )

        self.pipe.vram_management_enabled = self.pipe.check_vram_management_state()

        text_config = getattr(self.pipe.text_encoder.model.config, "text_config", self.pipe.text_encoder.model.config)
        text_hidden_size = int(text_config.hidden_size)
        configured_text_dim = self.complextro_model_config.get("text_embed_dim", None)
        if configured_text_dim is None:
            self.complextro_model_config["text_embed_dim"] = text_hidden_size
        elif int(configured_text_dim) != text_hidden_size:
            raise ValueError(
                f"complextro_model_config['text_embed_dim'] ({configured_text_dim}) must match text encoder hidden_size ({text_hidden_size})."
            )
        
        if complextro_dit_file:
            self.pipe.dit = load_model(
                ComplextroImageDiT,
                complextro_dit_file,
                config=self.complextro_model_config,
                torch_dtype=torch.bfloat16,
                device=device,
            )
        else:
            self.pipe.dit = ComplextroImageDiT(**self.complextro_model_config)
            self.pipe.dit = self.pipe.dit.to(device=device)
            self.pipe.dit = self.pipe.dit.to(dtype=torch.bfloat16)
        vae_latent_channels = infer_complextro_vae_latent_channels(self.pipe.vae)
        dit_latent_channels = int(self.pipe.dit.latent_channels)
        if vae_latent_channels is not None and vae_latent_channels != dit_latent_channels:
            raise ValueError(
                f"Selected VAE latent channels ({vae_latent_channels}) do not match ComplextroImageDiT in_channels "
                f"({dit_latent_channels})."
            )

        dit_text_dim = int(self.pipe.dit.txt_in.in_features)
        if text_hidden_size != dit_text_dim:
            raise ValueError(
                f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
                f"Please align QwenImageTextEncoder(model_type='qwen3_5', model_size='{qwen_model_size}') and ComplextroImageDiT(text_embed_dim=...)."
            )

        self.pipe.freeze_except(["dit"])
        self.pipe.scheduler.set_timesteps(1000, training=True)

    @staticmethod
    def _is_nan_scalar(value: Any) -> bool:
        return isinstance(value, float) and value != value

    @classmethod
    def _is_missing_value(cls, value: Any) -> bool:
        return value is None or cls._is_nan_scalar(value)

    def _infer_batch_size(self, prompt_value: Any, image_value: Any) -> int:
        image_batch = int(image_value.shape[0]) if isinstance(image_value, torch.Tensor) and image_value.ndim >= 1 else (
            len(image_value) if isinstance(image_value, (list, tuple)) else 1
        )
        prompt_batch = len(prompt_value) if isinstance(prompt_value, (list, tuple)) and not isinstance(prompt_value, str) else 1
        return max(prompt_batch, image_batch)

    def _split_batch_value(self, value: Any, batch_size: int) -> List[Any]:
        if batch_size <= 1:
            return [value]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value for _ in range(batch_size)]
            if int(value.shape[0]) != batch_size:
                raise ValueError(f"Tensor batch dimension {value.shape[0]} does not match batch_size {batch_size}.")
            return [value[i] for i in range(batch_size)]
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if len(value) == batch_size:
                return value
            if len(value) == 1:
                return value * batch_size
        return [value for _ in range(batch_size)]

    def _normalize_prompt_entries(self, prompt_entries: List[Any]) -> List[str]:
        prompts = []
        for entry in prompt_entries:
            if self._is_missing_value(entry):
                prompts.append("")
            else:
                prompts.append(entry if isinstance(entry, str) else str(entry))
        return prompts

    def _apply_condition_dropout(self, prompts: List[str]) -> List[str]:
        if self.condition_drop_prob <= 0:
            return prompts
        if len(prompts) == 0:
            return prompts
        keep = torch.rand(len(prompts)) >= self.condition_drop_prob
        return [prompt if bool(keep[idx]) else "" for idx, prompt in enumerate(prompts)]

    def _normalize_edit_image_entry(self, entry: Any) -> List[Any]:
        if self._is_missing_value(entry):
            return []
        if not isinstance(entry, (list, tuple)):
            return [entry]
        images = []
        for item in entry:
            if self._is_missing_value(item):
                continue
            images.append(item)
        return images

    def _normalize_edit_latent_entry(self, entry: Any) -> List[Any]:
        if self._is_missing_value(entry):
            return []
        if not isinstance(entry, (list, tuple)):
            return [entry]
        latents = []
        for item in entry:
            if self._is_nan_scalar(item):
                continue
            latents.append(item)
        return latents

    def _normalize_single_noise_mask(self, entry: Any, cond_num: int) -> List[int]:
        if self._is_missing_value(entry):
            return [0] * cond_num + [1]
        if not isinstance(entry, (list, tuple)):
            local = [int(entry)]
        else:
            local = [int(v) for v in entry if not self._is_missing_value(v)]
        need_len = cond_num + 1
        if len(local) == 0:
            return [0] * cond_num + [1]
        if len(local) < need_len:
            local = local + [local[-1]] * (need_len - len(local))
        return local[:need_len]

    def _infer_omni_condition_count(
        self,
        edit_image_group: Optional[List[Any]],
        edit_latent_group: Optional[List[Any]],
    ) -> int:
        image_count = 0 if edit_image_group is None else len(edit_image_group)
        latent_count = 0 if edit_latent_group is None else len(edit_latent_group)
        return max(image_count, latent_count)

    def _infer_image_hw(self, image_entries: List[Any]):
        first_image = image_entries[0]
        if hasattr(first_image, "size"):
            width, height = first_image.size
            return int(height), int(width)
        if hasattr(first_image, "shape") and len(first_image.shape) >= 2:
            return int(first_image.shape[-2]), int(first_image.shape[-1])
        raise ValueError("Cannot infer image height/width from data['image']")

    def _validate_batched_image_shapes(self, image_entries: List[Any]):
        if len(image_entries) <= 1:
            return
        reference_hw = self._infer_image_hw([image_entries[0]])
        for idx, image in enumerate(image_entries[1:], start=1):
            local_hw = self._infer_image_hw([image])
            if local_hw != reference_hw:
                raise ValueError(
                    f"Batch size > 1 requires all target images in a batch to share the same resolution. "
                    f"Got sample0={reference_hw}, sample{idx}={local_hw}. Enable bucket batching or use a fixed training resolution."
                )

    def forward(self, data):
        batch_size = self._infer_batch_size(data["prompt"], data["image"])
        prompt_entries = self._split_batch_value(data["prompt"], batch_size)
        image_entries = self._split_batch_value(data["image"], batch_size)
        neg_prompt_entries = self._split_batch_value(data.get("neg_prompt", ""), batch_size)
        edit_image_entries = self._split_batch_value(data.get("edit_image", None), batch_size)
        edit_latent_entries = self._split_batch_value(data.get("edit_latent", None), batch_size)
        noise_mask_entries = self._split_batch_value(data.get("image_noise_mask", None), batch_size)

        prompt_value = self._normalize_prompt_entries(prompt_entries)
        prompt_value = self._apply_condition_dropout(prompt_value)
        neg_prompt_value = self._normalize_prompt_entries(neg_prompt_entries)
        self._validate_batched_image_shapes(image_entries)
        height, width = self._infer_image_hw(image_entries)

        edit_images = [self._normalize_edit_image_entry(entry) for entry in edit_image_entries]
        if all(len(group) == 0 for group in edit_images):
            edit_images = None

        edit_latent_inputs = [self._normalize_edit_latent_entry(entry) for entry in edit_latent_entries]
        if all(len(group) == 0 for group in edit_latent_inputs):
            edit_latent_inputs = None

        omni_condition_counts = [
            self._infer_omni_condition_count(
                edit_images[i] if edit_images is not None else None,
                edit_latent_inputs[i] if edit_latent_inputs is not None else None,
            )
            for i in range(batch_size)
        ]
        has_any_omni_condition = any(cond_num > 0 for cond_num in omni_condition_counts)
        omni_noise_mask = None
        if self.train_omni and has_any_omni_condition:
            omni_noise_mask = [
                self._normalize_single_noise_mask(noise_mask_entries[i], omni_condition_counts[i])
                for i in range(batch_size)
            ]

        inputs_posi = {"prompt": prompt_value}
        inputs_nega = {"negative_prompt": neg_prompt_value}
        inputs_shared = {
            "input_image": image_entries if batch_size > 1 else image_entries[0],
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "cfg_scale": 1,
            "edit_image": edit_images,
            "edit_latent": edit_latent_inputs,
            "omni_mode": self.train_omni and has_any_omni_condition,
            "image_noise_mask": omni_noise_mask,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": True,
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        if self.pipe.prediction_type == "jit_xpred":
            loss = JiTXPredLoss(self.pipe, **inputs_shared, **inputs_posi)
        elif self.pipe.prediction_type == "bridge_xpred":
            loss = BridgeXPredLoss(self.pipe, **inputs_shared, **inputs_posi)
        else:
            loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        self.latest_loss_metrics = dict(getattr(self.pipe, "_last_loss_metrics", {}))
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complextro Training Script")
    parser.add_argument(
        "--vae_type",
        type=str,
        default="flux2",
        help="Complextro VAE type: flux2 / qwen_image / pixel / pixel_logit / pixel_norm / <type>:<patch_size>",
    )
    parser.add_argument(
        "--vae_file",
        type=str,
        default="/mnt/raid0/linux-train/diffusion-model-v1/flux2-vae/diffusion_pytorch_model.safetensors",
        help="Complextro VAE model file; ignored when --vae_type is pixel / pixel_logit / pixel_norm",
    )
    parser.add_argument("--use_image_text_pairs", action="store_true", help="True: 使用 ImageTextPairDataset（图片+txt目录），False: 使用 UnifiedDataset（metadata文件）")
    parser.add_argument("--train_omni", action="store_true", default=True, help="是否开启 Omni/编辑训练模式")
    parser.add_argument("--use_alpha_layer_vae", action="store_true", help="是否使用带 alpha 层 VAE")
    parser.add_argument("--siglip_model_file", type=str, default="", help="SigLIP 模型文件路径")
    parser.add_argument("--qwen_model_file", type=str, default="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b/model.safetensors", help="Qwen 模型文件路径")
    parser.add_argument("--qwen_tokenizer_dir", type=str, default="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b", help="Qwen Tokenizer 目录")
    parser.add_argument("--base_path", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/images", help="数据集根路径")
    parser.add_argument("--metadata_path", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl", help="UnifiedDataset 的 metadata 路径")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/images", help="ImageTextPairDataset 的目录路径")
    parser.add_argument("--recursive", action="store_true", help="是否递归加载子文件夹")
    parser.add_argument("--output_dir", type=str, default="models/Complextro/edit", help="训练模型输出目录")
    parser.add_argument("--batch_size", type=int, default=2, help="训练 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--prediction_type", type=str, default="flow", help="Training target type: flow / jit_xpred / bridge_xpred")
    parser.add_argument("--condition_drop_prob", type=float, default=0.0, help="Classifier-free condition dropout probability used during training")
    parser.add_argument("--jit_p_mean", type=float, default=-0.8, help="JiT x-pred logit-normal P_mean")
    parser.add_argument("--jit_p_std", type=float, default=0.8, help="JiT x-pred logit-normal P_std")
    parser.add_argument("--jit_noise_scale", type=float, default=1.0, help="JiT x-pred noise scale")
    parser.add_argument("--jit_t_eps", type=float, default=5e-2, help="JiT x-pred denominator clamp epsilon")
    parser.add_argument("--jit_sampling_method", type=str, default="heun", help="JiT x-pred sampling method: euler / heun")
    parser.add_argument("--jit_cfg_interval_min", type=float, default=0.0, help="JiT x-pred CFG interval minimum t")
    parser.add_argument("--jit_cfg_interval_max", type=float, default=1.0, help="JiT x-pred CFG interval maximum t")
    parser.add_argument("--jit_loss_weighting", type=str, default="balanced", help="JiT x-pred loss weighting: velocity / balanced / x_pred")
    parser.add_argument("--freq_loss_enabled", action="store_true", help="Enable frequency-aware residual loss. DeCo 推荐开启；这里默认关闭以保持原行为")
    parser.add_argument("--freq_loss_weight", type=float, default=0.0, help="Multiplier for the frequency-aware residual loss term. DeCo 原值: 1.0")
    parser.add_argument("--freq_loss_mode", type=str, default="dct", help="Frequency loss mode. Currently supports: dct. DeCo 原值: dct")
    parser.add_argument("--freq_loss_block_size", type=int, default=8, help="Block size used by the DCT frequency-aware residual loss. DeCo 原值: 8")
    parser.add_argument("--freq_loss_profile", type=str, default="jpeg", help="Frequency weighting profile: jpeg / linear / uniform. DeCo 风格: jpeg")
    parser.add_argument("--freq_loss_quality", type=int, default=85, help="JPEG quality proxy used by the DeCo-style frequency weighting. DeCo 原值: 85")
    parser.add_argument("--freq_loss_jpeg_mode", type=str, default="inv_gamma", help="JPEG weight mapping: inv / inv_gamma. DeCo 原值: inv_gamma")
    parser.add_argument("--freq_loss_gamma", type=float, default=1.0, help="Gamma used by inv_gamma JPEG weighting. DeCo 原值: 1.0")
    parser.add_argument("--freq_loss_color_space", type=str, default="rgb", help="Color space before block DCT: rgb / ycbcr. DeCo 推荐: ycbcr")
    parser.add_argument("--freq_loss_weight_floor", type=float, default=0.1, help="Minimum normalized frequency weight before LF/HF scaling")
    parser.add_argument("--freq_loss_hf_scale", type=float, default=0.25, help="Base multiplier applied to high-frequency residual energy")
    parser.add_argument("--freq_loss_lf_scale", type=float, default=1.0, help="Base multiplier applied to low-frequency residual energy")
    parser.add_argument("--freq_loss_t_adaptive", action="store_true", help="Make high-frequency weighting depend on normalized timestep t")
    parser.add_argument("--freq_loss_t_min_hf_scale", type=float, default=0.25, help="High-frequency scale used near t=0 when t-adaptive weighting is enabled")
    parser.add_argument("--freq_loss_t_max_hf_scale", type=float, default=1.0, help="High-frequency scale used near t=1 when t-adaptive weighting is enabled")
    parser.add_argument("--freq_loss_t_gamma", type=float, default=1.0, help="Exponent for timestep-adaptive high-frequency scaling")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="学习率调度器名称，例如 constant / warmup_stable_decay")
    parser.add_argument("--lr_warmup_steps", type=float, default=0, help="warmup 步数；传小数时按总步数比例计算")
    parser.add_argument("--lr_decay_steps", type=float, default=0, help="decay 步数；传小数时按总步数比例计算")
    parser.add_argument("--lr_scheduler_min_lr_ratio", type=float, default=None, help="最小学习率相对初始学习率的比例")
    parser.add_argument("--num_workers", type=int, default=4, help="数据读取线程数")
    parser.add_argument("--save_epochs", type=int, default=None, help="每多少个 epoch 保存一次模型")
    parser.add_argument("--save_steps", type=int, default=2000, help="每多少个 steps 保存一次模型")
    parser.add_argument("--num_epochs", type=int, default=999999999, help="总训练 epoch 数")
    parser.add_argument("--train_resolution", type=int, nargs=2, default=[256, 256], help="训练基础分辨率")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="分桶最大分辨率")
    parser.add_argument("--prebucket_index_path", type=str, default=None, help="预分桶索引 jsonl 路径")
    parser.add_argument("--jsonl_index_path", type=str, default=None, help="jsonl metadata 行偏移索引路径")
    
    # Model config parameters
    parser.add_argument("--num_layers", type=int, default=10, help="DiT 层数")
    parser.add_argument("--hidden_size", type=int, default=2304, help="DiT hidden size")
    parser.add_argument("--num_attention_heads", type=int, default=24, help="DiT attention heads")
    parser.add_argument("--attention_head_dim", type=int, default=96, help="DiT attention head dim")
    parser.add_argument("--enable_vram_offload", action="store_true", help="Enable VRAM offload for frozen auxiliary models")
    parser.add_argument("--offload_device", type=str, default="cpu", help="Offload device when VRAM offload is enabled")

    # 使用 WSD 调度器示例:
    # accelerate launch train_complextro_ddp.py \
    #   --lr_scheduler warmup_stable_decay \
    #   --lr_warmup_steps 1000 \
    #   --lr_decay_steps 10000 \
    #   --lr_scheduler_min_lr_ratio 0.1
    #
    # ps:
    # - lr_warmup_steps / lr_decay_steps 可以是整数或比例.
    # - 如果传入浮点数，它将被解释为总训练步数的比例.
    # - 稳定步数会被自动计算为总训练步数减去 warmup 和 decay 步数. 例如:
    #   max_train_steps - lr_warmup_steps - lr_decay_steps
    # - Unknown CLI args are forwarded into `args` as attributes, so you can also pass
    #   launch_training_task / runner fields directly with `--xxx value`.
    # - JiT-style pixel x-pred example:
    #   accelerate launch train_complextro_ddp.py \
    #     --vae_type pixel:32 \
    #     --prediction_type jit_xpred \
    #     --jit_sampling_method heun \
    #     --jit_cfg_interval_min 0.1 \
    #     --jit_cfg_interval_max 1.0

    def _parse_cli_value(value):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            if lowered == "none":
                return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def _apply_unknown_args(namespace, unknown_args):
        i = 0
        while i < len(unknown_args):
            token = unknown_args[i]
            if not token.startswith("--"):
                raise ValueError(f"Unexpected positional argument: {token}")

            key = token[2:]
            if "=" in key:
                key, raw_value = key.split("=", 1)
                setattr(namespace, key.replace("-", "_"), _parse_cli_value(raw_value))
                i += 1
                continue

            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                raw_value = unknown_args[i + 1]
                setattr(namespace, key.replace("-", "_"), _parse_cli_value(raw_value))
                i += 2
                continue

            if key.startswith("no-"):
                setattr(namespace, key[3:].replace("-", "_"), False)
            else:
                setattr(namespace, key.replace("-", "_"), True)
            i += 1

    args, unknown_args = parser.parse_known_args()
    _apply_unknown_args(args, unknown_args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    use_image_text_pairs = args.use_image_text_pairs
    train_omni = args.train_omni
    vae_type = args.vae_type
    use_alpha_layer_vae = args.use_alpha_layer_vae
    siglip_model_file = args.siglip_model_file
    data_vae_spec = get_complextro_vae_spec(
        vae_type=vae_type,
        vae_file=args.vae_file,
        use_alpha_layer_vae=use_alpha_layer_vae,
    )
    image_division_factor = int(data_vae_spec["latent_downsample_factor"]) * int(data_vae_spec["latent_patch_size"])
    enable_vram_offload = args.enable_vram_offload
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": args.offload_device,
        "onload_dtype": torch.bfloat16,
        "onload_device": accelerator.device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": accelerator.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": accelerator.device,
    }

    def build_optional_edit_latent_operator(base_path, max_pixels):
        resize_op = ImageCropAndResize(
            height=None,
            width=None,
            max_pixels=max_pixels,
            height_division_factor=image_division_factor,
            width_division_factor=image_division_factor,
        )
        to_abs = ToAbsolutePath(base_path)
        load_image = LoadImage()

        def operator(value):
            if value is None or (isinstance(value, float) and value != value):
                return None
            if isinstance(value, list):
                return [operator(item) for item in value]
            if isinstance(value, str):
                marker = value.strip()
                if marker in ("", "0", "1"):
                    return marker if marker != "" else "0"
                path = value if os.path.isabs(value) else to_abs(value)
                return resize_op(load_image(path))
            return value

        return operator

    complextro_model_config = {
        "num_layers": args.num_layers,
        "num_refiner_layers": 0,
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "attention_head_dim": args.attention_head_dim,
        "rope_axes_dim": [32, 32, 32],
        "enable_tread_routing": False,
        "tread_routes": [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 2,
                "end_layer_idx": 4,
            }
        ],
        "use_text_modulation": False,
    }

    train_resolution = tuple(args.train_resolution)
    max_bucket_reso = args.max_bucket_reso

    prebucket_index_path = args.prebucket_index_path
    jsonl_index_path = args.jsonl_index_path

    if use_image_text_pairs:
        if args.recursive:
            import glob, json
            print(f"递归扫描数据目录: {args.data_dir}")
            # 查找所有图片及其对应的文本文件
            image_exts = ["jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"]
            pairs = []
            for ext in image_exts:
                images = glob.glob(os.path.join(args.data_dir, "**", f"*.{ext}"), recursive=True)
                for img_path in images:
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            prompt = f.read().strip()
                        pairs.append({"image": img_path, "prompt": prompt})
            print(f"找到 {len(pairs)} 对数据")
            # 将递归扫描到的结果通过 UnifiedDataset 加载
            # 这里我们通过构建一个内存列表来模拟 metadata 文件
            # UnifiedDataset 默认需要 metadata_path 为文件，所以我们写一个临时文件
            temp_metadata_path = os.path.join(args.output_dir, "temp_metadata.jsonl")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(temp_metadata_path, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
            dataset = UnifiedDataset(
                base_path=args.data_dir,
                metadata_path=temp_metadata_path,
                max_data_items=10000000,
                data_file_keys=("image",),
                enable_bucket=True,
                bucket_no_upscale=False,
                min_bucket_reso=256,
                max_bucket_reso=max_bucket_reso,
                bucket_reso_steps=64,
                bucket_data_key="image",
                bucket_base_reso=train_resolution,
                bucket_index_path=prebucket_index_path,
                main_data_operator=UnifiedDataset.default_image_operator(
                    base_path=args.data_dir,
                    height=None,
                    width=None,
                    max_pixels=max_bucket_reso * max_bucket_reso,
                    height_division_factor=image_division_factor,
                    width_division_factor=image_division_factor,
                ),
            )
        else:
            dataset = ImageTextPairDataset(
                data_dir=args.data_dir,
                max_pixels=max_bucket_reso * max_bucket_reso,
                height_division_factor=image_division_factor,
                width_division_factor=image_division_factor,
                enable_bucket=True,
                bucket_no_upscale=False,
                min_bucket_reso=256,
                max_bucket_reso=max_bucket_reso,
                bucket_reso_steps=64,
                bucket_base_reso=train_resolution,
                bucket_index_path=prebucket_index_path,
            )
    else:
        data_file_keys = (
            ("image", "edit_image", "edit_latent")
            if train_omni
            else ("image", "edit_image")
        )
        dataset = UnifiedDataset(
            base_path=args.base_path,
            metadata_path=args.metadata_path,
            max_data_items=10000000,
            data_file_keys=data_file_keys,
            special_operator_map={
                "edit_latent": build_optional_edit_latent_operator(
                    base_path=args.base_path,
                    max_pixels=max_bucket_reso * max_bucket_reso,
                ),
            },
            enable_bucket=True,
            bucket_no_upscale=False,
            min_bucket_reso=256,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_steps=64,
            bucket_data_key="image",
            bucket_base_reso=train_resolution,
            bucket_index_path=prebucket_index_path,
            jsonl_index_path=jsonl_index_path,
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=args.base_path,
                height=None,
                width=None,
                max_pixels=max_bucket_reso * max_bucket_reso,
                height_division_factor=image_division_factor,
                width_division_factor=image_division_factor,
            ),
        )

    model = ComplextroTrainingModule(
        device=accelerator.device,
        qwen_model_file=args.qwen_model_file,
        vae_file=args.vae_file,
        qwen_tokenizer_dir=args.qwen_tokenizer_dir,
        qwen_model_size="2B",
        siglip_model_file=siglip_model_file,
        train_omni=train_omni,
        vae_type=vae_type,
        use_alpha_layer_vae=use_alpha_layer_vae,
        complextro_model_config=complextro_model_config,
        prediction_type=args.prediction_type,
        condition_drop_prob=args.condition_drop_prob,
        jit_p_mean=args.jit_p_mean,
        jit_p_std=args.jit_p_std,
        jit_noise_scale=args.jit_noise_scale,
        jit_t_eps=args.jit_t_eps,
        jit_sampling_method=args.jit_sampling_method,
        jit_cfg_interval_min=args.jit_cfg_interval_min,
        jit_cfg_interval_max=args.jit_cfg_interval_max,
        jit_loss_weighting=args.jit_loss_weighting,
        freq_loss_enabled=args.freq_loss_enabled,
        freq_loss_weight=args.freq_loss_weight,
        freq_loss_mode=args.freq_loss_mode,
        freq_loss_block_size=args.freq_loss_block_size,
        freq_loss_profile=args.freq_loss_profile,
        freq_loss_quality=args.freq_loss_quality,
        freq_loss_jpeg_mode=args.freq_loss_jpeg_mode,
        freq_loss_gamma=args.freq_loss_gamma,
        freq_loss_color_space=args.freq_loss_color_space,
        freq_loss_weight_floor=args.freq_loss_weight_floor,
        freq_loss_hf_scale=args.freq_loss_hf_scale,
        freq_loss_lf_scale=args.freq_loss_lf_scale,
        freq_loss_t_adaptive=args.freq_loss_t_adaptive,
        freq_loss_t_min_hf_scale=args.freq_loss_t_min_hf_scale,
        freq_loss_t_max_hf_scale=args.freq_loss_t_max_hf_scale,
        freq_loss_t_gamma=args.freq_loss_t_gamma,
        enable_vram_offload=enable_vram_offload,
        vram_config=vram_config,
    )
    model_logger = ModelLogger(
        args.output_dir,
        remove_prefix_in_ckpt="pipe.dit.",
    )

    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer_type="adamw",
        mup_scale=False,
        mup_base_dim=1.0,
        mup_dim=complextro_model_config.get("hidden_size", None),
        max_grad_norm=1.0,
        log_layer_grad_norms=True,
        num_workers=args.num_workers,
        save_epochs=args.save_epochs,
        save_steps=args.save_steps,
        num_epochs=args.num_epochs,
    )
