import argparse
import os
import importlib
import torch, accelerate
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
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/Qwen3_5_2b_claude_heretic/model.safetensors",
        vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/Qwen3_5_2b_claude_heretic",
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
        if self.pipe.prediction_type in ("jit_xpred", "bridge_xpred") and int(self.vae_spec["latent_downsample_factor"]) != 1:
            raise NotImplementedError(
                "JiT-style pixel-space training requires pixel-space Complextro "
                "(use vae_type='pixel', 'pixel_logit', 'pixel_norm', or their ':<patch_size>' variants)."
            )

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
    """
    Complextro 训练模式说明

    1) 普通训练模式（train_omni = False）
            - 必需字段：
                - image: 图像路径或路径列表（由 UnifiedDataset + default_image_operator 读取为 PIL）
                - prompt: 字符串或字符串列表
            - 可选字段：
                - neg_prompt: 负面提示词（支持字符串或字符串列表；列名固定为 neg_prompt）
            - 训练目标：单图生成（不使用编辑条件图）

    1.1) Prompt 扩展语法（正面/负面都支持）
            - `<prompt start>` 前为 system prompt，后为 user 正文。
                例："你是风格助手<prompt start>保留主体，做二次元化"
            - `<break>` 将正文拆成多段，分别编码后拼接。
                例："系统提示<prompt start>主体描述<break>风格描述<break>背景描述"
            - 若字符串中不含 `<prompt start>`，则按“无 system prompt”处理。

    2) Omni/编辑训练模式（train_omni = True）
            - 在普通字段基础上可选增加：
                - edit_image:
                    a) List[str] / List[PIL]，表示所有样本共享同一组条件图
                    b) List[List[str]] / List[List[PIL]]，每个样本独立条件图组
                    c) 当 batch>1 且长度等于 batch_size 的平铺 List，会自动视作每样本1张条件图
                - edit_latent:
                    a) 用于声明哪些 edit_image 槽位真正插入 condition latent，也可以提供独立于 edit_image 的另一套 latent 参考图
                    b) 固定写法 "0" 表示该槽位改为等长 pad token，底层会复用同位 edit_image 做 VAE 编码以保持长度一致
                    c) 固定写法 "1" 表示该槽位直接复用同位 edit_image 作为 cond latent 输入
                    d) 若写真实路径，则该路径对应的图会被单独读取并编码为 cond latent
                    e) 若长度少于 edit_image，会在尾部自动补齐为 pad；长度大于 edit_image 会报错
                - image_noise_mask:
                    a) List[int]（共享掩码）
                    b) List[List[int]]（每样本掩码）
            - 若未提供 image_noise_mask，会自动构建为 [0, ..., 0, 1]
                （条件图 token 为 0，目标图 token 为 1）
            - 注意：edit_latent 是“混合列”，既可能是路径，也可能是 "0" / "1" 占位；因此需要用 special_operator_map 单独处理
            - SigLIP 为可选分支：`siglip_model_file=""` 时默认不加载，也不会启用 SigLIP 图像特征。

    3) Batch 与分桶兼容性
            - 脚本中 batch_size 会由 prompt/image 两侧自动取最大值，避免单边长度触发错配。
            - UnifiedDataset 在 enable_bucket=True 时，会按 bucket_data_key='image' 建桶，并对 data_file_keys
                中存在的图像字段都应用 main_data_operator；因此 edit_image 也可自动读取与缩放。
            - edit_latent 在本脚本里会通过 special_operator_map 读取：真实路径会被转成 PIL，"0" / "1" 会保留为占位字符串，再由 pipeline 解释。
            - 建议 bucket_base_reso、min/max_bucket_reso 与 height_division_factor/width_division_factor 保持 16 的倍数。

    4) 关于“非 Omni 也可读参考图”的说明（重点）
            - 当前 Complextro 的 PromptEmbedder 已接入 Qwen3.5 多模态聊天模板。
            - 这意味着在 omni_mode=False 时，仍可向文本编码器传 edit_image，作为视觉上下文参与 prompt 编码。
            - 但 omni_mode=False 不会启用 omni latent 路径；它只影响 TE 编码，不做条件 latent 拼接。

    5) 元数据示例（有参考图）
            - CSV 示例（单参考图，最稳妥）
                列名建议至少包含: image,prompt,edit_image
                如需负面提示词可额外添加: neg_prompt
                如需控制哪些参考图进入 condition latent，可额外添加: edit_latent

                image,prompt,edit_image
                train/target_0001.jpg,"保持人物主体，改成二次元手办风格",refs/ref_0001.jpg
                train/target_0002.jpg,"保持构图，改成像素风",refs/ref_0002.jpg

                image,prompt,neg_prompt,edit_image
                train/target_0003.jpg,"你是写实风格助手<prompt start>保持主体<break>增强材质细节","你是约束助手<prompt start>低质量，模糊，水印",refs/ref_0003.jpg

                image,prompt,edit_image,edit_latent
                train/target_0004.jpg,"保留主体，增强金属质感",refs/ref_0004.jpg,0

                - 若需要在单条样本中写多张 edit_image / edit_latent，建议使用 JSON 或 JSONL metadata。

            - JSON 示例（支持单图或多图参考）
                [
                    {
                        "image": "train/target_0001.jpg",
                        "prompt": "保持人物主体，改成二次元手办风格",
                        "neg_prompt": "低质量，模糊，水印",
                        "edit_image": "refs/ref_0001.jpg",
                        "edit_latent": "1"
                    },
                    {
                        "image": "train/target_0002.jpg",
                        "prompt": "保持主体配色，增强金属质感",
                        "neg_prompt": "你是约束助手<prompt start>低质量<break>噪点<break>过饱和",
                        "edit_image": ["refs/ref_0002a.jpg", "refs/ref_0002b.jpg", "refs/ref_0002c.jpg"],
                        "edit_latent": ["1", "0", "refs/ref_latent_c.jpg"]
                    }
                ]

            - 路径规则：
                metadata 里的相对路径会拼接到 UnifiedDataset 的 base_path；
                如果你写绝对路径，也可以直接读取。

    6) 预分桶索引（大规模数据集优化，推荐 jsonl）
            - UnifiedDataset / ImageTextPairDataset 均新增可选参数 bucket_index_path，用于加载预先计算好的分桶索引，避免在训练启动阶段逐图 Image.open。
            - 适用场景：上千万级样本时，建议离线先跑一遍分桶脚本，生成 jsonl 索引文件，然后训练时直接读取索引。

            - UnifiedDataset 模式（有 metadata）：
                * bucket_index_path 索引文件为 jsonl，每行格式接受以下任意一种键名：
                    {"data_id": 0, "bucket": [1024, 576]}
                    {"idx": 0, "reso": [1024, 576]}
                * data_id / idx: 0-based 下标，对应 metadata 文件加载进来后的 self.data 列表下标；
                  即第 N 行 metadata（jsonl/CSV/json）在内存中是 data[N]，索引里就写 data_id=N。
                * bucket / reso: [width, height]，已经对齐 bucket_reso_steps，例如步长 64 时，宽高必须是 64 的倍数，且在 [min_bucket_reso, max_bucket_reso] 范围内。
                * 注意：索引文件中的行顺序可以、也推荐按桶分组，例如把同一个桶的 data_id 连续写在一起，方便后续你自己做流式扫描；
                  UnifiedDataset 内部仍会根据 batch_size 做 shuffle，所以不会依赖行顺序保证随机性。

            - ImageTextPairDataset 模式（纯目录 image+txt）：
                * bucket_index_path 同样是 jsonl，每行格式：
                    {"data_id": 0, "bucket": [1024, 576]}
                * 这里 data_id 是 _scan_pairs() 之后的 self.pairs 下标，即按文件名排序后的第 N 个 pair 对应 data_id=N。
                * bucket / reso 含义与 UnifiedDataset 完全相同。

            - 训练脚本中使用方式示例：
                * UnifiedDataset：
                    dataset = UnifiedDataset(
                        base_path=..., metadata_path=..., ...,
                        enable_bucket=True,
                        bucket_index_path="/path/to/prebucket_index.jsonl",
                    )
                * ImageTextPairDataset：
                    dataset = ImageTextPairDataset(
                        data_dir=..., ...,
                        enable_bucket=True,
                        bucket_index_path="/path/to/prebucket_index.jsonl",
                    )

            - 空间开销：
                * 仅保存 data_id 和 (w, h)，大约几十字节/样本；
                * 对于 1e7 级别样本，索引体积一般在数百 MB 量级，相对原始图像数据可以忽略。

    7) 预分桶脚本 & jsonl 行偏移索引脚本使用方式

            - 预分桶脚本（tools/build_bucket_index.py）

                a) UnifiedDataset 模式（metadata.jsonl/json/csv）：

                    python tools/build_bucket_index.py unified \
                        --base_path /root/autodl-tmp/DiffSynth-Studio/edit/images \
                        --metadata_path /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl \
                        --output /root/autodl-tmp/DiffSynth-Studio/edit/prebucket_index.jsonl \
                        --bucket_data_key image \
                        --max_bucket_reso 1024 \
                        --min_bucket_reso 256 \
                        --bucket_reso_steps 64 \
                        --bucket_base_reso 256 256

                    - base_path: UnifiedDataset 的 base_path（用于补全相对路径）。
                    - metadata_path: 你的 metadata 文件路径（支持 json/jsonl/csv）。
                    - output: 预分桶索引 jsonl 输出路径（传给 bucket_index_path）。
                    - 其他 bucket 参数需与训练脚本保持一致。

                b) ImageTextPairDataset 模式（纯目录 image+txt）：

                    python tools/build_bucket_index.py pairs \
                        --data_dir /root/autodl-tmp/DiffSynth-Studio/edit/images \
                        --output /root/autodl-tmp/DiffSynth-Studio/edit/prebucket_pairs_index.jsonl \
                        --max_bucket_reso 1024 \
                        --min_bucket_reso 256 \
                        --bucket_reso_steps 64 \
                        --bucket_base_reso 256 256

                    - data_dir: ImageTextPairDataset 的 data_dir。
                    - output: 预分桶索引 jsonl 输出路径（传给 bucket_index_path）。

            - jsonl 行偏移索引脚本（tools/build_jsonl_offset_index.py）

                适用于 UnifiedDataset + metadata 为 jsonl 时，给 metadata 做“行号 -> 文件偏移量”的索引，
                从而配合 jsonl_index_path 实现流式按行加载 metadata。

                使用示例：

                    python tools/build_jsonl_offset_index.py \
                        --metadata_path /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl \
                        --output /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl.offsets

                - metadata_path: 你的 jsonl metadata 文件路径。
                - output: 每行一个整数 offset 的索引文件路径（传给 UnifiedDataset.jsonl_index_path）。

                生成后在训练脚本里：

                    dataset = UnifiedDataset(
                        base_path=..., metadata_path=..., ...,
                        enable_bucket=True,
                        bucket_index_path="/root/autodl-tmp/DiffSynth-Studio/edit/prebucket_index.jsonl",
                        jsonl_index_path="/root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl.offsets",
                    )

                这样：
                    - 分桶信息来自 prebucket_index.jsonl（不再在训练时扫所有图片）；
                    - metadata 的内容通过行偏移索引按需读取，而不需要整份 jsonl 常驻内存。
    """

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1)
    use_image_text_pairs = False  # True: 使用 ImageTextPairDataset（图片+txt目录），False: 使用 UnifiedDataset（metadata文件）
    train_omni = True
    vae_type = "pixel:16"
    prediction_type = "flow"
    condition_drop_prob = 0.0
    # Frequency-aware loss knobs.
    # 推荐 preset 1: flow + pixel，最接近 DeCo 原论文
    #   prediction_type = "flow"
    #   freq_loss_enabled = True
    #   freq_loss_weight = 1.0
    #   freq_loss_mode = "dct"
    #   freq_loss_block_size = 8
    #   freq_loss_profile = "jpeg"
    #   freq_loss_quality = 85
    #   freq_loss_jpeg_mode = "inv_gamma"
    #   freq_loss_gamma = 1.0
    #   freq_loss_color_space = "ycbcr"
    #   freq_loss_t_adaptive = False
    #
    # 推荐 preset 2: jit_xpred + pixel，工程上更适合的扩展配置
    #   prediction_type = "jit_xpred"
    #   freq_loss_enabled = True
    #   freq_loss_weight = 1.0
    #   freq_loss_mode = "dct"
    #   freq_loss_block_size = 8
    #   freq_loss_profile = "jpeg"
    #   freq_loss_quality = 85
    #   freq_loss_jpeg_mode = "inv_gamma"
    #   freq_loss_gamma = 1.0
    #   freq_loss_color_space = "ycbcr"
    #   freq_loss_t_adaptive = True
    #   freq_loss_t_min_hf_scale = 0.25
    #   freq_loss_t_max_hf_scale = 1.0
    #   freq_loss_t_gamma = 1.0
    #
    # 当前示例默认使用 flow preset。
    freq_loss_enabled = True
    freq_loss_weight = 1.0
    freq_loss_mode = "dct"
    freq_loss_block_size = 8
    freq_loss_profile = "jpeg"
    freq_loss_quality = 85
    freq_loss_jpeg_mode = "inv_gamma"
    freq_loss_gamma = 1.0
    freq_loss_color_space = "ycbcr"
    freq_loss_weight_floor = 0.1
    freq_loss_hf_scale = 0.25
    freq_loss_lf_scale = 1.0
    freq_loss_t_adaptive = False
    freq_loss_t_min_hf_scale = 0.25
    freq_loss_t_max_hf_scale = 1.0
    freq_loss_t_gamma = 1.0
    jit_p_mean = -0.8
    jit_p_std = 0.8
    jit_noise_scale = 1.0
    jit_t_eps = 5e-2
    use_alpha_layer_vae = False
    vae_file = "/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors"
    siglip_model_file = ""
    data_vae_spec = get_complextro_vae_spec(
        vae_type=vae_type,
        vae_file=vae_file,
        use_alpha_layer_vae=use_alpha_layer_vae,
    )
    image_division_factor = int(data_vae_spec["latent_downsample_factor"]) * int(data_vae_spec["latent_patch_size"])

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

    # 你也可以改成更深更宽(一般是直接改num_layers和num_refiner_layers)；需要满足 hidden_size = num_attention_heads * attention_head_dim
    # 默认是num_layers=60，num_refiner_layers=2的配置
    """
    complextro_model_config = {
        "num_layers": 8,
        "num_refiner_layers": 0,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "rope_axes_dim": [16, 56, 56],
        "enable_tread_routing": False,
        "tread_routes": [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 2,
                "end_layer_idx": 4,
            }
        ],
    }
    """
    # 2.25B, 加 "num_refiner_layers": 1 时为 2.54B
    complextro_model_config = {
        "num_layers": 10,
        "num_refiner_layers": 0,
        "hidden_size": 2304,
        "num_attention_heads": 24,
        "attention_head_dim": 96,
        "rope_axes_dim": [32, 32, 32],
        "enable_tread_routing": False,
        "tread_routes": [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 2,
                "end_layer_idx": 4,
            }
        ],
        "use_text_modulation": True,
    }

    train_resolution = (256, 256)
    max_bucket_reso = 1024
    enable_vram_offload = False
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": accelerator.device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": accelerator.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": accelerator.device,
    }

    # 可选：预分桶索引（jsonl），大规模数据集建议提前生成
    # - UnifiedDataset 模式：索引中的 data_id 对应 metadata 的行号（0-based）
    # - ImageTextPairDataset 模式：索引中的 data_id 对应按文件名排序后的 pair 下标（0-based）
    prebucket_index_path = None  # 例如："/root/autodl-tmp/DiffSynth-Studio/edit/prebucket_index.jsonl"

    # 可选：jsonl metadata 行偏移索引，配合 UnifiedDataset.jsonl_index_path 使用
    # - 仅在 metadata_path 为 .jsonl 时生效
    # - 通过 tools/build_jsonl_offset_index.py 生成
    jsonl_index_path = None  # 例如："/root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl.offsets"

    if use_image_text_pairs:
        dataset = ImageTextPairDataset(
            data_dir="/root/autodl-tmp/DiffSynth-Studio/data/images",
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
            base_path="/root/autodl-tmp/DiffSynth-Studio/data/images",
            metadata_path="/root/autodl-tmp/DiffSynth-Studio/data/metadata_merged.csv",
            max_data_items=10000000,
            data_file_keys=data_file_keys,
            special_operator_map={
                "edit_latent": build_optional_edit_latent_operator(
                    base_path="/root/autodl-tmp/DiffSynth-Studio/data/images",
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
                base_path="/root/autodl-tmp/DiffSynth-Studio/data/images",
                height=None,
                width=None,
                max_pixels=max_bucket_reso * max_bucket_reso,
                height_division_factor=image_division_factor,
                width_division_factor=image_division_factor,
            ),
        )

    model = ComplextroTrainingModule(
        device=accelerator.device,
        qwen_model_size="2B",
        siglip_model_file=siglip_model_file,
        train_omni=train_omni,
        vae_file=vae_file,
        vae_type=vae_type,
        use_alpha_layer_vae=use_alpha_layer_vae,
        complextro_model_config=complextro_model_config,
        prediction_type=prediction_type,
        condition_drop_prob=condition_drop_prob,
        freq_loss_enabled=freq_loss_enabled,
        freq_loss_weight=freq_loss_weight,
        freq_loss_mode=freq_loss_mode,
        freq_loss_block_size=freq_loss_block_size,
        freq_loss_profile=freq_loss_profile,
        freq_loss_quality=freq_loss_quality,
        freq_loss_jpeg_mode=freq_loss_jpeg_mode,
        freq_loss_gamma=freq_loss_gamma,
        freq_loss_color_space=freq_loss_color_space,
        freq_loss_weight_floor=freq_loss_weight_floor,
        freq_loss_hf_scale=freq_loss_hf_scale,
        freq_loss_lf_scale=freq_loss_lf_scale,
        freq_loss_t_adaptive=freq_loss_t_adaptive,
        freq_loss_t_min_hf_scale=freq_loss_t_min_hf_scale,
        freq_loss_t_max_hf_scale=freq_loss_t_max_hf_scale,
        freq_loss_t_gamma=freq_loss_t_gamma,
        jit_p_mean=jit_p_mean,
        jit_p_std=jit_p_std,
        jit_noise_scale=jit_noise_scale,
        jit_t_eps=jit_t_eps,
        jit_sampling_method="heun",
        jit_cfg_interval_min=0.0,
        jit_cfg_interval_max=1.0,
        enable_vram_offload=enable_vram_offload,
        vram_config=vram_config,
    )
    model_logger = ModelLogger(
        "models/Complextro/edit", # dit输出文件夹
        remove_prefix_in_ckpt="pipe.dit.",
    )

    args = argparse.Namespace(
        #lr_scheduler="warmup_stable_decay",
        lr_scheduler="constant",
        #lr_warmup_steps=0.01,
        #lr_decay_steps=0.1,
        #lr_scheduler_min_lr_ratio=0.1,
    )

    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
        batch_size=10,
        learning_rate=1e-4,
        optimizer_type="adamw",
        mup_scale=False,
        mup_base_dim=1.0,
        mup_dim=complextro_model_config.get("hidden_size", None),
        max_grad_norm=1.0,
        log_layer_grad_norms=True,
        num_workers=4,
        #save_steps=50000,
        save_epochs=1,
        num_epochs=99999999999,
        log_with="wandb",
        tracker_project_name="complextro-pretrain",
        tracker_run_name="complextro-256",
    )
