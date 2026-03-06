import os
import torch, accelerate
from typing import List, Optional, Any

from transformers import AutoProcessor
from diffsynth.core import UnifiedDataset, load_model
from diffsynth.core.data.operators import ImageCropAndResize, LoadImage, ToAbsolutePath
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    ModelLogger,
    launch_training_task,
)
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder428M
from diffsynth.pipelines.complextro import ComplextroPipeline


class ComplextroTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        device,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw/model.safetensors-00001-of-00001.safetensors",
        flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3_5_nsfw",
        siglip_model_file="",
        #complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e43-s19221.safetensors",
        complextro_dit_file="",
        train_omni: bool = False,
        use_alpha_layer_vae: bool = False,
        complextro_model_config: Optional[dict] = None,
    ):
        super().__init__()
        self.train_omni = train_omni
        self.complextro_model_config = {} if complextro_model_config is None else dict(complextro_model_config)
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

        self.pipe.text_encoder = load_model(
            QwenImageTextEncoder,
            qwen_model_file,
            config={"model_type": "qwen3_5", "model_size": "0.8B"},
            torch_dtype=torch.bfloat16,
            device=device,
            state_dict_converter=QwenImageTextEncoderStateDictConverter,
        )
        self.pipe.vae = load_model(
            Flux2VAE,
            flux2_vae_file,
            config={"use_alpha_layer": use_alpha_layer_vae},
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.pipe.processor = AutoProcessor.from_pretrained(qwen_tokenizer_dir)
        self.pipe.tokenizer = self.pipe.processor.tokenizer

        if siglip_enabled:
            self.pipe.image_encoder = load_model(
                Siglip2ImageEncoder428M,
                siglip_model_file,
                torch_dtype=torch.bfloat16,
                device=device,
            )

        self.pipe.vram_management_enabled = self.pipe.check_vram_management_state()
        
        if complextro_dit_file:
            self.pipe.dit = load_model(
                ComplextroImageDiT,
                complextro_dit_file,
                config=self.complextro_model_config,
                torch_dtype=torch.bfloat16,
                device=device,
            )
        else:
            self.pipe.dit = ComplextroImageDiT(**self.complextro_model_config).to(dtype=torch.bfloat16, device=device)

        text_config = getattr(self.pipe.text_encoder.model.config, "text_config", self.pipe.text_encoder.model.config)
        text_hidden_size = int(text_config.hidden_size)
        dit_text_dim = int(self.pipe.dit.txt_in.in_features)
        if text_hidden_size != dit_text_dim:
            raise ValueError(
                f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
                f"Please align QwenImageTextEncoder(model_type='qwen3_5', model_size='0.8B') and ComplextroImageDiT(text_embed_dim=...)."
            )

        self.pipe.freeze_except(["dit"])
        self.pipe.scheduler.set_timesteps(1000, training=True)

    def _normalize_edit_images(self, edit_value: Any, batch_size: int) -> Optional[List[List[Any]]]:
        if isinstance(edit_value, float) and edit_value != edit_value:
            return None
        if edit_value is None:
            return None
        if not isinstance(edit_value, list):
            return [[edit_value] for _ in range(batch_size)]
        if len(edit_value) == 0:
            return None
        if isinstance(edit_value[0], list):
            if len(edit_value) == batch_size:
                return edit_value
            if len(edit_value) == 1:
                return [edit_value[0] for _ in range(batch_size)]
            return edit_value[:batch_size]

        if batch_size > 1 and len(edit_value) == batch_size:
            return [[v] for v in edit_value]
        return [edit_value for _ in range(batch_size)]

    def _normalize_edit_latent_inputs(
        self,
        edit_latent_value: Any,
        batch_size: int,
    ) -> Optional[List[List[Any]]]:
        if edit_latent_value is None or (isinstance(edit_latent_value, float) and edit_latent_value != edit_latent_value):
            return None

        if not isinstance(edit_latent_value, list):
            return [[edit_latent_value] for _ in range(batch_size)]
        if len(edit_latent_value) == 0:
            return None
        if isinstance(edit_latent_value[0], list):
            if len(edit_latent_value) == batch_size:
                return edit_latent_value
            if len(edit_latent_value) == 1:
                return [edit_latent_value[0] for _ in range(batch_size)]
            latent_num = len(edit_latent_value)
            return [edit_latent_value[i % latent_num] for i in range(batch_size)]

        if batch_size > 1 and len(edit_latent_value) == batch_size:
            return [[v] for v in edit_latent_value]
        return [edit_latent_value for _ in range(batch_size)]

    def _normalize_omni_noise_mask(self, noise_mask_value: Any, condition_groups: Optional[List[List[Any]]], batch_size: int):
        if condition_groups is None:
            return None

        def fit_len(mask, cond_num):
            if mask is None:
                return [0] * cond_num + [1]
            local = [int(v) for v in mask]
            need_len = cond_num + 1
            if len(local) < need_len:
                tail = local[-1] if len(local) > 0 else 1
                local = local + [tail] * (need_len - len(local))
            return local[:need_len]

        if noise_mask_value is None:
            return [[0] * len(group) + [1] for group in condition_groups]
        if not isinstance(noise_mask_value, list):
            return [[int(noise_mask_value)] * len(group) + [1] for group in condition_groups]
        if len(noise_mask_value) == 0:
            return [[0] * len(group) + [1] for group in condition_groups]
        if isinstance(noise_mask_value[0], list):
            if len(noise_mask_value) == batch_size:
                return [fit_len(noise_mask_value[i], len(condition_groups[i])) for i in range(batch_size)]
            if len(noise_mask_value) == 1:
                return [fit_len(noise_mask_value[0], len(condition_groups[i])) for i in range(batch_size)]
            mask_num = len(noise_mask_value)
            return [fit_len(noise_mask_value[i % mask_num], len(condition_groups[i])) for i in range(batch_size)]
        return [fit_len(noise_mask_value, len(condition_groups[i])) for i in range(batch_size)]

    def _infer_batch_size(self, prompt_value: Any, image_value: Any) -> int:
        prompt_batch = 1 if isinstance(prompt_value, str) else len(prompt_value)
        image_batch = len(image_value) if isinstance(image_value, (list, tuple)) else 1
        return max(prompt_batch, image_batch)

    def _infer_image_hw(self, image_value: Any):
        first_image = image_value[0] if isinstance(image_value, (list, tuple)) else image_value
        if hasattr(first_image, "size"):
            width, height = first_image.size
            return int(height), int(width)
        if hasattr(first_image, "shape") and len(first_image.shape) >= 2:
            return int(first_image.shape[-2]), int(first_image.shape[-1])
        raise ValueError("Cannot infer image height/width from data['image']")

    def _normalize_negative_prompt(self, prompt_value: Any, neg_prompt_value: Any):
        if isinstance(prompt_value, str):
            if isinstance(neg_prompt_value, list):
                return neg_prompt_value[0] if len(neg_prompt_value) > 0 else ""
            return "" if neg_prompt_value is None else neg_prompt_value

        prompt_batch = len(prompt_value)
        if neg_prompt_value is None:
            return [""] * prompt_batch
        if isinstance(neg_prompt_value, str):
            return [neg_prompt_value] * prompt_batch
        if not isinstance(neg_prompt_value, list):
            return [str(neg_prompt_value)] * prompt_batch
        if len(neg_prompt_value) == prompt_batch:
            return neg_prompt_value
        if len(neg_prompt_value) == 1:
            return [neg_prompt_value[0]] * prompt_batch
        if len(neg_prompt_value) == 0:
            return [""] * prompt_batch
        return [neg_prompt_value[i % len(neg_prompt_value)] for i in range(prompt_batch)]

    def forward(self, data):
        prompt_value = data["prompt"]
        neg_prompt_value = self._normalize_negative_prompt(prompt_value, data.get("neg_prompt", ""))
        image_value = data["image"]
        batch_size = self._infer_batch_size(prompt_value, image_value)
        height, width = self._infer_image_hw(image_value)

        edit_images = self._normalize_edit_images(data.get("edit_image", None), batch_size)
        edit_latent_inputs = self._normalize_edit_latent_inputs(data.get("edit_latent", None), batch_size)

        omni_condition_groups = edit_images if edit_images is not None else edit_latent_inputs
        omni_noise_mask = None
        if self.train_omni and omni_condition_groups is not None:
            omni_noise_mask = self._normalize_omni_noise_mask(
                data.get("image_noise_mask", None),
                omni_condition_groups,
                batch_size,
            )

        inputs_posi = {"prompt": prompt_value}
        inputs_nega = {"negative_prompt": neg_prompt_value}
        inputs_shared = {
            "input_image": image_value,
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "cfg_scale": 1,
            "edit_image": edit_images,
            "edit_latent": edit_latent_inputs,
            "omni_mode": self.train_omni and omni_condition_groups is not None,
            "image_noise_mask": omni_noise_mask,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
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
    """

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1)
    train_omni = False
    use_alpha_layer_vae = False
    siglip_model_file = ""

    def build_optional_edit_latent_operator(base_path, max_pixels):
        resize_op = ImageCropAndResize(
            height=None,
            width=None,
            max_pixels=max_pixels,
            height_division_factor=16,
            width_division_factor=16,
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

    # 1.05 B 配置
    # 你也可以改成更深更宽(一般是直接改num_layers和num_refiner_layers)；需要满足 hidden_size = num_attention_heads * attention_head_dim
    # 默认是num_layers=60，num_refiner_layers=2的配置
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

    data_file_keys = (
        ("image", "edit_image", "edit_latent")
        if train_omni
        else ("image", "edit_image")
    )

    train_resolution = (256,256)
    max_bucket_reso = 1024
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
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path="/root/autodl-tmp/DiffSynth-Studio/data/images",
            height=None,
            width=None,
            max_pixels=max_bucket_reso * max_bucket_reso,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )

    model = ComplextroTrainingModule(
        device=accelerator.device,
        siglip_model_file=siglip_model_file,
        train_omni=train_omni,
        use_alpha_layer_vae=use_alpha_layer_vae,
        complextro_model_config=complextro_model_config,
    )
    model_logger = ModelLogger(
        "models/Complextro/v0", # dit输出文件夹
        remove_prefix_in_ckpt="pipe.dit.",
    )

    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        batch_size=2,
        learning_rate=1e-4,
        optimizer_type="adamw",
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
        mup_scale=False,
        mup_base_dim=1.0,
        mup_dim=complextro_model_config.get("hidden_size", None),
        max_grad_norm=1.0,
        num_workers=4,
        #save_steps=50000,
        save_epochs=1,
        num_epochs=99999999999,
    )
