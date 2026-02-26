import torch, accelerate
from typing import List, Optional, Any

from transformers import AutoTokenizer
from diffsynth.core import UnifiedDataset, load_model
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    ModelLogger,
    launch_training_task,
)
from diffsynth.models.z_image_text_encoder import ZImageTextEncoder
from diffsynth.utils.state_dict_converters.z_image_text_encoder import ZImageTextEncoderStateDictConverter
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.complextro_dit import ComplextroImageDiT
from diffsynth.pipelines.complextro import ComplextroPipeline


class ComplextroTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        device,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3/model.safetensors",
        flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3",
        complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e3-s10059.safetensors",
        train_omni: bool = False,
        complextro_model_config: Optional[dict] = None,
    ):
        super().__init__()
        self.train_omni = train_omni
        self.complextro_model_config = {} if complextro_model_config is None else dict(complextro_model_config)

        self.pipe = ComplextroPipeline(device=device, torch_dtype=torch.bfloat16)

        self.pipe.text_encoder = load_model(
            ZImageTextEncoder,
            qwen_model_file,
            config={"model_size": "0.6B"},
            torch_dtype=torch.bfloat16,
            device=device,
            state_dict_converter=ZImageTextEncoderStateDictConverter,
        )
        self.pipe.vae = load_model(
            Flux2VAE,
            flux2_vae_file,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.pipe.tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir)

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

        text_hidden_size = int(self.pipe.text_encoder.model.config.hidden_size)
        dit_text_dim = int(self.pipe.dit.txt_in.in_features)
        if text_hidden_size != dit_text_dim:
            raise ValueError(
                f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
                f"Please align ZImageTextEncoder model_size and ComplextroImageDiT(text_embed_dim=...)."
            )

        self.pipe.freeze_except(["dit"])
        self.pipe.scheduler.set_timesteps(1000, training=True)

    def _normalize_omni_edit_images(self, edit_value: Any, batch_size: int) -> Optional[List[List[Any]]]:
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

    def _normalize_omni_noise_mask(self, noise_mask_value: Any, edit_images: Optional[List[List[Any]]], batch_size: int):
        if edit_images is None:
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
            return [[0] * len(group) + [1] for group in edit_images]
        if not isinstance(noise_mask_value, list):
            return [[int(noise_mask_value)] * len(group) + [1] for group in edit_images]
        if len(noise_mask_value) == 0:
            return [[0] * len(group) + [1] for group in edit_images]
        if isinstance(noise_mask_value[0], list):
            if len(noise_mask_value) == batch_size:
                return [fit_len(noise_mask_value[i], len(edit_images[i])) for i in range(batch_size)]
            if len(noise_mask_value) == 1:
                return [fit_len(noise_mask_value[0], len(edit_images[i])) for i in range(batch_size)]
            mask_num = len(noise_mask_value)
            return [fit_len(noise_mask_value[i % mask_num], len(edit_images[i])) for i in range(batch_size)]
        return [fit_len(noise_mask_value, len(edit_images[i])) for i in range(batch_size)]

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

    def forward(self, data):
        prompt_value = data["prompt"]
        image_value = data["image"]
        batch_size = self._infer_batch_size(prompt_value, image_value)
        height, width = self._infer_image_hw(image_value)

        omni_edit_images = None
        omni_noise_mask = None
        if self.train_omni:
            omni_edit_images = self._normalize_omni_edit_images(
                data.get("edit_image", data.get("condition_images", None)),
                batch_size,
            )
            omni_noise_mask = self._normalize_omni_noise_mask(
                data.get("image_noise_mask", None),
                omni_edit_images,
                batch_size,
            )

        inputs_posi = {"prompt": prompt_value}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": image_value,
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "cfg_scale": 1,
            "edit_image": omni_edit_images,
            "omni_mode": self.train_omni and omni_edit_images is not None,
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
            - 训练目标：单图生成（不使用编辑条件图）

    2) Omni/编辑训练模式（train_omni = True）
            - 在普通字段基础上可选增加：
                - edit_image 或 condition_images:
                    a) List[str] / List[PIL]，表示所有样本共享同一组条件图
                    b) List[List[str]] / List[List[PIL]]，每个样本独立条件图组
                    c) 当 batch>1 且长度等于 batch_size 的平铺 List，会自动视作每样本1张条件图
                - image_noise_mask:
                    a) List[int]（共享掩码）
                    b) List[List[int]]（每样本掩码）
            - 若未提供 image_noise_mask，会自动构建为 [0, ..., 0, 1]
                （条件图 token 为 0，目标图 token 为 1）

    3) Batch 与分桶兼容性
            - 脚本中 batch_size 会由 prompt/image 两侧自动取最大值，避免单边长度触发错配。
            - UnifiedDataset 在 enable_bucket=True 时，会按 bucket_data_key='image' 建桶，并对 data_file_keys
                中存在的图像字段都应用 main_data_operator；因此 edit_image/condition_images 也可自动读取与缩放。
            - 建议 bucket_base_reso、min/max_bucket_reso 与 height_division_factor/width_division_factor 保持 16 的倍数。
    """

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1)
    train_omni = False

    # 1.05 B 配置
    # 你也可以改成更深更宽(一般是直接改num_layers和num_refiner_layers)；需要满足 hidden_size = num_attention_heads * attention_head_dim
    # 默认是num_layers=60，num_refiner_layers=2的配置
    complextro_model_config = {
        "num_layers": 12,
        "num_refiner_layers": 1,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "rope_axes_dim": [16, 56, 56],
        "enable_tread_routing": True,
        "tread_routes": [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 2,
                "end_layer_idx": 8,
            }
        ],
    }

    data_file_keys = ("image", "edit_image", "condition_images") if train_omni else ("image",)

    train_resolution = (256, 256)
    max_bucket_reso = 512
    dataset = UnifiedDataset(
        base_path="/root/autodl-tmp/DiffSynth-Studio/data/images",
        metadata_path="/root/autodl-tmp/DiffSynth-Studio/data/metadata_merged.csv",
        max_data_items=10000000,
        data_file_keys=data_file_keys,
        enable_bucket=True,
        bucket_no_upscale=False,
        min_bucket_reso=128,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=16,
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
        train_omni=train_omni,
        complextro_model_config=complextro_model_config,
    )
    model_logger = ModelLogger(
        "models/Complextro/v2", # dit输出文件夹
        remove_prefix_in_ckpt="pipe.dit.",
    )

    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        batch_size=10,
        learning_rate=1e-4,
        optimizer_type="pytorch_optimizer.Adan",
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
        mup_scale=True,
        mup_base_dim=1.0,
        mup_dim=complextro_model_config.get("hidden_size", None),
        max_grad_norm=1.0,
        num_workers=4,
        #save_steps=50000,
        save_epochs=1,
        num_epochs=99999999999,
    )
