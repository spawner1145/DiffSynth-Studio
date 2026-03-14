import os, argparse
import torch, accelerate
from accelerate import DistributedDataParallelKwargs
from typing import List, Optional, Any

from transformers import AutoProcessor
from diffsynth.core import UnifiedDataset, ImageTextPairDataset, load_model
from diffsynth.configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
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
        qwen_model_file="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b/model.safetensors",
        flux2_vae_file="/mnt/raid0/linux-train/diffusion-model-v1/flux2-vae/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b",
        qwen_model_size: str = "2B",
        siglip_model_file="",
        #complextro_dit_file="/root/autodl-tmp/DiffSynth-Studio/models/Complextro/v2/model-e43-s19221.safetensors",
        complextro_dit_file="",
        train_omni: bool = False,
        use_alpha_layer_vae: bool = False,
        complextro_model_config: Optional[dict] = None,
        enable_vram_offload: bool = False,
        vram_config: Optional[dict] = None,
        vram_limit: Optional[float] = None,
    ):
        super().__init__()
        self.train_omni = train_omni
        self.complextro_model_config = {} if complextro_model_config is None else dict(complextro_model_config)
        self.enable_vram_offload = enable_vram_offload
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
            model_class_path = f"{model_class.__module__}.{model_class.__name__}"
            if model_class_path in VERSION_CHECKER_MAPS:
                return VERSION_CHECKER_MAPS[model_class_path]()
            if model_class_path not in VRAM_MANAGEMENT_MODULE_MAPS:
                raise KeyError(f"No VRAM management module map registered for {model_class_path}.")
            return VRAM_MANAGEMENT_MODULE_MAPS[model_class_path]

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
        self.pipe.vae = load_aux_model(
            Flux2VAE,
            flux2_vae_file,
            config={"use_alpha_layer": use_alpha_layer_vae},
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

        dit_text_dim = int(self.pipe.dit.txt_in.in_features)
        if text_hidden_size != dit_text_dim:
            raise ValueError(
                f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
                f"Please align QwenImageTextEncoder(model_type='qwen3_5', model_size='{qwen_model_size}') and ComplextroImageDiT(text_embed_dim=...)."
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

    def _normalize_prompt(self, prompt_value: Any, batch_size: int):
        if isinstance(prompt_value, str):
            return prompt_value if batch_size == 1 else [prompt_value] * batch_size
        if not isinstance(prompt_value, list):
            return [str(prompt_value)] * batch_size
        if len(prompt_value) == batch_size:
            return prompt_value
        if len(prompt_value) == 1:
            return prompt_value * batch_size
        if len(prompt_value) == 0:
            return [""] * batch_size
        return [prompt_value[i % len(prompt_value)] for i in range(batch_size)]

    def _normalize_negative_prompt(self, prompt_value: Any, neg_prompt_value: Any):
        prompt_batch = 1 if isinstance(prompt_value, str) else len(prompt_value)
        if neg_prompt_value is None:
            return [""] * prompt_batch
        if isinstance(neg_prompt_value, str):
            return neg_prompt_value if prompt_batch == 1 else [neg_prompt_value] * prompt_batch
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
        image_value = data["image"]
        batch_size = self._infer_batch_size(prompt_value, image_value)
        prompt_value = self._normalize_prompt(prompt_value, batch_size)
        neg_prompt_value = self._normalize_negative_prompt(prompt_value, data.get("neg_prompt", ""))
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
    parser = argparse.ArgumentParser(description="Complextro Training Script")
    parser.add_argument("--use_image_text_pairs", action="store_true", help="True: 使用 ImageTextPairDataset（图片+txt目录），False: 使用 UnifiedDataset（metadata文件）")
    parser.add_argument("--train_omni", action="store_true", default=True, help="是否开启 Omni/编辑训练模式")
    parser.add_argument("--use_alpha_layer_vae", action="store_true", help="是否使用带 alpha 层 VAE")
    parser.add_argument("--siglip_model_file", type=str, default="", help="SigLIP 模型文件路径")
    parser.add_argument("--qwen_model_file", type=str, default="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b/model.safetensors", help="Qwen 模型文件路径")
    parser.add_argument("--flux2_vae_file", type=str, default="/mnt/raid0/linux-train/diffusion-model-v1/flux2-vae/diffusion_pytorch_model.safetensors", help="Flux2 VAE 文件路径")
    parser.add_argument("--qwen_tokenizer_dir", type=str, default="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b", help="Qwen Tokenizer 目录")
    parser.add_argument("--base_path", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/images", help="数据集根路径")
    parser.add_argument("--metadata_path", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl", help="UnifiedDataset 的 metadata 路径")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/DiffSynth-Studio/edit/images", help="ImageTextPairDataset 的目录路径")
    parser.add_argument("--recursive", action="store_true", help="是否递归加载子文件夹")
    parser.add_argument("--output_dir", type=str, default="models/Complextro/edit", help="训练模型输出目录")
    parser.add_argument("--batch_size", type=int, default=2, help="训练 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据读取线程数")
    parser.add_argument("--save_epochs", type=int, default=1, help="每多少个 epoch 保存一次模型")
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
    
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    use_image_text_pairs = args.use_image_text_pairs
    train_omni = args.train_omni
    use_alpha_layer_vae = args.use_alpha_layer_vae
    siglip_model_file = args.siglip_model_file
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
                    height_division_factor=16,
                    width_division_factor=16,
                ),
            )
        else:
            dataset = ImageTextPairDataset(
                data_dir=args.data_dir,
                max_pixels=max_bucket_reso * max_bucket_reso,
                height_division_factor=16,
                width_division_factor=16,
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
                height_division_factor=16,
                width_division_factor=16,
            ),
        )

    model = ComplextroTrainingModule(
        device=accelerator.device,
        qwen_model_file=args.qwen_model_file,
        flux2_vae_file=args.flux2_vae_file,
        qwen_tokenizer_dir=args.qwen_tokenizer_dir,
        qwen_model_size="2B",
        siglip_model_file=siglip_model_file,
        train_omni=train_omni,
        use_alpha_layer_vae=use_alpha_layer_vae,
        complextro_model_config=complextro_model_config,
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
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer_type="adamw",
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
        mup_scale=False,
        mup_base_dim=1.0,
        mup_dim=complextro_model_config.get("hidden_size", None),
        max_grad_norm=1.0,
        num_workers=args.num_workers,
        save_epochs=args.save_epochs,
        num_epochs=args.num_epochs,
    )
