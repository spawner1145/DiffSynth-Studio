import torch
from PIL import Image
from typing import Union, List, Optional, Any
from tqdm import tqdm
import inspect

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from transformers import AutoTokenizer, AutoProcessor
from ..models.complextro_dit import ComplextroImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.flux2_vae import Flux2VAE
from ..models.siglip2_image_encoder import Siglip2ImageEncoder428M


class ComplextroPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: ComplextroImageDiT = None
        self.vae: Flux2VAE = None
        self.image_encoder: Siglip2ImageEncoder428M = None
        self.tokenizer: AutoTokenizer = None
        self.processor: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            ComplextroUnit_ShapeChecker(),
            ComplextroUnit_PromptEmbedder(),
            ComplextroUnit_NoiseInitializer(),
            ComplextroUnit_InputImageEmbedder(),
            ComplextroUnit_EditImageAutoResize(),
            ComplextroUnit_EditImageEmbedder(),
            ComplextroUnit_EditImageEmbedderSiglip(),
        ]
        self.model_fn = model_fn_complextro

    def _get_vae_input_channels(self) -> Optional[int]:
        if self.vae is None:
            return None
        if hasattr(self.vae, "encoder") and hasattr(self.vae.encoder, "conv_in"):
            return int(self.vae.encoder.conv_in.in_channels)
        return None

    def _normalize_image_mode_for_vae(self, image):
        expected_channels = self._get_vae_input_channels()
        if expected_channels not in (3, 4):
            return image
        target_mode = "RGBA" if expected_channels == 4 else "RGB"
        if isinstance(image, list):
            return [self._normalize_image_mode_for_vae(i) for i in image]
        if isinstance(image, Image.Image) and image.mode != target_mode:
            return image.convert(target_mode)
        return image

    def _load_image(self, image, convert_mode: Optional[str] = None):
        if isinstance(image, str):
            image = Image.open(image)
        if convert_mode is not None and isinstance(image, Image.Image) and image.mode != convert_mode:
            image = image.convert(convert_mode)
        return image

    def _prepare_multimodal_image(self, image):
        return self._load_image(image, convert_mode="RGB")

    def _prepare_vae_image(self, image):
        image = self._load_image(image)
        return self._normalize_image_mode_for_vae(image)

    @staticmethod
    def _is_nested_list(value) -> bool:
        return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)

    @classmethod
    def _normalize_grouped_batch_input(cls, value, batch_size: int):
        if value is None or batch_size <= 1:
            return value
        if cls._is_nested_list(value):
            if len(value) == batch_size:
                return value
            if len(value) == 1:
                return [value[0] for _ in range(batch_size)]
            return [value[i % len(value)] for i in range(batch_size)]

        values = value if isinstance(value, list) else [value]
        if len(values) == batch_size:
            return [[item] for item in values]
        return [list(values) for _ in range(batch_size)]

    @staticmethod
    def _get_text_hidden_size(text_encoder: Optional[QwenImageTextEncoder]) -> Optional[int]:
        if text_encoder is None:
            return None
        text_config = getattr(text_encoder.model.config, "text_config", text_encoder.model.config)
        return int(text_config.hidden_size)

    @classmethod
    def _validate_text_encoder_dit_alignment(cls, text_encoder: Optional[QwenImageTextEncoder], dit: Optional[ComplextroImageDiT]):
        if text_encoder is None or dit is None:
            return
        text_hidden_size = cls._get_text_hidden_size(text_encoder)
        dit_text_dim = int(dit.txt_in.in_features)
        if text_hidden_size != dit_text_dim:
            raise ValueError(
                f"Text encoder hidden_size ({text_hidden_size}) != Complextro text_embed_dim ({dit_text_dim}). "
                "Please load a ComplextroImageDiT checkpoint/config whose text_embed_dim matches the text encoder."
            )

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen3.5-2B", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen3.5-2B", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        pipe = ComplextroPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("complextro_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        pipe.image_encoder = model_pool.fetch_model("siglip_vision_model_428m")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            if pipe.tokenizer is None and hasattr(pipe.processor, "tokenizer"):
                pipe.tokenizer = pipe.processor.tokenizer

        pipe._validate_text_encoder_dit_alignment(pipe.text_encoder, pipe.dit)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        cfg_scale: float = 1.0,
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_latent: Optional[Any] = None,
        edit_image_auto_resize: bool = True,
        omni_mode: bool = False,
        image_noise_mask: Optional[Union[List[int], List[List[int]]]] = None,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 30,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=(height // 16) * (width // 16),
        )

        batch_size = 1
        if isinstance(prompt, list):
            batch_size = max(batch_size, len(prompt))
        if isinstance(input_image, list):
            batch_size = max(batch_size, len(input_image))
        if self._is_nested_list(edit_image):
            batch_size = max(batch_size, len(edit_image))
        if self._is_nested_list(edit_latent):
            batch_size = max(batch_size, len(edit_latent))
        if self._is_nested_list(image_noise_mask):
            batch_size = max(batch_size, len(image_noise_mask))

        if isinstance(prompt, str) and batch_size > 1:
            prompt = [prompt] * batch_size
        elif isinstance(prompt, list) and len(prompt) == 1 and batch_size > 1:
            prompt = prompt * batch_size

        if isinstance(negative_prompt, str) and batch_size > 1:
            negative_prompt = [negative_prompt] * batch_size
        elif isinstance(negative_prompt, list) and len(negative_prompt) == 1 and batch_size > 1:
            negative_prompt = negative_prompt * batch_size

        edit_image = self._normalize_grouped_batch_input(edit_image, batch_size)
        edit_latent = self._normalize_grouped_batch_input(edit_latent, batch_size)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "denoising_strength": denoising_strength,
            "edit_image": edit_image,
            "edit_latent": edit_latent,
            "edit_image_auto_resize": edit_image_auto_resize,
            "omni_mode": omni_mode,
            "image_noise_mask": image_noise_mask,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep,
                progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(
                self.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                **inputs_shared,
            )

        self.load_models_to_device(["vae"])
        image = self.vae.decode(inputs_shared["latents"])
        if image.shape[0] == 1:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])
        return image


class ComplextroUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: ComplextroPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ComplextroUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",),
        )

    def _normalize_image_groups(self, edit_image, batch_size):
        if edit_image is None:
            return [None] * batch_size
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            if len(edit_image) == batch_size:
                return edit_image
            if len(edit_image) == 1:
                return [edit_image[0] for _ in range(batch_size)]
            return [edit_image[i % len(edit_image)] for i in range(batch_size)]
        images = edit_image if isinstance(edit_image, list) else [edit_image]
        if batch_size > 1 and len(images) == batch_size:
            return [[image] for image in images]
        return [images for _ in range(batch_size)]

    def _build_chat_content(self, prompt: str, images):
        if images is None or len(images) == 0:
            return [{"type": "text", "text": prompt}]
        content = [{"type": "image", "image": pipe_image} for pipe_image in images]
        content.append({"type": "text", "text": prompt})
        return content

    def _split_prompt_segments(self, prompt_text: str):
        if prompt_text is None:
            return [""]
        segments = [segment.strip() for segment in str(prompt_text).split("<break>")]
        segments = [segment for segment in segments if segment != ""]
        return segments if len(segments) > 0 else [""]

    def _parse_system_and_user(self, text_segment: str):
        marker = "<prompt start>"
        if marker not in text_segment:
            return None, text_segment
        system_prompt, user_prompt = text_segment.split(marker, 1)
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()
        if system_prompt == "":
            system_prompt = None
        return system_prompt, user_prompt

    def process(self, pipe: ComplextroPipeline, prompt, edit_image=None):
        pipe.load_models_to_device(self.onload_model_names)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        image_groups = self._normalize_image_groups(edit_image, len(prompts))

        template_source = pipe.processor if pipe.processor is not None else pipe.tokenizer
        if template_source is None:
            raise ValueError("ComplextroPipeline requires tokenizer or processor for prompt encoding.")

        has_any_image = False
        conversations = []
        segment_owner = []
        for owner_id, (prompt_item, images) in enumerate(zip(prompts, image_groups)):
            local_images = images
            if local_images is not None and len(local_images) > 0:
                local_images = [pipe._prepare_multimodal_image(image) for image in local_images]
                has_any_image = True
            prompt_segments = self._split_prompt_segments(prompt_item)
            for prompt_segment in prompt_segments:
                system_prompt, user_prompt = self._parse_system_and_user(prompt_segment)
                user_prompt = "" if user_prompt is None else user_prompt
                messages = []
                if system_prompt is not None:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    })
                messages.append({
                    "role": "user",
                    "content": self._build_chat_content(user_prompt, local_images),
                })
                conversations.append(messages)
                segment_owner.append(owner_id)

        if len(prompts) == 0:
            return {"prompt_emb": torch.empty(0, device=pipe.device), "prompt_emb_mask": torch.empty(0, device=pipe.device, dtype=torch.long)}

        if has_any_image and pipe.processor is None:
            raise ValueError("Image prompts require an AutoProcessor; tokenizer-only mode cannot encode images.")
        if not hasattr(template_source, "apply_chat_template"):
            raise ValueError("Selected tokenizer/processor does not support apply_chat_template.")

        template_kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 1024,
            "add_generation_prompt": True,
        }
        signature = inspect.signature(template_source.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            template_kwargs["enable_thinking"] = False

        model_inputs = template_source.apply_chat_template(conversations, **template_kwargs).to(pipe.device)

        model_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if hasattr(model_inputs, "pixel_values"):
            model_kwargs["pixel_values"] = model_inputs.pixel_values
        if hasattr(model_inputs, "pixel_values_videos"):
            model_kwargs["pixel_values_videos"] = model_inputs.pixel_values_videos
        if hasattr(model_inputs, "image_grid_thw"):
            model_kwargs["image_grid_thw"] = model_inputs.image_grid_thw
        if hasattr(model_inputs, "video_grid_thw"):
            model_kwargs["video_grid_thw"] = model_inputs.video_grid_thw
        if hasattr(model_inputs, "mm_token_type_ids"):
            model_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        output = pipe.text_encoder(**model_kwargs)
        hidden_states = output.hidden_states if hasattr(output, "hidden_states") else output
        segment_emb = hidden_states[-2].to(dtype=pipe.torch_dtype, device=pipe.device) # 暂时保留意见，-1好像效果一般
        segment_mask = model_inputs.attention_mask.to(device=pipe.device, dtype=torch.long)

        emb_groups = [[] for _ in range(len(prompts))]
        mask_groups = [[] for _ in range(len(prompts))]
        for seg_id, owner_id in enumerate(segment_owner):
            valid_length = int(segment_mask[seg_id].sum().item())
            if valid_length <= 0:
                continue
            emb_groups[owner_id].append(segment_emb[seg_id, :valid_length])
            mask_groups[owner_id].append(segment_mask[seg_id, :valid_length])

        hidden_dim = segment_emb.shape[-1]
        merged_emb = []
        merged_mask = []
        for owner_id in range(len(prompts)):
            if len(emb_groups[owner_id]) == 0:
                merged_emb.append(torch.zeros((1, hidden_dim), dtype=pipe.torch_dtype, device=pipe.device))
                merged_mask.append(torch.ones((1,), dtype=torch.long, device=pipe.device))
                continue
            merged_emb.append(torch.cat(emb_groups[owner_id], dim=0))
            merged_mask.append(torch.cat(mask_groups[owner_id], dim=0))

        target_seq_len = 1024
        prompt_emb = torch.zeros((len(prompts), target_seq_len, hidden_dim), dtype=pipe.torch_dtype, device=pipe.device)
        prompt_emb_mask = torch.zeros((len(prompts), target_seq_len), dtype=torch.long, device=pipe.device)
        for batch_id, (emb_item, mask_item) in enumerate(zip(merged_emb, merged_mask)):
            local_len = min(int(emb_item.shape[0]), target_seq_len)
            prompt_emb[batch_id, :local_len] = emb_item[:local_len]
            prompt_emb_mask[batch_id, :local_len] = mask_item[:local_len]

        return {"prompt_emb": prompt_emb, "prompt_emb_mask": prompt_emb_mask}


class ComplextroUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "batch_size"),
            output_params=("noise",),
        )

    def process(self, pipe: ComplextroPipeline, height, width, seed, rand_device, batch_size=1):
        latent_channels = int(pipe.dit.img_in.in_features) if pipe.dit is not None else 128
        noise = pipe.generate_noise(
            (int(batch_size), latent_channels, height // 16, width // 16),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=pipe.torch_dtype,
        )
        return {"noise": noise}


class ComplextroUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: ComplextroPipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        input_image = pipe._normalize_image_mode_for_vae(input_image)
        if isinstance(input_image, list):
            image = torch.cat([pipe.preprocess_image(img) for img in input_image], dim=0)
        else:
            image = pipe.preprocess_image(input_image)
        image = image.to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents, "input_latents": input_latents}


class ComplextroUnit_EditImageAutoResize(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_image_auto_resize"),
            output_params=("edit_image",),
        )

    def process(self, pipe: ComplextroPipeline, edit_image, edit_image_auto_resize):
        if edit_image is None or not edit_image_auto_resize:
            return {}
        from ..core.data.operators import ImageCropAndResize
        operator = ImageCropAndResize(max_pixels=1024 * 1024, height_division_factor=16, width_division_factor=16)
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            edit_image = [[operator(pipe._prepare_multimodal_image(img)) for img in image_group] for image_group in edit_image]
        elif isinstance(edit_image, list):
            edit_image = [operator(pipe._prepare_multimodal_image(img)) for img in edit_image]
        else:
            edit_image = operator(pipe._prepare_multimodal_image(edit_image))
        return {"edit_image": edit_image}


class ComplextroUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_latent", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_latent_mask"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _is_pad_marker(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in ("", "0")
        return value == 0

    @staticmethod
    def _is_same_as_edit_image_marker(value) -> bool:
        return isinstance(value, str) and value.strip() == "1"

    def _normalize_latent_groups(self, edit_latent, batch_size):
        if edit_latent is None:
            return [None] * batch_size
        if isinstance(edit_latent, list) and len(edit_latent) > 0 and isinstance(edit_latent[0], list):
            if len(edit_latent) == batch_size:
                return edit_latent
            if len(edit_latent) == 1:
                return [edit_latent[0] for _ in range(batch_size)]
            return [edit_latent[i % len(edit_latent)] for i in range(batch_size)]
        values = edit_latent if isinstance(edit_latent, list) else [edit_latent]
        if batch_size > 1 and len(values) == batch_size:
            return [[value] for value in values]
        return [values for _ in range(batch_size)]

    @staticmethod
    def _normalize_image_groups(edit_image, batch_size):
        if edit_image is None:
            return [None] * batch_size
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            if len(edit_image) == batch_size:
                return edit_image
            if len(edit_image) == 1:
                return [edit_image[0] for _ in range(batch_size)]
            return [edit_image[i % len(edit_image)] for i in range(batch_size)]
        values = edit_image if isinstance(edit_image, list) else [edit_image]
        if batch_size > 1 and len(values) == batch_size:
            return [[value] for value in values]
        return [values for _ in range(batch_size)]

    @staticmethod
    def _infer_group_count(edit_image, edit_latent) -> int:
        candidates = []
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            candidates.append(len(edit_image))
        if isinstance(edit_latent, list) and len(edit_latent) > 0 and isinstance(edit_latent[0], list):
            candidates.append(len(edit_latent))
        return max(candidates) if len(candidates) > 0 else 1

    def _resolve_group_latent_inputs(self, edit_image_group, edit_latent_group):
        image_group = [] if edit_image_group is None else (edit_image_group if isinstance(edit_image_group, list) else [edit_image_group])
        latent_group = [] if edit_latent_group is None else edit_latent_group
        if not isinstance(latent_group, list):
            latent_group = [latent_group]

        if len(image_group) == 0:
            resolved_group = []
            keep_mask = []
            for latent_value in latent_group:
                if self._is_pad_marker(latent_value) or self._is_same_as_edit_image_marker(latent_value):
                    raise ValueError("edit_latent markers '0'/'1' require matching edit_image inputs.")
                resolved_group.append(latent_value)
                keep_mask.append(True)
            return resolved_group, keep_mask

        if len(latent_group) > len(image_group):
            raise ValueError("edit_latent entries must be less than or equal to edit_image entries.")

        resolved_group = []
        keep_mask = []
        for idx, image_value in enumerate(image_group):
            if idx >= len(latent_group):
                resolved_group.append(image_value)
                keep_mask.append(False)
                continue

            latent_value = latent_group[idx]
            if self._is_pad_marker(latent_value):
                resolved_group.append(image_value)
                keep_mask.append(False)
            elif self._is_same_as_edit_image_marker(latent_value):
                resolved_group.append(image_value)
                keep_mask.append(True)
            else:
                resolved_group.append(latent_value)
                keep_mask.append(True)
        return resolved_group, keep_mask

    def process(self, pipe: ComplextroPipeline, edit_image, edit_latent, edit_image_auto_resize):
        if edit_image is None and edit_latent is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        from ..core.data.operators import ImageCropAndResize
        resize_operator = ImageCropAndResize(max_pixels=1024 * 1024, height_division_factor=16, width_division_factor=16)

        group_count = self._infer_group_count(edit_image, edit_latent)
        image_groups = self._normalize_image_groups(edit_image, group_count)
        latent_groups = self._normalize_latent_groups(edit_latent, group_count)

        edit_latents = []
        edit_latent_mask = []
        for image_group, latent_group in zip(image_groups, latent_groups):
            resolved_group, keep_mask = self._resolve_group_latent_inputs(image_group, latent_group)
            group_latents = []
            for latent_source in resolved_group:
                latent_image = pipe._prepare_vae_image(latent_source)
                if edit_image_auto_resize:
                    latent_image = resize_operator(latent_image)
                image_tensor = pipe.preprocess_image(latent_image).to(device=pipe.device, dtype=pipe.torch_dtype)
                group_latents.append(pipe.vae.encode(image_tensor))
            edit_latents.append(group_latents)
            edit_latent_mask.append(keep_mask)

        if group_count == 1 and not pipe._is_nested_list(edit_image) and not pipe._is_nested_list(edit_latent):
            edit_latents = edit_latents[0]
            edit_latent_mask = edit_latent_mask[0]
        return {"edit_latents": edit_latents, "edit_latent_mask": edit_latent_mask}


class ComplextroUnit_EditImageEmbedderSiglip(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("image_embeds",),
            onload_model_names=("image_encoder",),
        )

    def process(self, pipe: ComplextroPipeline, edit_image):
        if edit_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            flat_images = []
            group_sizes = []
            for image_group in edit_image:
                prepared_group = [pipe._prepare_multimodal_image(image) for image in image_group]
                flat_images.extend(prepared_group)
                group_sizes.append(len(prepared_group))
            flat_embeds = pipe.image_encoder(flat_images, device=pipe.device)
            image_embeds = []
            offset = 0
            for group_size in group_sizes:
                image_embeds.append(flat_embeds[offset : offset + group_size])
                offset += group_size
        else:
            images = edit_image if isinstance(edit_image, list) else [edit_image]
            prepared_images = [pipe._prepare_multimodal_image(image) for image in images]
            image_embeds = pipe.image_encoder(prepared_images, device=pipe.device)
        return {"image_embeds": image_embeds}


def _build_omni_noise_mask(num_cond: int, image_noise_mask: Optional[List[int]]):
    default_mask = [0] * num_cond + [1]
    if image_noise_mask is None:
        return default_mask
    mask = [int(v) for v in image_noise_mask]
    if len(mask) < num_cond + 1:
        mask = mask + [mask[-1] if len(mask) > 0 else 1] * (num_cond + 1 - len(mask))
    return mask[: num_cond + 1]


def model_fn_complextro(
    dit: ComplextroImageDiT,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    edit_latents=None,
    edit_latent_mask=None,
    image_embeds=None,
    omni_mode: bool = False,
    image_noise_mask: Optional[Union[List[int], List[List[int]]]] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    timestep = timestep / 1000

    if omni_mode and edit_latents is not None and len(edit_latents) > 0:
        batch_size = latents.shape[0]

        def to_batch_groups(items, batch):
            if items is None:
                return [[] for _ in range(batch)]
            if len(items) == 0:
                return [[] for _ in range(batch)]
            if isinstance(items[0], list):
                if len(items) == batch:
                    return items
                if len(items) == 1:
                    return [items[0] for _ in range(batch)]
                raise ValueError("Omni mode expects nested list length == batch_size or 1.")
            return [items for _ in range(batch)]

        lat_groups = to_batch_groups(edit_latents, batch_size)
        sig_groups = to_batch_groups(image_embeds, batch_size)
        keep_groups = to_batch_groups(edit_latent_mask, batch_size) if edit_latent_mask is not None else None

        if image_noise_mask is None:
            mask = [_build_omni_noise_mask(len(g), None) for g in lat_groups]
        elif len(image_noise_mask) > 0 and isinstance(image_noise_mask[0], list):
            if len(image_noise_mask) == batch_size:
                mask = [_build_omni_noise_mask(len(lat_groups[i]), image_noise_mask[i]) for i in range(batch_size)]
            elif len(image_noise_mask) == 1:
                mask = [_build_omni_noise_mask(len(lat_groups[i]), image_noise_mask[0]) for i in range(batch_size)]
            else:
                raise ValueError("Nested image_noise_mask length must be batch_size or 1.")
        else:
            mask = [_build_omni_noise_mask(len(lat_groups[i]), image_noise_mask) for i in range(batch_size)]

        latents_omni = []
        siglip_omni = []
        latent_keep_mask = []
        for b in range(batch_size):
            cond_list = [latent_item[0] for latent_item in lat_groups[b]]
            latents_omni.append(cond_list + [latents[b]])

            if keep_groups is not None:
                local_keep = [bool(v) for v in keep_groups[b][: len(cond_list)]]
                if len(local_keep) < len(cond_list):
                    local_keep = local_keep + [False] * (len(cond_list) - len(local_keep))
                latent_keep_mask.append(local_keep + [True])

            if image_embeds is not None:
                cond_sig = sig_groups[b]
                siglip_omni.append(cond_sig + [None])

        siglip_arg = siglip_omni if image_embeds is not None else None

        model_output = dit(
            latents=latents_omni,
            timestep=timestep,
            prompt_emb=prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            image_noise_mask=mask,
            edit_latent_mask=latent_keep_mask if keep_groups is not None else None,
            siglip_feats=siglip_arg,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return model_output

    model_output = dit(
        latents=latents,
        timestep=timestep,
        prompt_emb=prompt_emb,
        prompt_emb_mask=prompt_emb_mask,
        siglip_feats=None,
        image_noise_mask=None,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return model_output
