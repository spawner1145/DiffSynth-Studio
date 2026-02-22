import torch
from PIL import Image
from typing import Union, List, Optional
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from transformers import AutoTokenizer
from ..models.complextro_dit import ComplextroImageDiT
from ..models.z_image_text_encoder import ZImageTextEncoder
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
        self.text_encoder: ZImageTextEncoder = None
        self.dit: ComplextroImageDiT = None
        self.vae: Flux2VAE = None
        self.image_encoder: Siglip2ImageEncoder428M = None
        self.tokenizer: AutoTokenizer = None
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

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        pipe = ComplextroPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("complextro_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        pipe.image_encoder = model_pool.fetch_model("siglip_vision_model_428m")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        cfg_scale: float = 1.0,
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_image_auto_resize: bool = True,
        omni_mode: bool = False,
        image_noise_mask: Optional[Union[List[int], List[List[int]]]] = None,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 30,
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

        if isinstance(prompt, str) and batch_size > 1:
            prompt = [prompt] * batch_size
        if isinstance(negative_prompt, str) and batch_size > 1:
            negative_prompt = [negative_prompt] * batch_size

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "denoising_strength": denoising_strength,
            "edit_image": edit_image,
            "edit_image_auto_resize": edit_image_auto_resize,
            "omni_mode": omni_mode,
            "image_noise_mask": image_noise_mask,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
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
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",),
        )

    def _apply_template(self, tokenizer, prompt: str):
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return prompt

    def process(self, pipe: ComplextroPipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        texts = [self._apply_template(pipe.tokenizer, p) for p in prompts]

        model_inputs = pipe.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(pipe.device)

        output = pipe.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        prompt_emb = output.hidden_states[-2].to(dtype=pipe.torch_dtype, device=pipe.device) # 暂时保留意见，-1好像效果一般
        prompt_emb_mask = model_inputs.attention_mask.to(device=pipe.device, dtype=torch.long)
        return {"prompt_emb": prompt_emb, "prompt_emb_mask": prompt_emb_mask}


class ComplextroUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "batch_size"),
            output_params=("noise",),
        )

    def process(self, pipe: ComplextroPipeline, height, width, seed, rand_device, batch_size=1):
        noise = pipe.generate_noise(
            (int(batch_size), 128, height // 16, width // 16),
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
        if not isinstance(edit_image, list):
            edit_image = [edit_image]
        edit_image = [operator(img) for img in edit_image]
        return {"edit_image": edit_image}


class ComplextroUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("edit_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: ComplextroPipeline, edit_image):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            edit_latents = []
            for image_group in edit_image:
                group_latents = []
                for image in image_group:
                    image_tensor = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                    group_latents.append(pipe.vae.encode(image_tensor))
                edit_latents.append(group_latents)
        else:
            images = edit_image if isinstance(edit_image, list) else [edit_image]
            edit_latents = []
            for image in images:
                image_tensor = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                edit_latents.append(pipe.vae.encode(image_tensor))
        return {"edit_latents": edit_latents}


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
            image_embeds = []
            for image_group in edit_image:
                group_embeds = [pipe.image_encoder(image, device=pipe.device).to(pipe.torch_dtype) for image in image_group]
                image_embeds.append(group_embeds)
        else:
            images = edit_image if isinstance(edit_image, list) else [edit_image]
            image_embeds = [pipe.image_encoder(image, device=pipe.device).to(pipe.torch_dtype) for image in images]
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
        for b in range(batch_size):
            cond_list = [latent_item[0] for latent_item in lat_groups[b]]
            latents_omni.append(cond_list + [latents[b]])

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
