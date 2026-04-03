import torch
from transformers.feature_extraction_utils import BatchFeature
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
from ..models.pixel_identity_vae import PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE
from ..models.qwen_image_vae import QwenImageVAE
from ..models.siglip2_image_encoder import Siglip2ImageEncoder428M
from .complextro_vae_utils import (
    infer_complextro_vae_downsample_factor,
    infer_complextro_vae_latent_channels,
)


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
        self.vae: Flux2VAE | QwenImageVAE | PixelIdentityVAE = None
        self.image_encoder: Siglip2ImageEncoder428M = None
        self.tokenizer: AutoTokenizer = None
        self.processor: AutoProcessor = None
        self.prediction_type = "flow"
        self.jit_p_mean = -0.8
        self.jit_p_std = 0.8
        self.jit_noise_scale = 1.0
        self.jit_t_eps = 5e-2
        self.jit_sampling_method = "heun"
        self.jit_cfg_interval_min = 0.0
        self.jit_cfg_interval_max = 1.0
        self.jit_loss_weighting = "velocity"  # "velocity" | "balanced" | "x_pred"
        self.freq_loss_enabled = False
        self.freq_loss_weight = 0.0
        self.freq_loss_mode = "dct"
        self.freq_loss_block_size = 8
        self.freq_loss_profile = "jpeg"
        self.freq_loss_quality = 85
        self.freq_loss_jpeg_mode = "inv_gamma"
        self.freq_loss_gamma = 1.0
        self.freq_loss_color_space = "rgb"
        self.freq_loss_weight_floor = 0.1
        self.freq_loss_hf_scale = 0.25
        self.freq_loss_lf_scale = 1.0
        self.freq_loss_t_adaptive = False
        self.freq_loss_t_min_hf_scale = 0.25
        self.freq_loss_t_max_hf_scale = 1.0
        self.freq_loss_t_gamma = 1.0
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
        if hasattr(self.vae, "image_channels"):
            return int(self.vae.image_channels)
        if hasattr(self.vae, "encoder") and hasattr(self.vae.encoder, "conv_in"):
            return int(self.vae.encoder.conv_in.in_channels)
        return None

    def _get_vae_downsample_factor(self) -> int:
        return infer_complextro_vae_downsample_factor(self.vae)

    def _get_vae_token_downsample_factor(self) -> int:
        downsample = self._get_vae_downsample_factor()
        latent_patch_size = 1
        if self.dit is not None and hasattr(self.dit, "latent_patch_size"):
            latent_patch_size = int(self.dit.latent_patch_size)
        return int(downsample) * int(latent_patch_size)

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

    def _jit_cfg_guided_velocity(
        self,
        models,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        latents: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        timestep = t.flatten()
        model_inputs = dict(inputs_shared)
        model_latents = latents.to(device=self.device, dtype=self.torch_dtype)
        model_inputs["latents"] = model_latents
        x_pred_posi = self.model_fn(**models, **model_inputs, **inputs_posi, timestep=timestep)
        x_pred_posi = x_pred_posi.to(device=latents.device, dtype=torch.float32)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        denom = (1.0 - t).clamp_min(float(self.jit_t_eps)).to(device=latents.device, dtype=torch.float32)
        while denom.ndim < latents.ndim:
            denom = denom.unsqueeze(-1)
        v_pred_posi = (x_pred_posi - latents_fp32) / denom
        if cfg_scale == 1.0:
            return v_pred_posi

        x_pred_nega = self.model_fn(**models, **model_inputs, **inputs_nega, timestep=timestep)
        x_pred_nega = x_pred_nega.to(device=latents.device, dtype=torch.float32)
        v_pred_nega = (x_pred_nega - latents_fp32) / denom
        low = float(self.jit_cfg_interval_min)
        high = float(self.jit_cfg_interval_max)
        apply_cfg = bool(float(t.flatten()[0]) < high and (low == 0.0 or float(t.flatten()[0]) > low))
        if not apply_cfg:
            return v_pred_posi
        return v_pred_nega + cfg_scale * (v_pred_posi - v_pred_nega)

    def _bridge_cfg_guided_velocity(
        self,
        models,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        latents: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        timestep = t.flatten()
        model_inputs = dict(inputs_shared)
        model_inputs["latents"] = latents.to(device=self.device, dtype=self.torch_dtype)
        r_pred_posi = self.model_fn(**models, **model_inputs, **inputs_posi, timestep=timestep)
        r_pred_posi = r_pred_posi.to(device=latents.device, dtype=torch.float32)
        t_fp32 = t.to(device=latents.device, dtype=torch.float32)
        lambda2 = t_fp32.pow(2) + (1.0 - t_fp32).pow(2)
        # NOTE: unlike BridgeXPredLoss where mean_t = (t/λ²) * latents (includes
        # latents), here mean_t is just the scalar coefficient (t/λ²).  The actual
        # conditional mean is reconstructed as mean_t * latents_fp32 below.
        mean_t = (t_fp32 / lambda2.clamp_min(float(self.jit_t_eps))).to(dtype=torch.float32)
        scale = ((1.0 - t_fp32) / lambda2.clamp_min(float(self.jit_t_eps)).sqrt()).to(dtype=torch.float32)
        scale = scale.clamp_min(float(self.jit_t_eps))
        while mean_t.ndim < latents.ndim:
            mean_t = mean_t.unsqueeze(-1)
        while scale.ndim < latents.ndim:
            scale = scale.unsqueeze(-1)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        x_pred_posi = mean_t * latents_fp32 + scale * r_pred_posi
        denom = (1.0 - t).clamp_min(float(self.jit_t_eps)).to(device=latents.device, dtype=torch.float32)
        while denom.ndim < latents.ndim:
            denom = denom.unsqueeze(-1)
        v_pred_posi = (x_pred_posi - latents_fp32) / denom
        if cfg_scale == 1.0:
            return v_pred_posi

        r_pred_nega = self.model_fn(**models, **model_inputs, **inputs_nega, timestep=timestep)
        r_pred_nega = r_pred_nega.to(device=latents.device, dtype=torch.float32)
        x_pred_nega = mean_t * latents_fp32 + scale * r_pred_nega
        v_pred_nega = (x_pred_nega - latents_fp32) / denom
        low = float(self.jit_cfg_interval_min)
        high = float(self.jit_cfg_interval_max)
        apply_cfg = bool(float(t.flatten()[0]) < high and (low == 0.0 or float(t.flatten()[0]) > low))
        if not apply_cfg:
            return v_pred_posi
        return v_pred_nega + cfg_scale * (v_pred_posi - v_pred_nega)

    def _jit_euler_step(self, models, inputs_shared, inputs_posi, inputs_nega, latents, t, t_next, cfg_scale):
        v_pred = self._jit_cfg_guided_velocity(models, inputs_shared, inputs_posi, inputs_nega, latents, t, cfg_scale)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        delta = (t_next - t).to(device=latents.device, dtype=torch.float32)
        return latents_fp32 + delta * v_pred

    def _jit_heun_step(self, models, inputs_shared, inputs_posi, inputs_nega, latents, t, t_next, cfg_scale):
        v_pred_t = self._jit_cfg_guided_velocity(models, inputs_shared, inputs_posi, inputs_nega, latents, t, cfg_scale)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        delta = (t_next - t).to(device=latents.device, dtype=torch.float32)
        latents_euler = latents_fp32 + delta * v_pred_t
        v_pred_t_next = self._jit_cfg_guided_velocity(
            models, inputs_shared, inputs_posi, inputs_nega, latents_euler, t_next, cfg_scale
        )
        return latents_fp32 + delta * 0.5 * (v_pred_t + v_pred_t_next)

    def _bridge_euler_step(self, models, inputs_shared, inputs_posi, inputs_nega, latents, t, t_next, cfg_scale):
        v_pred = self._bridge_cfg_guided_velocity(models, inputs_shared, inputs_posi, inputs_nega, latents, t, cfg_scale)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        delta = (t_next - t).to(device=latents.device, dtype=torch.float32)
        return latents_fp32 + delta * v_pred

    def _bridge_heun_step(self, models, inputs_shared, inputs_posi, inputs_nega, latents, t, t_next, cfg_scale):
        v_pred_t = self._bridge_cfg_guided_velocity(models, inputs_shared, inputs_posi, inputs_nega, latents, t, cfg_scale)
        latents_fp32 = latents.to(device=latents.device, dtype=torch.float32)
        delta = (t_next - t).to(device=latents.device, dtype=torch.float32)
        latents_euler = latents_fp32 + delta * v_pred_t
        v_pred_t_next = self._bridge_cfg_guided_velocity(
            models, inputs_shared, inputs_posi, inputs_nega, latents_euler, t_next, cfg_scale
        )
        return latents_fp32 + delta * 0.5 * (v_pred_t + v_pred_t_next)

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
        if pipe.vae is None:
            pipe.vae = model_pool.fetch_model("qwen_image_vae")
        if pipe.vae is None and pipe.dit is not None:
            dit_latent_channels = int(pipe.dit.latent_channels)
            dit_downsample = int(getattr(pipe.dit, "latent_downsample_factor", 16))
            if dit_latent_channels in (3, 4) and dit_downsample == 1:
                pipe.vae = PixelIdentityVAE(image_channels=dit_latent_channels).to(device=device, dtype=torch.float32)
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
        if pipe.vae is not None and pipe.dit is not None:
            latent_channels = infer_complextro_vae_latent_channels(pipe.vae)
            dit_latent_channels = int(pipe.dit.latent_channels)
            if latent_channels is not None and int(latent_channels) != dit_latent_channels:
                raise ValueError(
                    f"Selected VAE latent channels ({latent_channels}) do not match ComplextroImageDiT in_channels "
                    f"({dit_latent_channels})."
                )
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
        jit_sampling_method: Optional[str] = None,
        jit_cfg_interval_min: Optional[float] = None,
        jit_cfg_interval_max: Optional[float] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        progress_bar_cmd=tqdm,
    ):
        if self.prediction_type in ("jit_xpred", "bridge_xpred"):
            self.scheduler.training = False
            t_start = 1.0 - float(denoising_strength)
            t_start = max(0.0, min(1.0, t_start))
            self._jit_t_start = t_start
            effective_sampling_method = self.jit_sampling_method if jit_sampling_method is None else str(jit_sampling_method)
            effective_cfg_interval_min = self.jit_cfg_interval_min if jit_cfg_interval_min is None else float(jit_cfg_interval_min)
            effective_cfg_interval_max = self.jit_cfg_interval_max if jit_cfg_interval_max is None else float(jit_cfg_interval_max)
        else:
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                dynamic_shift_len=(
                    (height // self._get_vae_token_downsample_factor())
                    * (width // self._get_vae_token_downsample_factor())
                ),
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
        if self.prediction_type in ("jit_xpred", "bridge_xpred"):
            old_cfg_min = self.jit_cfg_interval_min
            old_cfg_max = self.jit_cfg_interval_max
            try:
                if input_image is None:
                    inputs_shared["latents"] = (inputs_shared["latents"] * float(self.jit_noise_scale)).to(torch.float32)
                else:
                    inputs_shared["latents"] = inputs_shared["latents"].to(torch.float32)
                timesteps = torch.linspace(
                    float(self._jit_t_start), 1.0, int(num_inference_steps) + 1, device=self.device, dtype=torch.float32
                )
                step_method = str(effective_sampling_method).lower()
                self.jit_cfg_interval_min = float(effective_cfg_interval_min)
                self.jit_cfg_interval_max = float(effective_cfg_interval_max)
                for progress_id, _ in enumerate(progress_bar_cmd(timesteps[:-1])):
                    t = timesteps[progress_id].view(1)
                    t_next = timesteps[progress_id + 1].view(1)
                    if self.prediction_type == "jit_xpred":
                        euler_step = self._jit_euler_step
                        heun_step = self._jit_heun_step
                    else:
                        euler_step = self._bridge_euler_step
                        heun_step = self._bridge_heun_step
                    if step_method == "euler" or progress_id + 1 >= int(num_inference_steps):
                        inputs_shared["latents"] = euler_step(
                            models, inputs_shared, inputs_posi, inputs_nega, inputs_shared["latents"], t, t_next, cfg_scale
                        )
                    elif step_method == "heun":
                        inputs_shared["latents"] = heun_step(
                            models, inputs_shared, inputs_posi, inputs_nega, inputs_shared["latents"], t, t_next, cfg_scale
                        )
                    else:
                        raise ValueError(f"Unsupported JiT sampling method: {effective_sampling_method!r}. Expected 'euler' or 'heun'.")
            finally:
                self.jit_cfg_interval_min = old_cfg_min
                self.jit_cfg_interval_max = old_cfg_max
        else:
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
        image = self.vae.decode(inputs_shared["latents"].to(dtype=self.torch_dtype))
        if image.shape[0] == 1:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])
        if hasattr(self, "_jit_t_start"):
            delattr(self, "_jit_t_start")
        return image


class ComplextroUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: ComplextroPipeline, height, width):
        token_factor = max(1, int(pipe._get_vae_token_downsample_factor()))
        old_h_div = pipe.height_division_factor
        old_w_div = pipe.width_division_factor
        try:
            pipe.height_division_factor = max(int(old_h_div), token_factor)
            pipe.width_division_factor = max(int(old_w_div), token_factor)
            height, width = pipe.check_resize_height_width(height, width)
        finally:
            pipe.height_division_factor = old_h_div
            pipe.width_division_factor = old_w_div
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
        segment_images = []
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
                segment_images.append(local_images)
                segment_owner.append(owner_id)

        if len(prompts) == 0:
            return {"prompt_emb": torch.empty(0, device=pipe.device), "prompt_emb_mask": torch.empty(0, device=pipe.device, dtype=torch.long)}

        if has_any_image and pipe.processor is None:
            raise ValueError("Image prompts require an AutoProcessor; tokenizer-only mode cannot encode images.")
        if not hasattr(template_source, "apply_chat_template"):
            raise ValueError("Selected tokenizer/processor does not support apply_chat_template.")

        processor_kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 1024,
        }
        template_kwargs = {"add_generation_prompt": True}
        signature = inspect.signature(template_source.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            template_kwargs["enable_thinking"] = False
        if "processor_kwargs" in signature.parameters:
            template_kwargs["processor_kwargs"] = processor_kwargs
        else:
            template_kwargs.update(processor_kwargs)

        model_inputs = template_source.apply_chat_template(conversations, **template_kwargs)
        if isinstance(model_inputs, list):
            if pipe.processor is None:
                raise ValueError("apply_chat_template returned a list; an AutoProcessor is required to convert it to tensors.")
            if has_any_image:
                processed_inputs = []
                for text_item, images_item in zip(model_inputs, segment_images):
                    processed_inputs.append(
                        pipe.processor(
                            text=[text_item],
                            images=images_item,
                            padding="max_length",
                            truncation=True,
                            max_length=1024,
                            return_tensors="pt",
                        )
                    )
                merged_data = {}
                feature_keys = set()
                for item in processed_inputs:
                    feature_keys.update(item.keys())
                for key in feature_keys:
                    values = [item[key] for item in processed_inputs if key in item]
                    if len(values) == 0:
                        continue
                    first_value = values[0]
                    if torch.is_tensor(first_value):
                        merged_data[key] = torch.cat(values, dim=0)
                    elif isinstance(first_value, list):
                        merged = []
                        for value in values:
                            merged.extend(value)
                        merged_data[key] = merged
                    else:
                        merged_data[key] = values
                model_inputs = BatchFeature(data=merged_data)
            else:
                model_inputs = pipe.processor(
                    text=model_inputs,
                    images=None,
                    padding="max_length",
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
        model_inputs = model_inputs.to(pipe.device)

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

        target_seq_len = max(1, min(1024, max(int(emb_item.shape[0]) for emb_item in merged_emb)))
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
        latent_channels = int(pipe.dit.latent_channels) if pipe.dit is not None else 128
        downsample_factor = pipe._get_vae_downsample_factor()
        noise_dtype = torch.float32 if isinstance(pipe.vae, (PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE)) else pipe.torch_dtype
        noise = pipe.generate_noise(
            (int(batch_size), latent_channels, height // downsample_factor, width // downsample_factor),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=noise_dtype,
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
        image_dtype = torch.float32 if isinstance(pipe.vae, (PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE)) else pipe.torch_dtype
        image = image.to(device=pipe.device, dtype=image_dtype)
        input_latents = pipe.vae.encode(image)
        if pipe.prediction_type in ("jit_xpred", "bridge_xpred") and not pipe.scheduler.training:
            t_start = float(getattr(pipe, "_jit_t_start", 0.0))
            latents = t_start * input_latents.float() + (1.0 - t_start) * noise.float() * float(pipe.jit_noise_scale)
            return {"latents": latents, "input_latents": input_latents}
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
        token_factor = max(1, int(pipe._get_vae_token_downsample_factor()))
        operator = ImageCropAndResize(max_pixels=1024 * 1024, height_division_factor=token_factor, width_division_factor=token_factor)
        if isinstance(edit_image, list) and len(edit_image) > 0 and isinstance(edit_image[0], list):
            edit_image = [[operator(pipe._load_image(img)) for img in image_group] for image_group in edit_image]
        elif isinstance(edit_image, list):
            edit_image = [operator(pipe._load_image(img)) for img in edit_image]
        else:
            edit_image = operator(pipe._load_image(edit_image))
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
        token_factor = max(1, int(pipe._get_vae_token_downsample_factor()))
        resize_operator = ImageCropAndResize(max_pixels=1024 * 1024, height_division_factor=token_factor, width_division_factor=token_factor)

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
                image_dtype = (
                    torch.float32
                    if isinstance(pipe.vae, (PixelIdentityVAE, PixelLogitVAE, PixelNormalizedVAE))
                    else pipe.torch_dtype
                )
                image_tensor = pipe.preprocess_image(latent_image).to(device=pipe.device, dtype=image_dtype)
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
    # Timestep convention: both training (JiTXPredLoss / FlowMatchSFTLoss) and
    # inference (_jit_cfg_guided_velocity / scheduler) pass timestep in the
    # *original* scale (e.g. [0, 1] for x-pred, [0, 1000] for flow-match).
    # Dividing by 1000 here is intentional — it is compensated inside
    # ComplextroImageDiT.time_text_embed which uses TimestepEmbeddings(scale=1000),
    # so the sinusoidal embedding ultimately sees the correct value.
    timestep_model = timestep / 1000

    # Ensure latents dtype matches DiT parameters (e.g. PixelIdentityVAE
    # produces float32 latents but DiT weights are typically bfloat16).
    model_dtype = next(dit.parameters()).dtype
    if latents is not None and not isinstance(latents, list) and latents.dtype != model_dtype:
        latents = latents.to(dtype=model_dtype)

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
            timestep=timestep_model,
            prompt_emb=prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            image_noise_mask=mask,
            edit_latent_mask=latent_keep_mask if keep_groups is not None else None,
            siglip_feats=siglip_arg,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return model_output

    siglip_arg = None
    if image_embeds is not None:
        siglip_arg = image_embeds
        if isinstance(siglip_arg, list):
            siglip_arg = torch.stack(siglip_arg, dim=0)

    model_output = dit(
        latents=latents,
        timestep=timestep_model,
        prompt_emb=prompt_emb,
        prompt_emb_mask=prompt_emb_mask,
        siglip_feats=siglip_arg,
        image_noise_mask=None,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return model_output
