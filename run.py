import torch, accelerate
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange, repeat

from transformers import AutoProcessor, AutoTokenizer
from diffsynth.core import ModelConfig, gradient_checkpoint_forward, attention_forward, UnifiedDataset, load_model
from diffsynth.diffusion import FlowMatchScheduler, DiffusionTrainingModule, FlowMatchSFTLoss, ModelLogger, launch_training_task
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit
from diffsynth.models.general_modules import TimestepEmbeddings
from diffsynth.models.z_image_text_encoder import ZImageTextEncoder
from diffsynth.utils.state_dict_converters.z_image_text_encoder import ZImageTextEncoderStateDictConverter
from diffsynth.models.flux2_vae import Flux2VAE


class AAAPositionalEmbedding(torch.nn.Module):
    def __init__(self, height=16, width=16, dim=1024):
        super().__init__()
        self.image_emb = torch.nn.Parameter(torch.randn((1, dim, height, width)))
        self.text_emb = torch.nn.Parameter(torch.randn((dim,)))

    def forward(self, image, text):
        height, width = image.shape[-2:]
        image_emb = self.image_emb.to(device=image.device, dtype=image.dtype)
        image_emb = torch.nn.functional.interpolate(image_emb, size=(height, width), mode="bilinear")
        image_emb = rearrange(image_emb, "B C H W -> B (H W) C")
        image_emb = repeat(image_emb, "1 S C -> B S C", B=text.shape[0])
        text_emb = self.text_emb.to(device=text.device, dtype=text.dtype)
        text_emb = repeat(text_emb, "C -> B L C", B=text.shape[0], L=text.shape[1])
        emb = torch.concat([image_emb, text_emb], dim=1)
        return emb


class AAABlock(torch.nn.Module):
    def __init__(self, dim=1024, num_heads=32):
        super().__init__()
        self.norm_attn = torch.nn.RMSNorm(dim, elementwise_affine=False)
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.Linear(dim, dim)
        self.norm_mlp = torch.nn.RMSNorm(dim, elementwise_affine=False)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*3),
            torch.nn.SiLU(),
            torch.nn.Linear(dim*3, dim),
        )
        self.to_gate = torch.nn.Linear(dim, dim * 2)
        self.num_heads = num_heads

    def attention(self, emb, pos_emb):
        emb = self.norm_attn(emb + pos_emb)
        q, k, v = self.to_q(emb), self.to_k(emb), self.to_v(emb)
        emb = attention_forward(
            q, k, v,
            q_pattern="b s (n d)", k_pattern="b s (n d)", v_pattern="b s (n d)", out_pattern="b s (n d)",
            dims={"n": self.num_heads},
        )
        emb = self.to_out(emb)
        return emb
    
    def feed_forward(self, emb, pos_emb):
        emb = self.norm_mlp(emb + pos_emb)
        emb = self.ff(emb)
        return emb
    
    def forward(self, emb, pos_emb, t_emb):
        gate_attn, gate_mlp = self.to_gate(t_emb).chunk(2, dim=-1)
        emb = emb + self.attention(emb, pos_emb) * (1 + gate_attn)
        emb = emb + self.feed_forward(emb, pos_emb) * (1 + gate_mlp)
        return emb


class AAADiT(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.pos_embedder = AAAPositionalEmbedding(dim=dim)
        self.timestep_embedder = TimestepEmbeddings(256, dim)
        self.image_embedder = torch.nn.Sequential(torch.nn.Linear(128, dim), torch.nn.LayerNorm(dim))
        self.text_embedder = torch.nn.Sequential(torch.nn.Linear(1024, dim), torch.nn.LayerNorm(dim))
        self.blocks = torch.nn.ModuleList([AAABlock(dim) for _ in range(20)])
        self.proj_out = torch.nn.Linear(dim, 128)

    def forward(
        self,
        latents,
        prompt_embeds,
        timestep,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        pos_emb = self.pos_embedder(latents, prompt_embeds)
        t_emb = self.timestep_embedder(timestep, dtype=latents.dtype)
        if t_emb.ndim == 1:
            t_emb = t_emb.unsqueeze(0)
        t_emb = t_emb.unsqueeze(1)
        image = self.image_embedder(rearrange(latents, "B C H W -> B (H W) C"))
        text = self.text_embedder(prompt_embeds)
        emb = torch.concat([image, text], dim=1)
        for block_id, block in enumerate(self.blocks):
            emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                emb=emb,
                pos_emb=pos_emb,
                t_emb=t_emb,
            )
        emb = emb[:, :latents.shape[-1] * latents.shape[-2]]
        emb = self.proj_out(emb)
        emb = rearrange(emb, "B (H W) C -> B C H W", W=latents.shape[-1])
        return emb


class AAAImagePipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: AAADiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            AAAUnit_PromptEmbedder(),
            AAAUnit_NoiseInitializer(),
            AAAUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_aaa
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = AAAImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("aaa_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=height//16*width//16)

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        if isinstance(input_image, (list, tuple)):
            batch_size = len(input_image)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "batch_size": batch_size,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class AAAUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )
        self.hidden_states_layers = (-1,)

    def process(self, pipe: AAAImagePipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)
        texts = [
            pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": text_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text_prompt in prompts
        ]
        inputs = pipe.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(pipe.device)
        output = pipe.text_encoder(**inputs, output_hidden_states=True, use_cache=False)
        prompt_embeds = torch.concat([output.hidden_states[k] for k in self.hidden_states_layers], dim=-1)
        return {"prompt_embeds": prompt_embeds}


class AAAUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "batch_size"),
            output_params=("noise",),
        )

    def process(self, pipe: AAAImagePipeline, height, width, seed, rand_device, batch_size=1):
        if batch_size is None:
            batch_size = 1
        batch_size = max(1, int(batch_size))
        noise = pipe.generate_noise((batch_size, 128, height//16, width//16), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class AAAUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: AAAImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        if isinstance(input_image, (list, tuple)):
            image = torch.cat([pipe.preprocess_image(img) for img in input_image], dim=0)
        else:
            image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


def model_fn_aaa(
    dit: AAADiT,
    latents=None,
    prompt_embeds=None,
    timestep=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    model_output = dit(
        latents,
        prompt_embeds,
        timestep,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return model_output


class AAATrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        device,
        qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3/model.safetensors",
        flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors",
        qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3",
    ):
        super().__init__()
        self.pipe = AAAImagePipeline(device=device, torch_dtype=torch.bfloat16)
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
        self.pipe.dit = AAADiT().to(dtype=torch.bfloat16, device=device)
        self.pipe.freeze_except(["dit"])
        self.pipe.scheduler.set_timesteps(1000, training=True)

    def forward(self, data):
        prompt_value = data["prompt"]
        image_value = data["image"]
        if isinstance(prompt_value, str):
            batch_size = 1
        else:
            batch_size = len(prompt_value)

        if isinstance(image_value, (list, tuple)):
            first_image = image_value[0]
        else:
            first_image = image_value

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": image_value,
            "height": first_image.size[1],
            "width": first_image.size[0],
            "batch_size": batch_size,
            "cfg_scale": 1,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        return loss


if __name__ == "__main__":
    qwen_model_file="/root/autodl-tmp/DiffSynth-Studio/qwen3/model.safetensors"
    flux2_vae_file="/root/autodl-tmp/DiffSynth-Studio/diffusion_pytorch_model.safetensors"
    qwen_tokenizer_dir="/root/autodl-tmp/DiffSynth-Studio/qwen3"
    pipe = AAAImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.text_encoder = load_model(
        ZImageTextEncoder,
        qwen_model_file,
        config={"model_size": "0.6B"},
        torch_dtype=torch.bfloat16,
        device="cuda",
        state_dict_converter=ZImageTextEncoderStateDictConverter,
    )
    pipe.vae = load_model(
        Flux2VAE,
        flux2_vae_file,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir)
    pipe.vram_management_enabled = pipe.check_vram_management_state()
    pipe.dit = load_model(AAADiT, "/root/autodl-tmp/DiffSynth-Studio/models/AAA/v1/model-e3-s10059.safetensors", torch_dtype=torch.bfloat16, device="cuda")

    for seed, prompt in enumerate([
        "green, lizard, plant, Grass, Poison, seed on back, red eyes, smiling expression, short stout limbs, sharp claws",
        "orange, cream, lizard, Fire, flame on tail tip, large eyes, smiling expression, cream-colored belly patch, sharp claws",
        "蓝色，米色，棕色，乌龟，水系，龟壳，大眼睛，短四肢，卷曲尾巴",
    ]):
        image = pipe(
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=30,
            cfg_scale=10,
            seed=seed,
            height=256, width=256,
        )
        image.save(f"image_{seed}.jpg")