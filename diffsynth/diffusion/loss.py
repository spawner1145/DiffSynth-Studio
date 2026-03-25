from .base_pipeline import BasePipeline
import torch
import torch.nn.functional as F


_JPEG_LUMA_QTABLE_8 = torch.tensor(
    [
        [16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0],
        [12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0],
        [14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0],
        [14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0],
        [18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0],
        [24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0],
        [49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0],
        [72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0],
    ],
    dtype=torch.float32,
)


def _normalized_time_value(timestep: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if timestep is None:
        return None
    t = timestep.detach().to(device=device, dtype=torch.float32).flatten()
    if t.numel() == 0:
        return None
    if float(t.max()) > 1.5:
        t = t / 1000.0
    return t.clamp_(0.0, 1.0)


def _get_freq_loss_config(pipe: BasePipeline, inputs: dict) -> dict:
    return {
        "enabled": bool(inputs.get("freq_loss_enabled", getattr(pipe, "freq_loss_enabled", False))),
        "weight": float(inputs.get("freq_loss_weight", getattr(pipe, "freq_loss_weight", 0.0))),
        "mode": str(inputs.get("freq_loss_mode", getattr(pipe, "freq_loss_mode", "dct"))),
        "block_size": int(inputs.get("freq_loss_block_size", getattr(pipe, "freq_loss_block_size", 8))),
        "profile": str(inputs.get("freq_loss_profile", getattr(pipe, "freq_loss_profile", "jpeg"))),
        "quality": int(inputs.get("freq_loss_quality", getattr(pipe, "freq_loss_quality", 85))),
        "jpeg_mode": str(inputs.get("freq_loss_jpeg_mode", getattr(pipe, "freq_loss_jpeg_mode", "inv_gamma"))),
        "gamma": float(inputs.get("freq_loss_gamma", getattr(pipe, "freq_loss_gamma", 1.0))),
        "color_space": str(inputs.get("freq_loss_color_space", getattr(pipe, "freq_loss_color_space", "rgb"))),
        "weight_floor": float(inputs.get("freq_loss_weight_floor", getattr(pipe, "freq_loss_weight_floor", 0.1))),
        "hf_scale": float(inputs.get("freq_loss_hf_scale", getattr(pipe, "freq_loss_hf_scale", 0.25))),
        "lf_scale": float(inputs.get("freq_loss_lf_scale", getattr(pipe, "freq_loss_lf_scale", 1.0))),
        "t_adaptive": bool(inputs.get("freq_loss_t_adaptive", getattr(pipe, "freq_loss_t_adaptive", False))),
        "t_min_hf_scale": float(inputs.get("freq_loss_t_min_hf_scale", getattr(pipe, "freq_loss_t_min_hf_scale", 0.25))),
        "t_max_hf_scale": float(inputs.get("freq_loss_t_max_hf_scale", getattr(pipe, "freq_loss_t_max_hf_scale", 1.0))),
        "t_gamma": float(inputs.get("freq_loss_t_gamma", getattr(pipe, "freq_loss_t_gamma", 1.0))),
    }


def _make_base_frequency_weight(
    block_size: int,
    profile: str,
    quality: int,
    jpeg_mode: str,
    gamma: float,
    color_space: str,
    weight_floor: float,
    lf_scale: float,
    hf_scale: float,
    device: torch.device,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError(f"freq_loss_block_size must be positive, got {block_size}.")
    profile = profile.lower()
    if profile == "jpeg":
        q = max(1, min(100, int(quality)))
        if q < 50:
            scale = 5000.0 / q
        else:
            scale = 200.0 - 2.0 * q
        lum_q = torch.floor((_JPEG_LUMA_QTABLE_8 * scale + 50.0) / 100.0).clamp(1.0, 255.0)
        chr_q = torch.tensor(
            [
                [17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0],
                [18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0],
                [24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                [47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            ],
            dtype=torch.float32,
        )
        chr_q = torch.floor((chr_q * scale + 50.0) / 100.0).clamp(1.0, 255.0)

        def q_to_weight(qtable: torch.Tensor) -> torch.Tensor:
            if jpeg_mode == "inv":
                w = 1.0 / qtable.clamp_min(1e-6)
            elif jpeg_mode == "inv_gamma":
                w = (qtable.mean() / qtable.clamp_min(1e-6)) ** gamma
            else:
                raise ValueError("freq_loss_jpeg_mode must be 'inv' or 'inv_gamma'.")
            return w / w.mean().clamp_min(1e-6)

        color_space = color_space.lower()
        if color_space == "ycbcr":
            weight_y = F.interpolate(q_to_weight(lum_q).view(1, 1, 8, 8), size=(block_size, block_size), mode="bilinear", align_corners=False).view(block_size, block_size)
            weight_c = F.interpolate(q_to_weight(chr_q).view(1, 1, 8, 8), size=(block_size, block_size), mode="bilinear", align_corners=False).view(block_size, block_size)
            weight = torch.stack([weight_y, weight_c, weight_c], dim=0)
        else:
            weight = F.interpolate(q_to_weight(lum_q).view(1, 1, 8, 8), size=(block_size, block_size), mode="bilinear", align_corners=False).view(block_size, block_size)
    elif profile == "linear":
        coords = torch.arange(block_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx.pow(2) + yy.pow(2))
        weight = 1.0 - radius / radius.max().clamp_min(1.0)
    elif profile == "uniform":
        weight = torch.ones(block_size, block_size, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported freq_loss_profile: {profile!r}. Expected 'jpeg', 'linear', or 'uniform'.")

    weight = weight_floor + (1.0 - weight_floor) * weight
    weight = hf_scale + (lf_scale - hf_scale) * weight
    return weight.to(device=device, dtype=torch.float32)


def _make_dct_basis(block_size: int, device: torch.device) -> torch.Tensor:
    n = torch.arange(block_size, device=device, dtype=torch.float32)
    k = n.view(-1, 1)
    basis = torch.cos(torch.pi * (n + 0.5) * k / float(block_size))
    basis[0] = basis[0] * (1.0 / float(block_size)) ** 0.5
    if block_size > 1:
        basis[1:] = basis[1:] * (2.0 / float(block_size)) ** 0.5
    return basis


def _block_dct2(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.matmul(basis, torch.matmul(x, basis.transpose(-1, -2)))


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x
    rgb = x[:, :3]
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    transformed = torch.cat([y, cb, cr], dim=1)
    if x.shape[1] == 3:
        return transformed
    return torch.cat([transformed, x[:, 3:]], dim=1)


def _frequency_residual_loss(
    residual: torch.Tensor,
    timestep: torch.Tensor | None,
    pipe: BasePipeline,
    inputs: dict,
) -> torch.Tensor | None:
    cfg = _get_freq_loss_config(pipe, inputs)
    if not cfg["enabled"] or cfg["weight"] <= 0.0:
        return None
    if cfg["mode"].lower() != "dct":
        raise ValueError(f"Unsupported freq_loss_mode: {cfg['mode']!r}. Expected 'dct'.")
    if residual.ndim < 4:
        return None

    block_size = int(cfg["block_size"])
    base_weight = _make_base_frequency_weight(
        block_size=block_size,
        profile=cfg["profile"],
        quality=cfg["quality"],
        jpeg_mode=cfg["jpeg_mode"],
        gamma=cfg["gamma"],
        color_space=cfg["color_space"],
        weight_floor=cfg["weight_floor"],
        lf_scale=cfg["lf_scale"],
        hf_scale=cfg["hf_scale"],
        device=residual.device,
    )
    if cfg["t_adaptive"]:
        t = _normalized_time_value(timestep, residual.device)
        if t is not None:
            alpha = t.pow(float(cfg["t_gamma"]))
            hf_scale_t = float(cfg["t_min_hf_scale"]) + (
                float(cfg["t_max_hf_scale"]) - float(cfg["t_min_hf_scale"])
            ) * alpha
            hf_scale_t = hf_scale_t.view(-1, 1, 1)
            denom = max(float(cfg["lf_scale"]) - float(cfg["hf_scale"]), 1e-6)
            normalized = (base_weight - float(cfg["hf_scale"])) / denom
            base_weight = hf_scale_t + (float(cfg["lf_scale"]) - hf_scale_t) * normalized.unsqueeze(0)
        else:
            base_weight = base_weight.unsqueeze(0)
    else:
        base_weight = base_weight.unsqueeze(0)

    if cfg["color_space"].lower() == "ycbcr":
        residual = _rgb_to_ycbcr(residual)

    b, c, h, w = residual.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    residual = residual.float()
    if pad_h > 0 or pad_w > 0:
        residual = F.pad(residual, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, hp, wp = residual.shape
    patches = residual.view(b, c, hp // block_size, block_size, wp // block_size, block_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    basis = _make_dct_basis(block_size, residual.device)
    coeff = _block_dct2(patches, basis)

    weight = base_weight.to(device=residual.device, dtype=coeff.dtype)
    if weight.ndim == 2:
        weight = weight.view(1, 1, 1, 1, block_size, block_size)
    elif weight.ndim == 3:
        weight = weight.view(1, weight.shape[0], 1, 1, block_size, block_size)
    elif weight.ndim == 4:
        # [B, bs, bs] or [1/ B, C, bs, bs] are both allowed.
        if weight.shape[-2:] != (block_size, block_size):
            raise ValueError("Frequency weight spatial shape must match the configured DCT block size.")
        if weight.shape[1] == block_size and weight.shape[2] == block_size:
            weight = weight.view(weight.shape[0], 1, 1, 1, block_size, block_size)
        else:
            weight = weight.view(weight.shape[0], weight.shape[1], 1, 1, block_size, block_size)
    elif weight.ndim == 5:
        weight = weight.unsqueeze(2)
    elif weight.ndim != 6:
        raise ValueError(f"Unsupported frequency weight rank: {weight.ndim}.")

    if weight.shape[0] == 1 and coeff.shape[0] > 1:
        expand_shape = (coeff.shape[0],) + tuple(weight.shape[1:])
        weight = weight.expand(*expand_shape)
    if weight.shape[1] == 1 and coeff.shape[1] > 1:
        expand_shape = (weight.shape[0], coeff.shape[1]) + tuple(weight.shape[2:])
        weight = weight.expand(*expand_shape)
    elif coeff.shape[1] > weight.shape[1]:
        pad_channels = coeff.shape[1] - weight.shape[1]
        tail = weight[:, -1:].expand(weight.shape[0], pad_channels, weight.shape[2], weight.shape[3], weight.shape[4], weight.shape[5])
        weight = torch.cat([weight, tail], dim=1)
    return (coeff.pow(2) * weight).mean()


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    batch_size = inputs["input_latents"].shape[0]
    timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (batch_size,))
    timesteps = pipe.scheduler.timesteps[timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timesteps)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timesteps)

    inputs["latents"] = inputs["latents"].to(dtype=pipe.torch_dtype)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timesteps)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    residual = noise_pred.float() - training_target.float()
    per_sample_mse = residual.pow(2).flatten(1).mean(1)
    weights = pipe.scheduler.training_weight(timesteps).to(device=per_sample_mse.device, dtype=per_sample_mse.dtype)
    base_loss = (per_sample_mse * weights).mean()
    loss = base_loss
    freq_loss = _frequency_residual_loss(residual, timesteps, pipe, inputs)
    aux_loss = None
    if freq_loss is not None:
        aux_weight = float(inputs.get("freq_loss_weight", getattr(pipe, "freq_loss_weight", 0.0)))
        aux_loss = aux_weight * freq_loss
        loss = loss + aux_loss
    pipe._last_loss_metrics = {
        "base_loss": float(base_loss.detach().float().item()),
        "total_loss": float(loss.detach().float().item()),
    }
    if aux_loss is not None:
        pipe._last_loss_metrics["freq_aux_loss"] = float(aux_loss.detach().float().item())
        pipe._last_loss_metrics["freq_aux_unweighted"] = float(freq_loss.detach().float().item())
    return loss


def JiTXPredLoss(pipe: BasePipeline, **inputs):
    """JiT x-pred loss
    三种权重策略
      - "velocity"  : ||x - x_pred||² / (1-t)²   — 原版jit
      - "balanced"  : ||x - x_pred||² / (1-t)    — 部分修正
      - "x_pred"    : ||x - x_pred||²            — 直接 x 预测 MSE
    """
    x = inputs["input_latents"]
    batch_size = x.shape[0]
    p_mean = float(inputs.get("jit_p_mean", getattr(pipe, "jit_p_mean", -0.8)))
    p_std = float(inputs.get("jit_p_std", getattr(pipe, "jit_p_std", 0.8)))
    noise_scale = float(inputs.get("jit_noise_scale", getattr(pipe, "jit_noise_scale", 1.0)))
    t_eps = float(inputs.get("jit_t_eps", getattr(pipe, "jit_t_eps", 5e-2)))

    t = torch.randn(batch_size, device=x.device, dtype=torch.float32) * p_std + p_mean
    t = torch.sigmoid(t).view(batch_size, *([1] * (x.ndim - 1)))
    x_fp32 = x.float()
    noise = torch.randn_like(x_fp32) * noise_scale
    latents = t * x_fp32 + (1 - t) * noise

    inputs["latents"] = latents.to(dtype=pipe.torch_dtype)
    timestep = t.flatten()
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    x_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss_weighting = str(inputs.get("jit_loss_weighting", getattr(pipe, "jit_loss_weighting", "x_pred")))
    t_flat = t.flatten()
    one_minus_t = (1.0 - t).clamp_min(t_eps)

    residual = x_fp32 - x_pred.float()
    if loss_weighting == "x_pred":
        # 直接 x 预测 MSE: ||x - x_pred||²
        # 所有 t 值贡献相同
        # 高噪（低 t）会获得公平的梯度信号
        per_sample_loss = residual.pow(2).flatten(1).mean(1)
        base_loss = per_sample_loss.mean()
    elif loss_weighting == "balanced":
        # 部分修正下的速度损失: effective = ||x - x_pred||² / (1-t)
        # 高t(低噪)仍然比低t(高噪)获得大约10倍的重量（相比原始速度下的100倍）
        v_target = (x_fp32 - latents) / one_minus_t
        v_pred = (x_pred.float() - latents) / one_minus_t
        residual = v_target - v_pred
        per_sample_loss = residual.pow(2).flatten(1).mean(1)
        weight = one_minus_t.flatten()
        base_loss = (per_sample_loss * weight).mean()
    else:
        # "velocity": 原版jit = ||x - x_pred||² / (1-t)²
        v_target = (x_fp32 - latents) / one_minus_t
        v_pred = (x_pred.float() - latents) / one_minus_t
        residual = v_target - v_pred
        per_sample_loss = residual.pow(2).flatten(1).mean(1)
        base_loss = per_sample_loss.mean()
    loss = base_loss
    freq_loss = _frequency_residual_loss(residual, t_flat, pipe, inputs)
    aux_loss = None
    if freq_loss is not None:
        aux_weight = float(inputs.get("freq_loss_weight", getattr(pipe, "freq_loss_weight", 0.0)))
        aux_loss = aux_weight * freq_loss
        loss = loss + aux_loss
    pipe._last_loss_metrics = {
        "base_loss": float(base_loss.detach().float().item()),
        "total_loss": float(loss.detach().float().item()),
    }
    if aux_loss is not None:
        pipe._last_loss_metrics["freq_aux_loss"] = float(aux_loss.detach().float().item())
        pipe._last_loss_metrics["freq_aux_unweighted"] = float(freq_loss.detach().float().item())
    return loss


def BridgeXPredLoss(pipe: BasePipeline, **inputs):
    x = inputs["input_latents"]
    batch_size = x.shape[0]
    p_mean = float(inputs.get("jit_p_mean", getattr(pipe, "jit_p_mean", -0.8)))
    p_std = float(inputs.get("jit_p_std", getattr(pipe, "jit_p_std", 0.8)))
    noise_scale = float(inputs.get("jit_noise_scale", getattr(pipe, "jit_noise_scale", 1.0)))
    t_eps = float(inputs.get("jit_t_eps", getattr(pipe, "jit_t_eps", 5e-2)))

    t = torch.randn(batch_size, device=x.device, dtype=torch.float32) * p_std + p_mean
    t = torch.sigmoid(t).view(batch_size, *([1] * (x.ndim - 1)))
    x_fp32 = x.float()
    noise = torch.randn_like(x_fp32) * noise_scale
    latents = t * x_fp32 + (1 - t) * noise

    inputs["latents"] = latents.to(dtype=pipe.torch_dtype)
    timestep = t.flatten()
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    bridge_pred = pipe.model_fn(**models, **inputs, timestep=timestep).float()

    # 灵感可以参考这两篇：
    # Albergo et al., "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" (JMLR 2025)
    # Shaul et al., "Flow Map Matching with Stochastic Interpolants" (TMLR 2025)
    
    lambda2 = t.pow(2) + (1 - t).pow(2)
    mean_t = (t / lambda2.clamp_min(t_eps)) * latents
    scale = (1 - t) / lambda2.clamp_min(t_eps).sqrt()
    scale = scale.clamp_min(t_eps)
    bridge_target = (x_fp32 - mean_t) / scale
    residual = bridge_pred - bridge_target
    base_loss = residual.pow(2).flatten(1).mean(1).mean()
    loss = base_loss
    freq_loss = _frequency_residual_loss(residual, timestep, pipe, inputs)
    aux_loss = None
    if freq_loss is not None:
        aux_weight = float(inputs.get("freq_loss_weight", getattr(pipe, "freq_loss_weight", 0.0)))
        aux_loss = aux_weight * freq_loss
        loss = loss + aux_loss
    pipe._last_loss_metrics = {
        "base_loss": float(base_loss.detach().float().item()),
        "total_loss": float(loss.detach().float().item()),
    }
    if aux_loss is not None:
        pipe._last_loss_metrics["freq_aux_loss"] = float(aux_loss.detach().float().item())
        pipe._last_loss_metrics["freq_aux_unweighted"] = float(freq_loss.detach().float().item())
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
