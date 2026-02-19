import torch, math
from typing_extensions import Literal


class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image", "LTX-2", "Qwen-Image-Lightning", "Optical"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
            "LTX-2": FlowMatchScheduler.set_timesteps_ltx2,
            "Qwen-Image-Lightning": FlowMatchScheduler.set_timesteps_qwen_image_lightning,
            "Optical": FlowMatchScheduler.set_timesteps_optical,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_flux(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.003/1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    @staticmethod
    def set_timesteps_qwen_image(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_qwen_image_lightning(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        base_shift = math.log(3)
        max_shift = math.log(3)
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len, base_shift=base_shift, max_shift=max_shift)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)
    
    @staticmethod
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        if dynamic_shift_len is None:
            # If you ask me why I set mu=0.8,
            # I can only say that it yields better training results.
            mu = 0.8
        else:
            mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None, target_timesteps=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ltx2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None):
        num_train_timesteps = 1000
        if special_case == "stage2":
            sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
        elif special_case == "ditilled_stage1":
            sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
        else:
            dynamic_shift_len = dynamic_shift_len or 4096
            sigma_shift = FlowMatchScheduler._calculate_shift_qwen_image(
                image_seq_len=dynamic_shift_len,
                base_seq_len=1024,
                max_seq_len=4096,
                base_shift=0.95,
                max_shift=2.05,
            )
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
            # Shift terminal
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1 - terminal)
            sigmas = 1.0 - (one_minus_z / scale_factor)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def _compute_refractive_index_profile(
        t: torch.Tensor,
        n_center: float = 1.5,
        n_edge: float = 1.0,
        profile: str = "sech",
        grin_constant: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute the refractive index n(t) for t in [0, 1], modeling the
        "information density" at each stage of the denoising process.

        In a GRIN (Gradient-Index) medium:
          - High n(t) => light travels slower => more "time" spent in that region
          - Low  n(t) => light travels faster => less "time" spent in that region

        Analogy to Flow Matching:
          - High n(t) => more sigma steps clustered here (critical denoising zone)
          - Low  n(t) => fewer sigma steps (easy/trivial denoising zone)

        Three profiles inspired by real GRIN lens designs:
          "sech"     : n(t) = n_edge + (n_center - n_edge) * sech(g * (t - 0.5))
                       Hyperbolic secant — the standard GRIN rod lens profile.
                       Peaks at center, smoothly decays. Focuses steps on mid-noise.

          "parabolic": n(t) = n_center - (n_center - n_edge) * (2t - 1)^2
                       Parabolic — the paraxial approximation of sech.
                       Simpler, nearly identical behavior for small g.

          "maxwell"  : n(t) = n_center / (1 + (g * (t - 0.5))^2)
                       Maxwell fish-eye lens profile.
                       Heavier tails than sech — keeps more steps near boundaries.
        """
        t_centered = t - 0.5

        if profile == "sech":
            raw = 1.0 / torch.cosh(grin_constant * t_centered)
            n_t = n_edge + (n_center - n_edge) * raw / raw.max()
        elif profile == "parabolic":
            n_t = n_center - (n_center - n_edge) * (2 * t_centered) ** 2
        elif profile == "maxwell":
            raw = 1.0 / (1.0 + (grin_constant * t_centered) ** 2)
            n_t = n_edge + (n_center - n_edge) * raw / raw.max()
        else:
            raise ValueError(f"Unknown GRIN profile: {profile}")

        return n_t

    @staticmethod
    def set_timesteps_optical(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        n_center: float = 1.5,
        n_edge: float = 1.0,
        grin_constant: float = 2.0,
        grin_profile: str = "maxwell",
        exponential_shift_mu: float = None,
        terminal: float = 0.0,
        dynamic_shift_len: int = None,
    ):
        """
        Optical-inspired Flow Matching scheduler based on GRIN (Gradient-Index) optics.
        === Physical Analogy ===
        In a Gradient-Index (GRIN) medium, the refractive index n(x) varies continuously.
        By Fermat's Principle, light follows the path of least optical path length:
            Optical Path Length = ∫ n(s) ds
        In regions of HIGH refractive index, light travels SLOWER and spends MORE TIME.
        In regions of LOW  refractive index, light travels FASTER and spends LESS TIME.
        This is exactly what a good sigma schedule should do:
            - Spend more steps where the velocity field is complex (mid-noise region)
            - Spend fewer steps where the velocity field is simple (pure noise / clean)
        === Algorithm ===
        1. Define a refractive index profile n(t) over t ∈ [0, 1]
           (inspired by real GRIN lens profiles: sech, parabolic, Maxwell fish-eye)
        2. Compute the cumulative optical path length (phase):
           Φ(t) = ∫₀ᵗ n(s) ds
        3. Distribute sigma values at equal optical-path-length intervals.
           This is the Eikonal equation's discrete analog: wavefronts (steps)
           are equally spaced in PHASE, not in coordinate distance.
           → Regions with high n(t) get denser sigma spacing.
           → Regions with low  n(t) get sparser sigma spacing.
        4. Optionally apply an exponential shift (dispersive medium propagation)
           and terminal correction.
        === Parameters ===
        num_inference_steps   : Number of denoising steps.
        denoising_strength    : For img2img; 1.0 = full denoising.
        n_center              : Refractive index at t=0.5 (mid-noise, highest info density).
                                Higher → more steps concentrated at mid-noise.
        n_edge                : Refractive index at t=0 and t=1 (boundaries).
        grin_constant         : Controls how sharply n(t) peaks. Like the gradient constant
                                'g' in a GRIN rod lens: n(r) = n₀ · sech(g·r).
                                Higher g → sharper focus of steps around t=0.5.
        grin_profile          : "sech" (standard GRIN), "parabolic", or "maxwell" (fish-eye).
        exponential_shift_mu  : Dispersive shift in logit-space: logit(σ') = logit(σ) + μ.
                                Positive μ → schedule biased toward high-noise (slower start).
                                Negative μ → schedule biased toward low-noise (faster start).
                                Analogous to chromatic dispersion: different "wavelengths"
                                (noise levels) experience different phase velocities.
                                None or 0.0 → no shift applied.
        terminal              : Minimum sigma at the last step (terminal correction).
        dynamic_shift_len     : If provided, automatically compute grin_constant from the
                                image sequence length, like adaptive-aperture optics.
        """
        num_train_timesteps = 1000
        sigma_min = 0.0
        sigma_max = 1.0
        # === Step 0: Adaptive optics ===
        if dynamic_shift_len is not None:
            base_seq_len = 256
            max_seq_len = 8192
            min_g = 1.0
            max_g = 4.0
            alpha = max(0.0, min(1.0, (dynamic_shift_len - base_seq_len) / (max_seq_len - base_seq_len)))
            grin_constant = max_g - alpha * (max_g - min_g)
            n_center = 1.2 + (1.8 - 1.2) * (1.0 - alpha)
        # === Step 1: Refractive index profile ===
        fine_steps = 10000
        t_fine = torch.linspace(0.0, 1.0, fine_steps)
        n_profile = FlowMatchScheduler._compute_refractive_index_profile(
            t_fine, n_center=n_center, n_edge=n_edge,
            profile=grin_profile, grin_constant=grin_constant,
        )
        # === Step 2: Cumulative optical path ===
        dt = t_fine[1] - t_fine[0]
        optical_path = torch.cumsum(n_profile * dt, dim=0)
        optical_path = optical_path / optical_path[-1]
        # === Step 3: Equal-phase distribution ===
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        target_phases = torch.linspace(0.0, 1.0, num_inference_steps + 1)[:-1]
        t_values = torch.zeros(num_inference_steps)
        for i, phase in enumerate(target_phases):
            idx = torch.searchsorted(optical_path, phase)
            idx = idx.clamp(0, fine_steps - 1)
            t_values[i] = t_fine[idx]
        sigmas = sigma_start * (1.0 - t_values) + sigma_min * t_values
        # === Step 4: Exponential shift===
        if exponential_shift_mu is not None and exponential_shift_mu != 0.0:
            mu = exponential_shift_mu
            # 避免 sigma=0 导致除零
            sigmas = sigmas.clamp(min=1e-8)
            sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
        # === Step 5: Terminal correction ===
        if terminal > 0.0:
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            sigmas = 1.0 - (one_minus_z / scale_factor)
        # === Step 6: Timesteps ===
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights