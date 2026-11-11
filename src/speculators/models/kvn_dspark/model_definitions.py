from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = [
    "BornConfidenceHead",
    "CausalKvNRFFLayer",
    "DataReuploadingLinear",
    "DyadicSymplecticInterferometer",
    "IntraBlockCausalMixer",
    "ModalDataReuploadingRFF",
    "MultiHorizonKoopmanRollout",
    "compute_vacuum_born_observables",
    "parallel_segmented_real_scan",
]


class DataReuploadingLinear(nn.Module):
    """Legacy dense multi-stage Fourier Data Re-uploading encoder (retained for v1 ablation)."""

    def __init__(self, d_model: int, reupload_layers_l: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.reupload_layers_l = reupload_layers_l
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
                for _ in range(reupload_layers_l)
            ]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(reupload_layers_l)]
        )
        self.phase_Mix = nn.ParameterList(
            [
                nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
                for _ in range(max(0, reupload_layers_l - 1))
            ]
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros_like(x)
        for l_idx in range(self.reupload_layers_l):
            proj = F.linear(x, self.weights[l_idx], self.biases[l_idx])
            h = h + proj
            if l_idx < self.reupload_layers_l - 1:
                h = torch.sin(h) + F.linear(torch.cos(h), self.phase_Mix[l_idx])
        return self.norm(h)


class ModalDataReuploadingRFF(nn.Module):
    """Dequantized Modal Data-Reuploading RFF Encoder (Barthe 2026, Thm. 2.1 & Cor. 3.2).

    Projects x in R^d into a spectral bottleneck of width d_re, applies normalized
    sin/cos activations across ALL L re-uploading stages, initializes frequencies with
    asymptotic Gaussian variance sigma_g^2 / (L * d_re), and routes learnable RFF
    frequencies omega_rff to AdamW for spectral alignment.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int,
        reupload_layers_l: int = 2,
        d_re: int = 512,
        sigma_g: float = 1.0,
        learnable_rff: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.reupload_layers_l = reupload_layers_l
        self.eff_d_re = d_re if (0 < d_re < d_model) else d_model
        self.learnable_rff = learnable_rff

        self.in_proj: nn.Module = (
            nn.Linear(d_model, self.eff_d_re, bias=False)
            if self.eff_d_re < d_model
            else nn.Identity()
        )
        self.stage_norms = nn.ModuleList(
            [nn.RMSNorm(self.eff_d_re) for _ in range(reupload_layers_l)]
        )
        self.reupload_x = nn.ModuleList(
            [
                nn.Linear(self.eff_d_re, self.eff_d_re, bias=True)
                for _ in range(reupload_layers_l)
            ]
        )
        self.reupload_h = nn.ModuleList(
            [
                nn.Linear(self.eff_d_re, self.eff_d_re, bias=False)
                for _ in range(max(0, reupload_layers_l - 1))
            ]
        )

        # Barthe Ch. 3 Cor. 3.2 (Eq. 3.28): L-fold convolution scales std as sigma_g / sqrt(L * d_re)
        rff_std = sigma_g / math.sqrt(max(1, reupload_layers_l * self.eff_d_re))
        omega_init = torch.randn(self.eff_d_re, n_modes) * rff_std
        phase_init = torch.rand(n_modes) * (2.0 * math.pi)
        if learnable_rff:
            self.omega_rff = nn.Parameter(omega_init)
            self.phase_rff = nn.Parameter(phase_init)
        else:
            self.register_buffer("omega_rff", omega_init)
            self.register_buffer("phase_rff", phase_init)

        self.raw_alpha = nn.Parameter(torch.zeros(n_modes))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_low = self.in_proj(x)
        h = torch.zeros_like(x_low)
        inv_sqrt2 = 0.7071067811865475
        for l_idx in range(self.reupload_layers_l):
            z = self.reupload_x[l_idx](x_low)
            if l_idx > 0:
                z = z + self.reupload_h[l_idx - 1](h)
            z = self.stage_norms[l_idx](z)
            h = h + inv_sqrt2 * (torch.sin(z) + torch.cos(z))

        proj = (h.float() @ self.omega_rff.float() + self.phase_rff.float()).to(x.dtype)
        cos_p = torch.cos(proj)
        sin_p = torch.sin(proj)
        alpha = torch.tanh(self.raw_alpha.to(x.dtype))
        scale = 1.0 / math.sqrt(max(1, self.n_modes))
        r_q = scale * (cos_p + alpha * (cos_p * cos_p - sin_p * sin_p))
        r_p = scale * (sin_p + 2.0 * alpha * (cos_p * sin_p))
        return r_q, r_p

    @torch.no_grad()
    def prune_to_indices(self, top_idx: Tensor) -> None:
        idx_dev = top_idx.to(self.omega_rff.device)
        if isinstance(self.omega_rff, nn.Parameter):
            self.omega_rff = nn.Parameter(self.omega_rff.data[:, idx_dev].clone())
            self.phase_rff = nn.Parameter(self.phase_rff.data[idx_dev].clone())
        else:
            self.omega_rff = self.omega_rff[:, idx_dev].clone()
            self.phase_rff = self.phase_rff[idx_dev].clone()
        self.raw_alpha = nn.Parameter(
            self.raw_alpha.data[top_idx.to(self.raw_alpha.device)].clone()
        )
        self.n_modes = int(top_idx.numel())


def parallel_segmented_real_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
) -> tuple[Tensor, Tensor]:
    """Work-parallel O(T log T) Kogge-Stone associative scan in real float32 quadratures.

    Computes z_t = (a_cos_t + i * a_sin_t) * z_{t-1} + (u_q_t + i * u_p_t) across
    arbitrary document-reset boundaries (where a_cos_t = a_sin_t = 0) in ceil(log2(T))
    parallel GPU steps, avoiding complex64 Inductor/FSDP pitfalls.
    """
    seq_len = u_q.size(1)
    c = a_cos
    s = a_sin
    q = u_q
    p = u_p
    stride = 1
    while stride < seq_len:
        q_prev = q[:, :-stride, :]
        p_prev = p[:, :-stride, :]
        c_prev = c[:, :-stride, :]
        s_prev = s[:, :-stride, :]

        c_curr = c[:, stride:, :]
        s_curr = s[:, stride:, :]
        q_curr = q[:, stride:, :]
        p_curr = p[:, stride:, :]

        q_upd = c_curr * q_prev - s_curr * p_prev + q_curr
        p_upd = s_curr * q_prev + c_curr * p_prev + p_curr
        c_upd = c_curr * c_prev - s_curr * s_prev
        s_upd = s_curr * c_prev + c_curr * s_prev

        q = torch.cat([q[:, :stride, :], q_upd], dim=1)
        p = torch.cat([p[:, :stride, :], p_upd], dim=1)
        c = torch.cat([c[:, :stride, :], c_upd], dim=1)
        s = torch.cat([s[:, :stride, :], s_upd], dim=1)
        stride <<= 1
    return q, p


class DyadicSymplecticInterferometer(nn.Module):
    """Pairwise passive Bosonic beamsplitter bank U in U(2)^{M/2} (Barthe Ch. 7 Prop. 7.2).

    Couples mode dyads (2g, 2g+1) in O(M) ops with exact local norm preservation,
    restoring Koopman phase interference in Born energies while keeping dyads
    independently prunable under Group-LASSO.
    """

    def __init__(self, n_modes: int) -> None:
        super().__init__()
        if n_modes % 2 != 0:
            raise ValueError(
                f"DyadicSymplecticInterferometer requires even n_modes, got {n_modes}."
            )
        self.n_modes = n_modes
        n_dyads = n_modes // 2
        self.dyad_theta = nn.Parameter(torch.full((n_dyads,), math.pi / 8.0))
        self.dyad_phase = nn.Parameter(torch.zeros(n_dyads))

    def forward(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        orig_shape = q.shape
        n_dyads = self.n_modes // 2
        q2 = q.reshape(*orig_shape[:-1], n_dyads, 2)
        p2 = p.reshape(*orig_shape[:-1], n_dyads, 2)

        q0, q1 = q2[..., 0], q2[..., 1]
        p0, p1 = p2[..., 0], p2[..., 1]

        c = torch.cos(self.dyad_theta.to(q.dtype))
        s = torch.sin(self.dyad_theta.to(q.dtype))
        cp = torch.cos(self.dyad_phase.to(q.dtype))
        sp = torch.sin(self.dyad_phase.to(q.dtype))

        rot_q1 = cp * q1 + sp * p1
        rot_p1 = cp * p1 - sp * q1
        rot_q0 = cp * q0 - sp * p0
        rot_p0 = cp * p0 + sp * q0

        out_q0 = c * q0 - s * rot_q1
        out_p0 = c * p0 - s * rot_p1
        out_q1 = s * rot_q0 + c * q1
        out_p1 = s * rot_p0 + c * p1

        q_out = torch.stack([out_q0, out_q1], dim=-1).reshape(orig_shape)
        p_out = torch.stack([out_p0, out_p1], dim=-1).reshape(orig_shape)
        return q_out, p_out

    @torch.no_grad()
    def prune_to_dyad_indices(self, dyad_idx: Tensor) -> None:
        idx_d = dyad_idx.to(self.dyad_theta.device)
        self.dyad_theta = nn.Parameter(self.dyad_theta.data[idx_d].clone())
        self.dyad_phase = nn.Parameter(self.dyad_phase.data[idx_d].clone())
        self.n_modes = int(dyad_idx.numel() * 2)


def compute_vacuum_born_observables(
    q: Tensor,
    p: Tensor,
    vacuum_log_kappa: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dyadic vacuum-mode normalized Born observables (Barthe Ch. 4 Thm. 4.2 Eq. 4.34).

    Uses M-invariant scaled energies E_hat_m = M * (q_m^2 + p_m^2) on each Bosonic
    dyad (2g, 2g+1) coupled to local vacuum mode |vac_g> with amplitude kappa_g > 0:
        p_{2g} + p_{2g+1} + p_{vac,g} = 1,   born_obs_m = 2 * p_m in [0, 2).
    """
    orig_shape = q.shape
    n_modes = orig_shape[-1]
    if n_modes % 2 != 0:
        raise ValueError(f"compute_vacuum_born_observables requires even M, got {n_modes}.")
    n_dyads = n_modes // 2

    e_scaled = float(n_modes) * (q.float().square() + p.float().square())
    e_dyad = e_scaled.reshape(*orig_shape[:-1], n_dyads, 2)
    e_sum = e_dyad.sum(dim=-1, keepdim=True)  # [..., M//2, 1]

    kappa_sq = torch.exp(2.0 * vacuum_log_kappa.float()).clamp(min=1e-4, max=1e4)
    if kappa_sq.numel() == n_dyads:
        view_shape = ([1] * (q.ndim - 1)) + [n_dyads, 1]
        kappa_sq = kappa_sq.view(*view_shape)
    else:
        kappa_sq = kappa_sq.view(*([1] * (q.ndim - 1)), 1, 1)

    denom = kappa_sq + e_sum
    born_obs = (2.0 * (e_dyad / denom)).reshape(orig_shape)
    p_vac = (kappa_sq / denom).squeeze(-1)
    return born_obs, p_vac


class CausalKvNRFFLayer(nn.Module):
    """Causal Dequantized Damped Bosonic RFF Layer with Dyadic Group-LASSO Pruning.

    Operates in O(T log T) parallel segmented Kogge-Stone scan during training (with
    exact state resets across multipacked document boundaries) and O(1) recurrent state
    (q_t, p_t) in R^{2M} during autoregressive/speculative inference.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int = 256,
        reupload_layers_l: int = 2,
        sigma_g: float = 1.0,
        d_re: int = 512,
        learnable_rff: bool = True,
        damping_min: float = 1e-3,
        damping_max: float = 2.0,
        enable_interferometer: bool = True,
        vacuum_born_norm: bool = True,
        ffn_mult: float = 1.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.reupload_layers_l = reupload_layers_l
        self.sigma_g = sigma_g
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.use_interferometer = enable_interferometer and (n_modes % 2 == 0)
        self.vacuum_born_norm = vacuum_born_norm and (n_modes % 2 == 0)

        self.norm1 = nn.RMSNorm(d_model)
        self.reupload = ModalDataReuploadingRFF(
            d_model=d_model,
            n_modes=n_modes,
            reupload_layers_l=reupload_layers_l,
            d_re=d_re,
            sigma_g=sigma_g,
            learnable_rff=learnable_rff,
        )

        self.dt_proj = nn.Linear(d_model, n_modes)
        nn.init.constant_(self.dt_proj.bias, -2.0)

        self.log_eta = nn.Parameter(torch.full((n_modes,), -2.5))
        self.kvn_omega = nn.Parameter(torch.randn(n_modes) * 0.5)
        self.in_gain_q = nn.Parameter(torch.ones(n_modes) * 0.5)
        self.in_gain_p = nn.Parameter(torch.ones(n_modes) * 0.5)

        self.interferometer: DyadicSymplecticInterferometer | None = (
            DyadicSymplecticInterferometer(n_modes) if self.use_interferometer else None
        )
        self.vacuum_log_kappa: nn.Parameter | None = (
            nn.Parameter(torch.zeros(n_modes // 2)) if self.vacuum_born_norm else None
        )

        self.readout_norm = nn.RMSNorm(d_model)
        self.readout_linear = nn.Linear(3 * n_modes, d_model)

        self.norm2 = nn.RMSNorm(d_model)
        ffn_hidden = max(64, int(round(d_model * ffn_mult)))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def _compute_readout(self, q_f: Tensor, p_f: Tensor, out_dtype: torch.dtype) -> Tensor:
        if self.interferometer is not None:
            q_obs_f, p_obs_f = self.interferometer(q_f, p_f)
        else:
            q_obs_f, p_obs_f = q_f, p_f

        if self.vacuum_born_norm and self.vacuum_log_kappa is not None:
            born_f, _ = compute_vacuum_born_observables(
                q_obs_f, p_obs_f, self.vacuum_log_kappa
            )
        else:
            born_f = (2.0 * self.n_modes) * (q_obs_f.square() + p_obs_f.square()) - 1.0

        obs = torch.cat(
            [q_obs_f.to(out_dtype), p_obs_f.to(out_dtype), born_f.to(out_dtype)],
            dim=-1,
        )
        return self.readout_norm(self.readout_linear(obs))

    def recurrent_step(
        self,
        x_step: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """O(1) Zero-KV-Cache damped Bosonic step for a single token x_step [B, 1, D]."""
        h_norm = self.norm1(x_step)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))

        with torch.amp.autocast(device_type=x_step.device.type, enabled=False):
            dt_f = dt.float()
            eta = (
                torch.exp(self.log_eta.float())
                .clamp(min=self.damping_min, max=self.damping_max)
                .view(1, 1, -1)
            )
            omega_pole = self.kvn_omega.float().view(1, 1, -1)
            dtheta_f = dt_f * omega_pole
            decay = torch.exp(-eta * dt_f)
            a_cos = decay * torch.cos(dtheta_f)
            a_sin = decay * torch.sin(dtheta_f)

            if state is None:
                q_prev = torch.zeros_like(dt_f)
                p_prev = torch.zeros_like(dt_f)
            else:
                q_prev = state[0].float()
                p_prev = state[1].float()

            if reset_mask is not None:
                keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
                q_prev = q_prev * keep
                p_prev = p_prev * keep

            u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
            u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()

            q_f = a_cos * q_prev - a_sin * p_prev + u_q_f
            p_f = a_sin * q_prev + a_cos * p_prev + u_p_f

        readout = self._compute_readout(q_f, p_f, x_step.dtype)
        q_t = q_f.to(x_step.dtype)
        p_t = p_f.to(x_step.dtype)
        dtheta = dtheta_f.to(x_step.dtype)

        x_out = x_step + readout
        x_out = x_out + self.ffn(self.norm2(x_out))
        return x_out, (q_t, p_t), dtheta

    def forward(
        self,
        x: Tensor,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (x_out [B, T, D], q_seq [B, T, M], p_seq [B, T, M], dtheta_seq [B, T, M])."""
        h_norm = self.norm1(x)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            dt_f = dt.float()
            eta = (
                torch.exp(self.log_eta.float())
                .clamp(min=self.damping_min, max=self.damping_max)
                .view(1, 1, -1)
            )
            omega_pole = self.kvn_omega.float().view(1, 1, -1)
            dtheta_f = dt_f * omega_pole
            decay = torch.exp(-eta * dt_f)
            a_cos = decay * torch.cos(dtheta_f)
            a_sin = decay * torch.sin(dtheta_f)

            if reset_mask is not None:
                keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
                a_cos = a_cos * keep
                a_sin = a_sin * keep

            u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
            u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()

            q_f, p_f = parallel_segmented_real_scan(a_cos, a_sin, u_q_f, u_p_f)

        readout = self._compute_readout(q_f, p_f, x.dtype)
        x_out = x + readout
        x_out = x_out + self.ffn(self.norm2(x_out))
        return x_out, q_f.to(x.dtype), p_f.to(x.dtype), dtheta_f.to(x.dtype)

    def mode_importance(self) -> Tensor:
        """Compute L2 group importance score of each harmonic mode m in [0 .. M-1]."""
        w = self.readout_linear.weight.float()
        m = self.n_modes
        imp_q = w[:, :m].square().mean(dim=0)
        imp_p = w[:, m : 2 * m].square().mean(dim=0)
        imp_b = w[:, 2 * m :].square().mean(dim=0)
        return torch.sqrt(imp_q + imp_p + imp_b + 1e-12)

    def lasso_penalty(self) -> Tensor:
        """Dyadic Group-LASSO (L_{2,1}) + L1 penalty over mode/dyad readouts."""
        imp = self.mode_importance()
        if (self.interferometer is not None or self.vacuum_born_norm) and self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(imp[0::2].square() + imp[1::2].square() + 1e-12).mean()
        else:
            group_l21 = imp.mean()
        elem_l1 = self.readout_linear.weight.abs().mean()
        return 0.5 * (group_l21 + elem_l1)

    @torch.no_grad()
    def select_top_indices(
        self,
        target_modes: int,
        extra_importance: Tensor | None = None,
    ) -> Tensor:
        """Select sorted mode indices (dyad-aligned when interferometer/vacuum Born is active)."""
        dev = self.readout_linear.weight.device
        if target_modes <= 0 or target_modes >= self.n_modes:
            return torch.arange(self.n_modes, device=dev)

        importance = self.mode_importance()
        if extra_importance is not None:
            importance = importance + extra_importance.to(importance.device)

        use_dyads = (
            (self.interferometer is not None or self.vacuum_born_norm)
            and self.n_modes % 2 == 0
            and target_modes % 2 == 0
        )
        if use_dyads:
            dyad_imp = torch.sqrt(
                importance[0::2].square() + importance[1::2].square() + 1e-12
            )
            _, top_dyads = torch.topk(
                dyad_imp, k=target_modes // 2, largest=True, sorted=True
            )
            top_dyads = torch.sort(top_dyads).values
            return torch.stack([2 * top_dyads, 2 * top_dyads + 1], dim=-1).reshape(-1)

        _, top_idx = torch.topk(importance, k=target_modes, largest=True, sorted=True)
        return torch.sort(top_idx).values

    @torch.no_grad()
    def prune_to_indices(
        self,
        top_idx: Tensor,
        compensate_zero_state_bias: bool = True,
    ) -> Tensor:
        """Structurally prune this layer to an explicit sorted index tensor ``top_idx``."""
        target_modes = int(top_idx.numel())
        old_m = self.n_modes
        if target_modes <= 0 or target_modes >= old_m:
            return torch.arange(old_m, device=self.readout_linear.weight.device)

        dev = self.readout_linear.weight.device
        top_idx = top_idx.to(dev)
        scale_ratio = math.sqrt(target_modes / old_m)

        self.reupload.prune_to_indices(top_idx)

        new_dt = nn.Linear(
            self.d_model,
            target_modes,
            device=dev,
            dtype=self.dt_proj.weight.dtype,
        )
        new_dt.weight.copy_(self.dt_proj.weight.data[top_idx])
        new_dt.bias.copy_(self.dt_proj.bias.data[top_idx])
        self.dt_proj = new_dt

        self.log_eta = nn.Parameter(self.log_eta.data[top_idx].clone())
        self.kvn_omega = nn.Parameter(self.kvn_omega.data[top_idx].clone())
        self.in_gain_q = nn.Parameter(self.in_gain_q.data[top_idx].clone())
        self.in_gain_p = nn.Parameter(self.in_gain_p.data[top_idx].clone())

        if self.interferometer is not None:
            dyad_idx = top_idx[0::2] // 2
            self.interferometer.prune_to_dyad_indices(dyad_idx)
        if self.vacuum_log_kappa is not None:
            dyad_idx = top_idx[0::2] // 2
            self.vacuum_log_kappa = nn.Parameter(
                self.vacuum_log_kappa.data[dyad_idx].clone()
            )

        w_old = self.readout_linear.weight.data
        w_q = w_old[:, top_idx] * scale_ratio
        w_p = w_old[:, old_m + top_idx] * scale_ratio
        w_b = w_old[:, 2 * old_m + top_idx]

        bias_new = self.readout_linear.bias.data.clone()
        if compensate_zero_state_bias and not self.vacuum_born_norm:
            pruned_mask = torch.ones(old_m, dtype=torch.bool, device=dev)
            pruned_mask[top_idx] = False
            w_b_pruned = w_old[:, 2 * old_m :][:, pruned_mask]
            bias_new.sub_(w_b_pruned.sum(dim=-1))

        new_readout = nn.Linear(
            3 * target_modes,
            self.d_model,
            device=dev,
            dtype=self.readout_linear.weight.dtype,
        )
        new_readout.weight.copy_(torch.cat([w_q, w_p, w_b], dim=-1))
        new_readout.bias.copy_(bias_new)
        self.readout_linear = new_readout
        self.n_modes = target_modes
        return top_idx

    @torch.no_grad()
    def prune_to_top_modes(
        self,
        target_modes: int,
        extra_importance: Tensor | None = None,
        compensate_zero_state_bias: bool = True,
    ) -> Tensor:
        """Select top active modes/dyads and structurally prune (M_train -> M_infer)."""
        top_idx = self.select_top_indices(target_modes, extra_importance=extra_importance)
        return self.prune_to_indices(
            top_idx, compensate_zero_state_bias=compensate_zero_state_bias
        )


class MultiHorizonKoopmanRollout(nn.Module):
    """One-shot analytical Koopman spectral rollout for k = 0 .. block_size - 1.

    Supports multi-layer modal state aggregation, low-rank anchor conditioning,
    pairwise symplectic interferometry, and Barthe Eq. 4.34 vacuum-mode Born calibration.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int,
        block_size: int = 8,
        anchor_rank: int = 0,
        enable_interferometer: bool = True,
        vacuum_born_norm: bool = True,
        num_layers: int = 1,
        multilayer_rollout: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.block_size = block_size
        self.use_low_rank_anchor = 0 < anchor_rank < d_model
        self.anchor_rank = anchor_rank if self.use_low_rank_anchor else 0
        self.use_interferometer = enable_interferometer and (n_modes % 2 == 0)
        self.vacuum_born_norm = vacuum_born_norm and (n_modes % 2 == 0)
        self.num_layers = num_layers
        self.multilayer_rollout = multilayer_rollout and (num_layers > 1)

        self.horizon_freq_scale = nn.Parameter(torch.ones(block_size, n_modes))
        self.horizon_log_decay = nn.Parameter(torch.full((block_size, n_modes), -2.5))
        self.horizon_phase_bias = nn.Parameter(torch.zeros(block_size, n_modes))

        if self.multilayer_rollout:
            init_logits = torch.zeros(num_layers, n_modes)
            init_logits[-1, :] = 2.0
            self.layer_mode_logits: nn.Parameter | None = nn.Parameter(init_logits)
        else:
            self.layer_mode_logits = None

        self.interferometer: DyadicSymplecticInterferometer | None = (
            DyadicSymplecticInterferometer(n_modes) if self.use_interferometer else None
        )
        self.vacuum_log_kappa: nn.Parameter | None = (
            nn.Parameter(torch.zeros(n_modes // 2)) if self.vacuum_born_norm else None
        )

        if self.use_low_rank_anchor:
            self.anchor_proj: nn.Module = nn.Sequential(
                nn.Linear(d_model, self.anchor_rank, bias=False),
                nn.Linear(self.anchor_rank, block_size * d_model),
            )
        else:
            self.anchor_proj = nn.Linear(d_model, block_size * d_model)

        self.obs_linear = nn.Linear(3 * n_modes, d_model)
        self.obs_norm = nn.RMSNorm(d_model)
        self.out_norm = nn.RMSNorm(d_model)

    def aggregate_multilayer_states(
        self,
        q_list: list[Tensor],
        p_list: list[Tensor],
        dtheta_list: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Aggregate modal states across draft layers with per-mode softmax weights."""
        if not self.multilayer_rollout or self.layer_mode_logits is None or len(q_list) == 1:
            return q_list[-1], p_list[-1], dtheta_list[-1]

        weights = torch.softmax(self.layer_mode_logits.float(), dim=0).unsqueeze(1)
        q_stack = torch.stack([q.float() for q in q_list], dim=0)
        p_stack = torch.stack([p.float() for p in p_list], dim=0)
        dt_stack = torch.stack([dt.float() for dt in dtheta_list], dim=0)
        dtype = q_list[-1].dtype
        return (
            (weights * q_stack).sum(dim=0).to(dtype),
            (weights * p_stack).sum(dim=0).to(dtype),
            (weights * dt_stack).sum(dim=0).to(dtype),
        )

    def forward(
        self,
        anchor_h: Tensor,  # [N_blocks, d_model]
        anchor_q: Tensor,  # [N_blocks, n_modes]
        anchor_p: Tensor,  # [N_blocks, n_modes]
        anchor_dtheta: Tensor,  # [N_blocks, n_modes]
    ) -> tuple[Tensor, Tensor]:
        """Return (hidden_blocks [N_blocks, K, d_model], born_features [N_blocks, K, n_modes])."""
        n_blocks = anchor_h.size(0)
        k_steps = torch.arange(
            1, self.block_size + 1, device=anchor_h.device, dtype=torch.float32
        ).view(1, self.block_size, 1)

        angles = (
            k_steps
            * anchor_dtheta.float().unsqueeze(1)
            * self.horizon_freq_scale.float().unsqueeze(0)
            + self.horizon_phase_bias.float().unsqueeze(0)
        )
        decay = torch.exp(
            -k_steps * torch.exp(self.horizon_log_decay.float()).unsqueeze(0)
        )

        cos_a = decay * torch.cos(angles)
        sin_a = decay * torch.sin(angles)

        q_0 = anchor_q.float().unsqueeze(1)
        p_0 = anchor_p.float().unsqueeze(1)
        q_k = cos_a * q_0 - sin_a * p_0
        p_k = sin_a * q_0 + cos_a * p_0

        if self.interferometer is not None:
            q_obs, p_obs = self.interferometer(q_k, p_k)
        else:
            q_obs, p_obs = q_k, p_k

        if self.vacuum_born_norm and self.vacuum_log_kappa is not None:
            born_f, _ = compute_vacuum_born_observables(
                q_obs, p_obs, self.vacuum_log_kappa
            )
        else:
            born_f = (2.0 * self.n_modes) * (q_obs.square() + p_obs.square()) - 1.0

        born_features = born_f.to(anchor_h.dtype)
        obs_k = torch.cat(
            [q_obs.to(anchor_h.dtype), p_obs.to(anchor_h.dtype), born_features],
            dim=-1,
        )
        kvn_hidden = self.obs_norm(self.obs_linear(obs_k))

        anchor_cond = self.anchor_proj(anchor_h).view(
            n_blocks, self.block_size, self.d_model
        )
        hidden_blocks = self.out_norm(anchor_cond + kvn_hidden)
        return hidden_blocks, born_features

    def mode_importance(self) -> Tensor:
        """Compute L2 group importance of each mode in rollout obs_linear."""
        w = self.obs_linear.weight.float()
        m = self.n_modes
        imp_q = w[:, :m].square().mean(dim=0)
        imp_p = w[:, m : 2 * m].square().mean(dim=0)
        imp_b = w[:, 2 * m :].square().mean(dim=0)
        return torch.sqrt(imp_q + imp_p + imp_b + 1e-12)

    def lasso_penalty(self) -> Tensor:
        """Dyadic Group-LASSO + L1 penalty on rollout mode readout weights."""
        imp = self.mode_importance()
        if (self.interferometer is not None or self.vacuum_born_norm) and self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(imp[0::2].square() + imp[1::2].square() + 1e-12).mean()
        else:
            group_l21 = imp.mean()
        return 0.5 * (group_l21 + self.obs_linear.weight.abs().mean())

    @torch.no_grad()
    def prune_to_indices(
        self,
        top_idx: Tensor,
        compensate_zero_state_bias: bool = True,
    ) -> None:
        """Align rollout mode dimension after structural LASSO pruning."""
        old_m = self.n_modes
        target_modes = int(top_idx.numel())
        if target_modes >= old_m:
            return

        dev = self.obs_linear.weight.device
        top_idx = top_idx.to(dev)
        scale_ratio = math.sqrt(target_modes / old_m)

        self.horizon_freq_scale = nn.Parameter(
            self.horizon_freq_scale.data[:, top_idx].clone()
        )
        self.horizon_log_decay = nn.Parameter(
            self.horizon_log_decay.data[:, top_idx].clone()
        )
        self.horizon_phase_bias = nn.Parameter(
            self.horizon_phase_bias.data[:, top_idx].clone()
        )
        if self.layer_mode_logits is not None:
            self.layer_mode_logits = nn.Parameter(
                self.layer_mode_logits.data[:, top_idx].clone()
            )
        if self.interferometer is not None:
            dyad_idx = top_idx[0::2] // 2
            self.interferometer.prune_to_dyad_indices(dyad_idx)
        if self.vacuum_log_kappa is not None:
            dyad_idx = top_idx[0::2] // 2
            self.vacuum_log_kappa = nn.Parameter(
                self.vacuum_log_kappa.data[dyad_idx].clone()
            )

        w_old = self.obs_linear.weight.data
        w_q = w_old[:, top_idx] * scale_ratio
        w_p = w_old[:, old_m + top_idx] * scale_ratio
        w_b = w_old[:, 2 * old_m + top_idx]

        bias_new = self.obs_linear.bias.data.clone()
        if compensate_zero_state_bias and not self.vacuum_born_norm:
            pruned_mask = torch.ones(old_m, dtype=torch.bool, device=dev)
            pruned_mask[top_idx] = False
            w_b_pruned = w_old[:, 2 * old_m :][:, pruned_mask]
            bias_new.sub_(w_b_pruned.sum(dim=-1))

        new_lin = nn.Linear(
            3 * target_modes,
            self.d_model,
            device=dev,
            dtype=self.obs_linear.weight.dtype,
        )
        new_lin.weight.copy_(torch.cat([w_q, w_p, w_b], dim=-1))
        new_lin.bias.copy_(bias_new)

        self.obs_linear = new_lin
        self.n_modes = target_modes


class IntraBlockCausalMixer(nn.Module):
    """Zero-KV-Cache intra-block causal operator over speculative horizons k = 0..K-1.

    Restores DSpark's intra-block multi-step syntactic coherence on k = 2..K-1 with
    ZERO persistent KV cache (0 bytes/seq) and negligible K x K FLOPs.
    """

    def __init__(self, d_model: int, block_size: int = 8, n_heads: int = 8) -> None:
        super().__init__()
        eff_heads = n_heads if d_model % n_heads == 0 else math.gcd(d_model, 8) or 1
        self.d_model = d_model
        self.block_size = block_size
        self.n_heads = eff_heads
        self.head_dim = d_model // eff_heads

        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.horizon_pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        nn.init.normal_(self.horizon_pos_emb, std=0.02)

        self.norm2 = nn.RMSNorm(d_model)
        ffn_dim = max(64, d_model // 2)
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, h_block: Tensor) -> Tensor:
        n_anc, k_len, d = h_block.shape
        x = h_block + self.horizon_pos_emb[:, :k_len, :].to(h_block.dtype)
        normed = self.norm1(x)
        qkv = self.qkv(normed).view(n_anc, k_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(n_anc, k_len, d)
        h_out = h_block + self.o_proj(attn_out)

        normed2 = self.norm2(h_out)
        h_out = h_out + self.down_proj(
            F.silu(self.gate_proj(normed2)) * self.up_proj(normed2)
        )
        return h_out


class BornConfidenceHead(nn.Module):
    """Per-position acceptance probability head with interferometric vacuum-Born calibration."""

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int = 0,
        n_modes: int = 0,
        use_born: bool = True,
        intermediate_size: int | None = None,
        vacuum_born_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.n_modes = n_modes
        self.use_born = use_born and (n_modes > 0)
        self.vacuum_born_norm = vacuum_born_norm
        base_dim = hidden_size + markov_rank
        mid = intermediate_size or min(512, max(64, hidden_size // 2))
        self.base_norm = nn.RMSNorm(base_dim)
        self.base_linear = nn.Linear(base_dim, mid)
        self.born_proj: nn.Linear | None = (
            nn.Linear(n_modes, mid, bias=False) if self.use_born else None
        )
        self.out_linear = nn.Linear(mid, 1)

    def forward(
        self,
        hidden_blocks: Tensor,
        prev_emb: Tensor | None = None,
        born_features: Tensor | None = None,
    ) -> Tensor:
        if prev_emb is not None and self.markov_rank > 0:
            base = torch.cat([hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1)
        else:
            base = hidden_blocks
        h = self.base_linear(self.base_norm(base))
        if self.use_born and self.born_proj is not None and born_features is not None:
            h = h + self.born_proj(born_features.to(h.dtype))
        return self.out_linear(F.silu(h)).squeeze(-1)

    def mode_importance(self) -> Tensor | None:
        if not self.use_born or self.born_proj is None:
            return None
        return torch.sqrt(self.born_proj.weight.float().square().mean(dim=0) + 1e-12)

    def lasso_penalty(self) -> Tensor:
        if not self.use_born or self.born_proj is None:
            return torch.tensor(0.0, device=self.base_linear.weight.device)
        imp = self.mode_importance()
        assert imp is not None
        if self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(imp[0::2].square() + imp[1::2].square() + 1e-12).mean()
        else:
            group_l21 = imp.mean()
        return 0.5 * (group_l21 + self.born_proj.weight.abs().mean())

    @torch.no_grad()
    def prune_born_modes(
        self,
        top_idx: Tensor,
        compensate_zero_state_bias: bool = True,
    ) -> None:
        if not self.use_born or self.born_proj is None:
            return
        old_m = self.n_modes
        target_modes = int(top_idx.numel())
        if target_modes >= old_m:
            return

        dev = self.born_proj.weight.device
        top_idx = top_idx.to(dev)
        w_old = self.born_proj.weight.data

        if compensate_zero_state_bias and not self.vacuum_born_norm:
            pruned_mask = torch.ones(old_m, dtype=torch.bool, device=dev)
            pruned_mask[top_idx] = False
            w_pruned = w_old[:, pruned_mask]
            self.base_linear.bias.data.sub_(
                w_pruned.sum(dim=-1).to(self.base_linear.bias.dtype)
            )

        new_born = nn.Linear(
            target_modes,
            self.born_proj.out_features,
            bias=False,
            device=dev,
            dtype=self.born_proj.weight.dtype,
        )
        new_born.weight.copy_(w_old[:, top_idx])
        self.born_proj = new_born
        self.n_modes = target_modes
