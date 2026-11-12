# ruff: noqa: C901, PLR2004, PT018, S101

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from transformers.models.qwen3.modeling_qwen3 import GradientCheckpointingLayer

from speculators.models.kvn_dspark.triton_kernels import (
    fused_multistep_complex_decode,
    fused_multistep_sp4_decode,
    segmented_complex_scan,
    segmented_scalar_scan,
    segmented_sp4_scan,
)

__all__ = [
    "BornConfidenceHead",
    "CausalKvNRFFLayer",
    "DataReuploadingLinear",
    "DyadicSymplecticInterferometer",
    "IntraBlockCausalMixer",
    "KrausPOVMHead",
    "ModalDataReuploadingRFF",
    "MultiHorizonKoopmanRollout",
    "SpeculativeEngram",
    "compute_butterfly_pairs",
    "compute_vacuum_born_observables",
    "parallel_segmented_real_scan",
    "specinfer_verify_single_step",
    "specinfer_verify_tree",
]


class DataReuploadingLinear(nn.Module):
    """Dense multi-stage Fourier Data Re-uploading encoder (wired for d_re=0,
    dyadic_rff=False).
    """

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
    """Modal Data-Reuploading RFF Encoder (Barthe 2026, Ch. 2 Thm. 2.1 & Ch. 3 Lem.
    3.2).

    Supports two spectral regimes:
    - - ``dyadic_rff=True`` (v3 default, Barthe Ch. 3 Lemma 3.2 Eq. 3.26): composes L
      stages
      multiplicatively in complex phase ``prod_{l=0}^{L-1} exp(i (2^l W_l z_bar^(l) +
      b_l))``
      via real ``(cos, sin)`` recursion, yielding exponential ``Theta(2^L)`` frequency
      cover
      with zero DC drift and zero ``torch.complex64`` operations.
    - ``dyadic_rff=False`` (v2 / v1 ablation, Barthe Ch. 3 Cor. 3.2 Eq. 3.28): additive
      L-stage re-uploading with ``sigma_g / sqrt(L * d_re)`` scaling, wiring
      ``DataReuploadingLinear`` when ``d_re == 0`` so full-width A0 ablations remain
      active.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int,
        reupload_layers_l: int = 2,
        d_re: int = 512,
        sigma_g: float = 1.0,
        learnable_rff: bool = True,
        dyadic_rff: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.reupload_layers_l = reupload_layers_l
        self.d_re = d_re
        self.eff_d_re = d_re if (0 < d_re < d_model) else d_model
        self.learnable_rff = learnable_rff
        self.dyadic_rff = dyadic_rff
        self.use_legacy_dense = (d_re == 0) and (not dyadic_rff)

        self.legacy_reupload: DataReuploadingLinear | None = None
        if self.use_legacy_dense:
            self.legacy_reupload = DataReuploadingLinear(
                d_model=d_model, reupload_layers_l=reupload_layers_l
            )
            self.in_proj: nn.Module = nn.Identity()
            self.stage_norms = nn.ModuleList()
            self.reupload_x = nn.ModuleList()
            self.reupload_h = nn.ModuleList()
        else:
            self.in_proj = (
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

        if self.dyadic_rff:
            # Barthe Ch. 3 Lem. 3.2 Eq. (3.26): sum_{l=0}^{L-1} 4^l = (4^L - 1) / 3
            geom_sum = max(1.0, ((4**reupload_layers_l) - 1) / 3.0)
            rff_std = sigma_g / math.sqrt(geom_sum * self.eff_d_re)
            omega_init = (
                torch.randn(reupload_layers_l, self.eff_d_re, n_modes) * rff_std
            )
            phase_init = torch.rand(reupload_layers_l, n_modes) * (2.0 * math.pi)
        else:
            # Barthe Ch. 3 Cor. 3.2 Eq. (3.28): L-fold additive scaling sigma_g / sqrt(L
            # * d_re)
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
        inv_sqrt2 = 0.7071067811865475
        if self.dyadic_rff:
            x_low = self.in_proj(x)
            h_mod = torch.zeros_like(x_low)
            c_acc: Tensor | None = None
            s_acc: Tensor | None = None
            omega_f = self.omega_rff.float()
            phase_f = self.phase_rff.float()

            for l_idx in range(self.reupload_layers_l):
                z = self.reupload_x[l_idx](x_low)
                if l_idx > 0:
                    z = z + self.reupload_h[l_idx - 1](h_mod)
                z_bar = self.stage_norms[l_idx](z)
                if l_idx < self.reupload_layers_l - 1:
                    h_mod = h_mod + inv_sqrt2 * (torch.sin(z_bar) + torch.cos(z_bar))

                dyadic_scale = float(1 << l_idx)
                phi_l = dyadic_scale * (z_bar.float() @ omega_f[l_idx]) + phase_f[l_idx]
                c_l = torch.cos(phi_l)
                s_l = torch.sin(phi_l)
                if c_acc is None or s_acc is None:
                    c_acc, s_acc = c_l, s_l
                else:
                    c_next = c_acc * c_l - s_acc * s_l
                    s_next = c_acc * s_l + s_acc * c_l
                    c_acc, s_acc = c_next, s_next

            assert c_acc is not None and s_acc is not None
            cos_p = c_acc.to(x.dtype)
            sin_p = s_acc.to(x.dtype)
        elif self.use_legacy_dense and self.legacy_reupload is not None:
            h = self.legacy_reupload(x)
            proj = (h.float() @ self.omega_rff.float() + self.phase_rff.float()).to(
                x.dtype
            )
            cos_p = torch.cos(proj)
            sin_p = torch.sin(proj)
        else:
            x_low = self.in_proj(x)
            h = torch.zeros_like(x_low)
            for l_idx in range(self.reupload_layers_l):
                z = self.reupload_x[l_idx](x_low)
                if l_idx > 0:
                    z = z + self.reupload_h[l_idx - 1](h)
                z = self.stage_norms[l_idx](z)
                h = h + inv_sqrt2 * (torch.sin(z) + torch.cos(z))

            proj = (h.float() @ self.omega_rff.float() + self.phase_rff.float()).to(
                x.dtype
            )
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
        omega_sliced = self.omega_rff.data[..., idx_dev].clone()
        phase_sliced = self.phase_rff.data[..., idx_dev].clone()
        if isinstance(self.omega_rff, nn.Parameter):
            self.omega_rff = nn.Parameter(omega_sliced)
            self.phase_rff = nn.Parameter(phase_sliced)
        else:
            self.omega_rff = omega_sliced
            self.phase_rff = phase_sliced
        self.raw_alpha = nn.Parameter(
            self.raw_alpha.data[top_idx.to(self.raw_alpha.device)].clone()
        )
        self.n_modes = int(top_idx.numel())


class SpeculativeEngram(nn.Module):
    """O(1) Multi-Head Hashed 2-gram & 3-gram Sub-Token Memory (Axe 1b).

    Hashes local token histories ``[prev3_id, prev_prev_id, prev_id]`` across 4
    prime-mixed
    hash heads into a shared embedding table and projects a near-zero residual into
    ``MarkovHead`` hidden space (or ``d_model`` when ``markov_rank == 0``), capturing
    crisp
    sub-token/idiom patterns with 0 bytes of per-sequence KV cache.
    """

    _P1 = 1315423911
    _P2 = 2654435761
    _P3 = 2246822519
    _P4 = 3266489917
    _P5 = 668265263
    _P6 = 374761393
    _P7 = 2862933555777941757
    _S0 = 0x9E3779B9
    _S1 = 0x85EBCA6B
    _S2 = 0xC2B2AE35
    _S3 = 0x27D4EB2F

    def __init__(
        self,
        out_dim: int,
        engram_buckets: int = 65536,
        engram_dim: int = 128,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.engram_buckets = engram_buckets
        self.engram_dim = engram_dim

        self.engram_table = nn.Embedding(engram_buckets, engram_dim)
        nn.init.normal_(self.engram_table.weight, mean=0.0, std=0.02)
        self.engram_norm = nn.RMSNorm(engram_dim)
        self.engram_proj = nn.Linear(engram_dim, out_dim, bias=False)
        nn.init.normal_(self.engram_proj.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        prev_id: Tensor,
        prev_prev_id: Tensor | None = None,
        prev3_id: Tensor | None = None,
    ) -> Tensor:
        p1 = prev_id.long().clamp_min(0)
        if prev_prev_id is None:
            if p1.ndim >= 2 and p1.size(-1) > 1:
                p2 = torch.cat([torch.zeros_like(p1[..., :1]), p1[..., :-1]], dim=-1)
            else:
                p2 = torch.zeros_like(p1)
        else:
            p2 = prev_prev_id.long().clamp_min(0)

        if prev3_id is None:
            if p2.ndim >= 2 and p2.size(-1) > 1:
                p3 = torch.cat([torch.zeros_like(p2[..., :1]), p2[..., :-1]], dim=-1)
            else:
                p3 = torch.zeros_like(p2)
        else:
            p3 = prev3_id.long().clamp_min(0)

        b = self.engram_buckets
        # 2-gram heads over [prev_prev_id, prev_id]
        h0 = torch.remainder(p2 * self._P1 + p1 * self._P2 + self._S0, b)
        h1 = torch.remainder(p2 * self._P3 + p1 * self._P4 + self._S1, b)
        # 3-gram heads over [prev3_id, prev_prev_id, prev_id]
        h2 = torch.remainder(
            p3 * self._P5 + p2 * self._P6 + p1 * self._P2 + self._S2, b
        )
        h3 = torch.remainder(
            p3 * self._P6 + p2 * self._P1 + p1 * self._P5 + self._S3, b
        )

        emb = 0.25 * (
            self.engram_table(h0)
            + self.engram_table(h1)
            + self.engram_table(h2)
            + self.engram_table(h3)
        )
        return self.engram_proj(self.engram_norm(emb))


def parallel_segmented_real_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    initial_state: tuple[Tensor, Tensor] | None = None,
    use_triton: bool = False,
) -> tuple[Tensor, Tensor]:
    """Work-parallel O(T log T) associative scan in real float32 quadratures."""
    return segmented_complex_scan(
        a_cos,
        a_sin,
        u_q,
        u_p,
        initial_state=initial_state,
        use_triton=use_triton,
    )


def compute_butterfly_pairs(n_modes: int, layer_idx: int) -> tuple[Tensor, Tensor]:
    """Compute bit-structured butterfly mode permutation ``(perm, inv_perm)`` at
    ``layer_idx``.

    For even ``n_modes`` divisible by ``2^p`` (with ``p >= 1`` maximal), pairs each mode
    ``m`` with ``m ^ (1 << (layer_idx % p))``. Across ``log2(M)`` layers, achieves
    global
    all-to-all Bosonic modal coupling in O(M) ops per layer.
    """
    if n_modes % 2 != 0:
        raise ValueError(f"Butterfly pairing requires even n_modes, got {n_modes}.")
    largest_pow2 = n_modes & (-n_modes)
    n_stages = max(1, int(round(math.log2(largest_pow2))))
    stride = 1 << (layer_idx % n_stages)

    a_list: list[int] = []
    b_list: list[int] = []
    visited = [False] * n_modes
    for m in range(n_modes):
        if not visited[m]:
            partner = m ^ stride
            lo, hi = (m, partner) if m < partner else (partner, m)
            visited[lo] = True
            visited[hi] = True
            a_list.append(lo)
            b_list.append(hi)

    perm_list: list[int] = []
    for a_m, b_m in zip(a_list, b_list, strict=True):
        perm_list.extend([a_m, b_m])
    perm = torch.tensor(perm_list, dtype=torch.long)
    inv_perm = torch.argsort(perm)
    return perm, inv_perm


def _smooth_sinh_over_x(x: Tensor) -> Tensor:
    """Smooth C^infinity evaluation of sinh(x) / x in float32 without 0/0."""
    x_sq = x * x
    taylor = 1.0 + x_sq * ((1.0 / 6.0) + x_sq * (1.0 / 120.0))
    safe_x = torch.where(x.abs() < 1e-3, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-3, taylor, torch.sinh(safe_x) / safe_x)


class DyadicSymplecticInterferometer(nn.Module):
    """Pairwise passive Bosonic beamsplitter bank U in U(2)^{M/2} (Barthe Ch. 7 Eq.
    7.25).

    Couples mode dyads (2g, 2g+1) in O(M) ops with exact local norm preservation,
    restoring Koopman phase interference in Born energies while keeping dyads
    independently prunable under Group-LASSO (Barthe Ch. 5 Thm. 5.6).
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
        raise ValueError(
            f"compute_vacuum_born_observables requires even M, got {n_modes}."
        )
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


class CausalKvNRFFLayer(GradientCheckpointingLayer):
    """Causal Driven-Damped Symplectic-Squeezed Bosonic SSM Layer (KvN-DSpark v3).

    Integrates:
    - Multiplicative dyadic RFF re-uploading (Barthe Ch. 3 Lem. 3.2 Eq. 3.26);
    - - sp(4, R) butterfly symplectic-squeezed 4x4 dyad scan ``A_{t,d} = M_{t,d} B_{t,d}
      M_{t,d} R_{t,d}``
      with strict contractivity ``||A_{t,d}||_2 < 1`` (Barthe Ch. 7 Eq. 7.25 & 7.56);
    - Open-system Lindblad excess thermal variance ``delta_nu_{t,m} >= 0`` & purity;
    - Exact FP32 state preservation across both parallel scan and O(1) recurrent decode.
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
        dyadic_rff: bool = True,
        use_symplectic_butterfly: bool = True,
        squeezing_max_ratio: float = 0.8,
        use_lindblad_purity: bool = True,
        use_triton_scan: bool = True,
        layer_idx: int = 0,
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
        self.dyadic_rff = dyadic_rff
        self.use_symplectic_butterfly = use_symplectic_butterfly and (n_modes % 2 == 0)
        self.squeezing_max_ratio = float(squeezing_max_ratio)
        self.use_lindblad_purity = use_lindblad_purity
        self.use_triton_scan = use_triton_scan
        self.layer_idx = layer_idx

        self.norm1 = nn.RMSNorm(d_model)
        self.reupload = ModalDataReuploadingRFF(
            d_model=d_model,
            n_modes=n_modes,
            reupload_layers_l=reupload_layers_l,
            d_re=d_re,
            sigma_g=sigma_g,
            learnable_rff=learnable_rff,
            dyadic_rff=dyadic_rff,
        )

        self.dt_proj = nn.Linear(d_model, n_modes)
        nn.init.constant_(self.dt_proj.bias, -2.0)

        self.log_eta = nn.Parameter(torch.full((n_modes,), -2.5))
        self.kvn_omega = nn.Parameter(torch.randn(n_modes) * 0.5)
        self.in_gain_q = nn.Parameter(torch.ones(n_modes) * 0.5)
        self.in_gain_p = nn.Parameter(torch.ones(n_modes) * 0.5)

        if self.use_symplectic_butterfly:
            n_dyads = n_modes // 2
            perm, inv_perm = compute_butterfly_pairs(n_modes, layer_idx)
            self.register_buffer("butterfly_perm", perm, persistent=True)
            self.register_buffer("butterfly_inv_perm", inv_perm, persistent=True)
            self.butterfly_phase_proj = nn.Linear(d_model, n_modes)
            nn.init.normal_(self.butterfly_phase_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.butterfly_phase_proj.bias)

            self.butterfly_bs_proj = nn.Linear(d_model, 2 * n_dyads)
            nn.init.normal_(self.butterfly_bs_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.butterfly_bs_proj.bias)

            self.butterfly_couple_gate = nn.Parameter(torch.ones(n_modes))

            self.squeezing_proj = nn.Linear(d_model, 2 * n_modes)
            nn.init.normal_(self.squeezing_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.squeezing_proj.bias)
        else:
            self.register_buffer(
                "butterfly_perm",
                torch.arange(n_modes, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "butterfly_inv_perm",
                torch.arange(n_modes, dtype=torch.long),
                persistent=False,
            )
            self.butterfly_phase_proj = None
            self.butterfly_bs_proj = None
            self.butterfly_couple_gate = None
            self.squeezing_proj = None

        if self.use_lindblad_purity:
            self.lindblad_env_proj: nn.Linear | None = nn.Linear(d_model, n_modes)
            nn.init.normal_(self.lindblad_env_proj.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.lindblad_env_proj.bias, -3.0)
            self.lindblad_gain: nn.Parameter | None = nn.Parameter(torch.ones(n_modes))
        else:
            self.lindblad_env_proj = None
            self.lindblad_gain = None

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

    def _has_dyadic_structure(self) -> bool:
        return bool(
            self.use_symplectic_butterfly
            or self.interferometer is not None
            or self.vacuum_born_norm
        )

    def compute_observables(
        self,
        q_f: Tensor,
        p_f: Tensor,
        out_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        """Project FP32 (q_f, p_f) to out_dtype readout residual and rescued p_vac."""
        if self.interferometer is not None:
            q_obs_f, p_obs_f = self.interferometer(q_f, p_f)
        else:
            q_obs_f, p_obs_f = q_f, p_f

        p_vac_f: Tensor | None = None
        if self.vacuum_born_norm and self.vacuum_log_kappa is not None:
            born_f, p_vac_f = compute_vacuum_born_observables(
                q_obs_f, p_obs_f, self.vacuum_log_kappa
            )
        else:
            born_f = (2.0 * self.n_modes) * (q_obs_f.square() + p_obs_f.square()) - 1.0

        obs = torch.cat(
            [q_obs_f.to(out_dtype), p_obs_f.to(out_dtype), born_f.to(out_dtype)],
            dim=-1,
        )
        return self.readout_norm(self.readout_linear(obs)), p_vac_f

    def _compute_readout(
        self, q_f: Tensor, p_f: Tensor, out_dtype: torch.dtype
    ) -> Tensor:
        readout, _ = self.compute_observables(q_f, p_f, out_dtype)
        return readout

    def _build_sp4_transition_and_forcing(
        self,
        h_norm: Tensor,  # [B, T, D]
        r_q: Tensor,  # [B, T, M]
        r_p: Tensor,  # [B, T, M]
        dt: Tensor,  # [B, T, M]
        reset_mask: Tensor | None = None,  # [B, T]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Construct Strang 4x4 symplectic-squeezed A_{t,d} = M_{t,d} B_{t,d} M_{t,d}
        R_{t,d}.

        Guarantees strict contractivity ``||A_{t,d}||_2 < 1`` via bounded squeezing
        ``|xi_{t,m}| <= squeezing_max_ratio * gamma_{t,m}``, and exact ``Sp(4, R)``
        symplecticity in the dissipationless limit ``gamma -> 0``.
        """
        assert (
            self.butterfly_phase_proj is not None
            and self.butterfly_bs_proj is not None
            and self.butterfly_couple_gate is not None
            and self.squeezing_proj is not None
        )
        bsz, seq_len, _ = h_norm.shape
        n_dyads = self.n_modes // 2

        raw_dphi = self.butterfly_phase_proj(h_norm)
        raw_bs = self.butterfly_bs_proj(h_norm)
        raw_sq = self.squeezing_proj(h_norm)
        raw_env = (
            self.lindblad_env_proj(h_norm)
            if self.lindblad_env_proj is not None
            else None
        )

        with torch.amp.autocast(device_type=h_norm.device.type, enabled=False):
            dt_f = dt.float()
            eta = (
                torch.exp(self.log_eta.float())
                .clamp(min=self.damping_min, max=self.damping_max)
                .view(1, 1, -1)
            )
            gamma = eta * dt_f  # [B, T, M] > 0

            # Decoupled input-selective phase increment: theta = omega * dt +
            # delta_theta(h_t)
            omega_pole = self.kvn_omega.float().view(1, 1, -1)
            delta_theta = math.pi * torch.tanh(raw_dphi.float())
            dtheta_f = dt_f * omega_pole + delta_theta  # [B, T, M]

            u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
            u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()

            # Permute modes into butterfly pairs (a_d, b_d) for d = 0 .. M//2 - 1
            perm = self.butterfly_perm
            gamma_p = gamma.index_select(-1, perm).view(bsz, seq_len, n_dyads, 2)
            theta_p = dtheta_f.index_select(-1, perm).view(bsz, seq_len, n_dyads, 2)
            uq_p = u_q_f.index_select(-1, perm).view(bsz, seq_len, n_dyads, 2)
            up_p = u_p_f.index_select(-1, perm).view(bsz, seq_len, n_dyads, 2)
            u_vec = torch.stack(
                [uq_p[..., 0], up_p[..., 0], uq_p[..., 1], up_p[..., 1]], dim=-1
            )  # [B, T, D_dyad, 4]

            # Bounded active squeezing half-step M_{t,m} = exp(-gamma/2) * exp(S_half)
            sq_p = (
                raw_sq.float()
                .view(bsz, seq_len, self.n_modes, 2)
                .index_select(-2, perm)
                .view(bsz, seq_len, n_dyads, 2, 2)
            )
            s1 = sq_p[..., 0]  # [B, T, D_dyad, 2]
            s2 = sq_p[..., 1]
            s_norm = torch.sqrt(s1 * s1 + s2 * s2 + 1e-12)
            xi_scale = (
                self.squeezing_max_ratio * gamma_p * (torch.tanh(s_norm) / s_norm)
            )
            xi1_half = 0.5 * xi_scale * s1
            xi2_half = 0.5 * xi_scale * s2
            xi_half_mag = 0.5 * xi_scale * s_norm

            decay_half = torch.exp(-0.5 * gamma_p)
            ch = torch.cosh(xi_half_mag)
            sh_over = _smooth_sinh_over_x(xi_half_mag)

            m_00 = decay_half * (ch + sh_over * xi1_half)
            m_01 = decay_half * (sh_over * xi2_half)
            m_11 = decay_half * (ch - sh_over * xi1_half)
            # Symmetric 2x2 per mode: [[m_00, m_01], [m_01, m_11]]
            ma00, ma01, ma11 = m_00[..., 0], m_01[..., 0], m_11[..., 0]
            mb00, mb01, mb11 = m_00[..., 1], m_01[..., 1], m_11[..., 1]

            # Selective U(2) beamsplitter B(vartheta, varphi) gated by g_a * g_b
            gate_p = (
                self.butterfly_couple_gate.float()
                .index_select(0, perm)
                .view(1, 1, n_dyads, 2)
            )
            pair_gate = gate_p[..., 0] * gate_p[..., 1]
            bs_f = raw_bs.float().view(bsz, seq_len, n_dyads, 2)
            vartheta = pair_gate * (0.5 * math.pi * torch.tanh(bs_f[..., 0]))
            varphi = bs_f[..., 1]

            cv = torch.cos(vartheta)
            sv = torch.sin(vartheta)
            cp = torch.cos(varphi)
            sp = torch.sin(varphi)

            # B = [[cv * I_2, -sv * R(varphi)], [sv * R(-varphi), cv * I_2]]
            # Compute W = M_d @ B_d @ M_d in closed form (4x4):
            # Top-left 2x2: cv * M_a^2
            tl00 = cv * (ma00 * ma00 + ma01 * ma01)
            tl01 = cv * (ma00 * ma01 + ma01 * ma11)
            tl10 = tl01
            tl11 = cv * (ma01 * ma01 + ma11 * ma11)

            # Bottom-right 2x2: cv * M_b^2
            br00 = cv * (mb00 * mb00 + mb01 * mb01)
            br01 = cv * (mb00 * mb01 + mb01 * mb11)
            br10 = br01
            br11 = cv * (mb01 * mb01 + mb11 * mb11)

            # Top-right 2x2: -sv * (M_a @ R(varphi) @ M_b)
            # R(varphi) @ M_b:
            rm00 = cp * mb00 - sp * mb01
            rm01 = cp * mb01 - sp * mb11
            rm10 = sp * mb00 + cp * mb01
            rm11 = sp * mb01 + cp * mb11
            tr00 = -sv * (ma00 * rm00 + ma01 * rm10)
            tr01 = -sv * (ma00 * rm01 + ma01 * rm11)
            tr10 = -sv * (ma01 * rm00 + ma11 * rm10)
            tr11 = -sv * (ma01 * rm01 + ma11 * rm11)

            # Bottom-left 2x2: sv * (M_b @ R(-varphi) @ M_a) = -tr^T!
            # (Since M_a, M_b are symmetric and (R(varphi))^T = R(-varphi)!)
            bl00 = -tr00
            bl01 = -tr10
            bl10 = -tr01
            bl11 = -tr11

            # Right-multiply W by R_d = diag(R(theta_a), R(theta_b)):
            ca = torch.cos(theta_p[..., 0])
            sa = torch.sin(theta_p[..., 0])
            cb = torch.cos(theta_p[..., 1])
            sb = torch.sin(theta_p[..., 1])

            def _mul_rot(
                x00: Tensor, x01: Tensor, x10: Tensor, x11: Tensor, c: Tensor, s: Tensor
            ):
                return (
                    x00 * c + x01 * s,
                    -x00 * s + x01 * c,
                    x10 * c + x11 * s,
                    -x10 * s + x11 * c,
                )

            a00, a01, a10, a11 = _mul_rot(tl00, tl01, tl10, tl11, ca, sa)
            a02, a03, a12, a13 = _mul_rot(tr00, tr01, tr10, tr11, cb, sb)
            a20, a21, a30, a31 = _mul_rot(bl00, bl01, bl10, bl11, ca, sa)
            a22, a23, a32, a33 = _mul_rot(br00, br01, br10, br11, cb, sb)

            a_mat = torch.stack(
                [
                    torch.stack([a00, a01, a02, a03], dim=-1),
                    torch.stack([a10, a11, a12, a13], dim=-1),
                    torch.stack([a20, a21, a22, a23], dim=-1),
                    torch.stack([a30, a31, a32, a33], dim=-1),
                ],
                dim=-2,
            )  # [B, T, D_dyad, 4, 4]

            a_nu: Tensor | None = None
            u_nu: Tensor | None = None
            if raw_env is not None and self.lindblad_gain is not None:
                a_nu = torch.exp(-2.0 * gamma)
                g_sq = self.lindblad_gain.float().square().view(1, 1, -1)
                u_nu = dt_f * g_sq * F.softplus(raw_env.float())

            if reset_mask is not None:
                keep = (~reset_mask).to(torch.float32)
                a_mat = a_mat * keep.view(bsz, seq_len, 1, 1, 1)
                if a_nu is not None:
                    a_nu = a_nu * keep.unsqueeze(-1)

        return a_mat, u_vec, dtheta_f, a_nu, u_nu

    def _unpack_sp4_state(self, z_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Unpack [B, T, D_dyad, 4] butterfly state to canonical [B, T, M] (q, p)."""
        bsz, seq_len, n_dyads, _ = z_seq.shape
        q_perm = torch.stack([z_seq[..., 0], z_seq[..., 2]], dim=-1).reshape(
            bsz, seq_len, 2 * n_dyads
        )
        p_perm = torch.stack([z_seq[..., 1], z_seq[..., 3]], dim=-1).reshape(
            bsz, seq_len, 2 * n_dyads
        )
        inv_perm = self.butterfly_inv_perm
        return q_perm.index_select(-1, inv_perm), p_perm.index_select(-1, inv_perm)

    def _pack_sp4_state(self, q: Tensor, p: Tensor) -> Tensor:
        """Pack canonical [B, M] (q, p) into butterfly-paired [B, D_dyad, 4]."""
        bsz = q.size(0)
        n_dyads = self.n_modes // 2
        perm = self.butterfly_perm
        qp = q.index_select(-1, perm).view(bsz, n_dyads, 2)
        pp = p.index_select(-1, perm).view(bsz, n_dyads, 2)
        return torch.stack([qp[..., 0], pp[..., 0], qp[..., 1], pp[..., 1]], dim=-1)

    def compute_step_transition_matrices(self, x: Tensor) -> Tensor:
        """Diagnostic helper returning exact 4x4 dyad transition matrices A_{t,d} [B, T,
        D_dyad, 4, 4].
        """
        h_norm = self.norm1(x)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))
        a_mat, _, _, _, _ = self._build_sp4_transition_and_forcing(h_norm, r_q, r_p, dt)
        return a_mat

    def recurrent_step(
        self,
        x_step: Tensor,
        state: tuple[Tensor, ...] | None = None,
        reset_mask: Tensor | None = None,
        state_dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        """O(1) Zero-KV-Cache step for x_step [B, 1, D], preserving exact FP32 state
        parity.
        """
        h_norm = self.norm1(x_step)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))
        bsz = x_step.size(0)

        if self.use_symplectic_butterfly:
            a_mat, u_vec, dtheta_f, a_nu, u_nu = self._build_sp4_transition_and_forcing(
                h_norm, r_q, r_p, dt, reset_mask=reset_mask
            )
            with torch.amp.autocast(device_type=x_step.device.type, enabled=False):
                if state is None:
                    z_prev = torch.zeros(
                        (bsz, self.n_modes // 2, 4),
                        dtype=torch.float32,
                        device=x_step.device,
                    )
                    nu_prev = torch.zeros(
                        (bsz, 1, self.n_modes),
                        dtype=torch.float32,
                        device=x_step.device,
                    )
                else:
                    q_in = state[0].float()
                    p_in = state[1].float()
                    if q_in.ndim == 3:
                        q_in = q_in.squeeze(1)
                        p_in = p_in.squeeze(1)
                    z_prev = self._pack_sp4_state(q_in, p_in)
                    if len(state) >= 3 and state[2] is not None:
                        nu_prev = state[2].float()
                        if nu_prev.ndim == 2:
                            nu_prev = nu_prev.unsqueeze(1)
                    else:
                        nu_prev = torch.zeros(
                            (bsz, 1, self.n_modes),
                            dtype=torch.float32,
                            device=x_step.device,
                        )

                z_step = (
                    torch.matmul(a_mat[:, 0], z_prev.unsqueeze(-1)).squeeze(-1)
                    + u_vec[:, 0]
                ).unsqueeze(1)
                q_f, p_f = self._unpack_sp4_state(z_step)
                nu_f = (
                    a_nu * nu_prev + u_nu
                    if (a_nu is not None and u_nu is not None)
                    else None
                )
        else:
            raw_env = (
                self.lindblad_env_proj(h_norm)
                if self.lindblad_env_proj is not None
                else None
            )
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
                a_nu = torch.exp(-2.0 * eta * dt_f) if raw_env is not None else None

                if state is None:
                    q_prev = torch.zeros_like(dt_f)
                    p_prev = torch.zeros_like(dt_f)
                    nu_prev = torch.zeros_like(dt_f)
                else:
                    q_prev = state[0].float()
                    p_prev = state[1].float()
                    if q_prev.ndim == 2:
                        q_prev = q_prev.unsqueeze(1)
                        p_prev = p_prev.unsqueeze(1)
                    if len(state) >= 3 and state[2] is not None:
                        nu_prev = state[2].float()
                        if nu_prev.ndim == 2:
                            nu_prev = nu_prev.unsqueeze(1)
                    else:
                        nu_prev = torch.zeros_like(dt_f)

                if reset_mask is not None:
                    keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
                    q_prev = q_prev * keep
                    p_prev = p_prev * keep
                    nu_prev = nu_prev * keep

                u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
                u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()

                q_f = a_cos * q_prev - a_sin * p_prev + u_q_f
                p_f = a_sin * q_prev + a_cos * p_prev + u_p_f
                if (
                    raw_env is not None
                    and a_nu is not None
                    and self.lindblad_gain is not None
                ):
                    g_sq = self.lindblad_gain.float().square().view(1, 1, -1)
                    nu_f = a_nu * nu_prev + dt_f * g_sq * F.softplus(raw_env.float())
                else:
                    nu_f = None

        readout = self._compute_readout(q_f, p_f, x_step.dtype)
        q_t = q_f.to(state_dtype)
        p_t = p_f.to(state_dtype)
        dtheta = dtheta_f.to(x_step.dtype)

        x_out = x_step + readout
        x_out = x_out + self.ffn(self.norm2(x_out))
        if self.use_lindblad_purity and nu_f is not None:
            return x_out, (q_t, p_t, nu_f.to(state_dtype)), dtheta
        return x_out, (q_t, p_t), dtheta

    def multistep_decode(
        self,
        x_block: Tensor,  # [B, S_max, D]
        q_init: Tensor,  # [B, M]
        p_init: Tensor,  # [B, M]
        valid_lens: Tensor,  # [B]
        nu_init: Tensor | None = None,  # [B, M]
        state_dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Batched pre/post-GEMM + fused multi-step decode over [B, S_max, D]."""
        h_norm = self.norm1(x_block)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))

        if self.use_symplectic_butterfly:
            a_mat, u_vec, dtheta_f, a_nu, u_nu = self._build_sp4_transition_and_forcing(
                h_norm, r_q, r_p, dt, reset_mask=None
            )
            with torch.amp.autocast(device_type=x_block.device.type, enabled=False):
                z_init = self._pack_sp4_state(q_init.float(), p_init.float())
                z_seq, z_final, _, nu_final = fused_multistep_sp4_decode(
                    a_mat=a_mat,
                    u_vec=u_vec,
                    z_init=z_init,
                    valid_lens=valid_lens,
                    a_nu=a_nu,
                    u_nu=u_nu,
                    nu_init=nu_init.float() if nu_init is not None else None,
                    use_triton=self.use_triton_scan,
                )
                q_seq_f, p_seq_f = self._unpack_sp4_state(z_seq)
                q_fin_f, p_fin_f = self._unpack_sp4_state(z_final.unsqueeze(1))
                q_fin_f = q_fin_f.squeeze(1)
                p_fin_f = p_fin_f.squeeze(1)
        else:
            raw_env = (
                self.lindblad_env_proj(h_norm)
                if self.lindblad_env_proj is not None
                else None
            )
            with torch.amp.autocast(device_type=x_block.device.type, enabled=False):
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
                u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
                u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()
                if raw_env is not None and self.lindblad_gain is not None:
                    a_nu = torch.exp(-2.0 * eta * dt_f)
                    g_sq = self.lindblad_gain.float().square().view(1, 1, -1)
                    u_nu = dt_f * g_sq * F.softplus(raw_env.float())
                else:
                    a_nu, u_nu = None, None

                q_seq_f, p_seq_f, q_fin_f, p_fin_f, _, nu_final = (
                    fused_multistep_complex_decode(
                        a_cos=a_cos,
                        a_sin=a_sin,
                        u_q=u_q_f,
                        u_p=u_p_f,
                        q_init=q_init.float(),
                        p_init=p_init.float(),
                        valid_lens=valid_lens,
                        a_nu=a_nu,
                        u_nu=u_nu,
                        nu_init=nu_init.float() if nu_init is not None else None,
                    )
                )

        readout = self._compute_readout(q_seq_f, p_seq_f, x_block.dtype)
        x_out = x_block + readout
        x_out = x_out + self.ffn(self.norm2(x_out))
        return (
            x_out,
            q_fin_f.to(state_dtype),
            p_fin_f.to(state_dtype),
            nu_final.to(state_dtype) if nu_final is not None else None,
            dtheta_f.to(x_block.dtype),
        )

    def forward(
        self,
        x: Tensor,
        reset_mask: Tensor | None = None,
        initial_state: tuple[Tensor, ...] | None = None,
        state_dtype: torch.dtype = torch.float32,
        return_nu: bool = False,
    ):
        """Run O(T log T) parallel scan; returns FP32 (q_seq, p_seq[, nu_seq])."""
        h_norm = self.norm1(x)
        r_q, r_p = self.reupload(h_norm)
        dt = F.softplus(self.dt_proj(h_norm))

        init_q = (
            initial_state[0].float()
            if (initial_state and len(initial_state) >= 2)
            else None
        )
        init_p = (
            initial_state[1].float()
            if (initial_state and len(initial_state) >= 2)
            else None
        )
        init_nu = (
            initial_state[2].float()
            if (
                initial_state
                and len(initial_state) >= 3
                and initial_state[2] is not None
            )
            else None
        )

        if self.use_symplectic_butterfly:
            a_mat, u_vec, dtheta_f, a_nu, u_nu = self._build_sp4_transition_and_forcing(
                h_norm, r_q, r_p, dt, reset_mask=reset_mask
            )
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                z_init: Tensor | None = None
                if init_q is not None and init_p is not None:
                    q0 = init_q.squeeze(1) if init_q.ndim == 3 else init_q
                    p0 = init_p.squeeze(1) if init_p.ndim == 3 else init_p
                    z_init = self._pack_sp4_state(q0, p0)
                z_seq = segmented_sp4_scan(
                    a_mat, u_vec, initial_state=z_init, use_triton=self.use_triton_scan
                )
                q_f, p_f = self._unpack_sp4_state(z_seq)
                nu_f = (
                    segmented_scalar_scan(
                        a_nu,
                        u_nu,
                        initial_state=init_nu,
                        use_triton=self.use_triton_scan,
                    )
                    if (a_nu is not None and u_nu is not None)
                    else None
                )
        else:
            raw_env = (
                self.lindblad_env_proj(h_norm)
                if self.lindblad_env_proj is not None
                else None
            )
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
                a_nu = torch.exp(-2.0 * eta * dt_f) if raw_env is not None else None

                if reset_mask is not None:
                    keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
                    a_cos = a_cos * keep
                    a_sin = a_sin * keep
                    if a_nu is not None:
                        a_nu = a_nu * keep

                u_q_f = dt_f * self.in_gain_q.float().view(1, 1, -1) * r_q.float()
                u_p_f = dt_f * self.in_gain_p.float().view(1, 1, -1) * r_p.float()

                c_init = (
                    (init_q, init_p)
                    if (init_q is not None and init_p is not None)
                    else None
                )
                q_f, p_f = segmented_complex_scan(
                    a_cos,
                    a_sin,
                    u_q_f,
                    u_p_f,
                    initial_state=c_init,
                    use_triton=self.use_triton_scan,
                )
                if (
                    raw_env is not None
                    and a_nu is not None
                    and self.lindblad_gain is not None
                ):
                    g_sq = self.lindblad_gain.float().square().view(1, 1, -1)
                    u_nu = dt_f * g_sq * F.softplus(raw_env.float())
                    nu_f = segmented_scalar_scan(
                        a_nu,
                        u_nu,
                        initial_state=init_nu,
                        use_triton=self.use_triton_scan,
                    )
                else:
                    nu_f = None

        readout = self._compute_readout(q_f, p_f, x.dtype)
        x_out = x + readout
        x_out = x_out + self.ffn(self.norm2(x_out))

        q_ret = q_f.to(state_dtype)
        p_ret = p_f.to(state_dtype)
        dtheta_ret = dtheta_f.to(x.dtype)
        if return_nu:
            nu_ret = (
                nu_f.to(state_dtype) if nu_f is not None else torch.zeros_like(q_ret)
            )
            return x_out, q_ret, p_ret, dtheta_ret, nu_ret
        return x_out, q_ret, p_ret, dtheta_ret

    def mode_importance(self) -> Tensor:
        """Compute L2 group importance score of each harmonic mode m in [0 .. M-1]."""
        w = self.readout_linear.weight.float()
        m = self.n_modes
        imp_sq = (
            w[:, :m].square().mean(dim=0)
            + w[:, m : 2 * m].square().mean(dim=0)
            + w[:, 2 * m :].square().mean(dim=0)
        )
        return torch.sqrt(imp_sq + 1e-12)

    def lasso_penalty(self) -> Tensor:
        """Dyadic Group-LASSO (L_{2,1}) + L1 penalty over mode/dyad readouts."""
        imp = self.mode_importance()
        if self._has_dyadic_structure() and self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(
                imp[0::2].square() + imp[1::2].square() + 1e-12
            ).mean()
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
        """Select sorted mode indices (dyad-aligned when dyadic modules are active)."""
        dev = self.readout_linear.weight.device
        if target_modes <= 0 or target_modes >= self.n_modes:
            return torch.arange(self.n_modes, device=dev)

        if self._has_dyadic_structure() and target_modes % 2 != 0:
            raise ValueError(
                "target_modes must be even when dyadic modules are active, "
                f"got {target_modes}."
            )

        importance = self.mode_importance()
        if extra_importance is not None:
            importance = importance + extra_importance.to(importance.device)

        if self._has_dyadic_structure() and self.n_modes % 2 == 0:
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
        """Structurally prune this layer to a sorted index tensor ``top_idx``."""
        target_modes = int(top_idx.numel())
        old_m = self.n_modes
        dev = self.readout_linear.weight.device
        if target_modes <= 0 or target_modes >= old_m:
            return torch.arange(old_m, device=dev)

        if self._has_dyadic_structure() and target_modes % 2 != 0:
            raise ValueError(
                "target_modes must be even when dyadic modules are active, "
                f"got {target_modes}."
            )

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

        if (
            self.use_symplectic_butterfly
            and self.butterfly_phase_proj is not None
            and self.butterfly_bs_proj is not None
            and self.butterfly_couple_gate is not None
            and self.squeezing_proj is not None
        ):
            new_phase = nn.Linear(
                self.d_model,
                target_modes,
                device=dev,
                dtype=self.butterfly_phase_proj.weight.dtype,
            )
            new_phase.weight.copy_(self.butterfly_phase_proj.weight.data[top_idx])
            new_phase.bias.copy_(self.butterfly_phase_proj.bias.data[top_idx])
            self.butterfly_phase_proj = new_phase

            sq_rows = torch.stack([2 * top_idx, 2 * top_idx + 1], dim=-1).reshape(-1)
            new_sq = nn.Linear(
                self.d_model,
                2 * target_modes,
                device=dev,
                dtype=self.squeezing_proj.weight.dtype,
            )
            new_sq.weight.copy_(self.squeezing_proj.weight.data[sq_rows])
            new_sq.bias.copy_(self.squeezing_proj.bias.data[sq_rows])
            self.squeezing_proj = new_sq

            # Preserve exact butterfly pairs where both partners survive; pair
            # singletons with B = I_4.
            top_list = [int(i) for i in top_idx.tolist()]
            old_to_new = {old_i: new_i for new_i, old_i in enumerate(top_list)}
            old_perm_list = [int(i) for i in self.butterfly_perm.tolist()]

            new_perm_list: list[int] = []
            kept_bs_w: list[Tensor] = []
            kept_bs_b: list[Tensor] = []
            old_bs_w = self.butterfly_bs_proj.weight.data
            old_bs_b = self.butterfly_bs_proj.bias.data
            paired_new: set[int] = set()

            for d_old in range(old_m // 2):
                oa = old_perm_list[2 * d_old]
                ob = old_perm_list[2 * d_old + 1]
                if oa in old_to_new and ob in old_to_new:
                    na = old_to_new[oa]
                    nb = old_to_new[ob]
                    new_perm_list.extend([na, nb])
                    paired_new.add(na)
                    paired_new.add(nb)
                    kept_bs_w.append(old_bs_w[2 * d_old : 2 * d_old + 2])
                    kept_bs_b.append(old_bs_b[2 * d_old : 2 * d_old + 2])

            singletons = [ni for ni in range(target_modes) if ni not in paired_new]
            new_gate = self.butterfly_couple_gate.data[top_idx].clone()
            for s_idx in range(0, len(singletons), 2):
                na = singletons[s_idx]
                nb = singletons[s_idx + 1]
                new_perm_list.extend([na, nb])
                new_gate[na] = 0.0
                new_gate[nb] = 0.0
                kept_bs_w.append(torch.zeros_like(old_bs_w[:2]))
                kept_bs_b.append(torch.zeros_like(old_bs_b[:2]))

            new_perm = torch.tensor(new_perm_list, dtype=torch.long, device=dev)
            new_inv_perm = torch.argsort(new_perm)
            self.butterfly_perm = new_perm
            self.butterfly_inv_perm = new_inv_perm
            self.butterfly_couple_gate = nn.Parameter(new_gate)

            new_bs = nn.Linear(
                self.d_model,
                target_modes,  # 2 * (target_modes // 2)
                device=dev,
                dtype=old_bs_w.dtype,
            )
            new_bs.weight.copy_(torch.cat(kept_bs_w, dim=0))
            new_bs.bias.copy_(torch.cat(kept_bs_b, dim=0))
            self.butterfly_bs_proj = new_bs

        if self.lindblad_env_proj is not None and self.lindblad_gain is not None:
            new_env = nn.Linear(
                self.d_model,
                target_modes,
                device=dev,
                dtype=self.lindblad_env_proj.weight.dtype,
            )
            new_env.weight.copy_(self.lindblad_env_proj.weight.data[top_idx])
            new_env.bias.copy_(self.lindblad_env_proj.bias.data[top_idx])
            self.lindblad_env_proj = new_env
            self.lindblad_gain = nn.Parameter(self.lindblad_gain.data[top_idx].clone())

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
        top_idx = self.select_top_indices(
            target_modes, extra_importance=extra_importance
        )
        return self.prune_to_indices(
            top_idx, compensate_zero_state_bias=compensate_zero_state_bias
        )


class MultiHorizonKoopmanRollout(nn.Module):
    """One-shot analytical driven-damped Koopman spectral rollout for k = 0 .. K - 1.

    Supports multi-layer modal state aggregation (including Lindblad covariance),
    inhomogeneous horizon forcing u_{k,m}(h_anchor), pairwise symplectic interferometry
    (Barthe Ch. 7 Eq. 7.25), and Barthe Ch. 4 Thm. 4.2 Eq. 4.34 vacuum-Born + purity
    stats.
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
        use_lindblad_purity: bool = True,
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
        self.use_lindblad_purity = use_lindblad_purity

        self.horizon_freq_scale = nn.Parameter(torch.ones(block_size, n_modes))
        self.horizon_log_decay = nn.Parameter(torch.full((block_size, n_modes), -2.5))
        self.horizon_phase_bias = nn.Parameter(torch.zeros(block_size, n_modes))

        # Inhomogeneous horizon forcing u_{k,m}(h_anchor) [2 * K * M]: zero-init for
        # exact
        # dead-mode stability while receiving full step-1 gradients via obs_linear.
        self.horizon_force_proj = nn.Linear(d_model, 2 * block_size * n_modes)
        nn.init.zeros_(self.horizon_force_proj.weight)
        nn.init.zeros_(self.horizon_force_proj.bias)

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
        nu_list: list[Tensor] | None = None,
    ):
        """Aggregate modal states across draft layers with per-mode softmax weights."""
        if (
            not self.multilayer_rollout
            or self.layer_mode_logits is None
            or len(q_list) == 1
        ):
            if nu_list is not None:
                return q_list[-1], p_list[-1], dtheta_list[-1], nu_list[-1]
            return q_list[-1], p_list[-1], dtheta_list[-1]

        weights = torch.softmax(self.layer_mode_logits.float(), dim=0).unsqueeze(1)
        q_stack = torch.stack([q.float() for q in q_list], dim=0)
        p_stack = torch.stack([p.float() for p in p_list], dim=0)
        dt_stack = torch.stack([dt.float() for dt in dtheta_list], dim=0)
        q_agg = (weights * q_stack).sum(dim=0).to(q_list[-1].dtype)
        p_agg = (weights * p_stack).sum(dim=0).to(p_list[-1].dtype)
        dt_agg = (weights * dt_stack).sum(dim=0).to(dtheta_list[-1].dtype)

        if nu_list is not None:
            nu_stack = torch.stack([nu.float() for nu in nu_list], dim=0)
            nu_agg = (weights * nu_stack).sum(dim=0).to(nu_list[-1].dtype)
            return q_agg, p_agg, dt_agg, nu_agg
        return q_agg, p_agg, dt_agg

    def forward(
        self,
        anchor_h: Tensor,  # [N_blocks, d_model]
        anchor_q: Tensor,  # [N_blocks, n_modes]
        anchor_p: Tensor,  # [N_blocks, n_modes]
        anchor_dtheta: Tensor,  # [N_blocks, n_modes]
        anchor_nu: Tensor | None = None,  # [N_blocks, n_modes]
        return_quantum_stats: bool = False,
    ):
        """Roll out horizons k=0..K-1; returns (hidden_blocks, born_features[, p_vac,
        purity, q_obs, p_obs]).
        """
        n_blocks = anchor_h.size(0)
        k_steps = torch.arange(
            1, self.block_size + 1, device=anchor_h.device, dtype=torch.float32
        ).view(1, self.block_size, 1)

        angles = k_steps * anchor_dtheta.float().unsqueeze(
            1
        ) * self.horizon_freq_scale.float().unsqueeze(
            0
        ) + self.horizon_phase_bias.float().unsqueeze(0)
        eta_hor = torch.exp(self.horizon_log_decay.float()).unsqueeze(0)
        decay = torch.exp(-k_steps * eta_hor)

        cos_a = decay * torch.cos(angles)
        sin_a = decay * torch.sin(angles)

        force_raw = (
            self.horizon_force_proj(anchor_h)
            .float()
            .view(n_blocks, 2, self.block_size, self.n_modes)
        )
        inv_sqrt_m = 1.0 / math.sqrt(max(1, self.n_modes))
        u_q_hor = inv_sqrt_m * force_raw[:, 0]
        u_p_hor = inv_sqrt_m * force_raw[:, 1]

        q_0 = anchor_q.float().unsqueeze(1)
        p_0 = anchor_p.float().unsqueeze(1)
        q_k = cos_a * q_0 - sin_a * p_0 + u_q_hor
        p_k = sin_a * q_0 + cos_a * p_0 + u_p_hor

        if self.interferometer is not None:
            q_obs, p_obs = self.interferometer(q_k, p_k)
        else:
            q_obs, p_obs = q_k, p_k

        if self.vacuum_born_norm and self.vacuum_log_kappa is not None:
            born_f, p_vac_f = compute_vacuum_born_observables(
                q_obs, p_obs, self.vacuum_log_kappa
            )
        else:
            born_f = (2.0 * self.n_modes) * (q_obs.square() + p_obs.square()) - 1.0
            p_vac_f = torch.ones(
                (n_blocks, self.block_size, max(1, self.n_modes // 2)),
                dtype=torch.float32,
                device=anchor_h.device,
            )

        if anchor_nu is not None:
            nu_k = (decay * decay) * anchor_nu.float().clamp_min(0.0).unsqueeze(1)
            purity_f = 1.0 / (1.0 + 2.0 * nu_k)
        else:
            purity_f = torch.ones_like(q_obs)

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

        if return_quantum_stats:
            return (
                hidden_blocks,
                born_features,
                p_vac_f.to(anchor_h.dtype),
                purity_f.to(anchor_h.dtype),
                q_obs,
                p_obs,
            )
        return hidden_blocks, born_features

    def mode_importance(self) -> Tensor:
        """Compute L2 group importance of each mode in rollout obs_linear +
        horizon_force_proj.
        """
        w = self.obs_linear.weight.float()
        m = self.n_modes
        imp_q = w[:, :m].square().mean(dim=0)
        imp_p = w[:, m : 2 * m].square().mean(dim=0)
        imp_b = w[:, 2 * m :].square().mean(dim=0)
        wf = self.horizon_force_proj.weight.float().view(
            2 * self.block_size, m, self.d_model
        )
        imp_f = wf.square().mean(dim=(0, 2))
        return torch.sqrt(imp_q + imp_p + imp_b + imp_f + 1e-12)

    def lasso_penalty(self) -> Tensor:
        """Dyadic Group-LASSO + L1 penalty on rollout mode readout weights."""
        imp = self.mode_importance()
        if (
            self.interferometer is not None or self.vacuum_born_norm
        ) and self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(
                imp[0::2].square() + imp[1::2].square() + 1e-12
            ).mean()
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
        if target_modes <= 0 or target_modes >= old_m:
            return
        if (
            self.interferometer is not None or self.vacuum_born_norm
        ) and target_modes % 2 != 0:
            raise ValueError(
                "target_modes must be even when dyadic modules are active, "
                f"got {target_modes}."
            )

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

        wf_old = self.horizon_force_proj.weight.data.view(
            2, self.block_size, old_m, self.d_model
        )
        bf_old = self.horizon_force_proj.bias.data.view(2, self.block_size, old_m)
        wf_new = wf_old[:, :, top_idx, :].reshape(
            2 * self.block_size * target_modes, self.d_model
        )
        bf_new = bf_old[:, :, top_idx].reshape(2 * self.block_size * target_modes)
        new_force = nn.Linear(
            self.d_model,
            2 * self.block_size * target_modes,
            device=dev,
            dtype=self.horizon_force_proj.weight.dtype,
        )
        new_force.weight.copy_(wf_new)
        new_force.bias.copy_(bf_new)
        self.horizon_force_proj = new_force

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
    """Zero-KV-Cache intra-block causal mixer over speculative horizons k = 0..K-1."""

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
        return h_out + self.down_proj(
            F.silu(self.gate_proj(normed2)) * self.up_proj(normed2)
        )


class BornConfidenceHead(nn.Module):
    """Acceptance probability head fusing interferometric Born energy, vacuum p_vac, and
    Lindblad purity.
    """

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int = 0,
        n_modes: int = 0,
        use_born: bool = True,
        intermediate_size: int | None = None,
        vacuum_born_norm: bool = True,
        use_lindblad_purity: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.n_modes = n_modes
        self.use_born = use_born and (n_modes > 0)
        self.vacuum_born_norm = vacuum_born_norm and (n_modes % 2 == 0)
        self.use_lindblad_purity = use_lindblad_purity and self.use_born

        base_dim = hidden_size + markov_rank
        mid = intermediate_size or min(512, max(64, hidden_size // 2))
        self.base_norm = nn.RMSNorm(base_dim)
        self.base_linear = nn.Linear(base_dim, mid)
        self.born_proj: nn.Linear | None = (
            nn.Linear(n_modes, mid, bias=False) if self.use_born else None
        )
        self.vacuum_pvac_proj: nn.Linear | None = (
            nn.Linear(n_modes // 2, mid, bias=False)
            if (self.use_born and self.vacuum_born_norm)
            else None
        )
        self.lindblad_purity_proj: nn.Linear | None = (
            nn.Linear(n_modes, mid, bias=False) if self.use_lindblad_purity else None
        )
        if self.vacuum_pvac_proj is not None:
            nn.init.normal_(self.vacuum_pvac_proj.weight, mean=0.0, std=0.02)
        if self.lindblad_purity_proj is not None:
            nn.init.normal_(self.lindblad_purity_proj.weight, mean=0.0, std=0.02)
        self.out_linear = nn.Linear(mid, 1)

    def forward(
        self,
        hidden_blocks: Tensor,
        prev_emb: Tensor | None = None,
        born_features: Tensor | None = None,
        p_vac: Tensor | None = None,
        purity: Tensor | None = None,
    ) -> Tensor:
        if prev_emb is not None and self.markov_rank > 0:
            base = torch.cat([hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1)
        else:
            base = hidden_blocks
        h = self.base_linear(self.base_norm(base))
        if self.use_born and self.born_proj is not None and born_features is not None:
            m = self.n_modes
            half_m = m // 2
            if (
                born_features.size(-1) == (2 * m + half_m)
                and p_vac is None
                and purity is None
            ):
                born_only = born_features[..., :m]
                p_vac = born_features[..., m : m + half_m]
                purity = born_features[..., m + half_m :]
            else:
                born_only = born_features
            h = h + self.born_proj(born_only.to(h.dtype))
            if self.vacuum_pvac_proj is not None and p_vac is not None:
                # Dead dyad (q=p=0) has p_vac == 1.0 -> (p_vac - 1.0) == 0.0 (zero
                # pruning drift)
                h = h + self.vacuum_pvac_proj((p_vac.float() - 1.0).to(h.dtype))
            if self.lindblad_purity_proj is not None and purity is not None:
                # Pure un-driven mode (delta_nu == 0) has purity == 1.0 -> (purity -
                # 1.0) == 0.0
                h = h + self.lindblad_purity_proj((purity.float() - 1.0).to(h.dtype))
        return self.out_linear(F.silu(h)).squeeze(-1)

    def mode_importance(self) -> Tensor | None:
        if not self.use_born or self.born_proj is None:
            return None
        imp_sq = self.born_proj.weight.float().square().mean(dim=0)
        if self.lindblad_purity_proj is not None:
            imp_sq = imp_sq + self.lindblad_purity_proj.weight.float().square().mean(
                dim=0
            )
        if self.vacuum_pvac_proj is not None and self.n_modes % 2 == 0:
            pvac_imp = self.vacuum_pvac_proj.weight.float().square().mean(dim=0)
            imp_sq = imp_sq + 0.5 * pvac_imp.repeat_interleave(2)
        return torch.sqrt(imp_sq + 1e-12)

    def lasso_penalty(self) -> Tensor:
        if not self.use_born or self.born_proj is None:
            return torch.tensor(0.0, device=self.base_linear.weight.device)
        imp = self.mode_importance()
        assert imp is not None
        if self.n_modes % 2 == 0:
            group_l21 = torch.sqrt(
                imp[0::2].square() + imp[1::2].square() + 1e-12
            ).mean()
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
        if target_modes <= 0 or target_modes >= old_m:
            return
        if self.vacuum_born_norm and target_modes % 2 != 0:
            raise ValueError(
                "target_modes must be even when vacuum_born_norm=True, "
                f"got {target_modes}."
            )

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

        if self.vacuum_pvac_proj is not None:
            dyad_idx = top_idx[0::2] // 2
            new_pvac = nn.Linear(
                target_modes // 2,
                self.vacuum_pvac_proj.out_features,
                bias=False,
                device=dev,
                dtype=self.vacuum_pvac_proj.weight.dtype,
            )
            new_pvac.weight.copy_(self.vacuum_pvac_proj.weight.data[:, dyad_idx])
            self.vacuum_pvac_proj = new_pvac

        if self.lindblad_purity_proj is not None:
            new_pur = nn.Linear(
                target_modes,
                self.lindblad_purity_proj.out_features,
                bias=False,
                device=dev,
                dtype=self.lindblad_purity_proj.weight.dtype,
            )
            new_pur.weight.copy_(self.lindblad_purity_proj.weight.data[:, top_idx])
            self.lindblad_purity_proj = new_pur

        self.n_modes = target_modes


class KrausPOVMHead(nn.Module):
    """Diagonal Kraus POVM Quantum Measurement & Back-Action Head (Axe 5).

    Normalizes ``p_mode(y | j)`` via ``softmax`` across active top-C candidates ``y in
    C_k``
    for each mode ``j in {0..M}`` (M Bosonic modes + 1 vacuum mode), guaranteeing the
    exact POVM completeness identity ``sum_{y in C_k} M_y^dagger M_y = I_{M+1}``
    identically,
    and computes Lüders-Kraus post-measurement modal collapse for speculative tree
    expansion.
    """

    def __init__(self, d_model: int, n_modes: int, top_c: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.top_c = top_c

        self.kraus_mode_proj = nn.Linear(d_model, n_modes + 1, bias=True)
        nn.init.normal_(self.kraus_mode_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.kraus_mode_proj.bias)

        self.kraus_backaction_proj = nn.Linear(2 * n_modes, d_model, bias=False)
        nn.init.normal_(self.kraus_backaction_proj.weight, mean=0.0, std=1e-3)

        self.kraus_mode_gate = nn.Parameter(torch.ones(n_modes))
        self.kraus_log_kappa = nn.Parameter(torch.zeros(()))
        self.kraus_logit_scale = nn.Parameter(torch.tensor(0.05))

    def compute_povm(
        self,
        cand_logits: Tensor,  # [..., C]
        cand_vecs: Tensor,  # [..., C, D]
        q_state: Tensor,  # [..., M]
        p_state: Tensor,  # [..., M]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(p_kraus [..., C], povm_diag [..., C, M+1], psi_sq [..., M+1])``."""
        m = self.n_modes
        scores = self.kraus_mode_proj(
            cand_vecs
        ).float() + cand_logits.float().unsqueeze(-1)
        # Softmax over candidate axis (-2) guarantees sum_{y in C} povm_diag[y, j] == 1
        # for all j!
        povm_diag = torch.softmax(scores, dim=-2)

        gate_sq = self.kraus_mode_gate.float().square()
        e_modes = (
            float(m) * gate_sq * (q_state.float().square() + p_state.float().square())
        )
        e_vac = (
            torch.exp(2.0 * self.kraus_log_kappa.float())
            .clamp(1e-4, 1e4)
            .expand_as(e_modes[..., :1])
        )
        e_all = torch.cat([e_modes, e_vac], dim=-1)
        psi_sq = e_all / e_all.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        p_kraus = (povm_diag * psi_sq.unsqueeze(-2)).sum(dim=-1)
        return p_kraus, povm_diag, psi_sq

    def forward_topc_logit_delta(
        self,
        logits: Tensor,  # [N, K, V_draft]
        q_obs: Tensor,  # [N, K, M]
        p_obs: Tensor,  # [N, K, M]
        lm_head_weight: Tensor,  # [V_draft, D]
    ) -> Tensor:
        """Differentiable top-C Kraus POVM log-Born calibration residual over [N, K,
        V_draft].
        """
        v_draft = logits.size(-1)
        eff_c = min(self.top_c, v_draft)
        with torch.no_grad():
            top_idx = logits.topk(eff_c, dim=-1).indices  # [N, K, C]

        cand_logits = logits.gather(dim=-1, index=top_idx)
        cand_vecs = lm_head_weight.index_select(0, top_idx.reshape(-1)).view(
            *top_idx.shape, self.d_model
        )
        p_kraus, povm_diag, _ = self.compute_povm(cand_logits, cand_vecs, q_obs, p_obs)

        # Collapse top-1 modal amplitude to keep kraus_backaction_proj actively trained
        # end-to-end
        top1_gain = torch.sqrt(povm_diag[..., 0, : self.n_modes].clamp_min(1e-12))
        q_col = (q_obs.float() * top1_gain).to(lm_head_weight.dtype)
        p_col = (p_obs.float() * top1_gain).to(lm_head_weight.dtype)
        back_h = self.kraus_backaction_proj(torch.cat([q_col, p_col], dim=-1))
        back_cand = (cand_vecs.to(back_h.dtype) * back_h.unsqueeze(-2)).sum(dim=-1)

        scale = self.kraus_logit_scale.to(logits.dtype)
        cand_delta = scale * (
            torch.log(p_kraus.clamp_min(1e-8)).to(logits.dtype)
            + back_cand.to(logits.dtype)
        )
        delta_full = torch.zeros_like(logits)
        delta_full.scatter_add_(dim=-1, index=top_idx, src=cand_delta)
        return delta_full

    def collapse_branch_state(
        self,
        q_parent: Tensor,  # [..., M]
        p_parent: Tensor,  # [..., M]
        povm_row: Tensor,  # [..., M+1]
        out_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Lüders-Kraus post-measurement collapse M_y |psi> / ||M_y |psi>||."""
        mode_w = povm_row[..., : self.n_modes].float().clamp_min(1e-12)
        vac_w = povm_row[..., self.n_modes : self.n_modes + 1].float().clamp_min(1e-12)
        # Lüders-Kraus collapse relative to post-measurement vacuum amplitude
        # (pruning-invariant)
        normed_amp = torch.sqrt(mode_w / vac_w).clamp(min=0.1, max=10.0)
        q_child = q_parent.float() * normed_amp
        p_child = p_parent.float() * normed_amp
        qp_cat = torch.cat([q_child.to(out_dtype), p_child.to(out_dtype)], dim=-1)
        h_delta = self.kraus_backaction_proj(qp_cat)
        return q_child, p_child, h_delta

    @torch.no_grad()
    def prune_to_indices(self, top_idx: Tensor) -> None:
        old_m = self.n_modes
        target_modes = int(top_idx.numel())
        if target_modes <= 0 or target_modes >= old_m:
            return
        dev = self.kraus_mode_proj.weight.device
        top_idx = top_idx.to(dev)
        scale_ratio = math.sqrt(target_modes / old_m)

        mode_and_vac = torch.cat(
            [top_idx, torch.tensor([old_m], dtype=torch.long, device=dev)], dim=0
        )
        new_mode_proj = nn.Linear(
            self.d_model,
            target_modes + 1,
            bias=True,
            device=dev,
            dtype=self.kraus_mode_proj.weight.dtype,
        )
        new_mode_proj.weight.copy_(self.kraus_mode_proj.weight.data[mode_and_vac])
        new_mode_proj.bias.copy_(self.kraus_mode_proj.bias.data[mode_and_vac])
        self.kraus_mode_proj = new_mode_proj

        w_back = self.kraus_backaction_proj.weight.data
        wb_q = w_back[:, top_idx] * scale_ratio
        wb_p = w_back[:, old_m + top_idx] * scale_ratio
        new_back = nn.Linear(
            2 * target_modes,
            self.d_model,
            bias=False,
            device=dev,
            dtype=w_back.dtype,
        )
        new_back.weight.copy_(torch.cat([wb_q, wb_p], dim=-1))
        self.kraus_backaction_proj = new_back
        self.kraus_mode_gate = nn.Parameter(self.kraus_mode_gate.data[top_idx].clone())
        self.n_modes = target_modes


def specinfer_verify_single_step(
    candidate_ids: Tensor,  # [W] int64 candidate token IDs in verification order
    candidate_q_probs: Tensor,  # [W, V] float32 proposal distributions q^(r)
    target_probs: Tensor,  # [V] float32 verifier target distribution p
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Lossless SpecInfer multi-candidate rejection sampling at a single tree node.

    Returns ``(emitted_token_id, accepted_branch_idx)``, where ``accepted_branch_idx``
    is
    in ``0 .. W - 1`` if one of the drafted branches was accepted, or ``-1`` if all W
    branches were rejected and a bonus token was sampled from the exact residual target.
    """
    p_res = target_probs.float().clone()
    p_res = p_res / p_res.sum().clamp_min(1e-12)
    w = int(candidate_ids.numel())

    for r in range(w):
        tok = int(candidate_ids[r].item())
        q_r = candidate_q_probs[r].float()
        q_r = q_r / q_r.sum().clamp_min(1e-12)

        p_tok = float(p_res[tok].item())
        q_tok = float(q_r[tok].item())
        accept_prob = (
            1.0 if q_tok <= 0.0 and p_tok > 0.0 else min(1.0, p_tok / max(q_tok, 1e-12))
        )
        if p_tok <= 0.0:
            accept_prob = 0.0

        u = float(
            torch.rand((), device=target_probs.device, generator=generator).item()
        )
        if u < accept_prob:
            return tok, r

        p_res = (p_res - q_r).clamp_min(0.0)
        rem_mass = float(p_res.sum().item())
        if rem_mass <= 1e-12:
            # Target == proposal identically; fallback numerical guard
            return tok, r
        p_res = p_res / rem_mass

    bonus_tok = int(torch.multinomial(p_res, num_samples=1, generator=generator).item())
    return bonus_tok, -1


def specinfer_verify_tree(
    tree_draft_ids: Tensor,  # [N_nodes] int64
    tree_parent_idx: Tensor,  # [N_nodes] int64 (-1 for root children)
    tree_q_probs: Tensor,  # [N_nodes, V] float32
    node_target_probs: Tensor,  # [N_nodes + 1, V] float32 (0=root, i+1=after node i)
    generator: torch.Generator | None = None,
) -> tuple[list[int], list[int]]:
    """Verify a speculative tree root-to-leaf using lossless SpecInfer rejection
    sampling.

    Returns ``(emitted_tokens, accepted_node_indices)`` (including the final bonus
    token).
    """
    n_nodes = int(tree_draft_ids.numel())
    parents = [int(x) for x in tree_parent_idx.tolist()]
    emitted: list[int] = []
    accepted_nodes: list[int] = []

    cur_parent = -1
    while True:
        children = [i for i in range(n_nodes) if parents[i] == cur_parent]
        target_p = node_target_probs[0 if cur_parent == -1 else (cur_parent + 1)]
        if not children:
            p_norm = target_p.float() / target_p.float().sum().clamp_min(1e-12)
            bonus = int(
                torch.multinomial(p_norm, num_samples=1, generator=generator).item()
            )
            emitted.append(bonus)
            break

        child_idx_t = torch.tensor(
            children, dtype=torch.long, device=tree_draft_ids.device
        )
        tok, rel_r = specinfer_verify_single_step(
            candidate_ids=tree_draft_ids.index_select(0, child_idx_t),
            candidate_q_probs=tree_q_probs.index_select(0, child_idx_t),
            target_probs=target_p,
            generator=generator,
        )
        emitted.append(tok)
        if rel_r < 0:
            break
        chosen_node = children[rel_r]
        accepted_nodes.append(chosen_node)
        cur_parent = chosen_node

    return emitted, accepted_nodes
