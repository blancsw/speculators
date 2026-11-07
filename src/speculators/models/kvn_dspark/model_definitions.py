from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "BornConfidenceHead",
    "CausalKvNRFFLayer",
    "DataReuploadingLinear",
    "MultiHorizonKoopmanRollout",
]


class DataReuploadingLinear(nn.Module):
    """Multi-stage Fourier Data Re-uploading encoder (Barthe Ch. 3, Eq. 3.32)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(x)
        for l_idx in range(self.reupload_layers_l):
            proj = F.linear(x, self.weights[l_idx], self.biases[l_idx])
            h = h + proj
            if l_idx < self.reupload_layers_l - 1:
                h = torch.sin(h) + F.linear(torch.cos(h), self.phase_Mix[l_idx])
        return self.norm(h)


class CausalKvNRFFLayer(nn.Module):
    """Causal Koopman-von Neumann Bosonic RFF Layer with Barthe LASSO mode pruning.

    Operates in O(T log T) parallel FFT convolution during training and O(1)
    symplectic state (q_t, p_t) in R^{2M} during autoregressive inference.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int = 256,
        reupload_layers_l: int = 2,
        sigma_g: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.reupload_layers_l = reupload_layers_l
        self.sigma_g = sigma_g

        self.norm1 = nn.RMSNorm(d_model)
        self.reupload = DataReuploadingLinear(
            d_model=d_model,
            reupload_layers_l=reupload_layers_l,
        )

        omega_init = torch.randn(n_modes, d_model) * (
            sigma_g * reupload_layers_l / math.sqrt(d_model)
        )
        bias_init = torch.rand(n_modes) * (2.0 * math.pi)
        self.register_buffer("omega", omega_init)
        self.register_buffer("bias", bias_init)

        self.raw_alpha = nn.Parameter(torch.zeros(n_modes))
        self.dt_proj = nn.Linear(d_model, n_modes)
        nn.init.constant_(self.dt_proj.bias, -2.0)

        self.log_eta = nn.Parameter(torch.full((n_modes,), -2.5))
        self.kvn_omega = nn.Parameter(torch.randn(n_modes) * 0.5)
        self.in_gain_q = nn.Parameter(torch.ones(n_modes) * 0.5)
        self.in_gain_p = nn.Parameter(torch.ones(n_modes) * 0.5)

        # Readout over [q_t, p_t, amplified_born_energy] -> d_model
        self.readout_norm = nn.RMSNorm(d_model)
        self.readout_linear = nn.Linear(3 * n_modes, d_model)

        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def _fft_causal_dynamics(
        self,
        u_q: torch.Tensor,
        u_p: torch.Tensor,
        dtheta_bar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact causal damped harmonic oscillator convolution in float32 via FFT."""
        with torch.amp.autocast(device_type=u_q.device.type, enabled=False):
            u_q_f = u_q.float()
            u_p_f = u_p.float()
            dtheta_f = dtheta_bar.float()

            seq_len = u_q_f.size(1)
            t_idx = torch.arange(seq_len, device=u_q_f.device, dtype=torch.float32).view(
                1, seq_len, 1
            )

            eta = torch.exp(self.log_eta.float()).clamp(min=1e-4, max=2.0).view(1, 1, -1)
            decay = torch.exp(-eta * t_idx)
            phase = dtheta_f * t_idx

            k_cos = decay * torch.cos(phase)
            k_sin = decay * torch.sin(phase)

            fft_size = 1 << (2 * seq_len - 1).bit_length()
            f_uq = torch.fft.rfft(u_q_f, n=fft_size, dim=1)
            f_up = torch.fft.rfft(u_p_f, n=fft_size, dim=1)
            f_kc = torch.fft.rfft(k_cos, n=fft_size, dim=1)
            f_ks = torch.fft.rfft(k_sin, n=fft_size, dim=1)

            f_q = f_uq * f_kc - f_up * f_ks
            f_p = f_uq * f_ks + f_up * f_kc

            q_seq = torch.fft.irfft(f_q, n=fft_size, dim=1)[:, :seq_len, :]
            p_seq = torch.fft.irfft(f_p, n=fft_size, dim=1)[:, :seq_len, :]
        return q_seq.to(u_q.dtype), p_seq.to(u_p.dtype)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_out [B, T, D], q_seq [B, T, M], p_seq [B, T, M], dtheta_seq [B, T, M])."""
        h = self.reupload(self.norm1(x))

        proj = F.linear(h, self.omega.to(h.dtype), self.bias.to(h.dtype))
        scale = 1.0 / math.sqrt(self.n_modes)
        z_q = scale * torch.cos(proj)
        z_p = scale * torch.sin(proj)

        alpha = torch.tanh(self.raw_alpha.to(h.dtype))
        r_q = z_q + alpha * (z_q * z_q - z_p * z_p)
        r_p = z_p + 2.0 * alpha * (z_q * z_p)

        dt = F.softplus(self.dt_proj(h))
        dtheta = dt * self.kvn_omega.to(h.dtype)

        u_q = dt * self.in_gain_q.to(h.dtype) * r_q
        u_p = dt * self.in_gain_p.to(h.dtype) * r_p

        dtheta_bar = dtheta.mean(dim=1, keepdim=True)
        q_seq, p_seq = self._fft_causal_dynamics(u_q, u_p, dtheta_bar)

        # Barthe Ch. 4 Theorem 4.2 amplified Born-rule energy observable: 2M*(q^2 + p^2) - 1
        born_energy = (2.0 * self.n_modes) * (q_seq.square() + p_seq.square()) - 1.0
        obs = torch.cat([q_seq, p_seq, born_energy], dim=-1)
        readout = self.readout_norm(self.readout_linear(obs))

        x = x + readout
        x = x + self.ffn(self.norm2(x))
        return x, q_seq, p_seq, dtheta

    def mode_importance(self) -> torch.Tensor:
        """Compute L1 importance score of each harmonic mode m in [0 .. M-1]."""
        w = self.readout_linear.weight  # [d_model, 3 * n_modes]
        m = self.n_modes
        imp_q = w[:, :m].abs().sum(dim=0)
        imp_p = w[:, m : 2 * m].abs().sum(dim=0)
        imp_b = w[:, 2 * m :].abs().sum(dim=0)
        return imp_q + imp_p + imp_b

    def lasso_penalty(self) -> torch.Tensor:
        """Barthe Ch. 5 Theorem 5.6 L1 regularization over mode readout weights."""
        return self.readout_linear.weight.abs().mean()

    @torch.no_grad()
    def prune_to_top_modes(self, target_modes: int) -> torch.Tensor:
        """Structurally prune inactive RFF modes (M_train -> M_infer) for HBM Roofline speed."""
        if target_modes <= 0 or target_modes >= self.n_modes:
            return torch.arange(self.n_modes, device=self.omega.device)

        old_m = self.n_modes
        importance = self.mode_importance()
        _, top_idx = torch.topk(importance, k=target_modes, largest=True, sorted=True)

        scale_ratio = math.sqrt(target_modes / old_m)

        self.omega = self.omega[top_idx].clone()
        self.bias = self.bias[top_idx].clone()
        self.raw_alpha = nn.Parameter(self.raw_alpha.data[top_idx].clone())

        new_dt = nn.Linear(
            self.d_model,
            target_modes,
            device=self.omega.device,
            dtype=self.dt_proj.weight.dtype,
        )
        new_dt.weight.copy_(self.dt_proj.weight.data[top_idx])
        new_dt.bias.copy_(self.dt_proj.bias.data[top_idx])
        self.dt_proj = new_dt

        self.log_eta = nn.Parameter(self.log_eta.data[top_idx].clone())
        self.kvn_omega = nn.Parameter(self.kvn_omega.data[top_idx].clone())
        self.in_gain_q = nn.Parameter(self.in_gain_q.data[top_idx].clone())
        self.in_gain_p = nn.Parameter(self.in_gain_p.data[top_idx].clone())

        w_old = self.readout_linear.weight.data
        w_q = w_old[:, top_idx] * scale_ratio
        w_p = w_old[:, old_m + top_idx] * scale_ratio
        w_b = w_old[:, 2 * old_m + top_idx] * (scale_ratio * scale_ratio)

        new_readout = nn.Linear(
            3 * target_modes,
            self.d_model,
            device=self.omega.device,
            dtype=self.readout_linear.weight.dtype,
        )
        new_readout.weight.copy_(torch.cat([w_q, w_p, w_b], dim=-1))
        new_readout.bias.copy_(self.readout_linear.bias.data)
        self.readout_linear = new_readout
        self.n_modes = target_modes
        return top_idx


class MultiHorizonKoopmanRollout(nn.Module):
    """One-shot analytical Koopman phase rollout for k = 0 .. block_size - 1.

    Projects the anchor symplectic state (q_t, p_t) to K future speculative steps in
    parallel without running K sequential Transformer layers or storing a KV cache.
    """

    def __init__(self, d_model: int, n_modes: int, block_size: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.block_size = block_size

        self.horizon_freq_scale = nn.Parameter(torch.ones(block_size, n_modes))
        self.horizon_log_decay = nn.Parameter(torch.full((block_size, n_modes), -2.5))
        self.horizon_phase_bias = nn.Parameter(torch.zeros(block_size, n_modes))

        self.anchor_proj = nn.Linear(d_model, block_size * d_model)
        self.obs_to_hidden = nn.Sequential(
            nn.RMSNorm(3 * n_modes),
            nn.Linear(3 * n_modes, d_model),
        )
        self.out_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        anchor_h: torch.Tensor,  # [N_blocks, d_model]
        anchor_q: torch.Tensor,  # [N_blocks, n_modes]
        anchor_p: torch.Tensor,  # [N_blocks, n_modes]
        anchor_dtheta: torch.Tensor,  # [N_blocks, n_modes]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden_blocks [N_blocks, K, d_model], born_features [N_blocks, K, n_modes])."""
        n_blocks = anchor_h.size(0)
        k_steps = torch.arange(
            1, self.block_size + 1, device=anchor_h.device, dtype=anchor_h.dtype
        ).view(1, self.block_size, 1)

        angles = (
            k_steps
            * anchor_dtheta.unsqueeze(1)
            * self.horizon_freq_scale.to(anchor_h.dtype).unsqueeze(0)
            + self.horizon_phase_bias.to(anchor_h.dtype).unsqueeze(0)
        )
        decay = torch.exp(
            -k_steps
            * torch.exp(self.horizon_log_decay.to(anchor_h.dtype)).unsqueeze(0)
        )

        cos_a = decay * torch.cos(angles)
        sin_a = decay * torch.sin(angles)

        q_0 = anchor_q.unsqueeze(1)
        p_0 = anchor_p.unsqueeze(1)
        q_k = cos_a * q_0 - sin_a * p_0
        p_k = sin_a * q_0 + cos_a * p_0

        born_prob = q_k.square() + p_k.square()
        born_features = (2.0 * self.n_modes) * born_prob - 1.0

        obs_k = torch.cat([q_k, p_k, born_features], dim=-1)  # [N_blocks, K, 3M]
        kvn_hidden = self.obs_to_hidden(obs_k)

        anchor_cond = self.anchor_proj(anchor_h).view(
            n_blocks, self.block_size, self.d_model
        )
        hidden_blocks = self.out_norm(anchor_cond + kvn_hidden)
        return hidden_blocks, born_features

    @torch.no_grad()
    def prune_to_indices(self, top_idx: torch.Tensor) -> None:
        """Align rollout mode dimension after structural LASSO pruning of the last KvN layer."""
        old_m = self.n_modes
        target_modes = int(top_idx.numel())
        if target_modes >= old_m:
            return

        self.horizon_freq_scale = nn.Parameter(
            self.horizon_freq_scale.data[:, top_idx].clone()
        )
        self.horizon_log_decay = nn.Parameter(
            self.horizon_log_decay.data[:, top_idx].clone()
        )
        self.horizon_phase_bias = nn.Parameter(
            self.horizon_phase_bias.data[:, top_idx].clone()
        )

        old_norm = self.obs_to_hidden[0]
        old_lin = self.obs_to_hidden[1]
        assert isinstance(old_norm, nn.RMSNorm) and isinstance(old_lin, nn.Linear)

        cat_idx = torch.cat(
            [top_idx, old_m + top_idx, 2 * old_m + top_idx], dim=0
        )
        new_norm = nn.RMSNorm(
            3 * target_modes, device=top_idx.device, dtype=old_lin.weight.dtype
        )
        new_norm.weight.copy_(old_norm.weight.data[cat_idx])

        new_lin = nn.Linear(
            3 * target_modes,
            self.d_model,
            device=top_idx.device,
            dtype=old_lin.weight.dtype,
        )
        new_lin.weight.copy_(old_lin.weight.data[:, cat_idx])
        new_lin.bias.copy_(old_lin.bias.data)

        self.obs_to_hidden = nn.Sequential(new_norm, new_lin)
        self.n_modes = target_modes


class BornConfidenceHead(nn.Module):
    """Per-position acceptance probability head augmented with Born-rule energy observables."""

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int = 0,
        n_modes: int = 0,
        use_born: bool = True,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.n_modes = n_modes
        self.use_born = use_born and (n_modes > 0)
        input_dim = hidden_size + markov_rank + (n_modes if self.use_born else 0)
        mid = intermediate_size or max(64, hidden_size // 2)
        self.net = nn.Sequential(
            nn.RMSNorm(input_dim),
            nn.Linear(input_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, 1),
        )

    def forward(
        self,
        hidden_blocks: torch.Tensor,
        prev_emb: torch.Tensor | None = None,
        born_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [hidden_blocks]
        if prev_emb is not None and self.markov_rank > 0:
            parts.append(prev_emb.to(hidden_blocks.dtype))
        if self.use_born and born_features is not None:
            parts.append(born_features.to(hidden_blocks.dtype))
        features = torch.cat(parts, dim=-1)
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def prune_born_modes(self, top_idx: torch.Tensor) -> None:
        if not self.use_born:
            return
        target_modes = int(top_idx.numel())
        if target_modes >= self.n_modes:
            return

        prefix_dim = self.hidden_size + self.markov_rank
        keep_idx = torch.cat(
            [torch.arange(prefix_dim, device=top_idx.device), prefix_dim + top_idx],
            dim=0,
        )
        old_norm = self.net[0]
        old_lin1 = self.net[1]
        assert isinstance(old_norm, nn.RMSNorm) and isinstance(old_lin1, nn.Linear)

        new_in_dim = prefix_dim + target_modes
        new_norm = nn.RMSNorm(
            new_in_dim, device=top_idx.device, dtype=old_lin1.weight.dtype
        )
        new_norm.weight.copy_(old_norm.weight.data[keep_idx])

        new_lin1 = nn.Linear(
            new_in_dim,
            old_lin1.out_features,
            device=top_idx.device,
            dtype=old_lin1.weight.dtype,
        )
        new_lin1.weight.copy_(old_lin1.weight.data[:, keep_idx])
        new_lin1.bias.copy_(old_lin1.bias.data)

        self.net[0] = new_norm
        self.net[1] = new_lin1
        self.n_modes = target_modes
