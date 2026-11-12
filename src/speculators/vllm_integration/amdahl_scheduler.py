"""Purity-Guided Load-Aware Amdahl Speculative Scheduler (Axis 6).

Implements ``deepseekv4.1.pdf`` Sec. 2.4.3 dynamic verification truncation generalized
to open-system Lindblad Bosonic purity (Barthe Sec. 2.4.2 Eq. 2.39 & Sec. 7.6.1):

- Separates **aleatoric verifier uncertainty** at ``T > 0`` (high ``H(q_k)``, high
  Lindblad purity ``P_k -> 1``, low vacuum leakage ``p_vac,k -> 0`` -> keep drafting!)
  from **epistemic phase decoherence** (low ``P_k << 1`` or high ``p_vac,k`` ->
  truncate speculative horizon or throttle tree width!).
- Couples cumulative prefix survival ``S_k = prod_{j <= k} alpha_eff,j`` with GPU
  roofline batch-latency curves (H200 BF16 ridge point ``206`` tokens; RTX 3070 Ti
  ridge point ``99`` tokens; or empirical calibrated tables) to select optimal
  per-batch / per-request horizons ``K*(B)`` and Kraus tree budgets ``N_tree*(b, B)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

_RANK_2D = 2
_RANK_3D = 3

__all__ = [
    "HardwareRooflineProfile",
    "PurityLoadAwareAmdahlScheduler",
    "ScheduleDecision",
    "TreeScheduleDecision",
]


@dataclass(frozen=True)
class HardwareRooflineProfile:
    """Analytical + empirical roofline latency model for target verifier & KvN drafter.

    Attributes:
        name: Hardware identifier (e.g. ``"NVIDIA H200 SXM"``).
        peak_bf16_tflops: Peak Tensor-Core BF16 compute throughput in TFLOPS.
        hbm_bandwidth_tbs: Peak HBM/GDDR memory bandwidth in TB/s.
        ridge_tokens: Roofline arithmetic-intensity transition ``ceil(TFLOPS / TB/s)``.
        base_verify_ms: Memory-bandwidth-bound 1-step verifier latency below ridge.
        base_draft_chain_ms: Persistent-SRAM ``Mega-KvN`` linear-chain draft latency.
        draft_per_req_slope_ms: Marginal KvN draft latency per sequence in batch.
        tree_node_overhead_ms: Extra L2-resident Kraus POVM expansion cost per node.
    """

    name: str
    peak_bf16_tflops: float
    hbm_bandwidth_tbs: float
    ridge_tokens: int
    base_verify_ms: float
    base_draft_chain_ms: float
    draft_per_req_slope_ms: float = 0.0015
    tree_node_overhead_ms: float = 0.0006
    empirical_verify_ms: dict[tuple[int, int], float] | None = None
    empirical_draft_ms: dict[tuple[int, int], float] | None = None

    @classmethod
    def h200_sxm(cls) -> HardwareRooflineProfile:
        """Return calibrated roofline for H200 SXM (989 TFLOPS, 4.8 TB/s -> 206 tok)."""
        ridge = round(989.0 / 4.8)  # 206 tokens
        return cls(
            name="NVIDIA-H200-SXM-141GB",
            peak_bf16_tflops=989.0,
            hbm_bandwidth_tbs=4.8,
            ridge_tokens=ridge,
            base_verify_ms=3.45,
            base_draft_chain_ms=0.14,
            draft_per_req_slope_ms=0.0012,
            tree_node_overhead_ms=0.0006,
        )

    @classmethod
    def rtx_3070_ti_laptop(cls) -> HardwareRooflineProfile:
        """Return calibrated roofline for RTX 3070 Ti Laptop (44 TFLOPS -> 99 tok)."""
        ridge = math.ceil(44.0 / 0.448)  # 99 tokens
        return cls(
            name="NVIDIA-RTX-3070-Ti-Laptop-8GB",
            peak_bf16_tflops=44.0,
            hbm_bandwidth_tbs=0.448,
            ridge_tokens=ridge,
            base_verify_ms=14.20,
            base_draft_chain_ms=0.65,
            draft_per_req_slope_ms=0.0080,
            tree_node_overhead_ms=0.0035,
        )

    @classmethod
    def from_empirical_profile(
        cls,
        verify_ms_table: dict[tuple[int, int], float],
        draft_ms_table: dict[tuple[int, int], float] | None = None,
        base_profile: HardwareRooflineProfile | None = None,
    ) -> HardwareRooflineProfile:
        """Build profile backed by measured ``(batch_size, tokens_per_req)`` times."""
        ref = base_profile or cls.h200_sxm()
        return cls(
            name=f"{ref.name}-calibrated",
            peak_bf16_tflops=ref.peak_bf16_tflops,
            hbm_bandwidth_tbs=ref.hbm_bandwidth_tbs,
            ridge_tokens=ref.ridge_tokens,
            base_verify_ms=ref.base_verify_ms,
            base_draft_chain_ms=ref.base_draft_chain_ms,
            draft_per_req_slope_ms=ref.draft_per_req_slope_ms,
            tree_node_overhead_ms=ref.tree_node_overhead_ms,
            empirical_verify_ms=dict(verify_ms_table),
            empirical_draft_ms=dict(draft_ms_table) if draft_ms_table else None,
        )

    def estimate_verify_latency_ms(
        self,
        batch_size: int,
        tokens_per_req: int,
    ) -> float:
        """Estimate verifier latency for ``batch_size * tokens_per_req`` query tokens.

        Below ``ridge_tokens`` (memory-bandwidth bound), verifier latency grows only
        weakly; above ``ridge_tokens`` (Tensor-Core compute bound), latency scales
        linearly with ``total_tokens / ridge_tokens``.
        """
        b = max(1, int(batch_size))
        n_tok = max(1, int(tokens_per_req))
        if self.empirical_verify_ms and (b, n_tok) in self.empirical_verify_ms:
            return float(self.empirical_verify_ms[(b, n_tok)])

        total_tokens = b * n_tok
        ridge = float(max(1, self.ridge_tokens))
        ratio = total_tokens / ridge
        # Memory-bound floor + slight KV-read slope below ridge; linear ramp above
        scale = 1.0 + 0.06 * ratio if ratio <= 1.0 else 1.06 * ratio
        return self.base_verify_ms * scale

    def estimate_draft_latency_ms(
        self,
        batch_size: int,
        spec_budget: int,
        is_tree: bool = False,
    ) -> float:
        """Estimate proposer latency in ms (``0.0`` when ``spec_budget <= 0``)."""
        if spec_budget <= 0:
            return 0.0
        b = max(1, int(batch_size))
        budget = int(spec_budget)
        if self.empirical_draft_ms and (b, budget) in self.empirical_draft_ms:
            return float(self.empirical_draft_ms[(b, budget)])

        # Base O(1)-KV recurrent advance + Markov/Kraus readout
        horizon_scale = 0.65 + 0.35 * (min(budget, 8) / 8.0)
        lat = self.base_draft_chain_ms * horizon_scale + self.draft_per_req_slope_ms * b
        if is_tree:
            lat += self.tree_node_overhead_ms * budget * (1.0 + 0.02 * (b - 1))
        return lat


@dataclass(frozen=True)
class ScheduleDecision:
    """Dynamic speculative horizon decision for a runtime batch."""

    batch_horizon: int
    per_req_horizons: Tensor  # [B] int64 in [0, max_horizon]
    effective_alpha: Tensor  # [B, K] float32 in [0, 1]
    prefix_survival: Tensor  # [B, K] float32 in [0, 1]
    expected_tau_per_req: Tensor  # [B] float32 >= 1.0
    predicted_throughput_tps: float
    baseline_throughput_tps: float
    predicted_speedup: float


@dataclass(frozen=True)
class TreeScheduleDecision:
    """Dynamic Kraus POVM speculative tree budget decision."""

    batch_tree_budget: int
    per_req_tree_budgets: Tensor  # [B] int64 from candidate_tree_budgets
    expected_tau_per_req: Tensor  # [B] float32 >= 1.0
    predicted_throughput_tps: float
    predicted_speedup: float


class PurityLoadAwareAmdahlScheduler:
    """Axis 6 Dynamic Horizon & Kraus Tree Scheduler (Purity + Roofline Load)."""

    def __init__(
        self,
        roofline: HardwareRooflineProfile | None = None,
        max_horizon: int = 8,
        candidate_tree_budgets: Sequence[int] = (0, 4, 8, 16, 24),
        purity_exponent: float = 0.85,
        vacuum_exponent: float = 0.75,
        aleatoric_purity_threshold: float = 0.75,
        aleatoric_entropy_boost: float = 0.12,
        min_survival_floor: float = 0.08,
    ) -> None:
        if max_horizon <= 0:
            raise ValueError(f"max_horizon must be > 0, got {max_horizon}.")
        self.roofline = roofline or HardwareRooflineProfile.h200_sxm()
        self.max_horizon = int(max_horizon)
        self.candidate_tree_budgets = tuple(
            sorted({int(x) for x in candidate_tree_budgets})
        )
        if not self.candidate_tree_budgets or self.candidate_tree_budgets[0] < 0:
            raise ValueError("candidate_tree_budgets must be non-empty and >= 0.")
        self.purity_exponent = float(purity_exponent)
        self.vacuum_exponent = float(vacuum_exponent)
        self.aleatoric_purity_threshold = float(aleatoric_purity_threshold)
        self.aleatoric_entropy_boost = float(aleatoric_entropy_boost)
        self.min_survival_floor = float(min_survival_floor)

    @staticmethod
    def _slice_slot_metric(metric: Tensor | None, k_cap: int) -> Tensor | None:
        """Reduce optional 3D mode metric and align slot axis to ``k_cap``."""
        if metric is None:
            return None
        m = metric.mean(dim=-1) if metric.ndim == _RANK_3D else metric
        if m.ndim == _RANK_2D and m.size(1) > 0:
            if m.size(1) >= k_cap:
                return m[:, :k_cap]
            pad = m[:, -1:].expand(m.size(0), k_cap - m.size(1))
            return torch.cat([m, pad], dim=1)
        return m

    @staticmethod
    def _broadcast_slot_metric(
        metric: Tensor | None,
        ref: Tensor,
        default_val: float,
    ) -> Tensor:
        """Broadcast optional ``[B]``, ``[B, K]``, or ``[B, K, M]`` onto ``[B, K]``."""
        bsz, k_len = ref.shape
        if metric is None:
            return torch.full(
                (bsz, k_len), default_val, dtype=torch.float32, device=ref.device
            )
        m = metric.detach().to(device=ref.device, dtype=torch.float32)
        if m.ndim == _RANK_3D:
            m = m.mean(dim=-1)
        if m.ndim == 1:
            if m.numel() != bsz:
                raise ValueError(f"Expected length {bsz}, got {m.numel()}.")
            return m.unsqueeze(1).expand(bsz, k_len)
        if m.shape != (bsz, k_len):
            raise ValueError(f"Expected shape ({bsz}, {k_len}), got {tuple(m.shape)}.")
        return m

    def compute_effective_survival(
        self,
        confidences: Tensor,  # [B, K]
        purity: Tensor | None = None,  # [B, K] or [B] in (0, 1]
        p_vac: Tensor | None = None,  # [B, K] or [B] in [0, 1]
        draft_entropy: Tensor | None = None,  # [B, K] normalized entropy in [0, 1]
        temperature: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Compute coherence-gated acceptance ``alpha_eff`` and survival ``S_k``.

        At ``T > 0``, high entropy ``H(q_k)`` paired with high Lindblad purity
        ``P_k >= aleatoric_purity_threshold`` and low vacuum leakage ``p_vac``
        indicates shared verifier-drafter aleatoric sampling entropy (where true
        speculative acceptance ``1 - D_TV(p, q)`` stays high!), so confidence is
        protected against entropy-induced under-prediction rather than penalized.
        Conversely, low purity ``P_k << 1`` or high ``p_vac`` reflects epistemic
        decoherence and smoothly attenuates ``alpha_eff``.
        """
        if confidences.ndim != _RANK_2D:
            raise ValueError(
                f"confidences must be 2D [B, K], got shape {tuple(confidences.shape)}."
            )
        conf = confidences.detach().to(torch.float32).clamp(0.0, 1.0)
        pur = self._broadcast_slot_metric(purity, conf, default_val=1.0).clamp(
            1e-6, 1.0
        )
        vac = self._broadcast_slot_metric(p_vac, conf, default_val=0.0).clamp(
            0.0, 1.0 - 1e-6
        )
        ent = self._broadcast_slot_metric(draft_entropy, conf, default_val=0.0).clamp(
            0.0, 1.0
        )

        # Epistemic coherence gate: exactly 1.0 when pure (P_k = 1, p_vac = 0)
        coherence_gate = (pur**self.purity_exponent) * (
            (1.0 - vac) ** self.vacuum_exponent
        )
        alpha_eff = conf * coherence_gate

        if temperature > 0.0 and self.aleatoric_entropy_boost > 0.0:
            # Protect creative T > 0 drafting when state remains pure & in-simplex
            pure_excess = (
                (pur - self.aleatoric_purity_threshold)
                / max(1e-6, 1.0 - self.aleatoric_purity_threshold)
            ).clamp(0.0, 1.0)
            aleatoric_recovery = (
                self.aleatoric_entropy_boost * pure_excess * (1.0 - vac) * ent
            )
            alpha_eff = (alpha_eff + aleatoric_recovery * (1.0 - alpha_eff)).clamp(
                0.0, 1.0
            )

        prefix_survival = torch.cumprod(alpha_eff.clamp(0.0, 1.0), dim=-1)
        return alpha_eff, prefix_survival

    def select_horizon(
        self,
        confidences: Tensor,
        active_batch_size: int | None = None,
        purity: Tensor | None = None,
        p_vac: Tensor | None = None,
        draft_entropy: Tensor | None = None,
        temperature: float = 0.0,
    ) -> ScheduleDecision:
        """Select Amdahl-optimal horizon ``K*(B)`` and per-request ``K*_b``."""
        bsz, k_avail = confidences.shape
        eff_batch = max(bsz, int(active_batch_size or bsz))
        k_cap = min(k_avail, self.max_horizon)

        alpha_eff, prefix_survival = self.compute_effective_survival(
            confidences=confidences[:, :k_cap],
            purity=self._slice_slot_metric(purity, k_cap),
            p_vac=self._slice_slot_metric(p_vac, k_cap),
            draft_entropy=self._slice_slot_metric(draft_entropy, k_cap),
            temperature=temperature,
        )

        # Baseline K'=0 throughput (pure 1-token target decode, 0 draft overhead)
        t_base_ms = self.roofline.estimate_verify_latency_ms(
            eff_batch, tokens_per_req=1
        )
        baseline_tps = (1000.0 * eff_batch) / max(t_base_ms, 1e-6)

        best_k = 0
        best_tps = baseline_tps
        mean_surv_by_k = prefix_survival.mean(dim=0)  # [k_cap]

        for k_cand in range(1, k_cap + 1):
            mean_tau_k = 1.0 + float(mean_surv_by_k[:k_cand].sum().item())
            t_draft = self.roofline.estimate_draft_latency_ms(
                eff_batch, spec_budget=k_cand, is_tree=False
            )
            t_ver = self.roofline.estimate_verify_latency_ms(
                eff_batch, tokens_per_req=1 + k_cand
            )
            cand_tps = (1000.0 * eff_batch * mean_tau_k) / max(t_draft + t_ver, 1e-6)
            if cand_tps > best_tps:
                best_tps = cand_tps
                best_k = k_cand

        # Per-request horizon: marginal verifier cost threshold scales with load
        load_ratio = (eff_batch * max(1, 1 + best_k)) / float(
            max(1, self.roofline.ridge_tokens)
        )
        # Below ridge, marginal verify token is nearly free -> low survival floor;
        # above ridge, require survival >= marginal step time fraction.
        marginal_ver_delta = (
            self.roofline.estimate_verify_latency_ms(eff_batch, 2) - t_base_ms
        ) / max(t_base_ms, 1e-6)
        gamma_load = max(
            self.min_survival_floor,
            min(0.85, marginal_ver_delta * (1.0 if load_ratio > 1.0 else 0.5)),
        )
        keep_mask = torch.cumprod(
            (prefix_survival >= gamma_load).to(torch.int64), dim=-1
        )
        per_req_horizons = keep_mask.sum(dim=-1)
        if best_k == 0:
            per_req_horizons = torch.zeros_like(per_req_horizons)
        else:
            per_req_horizons = per_req_horizons.clamp(max=best_k)

        # Expected accepted tokens per request at its chosen horizon
        slot_idx = torch.arange(k_cap, device=confidences.device).unsqueeze(0)
        active_slots = (slot_idx < per_req_horizons.unsqueeze(1)).to(torch.float32)
        expected_tau = 1.0 + (prefix_survival * active_slots).sum(dim=-1)

        return ScheduleDecision(
            batch_horizon=best_k,
            per_req_horizons=per_req_horizons,
            effective_alpha=alpha_eff,
            prefix_survival=prefix_survival,
            expected_tau_per_req=expected_tau,
            predicted_throughput_tps=best_tps,
            baseline_throughput_tps=baseline_tps,
            predicted_speedup=best_tps / max(baseline_tps, 1e-6),
        )

    def select_tree_budget(
        self,
        confidences: Tensor,  # [B, K]
        active_batch_size: int | None = None,
        purity: Tensor | None = None,
        p_vac: Tensor | None = None,
        draft_entropy: Tensor | None = None,
        temperature: float = 0.0,
        branch_gain_factor: float = 0.55,
    ) -> TreeScheduleDecision:
        """Select Amdahl-optimal Kraus POVM tree budget ``N_tree*(b, B)``.

        Models expected multi-branch recovery at bimodal forks: when purity is high
        (low epistemic decoherence) and chain survival leaves uncaptured multi-modal
        mass ``1 - alpha_eff``, widening the Kraus tree from ``K`` to ``N`` nodes
        recovers ``branch_gain_factor * log2(N / K + 1)`` of residual pure mass while
        paying minimal extra verifier latency when ``B * (1 + N) <= ridge_tokens``.
        """
        bsz, k_avail = confidences.shape
        eff_batch = max(bsz, int(active_batch_size or bsz))
        k_cap = min(k_avail, self.max_horizon)

        alpha_eff, prefix_survival = self.compute_effective_survival(
            confidences=confidences[:, :k_cap],
            purity=self._slice_slot_metric(purity, k_cap),
            p_vac=self._slice_slot_metric(p_vac, k_cap),
            draft_entropy=self._slice_slot_metric(draft_entropy, k_cap),
            temperature=temperature,
        )
        pur = self._broadcast_slot_metric(
            self._slice_slot_metric(purity, k_cap),
            alpha_eff,
            default_val=1.0,
        ).clamp(1e-6, 1.0)
        vac = self._broadcast_slot_metric(
            self._slice_slot_metric(p_vac, k_cap),
            alpha_eff,
            default_val=0.0,
        ).clamp(0.0, 1.0 - 1e-6)

        t_base_ms = self.roofline.estimate_verify_latency_ms(
            eff_batch, tokens_per_req=1
        )
        baseline_tps = (1000.0 * eff_batch) / max(t_base_ms, 1e-6)

        tau_by_budget: dict[int, Tensor] = {}
        tps_by_budget: dict[int, float] = {}

        for budget in self.candidate_tree_budgets:
            if budget <= 0:
                tau_vec = torch.ones(
                    (bsz,), dtype=torch.float32, device=confidences.device
                )
                tau_by_budget[0] = tau_vec
                tps_by_budget[0] = baseline_tps
                continue

            depth = min(k_cap, budget)
            extra_branches = max(0, budget - depth)
            if extra_branches == 0:
                tau_vec = 1.0 + prefix_survival[:, :depth].sum(dim=-1)
            else:
                # Per-depth branching w >= 1 recovers bimodal mass on pure states
                branch_boost = branch_gain_factor * math.log2(
                    1.0 + extra_branches / float(max(1, depth))
                )
                pure_fork_cap = pur[:, :depth] * (1.0 - vac[:, :depth])
                alpha_tree = (
                    alpha_eff[:, :depth]
                    + branch_boost * pure_fork_cap * (1.0 - alpha_eff[:, :depth])
                ).clamp(0.0, 0.995)
                surv_tree = torch.cumprod(alpha_tree, dim=-1)
                tau_vec = 1.0 + surv_tree.sum(dim=-1)

            tau_by_budget[budget] = tau_vec
            t_draft = self.roofline.estimate_draft_latency_ms(
                eff_batch, spec_budget=budget, is_tree=True
            )
            t_ver = self.roofline.estimate_verify_latency_ms(
                eff_batch, tokens_per_req=1 + budget
            )
            tps_by_budget[budget] = (
                1000.0 * eff_batch * float(tau_vec.mean().item())
            ) / max(t_draft + t_ver, 1e-6)

        best_batch_budget = max(tps_by_budget, key=lambda n: tps_by_budget[n])
        best_tps = tps_by_budget[best_batch_budget]

        # Per-request budget optimization under the current batch's per-slot verify cost
        per_req_budgets = torch.zeros(
            (bsz,), dtype=torch.int64, device=confidences.device
        )
        best_req_tau = torch.ones(
            (bsz,), dtype=torch.float32, device=confidences.device
        )
        best_req_score = torch.full(
            (bsz,),
            1000.0 / max(t_base_ms, 1e-6),
            dtype=torch.float32,
            device=confidences.device,
        )
        for budget in self.candidate_tree_budgets:
            if budget == 0:
                continue
            t_step = self.roofline.estimate_draft_latency_ms(
                eff_batch, spec_budget=budget, is_tree=True
            ) + self.roofline.estimate_verify_latency_ms(
                eff_batch, tokens_per_req=1 + budget
            )
            score_vec = (1000.0 * tau_by_budget[budget]) / max(t_step, 1e-6)
            better = score_vec > best_req_score
            per_req_budgets = torch.where(
                better,
                torch.full_like(per_req_budgets, budget),
                per_req_budgets,
            )
            best_req_score = torch.where(better, score_vec, best_req_score)
            best_req_tau = torch.where(better, tau_by_budget[budget], best_req_tau)

        return TreeScheduleDecision(
            batch_tree_budget=best_batch_budget,
            per_req_tree_budgets=per_req_budgets,
            expected_tau_per_req=best_req_tau,
            predicted_throughput_tps=best_tps,
            predicted_speedup=best_tps / max(baseline_tps, 1e-6),
        )
