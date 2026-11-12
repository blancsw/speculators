"""Self-contained OOV-exact TV/CE losses, DEAO survival weights, and KvN metrics."""

from __future__ import annotations

import importlib.util
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from speculators.losses import (
    LossConfig,
    ce_loss,
    compound_loss,
    eager,
    resolve_loss_config,
    tv_loss,
)
from speculators.models.dspark.metrics import compute_metrics as dspark_compute_metrics
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "compute_kvn_metrics",
    "deao_loss_decay",
    "kvn_ce_loss",
    "kvn_tv_loss",
    "resolve_kvn_loss_config",
    "resolve_shortlist_oov_mass",
]

_HAS_TRITON = importlib.util.find_spec("triton") is not None


def resolve_shortlist_oov_mass(
    in_vocab_mass: Tensor | None,
    target_lse_full: Tensor | None,
    targets: Tensor,
) -> Tensor | None:
    """Resolve out-of-shortlist verifier probability mass M_oov = 1 - M_in in [0, 1]."""
    if in_vocab_mass is not None:
        return (1.0 - in_vocab_mass.float()).clamp(0.0, 1.0)
    if target_lse_full is not None:
        shortlist_lse = torch.logsumexp(targets.float(), dim=-1)
        m_in = torch.exp(shortlist_lse - target_lse_full.float()).clamp(0.0, 1.0)
        return (1.0 - m_in).clamp(0.0, 1.0)
    return None


def kvn_tv_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    oov_mass: Tensor | None = None,
    implementation: str = "fused",
) -> Tensor:
    """Exact full-vocab TV: 1 - sum_{v in V_d} min(q_v, M_in * p_{V_d}(v))."""
    if oov_mass is None:
        if implementation == "fused" and _HAS_TRITON and logits.is_cuda:
            return tv_loss(logits, targets)
        return eager.tv_loss(logits, targets)
    m_in = (1.0 - oov_mass.float()).clamp(0.0, 1.0)
    q = torch.softmax(logits.float(), dim=-1)
    p_in = torch.softmax(targets.float(), dim=-1) * m_in.unsqueeze(-1)
    return 1.0 - torch.minimum(q, p_in).sum(dim=-1)


def kvn_ce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    oov_mass: Tensor | None = None,
    implementation: str = "fused",
) -> Tensor:
    """Exact in-shortlist CE: -sum_{v in V_d} p_full(v) log q_v = M_in * CE_{V_d}."""
    use_fused = implementation == "fused" and _HAS_TRITON and logits.is_cuda
    base = ce_loss(logits, targets) if use_fused else eager.ce_loss(logits, targets)
    if oov_mass is None:
        return base
    return base * (1.0 - oov_mass.float()).clamp(0.0, 1.0).to(base.dtype)


def resolve_kvn_loss_config(spec: str, implementation: str = "fused") -> LossConfig:
    """Resolve compound loss spec, falling back cleanly when Triton is unavailable."""
    effective_impl = (
        "eager" if (implementation == "fused" and not _HAS_TRITON) else implementation
    )
    return resolve_loss_config(spec, effective_impl)


@torch.no_grad()
def deao_loss_decay(
    pos_idx: Tensor,  # noqa: ARG001
    loss_mask: Tensor,
    block_size: int,
    accept_rate: Tensor,
    sample_from_anchor: bool = True,
    **_kwargs: Any,
) -> Tensor:
    """Exact survival Jacobian dE[tau]/d(alpha_k) over speculative block horizons."""
    seq_len = accept_rate.shape[1]
    num_blocks = seq_len // block_size
    start_pos = 0 if sample_from_anchor else 1

    alpha = accept_rate.detach().float().clamp(0.0, 1.0).reshape(num_blocks, block_size)
    mask_bool = (loss_mask.reshape(num_blocks, block_size) > 0)[:, start_pos:]
    valid = mask_bool.to(torch.int64).cumprod(dim=-1).to(torch.float32)
    a = alpha[:, start_pos:] * valid

    prefix_excl = torch.cat(
        [torch.ones_like(a[:, :1]), torch.cumprod(a[:, :-1], dim=-1)], dim=-1
    )
    weights_draft = torch.zeros_like(a)
    after = torch.zeros_like(a[:, 0])
    for k in range(a.shape[1] - 1, -1, -1):
        suffix_k = 1.0 + after
        weights_draft[:, k] = valid[:, k] * prefix_excl[:, k] * suffix_k
        after = a[:, k] * suffix_k

    if start_pos == 0:
        return weights_draft.reshape(1, seq_len)
    full_weights = torch.zeros_like(alpha)
    full_weights[:, start_pos:] = weights_draft
    return full_weights.reshape(1, seq_len)


def compute_kvn_metrics(
    logits: Tensor,
    targets: Tensor,
    confidence_logits: Tensor | None,
    loss_mask: Tensor,
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
    *,
    in_vocab_mass: Tensor | None = None,
    target_lse_full: Tensor | None = None,
    target_argmax_in_vocab: Tensor | None = None,
    loss_implementation: str = "fused",
) -> tuple[Tensor, dict[str, Any]]:
    """Compute OOV-exact KvN-DSpark loss, c_star <= M_in BCE, DEAO, and metrics."""
    impl = (
        "eager"
        if (loss_implementation == "fused" and (not _HAS_TRITON or not logits.is_cuda))
        else loss_implementation
    )
    oov_mass = resolve_shortlist_oov_mass(in_vocab_mass, target_lse_full, targets)
    tv_fn = (
        partial(kvn_tv_loss, oov_mass=oov_mass, implementation=impl)
        if (oov_mass is not None or tv_loss_fn is None)
        else tv_loss_fn
    )

    effective_cfg: LossConfig = dict(loss_config)
    if oov_mass is not None:
        if "tv" in effective_cfg:
            effective_cfg["tv"] = (
                partial(kvn_tv_loss, oov_mass=oov_mass, implementation=impl),
                effective_cfg["tv"][1],
            )
        if "ce" in effective_cfg:
            effective_cfg["ce"] = (
                partial(kvn_ce_loss, oov_mass=oov_mass, implementation=impl),
                effective_cfg["ce"][1],
            )

    is_deao = per_position_loss_weight == "deao"
    loss, metrics = dspark_compute_metrics(
        logits=logits,
        targets=targets,
        confidence_logits=confidence_logits,
        loss_mask=loss_mask,
        block_size=block_size,
        loss_config={} if is_deao else effective_cfg,
        tv_loss_fn=lambda x, y: tv_fn(x, y).clamp(0.0, 1.0),
        gamma=gamma,
        confidence_head_alpha=confidence_head_alpha,
        per_position_loss_weight=(
            "fixed-exp-decay" if is_deao else per_position_loss_weight
        ),
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )

    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=logits.device) % block_size).unsqueeze(0)
    if is_deao:
        with torch.no_grad():
            accept_rate = (1.0 - tv_fn(logits, targets)).clamp(0.0, 1.0)
        decay_fn = partial(
            deao_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            accept_rate=accept_rate,
            sample_from_anchor=sample_from_anchor,
        )
        deao_loss, term_losses = compound_loss(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_config=effective_cfg,
            decay_fn=decay_fn,
        )
        loss = loss + deao_loss
        ones = torch.ones((), device=logits.device)
        metrics["loss_sum"] = loss.detach().clone()
        for term_name, term_val in term_losses.items():
            metrics[f"{term_name}_sum"] = term_val
            metrics[f"{term_name}_total"] = ones.clone()

    if target_argmax_in_vocab is not None:
        start_pos = 0 if sample_from_anchor else 1
        num_blocks = seq_len // block_size
        pred_ids = torch.argmax(logits, dim=-1)
        raw_target_ids = torch.argmax(targets, dim=-1)
        target_ids = torch.where(
            target_argmax_in_vocab.to(torch.bool),
            raw_target_ids,
            torch.full_like(raw_target_ids, -1),
        )
        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, block_size
        )
        metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
        metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
        for pos in range(start_pos, block_size):
            metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
            metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        draft_mask = loss_mask.to(torch.bool).view(num_blocks, block_size)[
            :, start_pos:
        ]
        eal_sum, eal_total = compute_accepted_length_counts(
            (pred_ids == target_ids).reshape(num_blocks, block_size)[:, start_pos:],
            draft_mask,
        )
        metrics["eal_sum"] = eal_sum
        metrics["eal_total"] = eal_total

    return loss, metrics
