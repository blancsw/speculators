"""Production vLLM V1 Proposer & Lossless SpecInfer Verifier for ``KvN-DSpark``.

Provides:
1. ``KvNDSparkVllmProposer``:
   - Slot-indexed FP32 recurrent state management via :class:`KvNStateSlotPool` (exact
     vLLM V1 chunked-prefill continuation and scheduler preemption/resume with zero
     boundary state corruption).
   - Axis 3 0-FLOP prefix state restoration via :class:`ZeroReplayAPCStateCache` on
     16-token PagedAttention block boundaries.
   - Constant-shape ``[B, block_size + 1]`` CUDA-Graph-safe decode via
     ``drafter.advance_and_draft_step(...)`` with zero bonus-token rollback.
   - Axis 5 Kraus POVM multi-branch tree proposal via ``drafter.draft_kraus_tree_step``
     and Axis 6 dynamic horizon / tree-budget truncation via
     :class:`PurityLoadAwareAmdahlScheduler`.
   - Dual-mode construction: instantiates directly from ``vllm.config.VllmConfig``
     (``method="custom_class"`` / ``vllm.general_plugins``) OR standalone with a
     ``KvNDSparkDraftModel`` instance/path when ``vllm`` C++ bindings are absent.
2. ``specinfer_verify_tree_step`` & ``compact_tree_kv_cache``:
   - Lossless multi-branch tree verification at both ``T = 0`` (bit-exact greedy path
     traversal) and ``T > 0`` (SpecInfer sequential residual rejection sampling over
     the full target vocabulary, preserving the exact target distribution ``p(y|x)``
     including out-of-shortlist verifier tokens).
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from speculators.models.kvn_dspark.core import KvNDSparkDraftModel
from speculators.vllm_integration.amdahl_scheduler import (
    HardwareRooflineProfile,
    PurityLoadAwareAmdahlScheduler,
    ScheduleDecision,
    TreeScheduleDecision,
)
from speculators.vllm_integration.apc_state_cache import (
    KvNStateSlotPool,
    ZeroReplayAPCStateCache,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_RANK_2D = 2
_RANK_3D = 3
_RANK_4D = 4
_LAYER_OUT_V3_LEN = 5
_STEP_STATE_V3_LEN = 3
_ADVANCE_V3_RETURN_LEN = 3
_PROB_EPS = 1e-12

logger = logging.getLogger(__name__)

__all__ = [
    "KvNDSparkVllmProposer",
    "KvNProposeOutput",
    "SpecInferStepResult",
    "build_tree_attention_mask",
    "compact_tree_kv_cache",
    "specinfer_verify_tree_step",
]


@dataclass
class SpecInferStepResult:
    """Result of verifying a batch of speculative token trees against target logits."""

    accepted_tokens: list[list[int]]
    accepted_node_indices: list[list[int]]
    bonus_tokens: Tensor  # [B] int64
    num_accepted: Tensor  # [B] int64


@dataclass
class KvNProposeOutput:
    """Structured output of a batched ``KvNDSparkVllmProposer`` proposal step."""

    draft_token_ids: list[list[int]]
    padded_draft_ids: Tensor  # [B, K] int64
    confidences: Tensor  # [B, K] float32
    schedule: ScheduleDecision
    draft_probs: Tensor | None = None
    tree_dict: dict[str, Tensor] | None = None
    tree_schedule: TreeScheduleDecision | None = None


def _compute_tree_ancestry_and_depths(
    parent_row: list[int],
) -> tuple[list[list[bool]], list[int]]:
    """Compute transitive ancestor matrix ``[N, N]`` and 1-based depths ``[N]``."""
    n_nodes = len(parent_row)
    anc = [[False] * n_nodes for _ in range(n_nodes)]
    depths = [0] * n_nodes
    for idx in range(n_nodes):
        par = parent_row[idx]
        if par < -1:
            continue
        anc[idx][idx] = True
        cur = par
        depth = 1
        hops = 0
        while cur >= 0 and hops < n_nodes:
            if cur >= n_nodes:
                break
            anc[idx][cur] = True
            cur = parent_row[cur]
            depth += 1
            hops += 1
        depths[idx] = depth
    return anc, depths


def _apply_custom_tree_mask(
    allow: Tensor,
    tree_attn_mask: Tensor,
    bsz: int,
    n_nodes: int,
    q_len: int,
    prefix_len: int,
    device: torch.device,
) -> None:
    """Populate intra-tree boolean mask entries from an explicit ``tree_attn_mask``."""
    if tree_attn_mask.shape == (bsz, n_nodes, n_nodes):
        allow[:, 1:, prefix_len + 1 :] = tree_attn_mask.to(
            device=device, dtype=torch.bool
        )
    elif tree_attn_mask.shape == (bsz, q_len, q_len):
        allow[:, :, prefix_len:] = tree_attn_mask.to(device=device, dtype=torch.bool)
    else:
        raise ValueError(
            f"Unexpected tree_attn_mask shape {tuple(tree_attn_mask.shape)}."
        )


def _populate_row_tree_ancestry(
    allow_row: Tensor,
    parent_row: list[int],
    anc: list[list[bool]],
    prefix_len: int,
    fill_edges: bool,
) -> None:
    """Mark root-anchor and ancestor visibility for one request's tree nodes."""
    n_nodes = len(parent_row)
    for i_idx in range(n_nodes):
        if parent_row[i_idx] < -1:
            continue
        allow_row[1 + i_idx, prefix_len] = True
        if fill_edges:
            for j_idx in range(n_nodes):
                if anc[i_idx][j_idx]:
                    allow_row[1 + i_idx, prefix_len + 1 + j_idx] = True


def build_tree_attention_mask(
    tree_parent_idx: Tensor,
    prefix_len: int,
    dtype: torch.dtype = torch.float32,
    tree_attn_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build verifier ``[B, 1, 1+N, prefix_len+1+N]`` mask and ``[B, N]`` node depths.

    Query slot ``0`` is the confirmed root anchor at position ``prefix_len``; query
    slots ``1..N`` correspond to ``tree_draft_ids[:, 0..N-1]``.
    """
    if tree_parent_idx.ndim != _RANK_2D:
        raise ValueError(
            f"tree_parent_idx must be 2D [B, N], got {tuple(tree_parent_idx.shape)}."
        )
    bsz, n_nodes = tree_parent_idx.shape
    device = tree_parent_idx.device
    q_len = 1 + n_nodes
    kv_len = int(prefix_len) + q_len

    allow = torch.zeros((bsz, q_len, kv_len), dtype=torch.bool, device=device)
    if prefix_len > 0:
        allow[:, :, :prefix_len] = True
    allow[:, 0, prefix_len] = True

    depths_out = torch.zeros((bsz, n_nodes), dtype=torch.long, device=device)
    parents_cpu = tree_parent_idx.detach().cpu().tolist()

    for b_idx, p_row in enumerate(parents_cpu):
        anc, depths = _compute_tree_ancestry_and_depths(p_row)
        depths_out[b_idx] = torch.tensor(depths, dtype=torch.long, device=device)
        _populate_row_tree_ancestry(
            allow[b_idx], p_row, anc, prefix_len, fill_edges=(tree_attn_mask is None)
        )

    if tree_attn_mask is not None:
        _apply_custom_tree_mask(
            allow, tree_attn_mask, bsz, n_nodes, q_len, prefix_len, device
        )

    min_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    attn_mask = torch.full((bsz, 1, q_len, kv_len), min_val, dtype=dtype, device=device)
    attn_mask = attn_mask.masked_fill(allow.unsqueeze(1), 0.0)
    return attn_mask, depths_out


def _build_children_map(parent_row: list[int]) -> dict[int, list[int]]:
    """Group active child node indices ``0..N-1`` by parent index (``-1`` = root)."""
    children: dict[int, list[int]] = {}
    for idx, par in enumerate(parent_row):
        if par < -1:
            continue
        children.setdefault(int(par), []).append(idx)
    return children


def _lift_draft_probs_to_full_vocab(
    q_vec: Tensor,
    vocab_size: int,
    d2t: Tensor | None,
) -> Tensor:
    """Map a draft probability vector (shortlist or full) onto ``[vocab_size]``."""
    q_f = q_vec.detach().to(torch.float32)
    if q_f.numel() == vocab_size:
        return q_f
    if d2t is None or d2t.numel() != q_f.numel():
        raise ValueError(
            f"Cannot lift draft probs of len {q_f.numel()} to {vocab_size} without d2t."
        )
    full_q = torch.zeros((vocab_size,), dtype=torch.float32, device=q_f.device)
    draft_idx = torch.arange(q_f.numel(), device=q_f.device, dtype=torch.long)
    target_idx = draft_idx + d2t.to(device=q_f.device, dtype=torch.long)
    full_q.scatter_add_(0, target_idx, q_f)
    return full_q


def _sample_categorical_1d(
    probs: Tensor,
    generator: torch.Generator | None,
) -> int:
    """Sample a single token ID from a normalized 1D float32 probability vector."""
    total = float(probs.sum().item())
    if total <= _PROB_EPS or not torch.isfinite(probs).all():
        return int(probs.argmax().item())
    norm_probs = probs / total
    return int(torch.multinomial(norm_probs, num_samples=1, generator=generator).item())


def _verify_single_tree_greedy(
    logits_b: Tensor,  # [1 + N, V]
    draft_row: list[int],
    children_map: dict[int, list[int]],
) -> tuple[list[int], list[int], int]:
    """Exact greedy tree verification (``T <= 0``)."""
    acc_tokens: list[int] = []
    acc_indices: list[int] = []
    cur_parent = -1

    while True:
        logit_slot = 0 if cur_parent == -1 else (1 + cur_parent)
        target_tok = int(logits_b[logit_slot].argmax(dim=-1).item())
        matched_child = -1
        for child_idx in children_map.get(cur_parent, ()):
            if draft_row[child_idx] == target_tok:
                matched_child = child_idx
                break
        if matched_child < 0:
            return acc_tokens, acc_indices, target_tok
        acc_tokens.append(target_tok)
        acc_indices.append(matched_child)
        cur_parent = matched_child


def _verify_single_tree_specinfer(
    logits_b: Tensor,  # [1 + N, V]
    draft_row: list[int],
    children_map: dict[int, list[int]],
    temperature: float,
    q_probs_b: Tensor | None,
    d2t: Tensor | None,
    stochastic_proposals: bool,
    generator: torch.Generator | None,
) -> tuple[list[int], list[int], int]:
    """Lossless SpecInfer multi-branch residual rejection sampling (``T > 0``).

    Supports both:
    - ``stochastic_proposals=False`` (default for Kraus POVM / top-k / beam trees):
      distinct child candidates ``y_{u_1}, ..., y_{u_m}`` represent conditional point
      masses ``delta_{y_{u_j}}``, accepting ``y_{u_j}`` with probability
      ``p_res(y_{u_j})`` and zeroing/renormalizing ``p_res`` on rejection.
    - ``stochastic_proposals=True`` (children sampled ``y_{u_j} ~ q^{(j)}``):
      accepts with ``min(1, p_res(y_{u_j}) / q^{(j)}(y_{u_j}))`` and updates
      ``p_res <- normalize(max(0, p_res - q^{(j)}))``.

    In both regimes, the marginal distribution of every emitted token is **identically**
    ``softmax(logits / temperature)`` across the full verifier vocabulary.
    """
    vocab_size = logits_b.size(-1)
    inv_temp = 1.0 / max(float(temperature), 1e-6)
    acc_tokens: list[int] = []
    acc_indices: list[int] = []
    cur_parent = -1

    while True:
        logit_slot = 0 if cur_parent == -1 else (1 + cur_parent)
        p_res = torch.softmax(logits_b[logit_slot].to(torch.float32) * inv_temp, dim=-1)
        accepted_child = -1
        seen_det_tokens: set[int] = set()

        for child_idx in children_map.get(cur_parent, ()):
            cand_tok = int(draft_row[child_idx])
            if (
                stochastic_proposals
                and q_probs_b is not None
                and q_probs_b.ndim == _RANK_2D
            ):
                q_full = _lift_draft_probs_to_full_vocab(
                    q_probs_b[child_idx], vocab_size, d2t
                )
            else:
                if cand_tok in seen_det_tokens:
                    continue
                seen_det_tokens.add(cand_tok)
                q_full = torch.zeros_like(p_res)
                q_full[cand_tok] = 1.0

            q_tok = float(q_full[cand_tok].item())
            p_tok = float(p_res[cand_tok].item())
            accept_prob = (
                min(1.0, p_tok / max(q_tok, _PROB_EPS)) if q_tok > 0.0 else 0.0
            )
            u_draw = float(
                torch.rand((), device=logits_b.device, generator=generator).item()
            )
            if u_draw < accept_prob:
                accepted_child = child_idx
                acc_tokens.append(cand_tok)
                acc_indices.append(child_idx)
                break

            residual = (p_res - q_full).clamp(min=0.0)
            res_mass = float(residual.sum().item())
            if res_mass <= _PROB_EPS:
                p_res = torch.zeros_like(p_res)
                p_res[cand_tok] = 1.0
                break
            p_res = residual / res_mass

        if accepted_child < 0:
            bonus_tok = _sample_categorical_1d(p_res, generator=generator)
            return acc_tokens, acc_indices, bonus_tok
        cur_parent = accepted_child


def specinfer_verify_tree_step(
    verifier_logits: Tensor,
    tree_draft_ids: Tensor,
    tree_parent_idx: Tensor,
    temperature: float = 0.0,
    tree_q_probs: Tensor | None = None,
    d2t: Tensor | None = None,
    stochastic_proposals: bool = False,
    generator: torch.Generator | None = None,
) -> SpecInferStepResult:
    """Verify a batch of speculative token trees against target ``verifier_logits``."""
    if verifier_logits.ndim != _RANK_3D:
        raise ValueError(
            f"verifier_logits must be [B, 1+N, V], got {tuple(verifier_logits.shape)}."
        )
    bsz, n_nodes = tree_draft_ids.shape
    if verifier_logits.size(0) != bsz or verifier_logits.size(1) != 1 + n_nodes:
        raise ValueError(
            "verifier_logits shape must match [B, 1 + N, V] for tree_draft_ids [B, N]."
        )

    draft_cpu = tree_draft_ids.detach().cpu().tolist()
    parents_cpu = tree_parent_idx.detach().cpu().tolist()

    all_acc_tokens: list[list[int]] = []
    all_acc_indices: list[list[int]] = []
    bonus_list: list[int] = []

    for b_idx in range(bsz):
        children_map = _build_children_map(parents_cpu[b_idx])
        if temperature <= 0.0:
            acc_toks, acc_idx, bonus_tok = _verify_single_tree_greedy(
                logits_b=verifier_logits[b_idx],
                draft_row=draft_cpu[b_idx],
                children_map=children_map,
            )
        else:
            q_b = tree_q_probs[b_idx] if tree_q_probs is not None else None
            acc_toks, acc_idx, bonus_tok = _verify_single_tree_specinfer(
                logits_b=verifier_logits[b_idx],
                draft_row=draft_cpu[b_idx],
                children_map=children_map,
                temperature=temperature,
                q_probs_b=q_b,
                d2t=d2t,
                stochastic_proposals=stochastic_proposals,
                generator=generator,
            )
        all_acc_tokens.append(acc_toks)
        all_acc_indices.append(acc_idx)
        bonus_list.append(bonus_tok)

    device = verifier_logits.device
    bonus_tokens = torch.tensor(bonus_list, dtype=torch.long, device=device)
    num_accepted = torch.tensor(
        [len(x) for x in all_acc_tokens], dtype=torch.long, device=device
    )
    return SpecInferStepResult(
        accepted_tokens=all_acc_tokens,
        accepted_node_indices=all_acc_indices,
        bonus_tokens=bonus_tokens,
        num_accepted=num_accepted,
    )


def _extract_cache_kv_lists(past_key_values: Any) -> list[tuple[Tensor, Tensor]]:
    """Extract mutable ``(key_cache, value_cache)`` pairs from ``DynamicCache``."""
    if hasattr(past_key_values, "layers"):
        return [
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if getattr(layer, "keys", None) is not None
        ]
    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        return list(
            zip(past_key_values.key_cache, past_key_values.value_cache, strict=True)
        )
    raise TypeError(
        f"Unsupported past_key_values type {type(past_key_values).__name__}."
    )


def compact_tree_kv_cache(
    past_key_values: Any,
    prefix_len: int,
    accepted_node_indices: Sequence[Sequence[int]],
) -> Tensor:
    """Compact verified tree KV entries in-place into contiguous causal KV positions.

    Because accepted tree node ``m`` (0-indexed along the accepted branch) was already
    evaluated at RoPE position ``prefix_len + 1 + m``, gathering KV slot
    ``prefix_len + 1 + node_idx`` into ``prefix_len + 1 + m`` and cropping trailing
    rejected tree nodes yields **100% bit-exact** KV cache state with zero re-compute.

    Returns:
        ``new_kv_lens`` ``[B]`` int64 tensor (``prefix_len + 1 + len(acc_b)``).
    """
    kv_pairs = _extract_cache_kv_lists(past_key_values)
    if not kv_pairs:
        return torch.tensor(
            [prefix_len + 1 + len(idx) for idx in accepted_node_indices],
            dtype=torch.long,
        )

    device = kv_pairs[0][0].device
    bsz = len(accepted_node_indices)
    max_acc = max((len(idx) for idx in accepted_node_indices), default=0)
    keep_len = int(prefix_len) + 1 + max_acc

    for k_tensor, v_tensor in kv_pairs:
        for b_idx in range(bsz):
            acc_b = accepted_node_indices[b_idx]
            if not acc_b:
                continue
            src_pos = torch.tensor(
                [prefix_len + 1 + int(node_i) for node_i in acc_b],
                dtype=torch.long,
                device=device,
            )
            dst_end = prefix_len + 1 + len(acc_b)
            k_selected = k_tensor[b_idx].index_select(dim=-2, index=src_pos).clone()
            v_selected = v_tensor[b_idx].index_select(dim=-2, index=src_pos).clone()
            k_tensor[b_idx, :, prefix_len + 1 : dst_end, :] = k_selected
            v_tensor[b_idx, :, prefix_len + 1 : dst_end, :] = v_selected

    if hasattr(past_key_values, "crop") and hasattr(past_key_values, "get_seq_length"):
        remove_cnt = int(past_key_values.get_seq_length()) - keep_len
        if remove_cnt > 0:
            past_key_values.crop(-remove_cnt)
    return torch.tensor(
        [prefix_len + 1 + len(idx) for idx in accepted_node_indices],
        dtype=torch.long,
        device=device,
    )


def _unpack_layer_forward_outputs(
    layer_out: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Normalize v2 ``(x, q, p, dtheta)`` and v3 ``(x, q, p, dtheta, nu)`` outputs."""
    x_out, q_seq, p_seq, dtheta_seq = (
        layer_out[0],
        layer_out[1],
        layer_out[2],
        layer_out[3],
    )
    nu_seq = layer_out[4] if len(layer_out) >= _LAYER_OUT_V3_LEN else None
    return x_out, q_seq, p_seq, dtheta_seq, nu_seq


def _unpack_recurrent_step_outputs(
    step_out: tuple[Any, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    """Normalize ``recurrent_step`` across v2 ``(q, p)`` and v3 ``(q, p, nu)``."""
    h_step = step_out[0]
    state_tup = step_out[1]
    dtheta_step = step_out[2]
    q_s = state_tup[0]
    p_s = state_tup[1]
    nu_s = state_tup[2] if len(state_tup) >= _STEP_STATE_V3_LEN else None
    return h_step, q_s, p_s, nu_s, dtheta_step


class KvNDSparkVllmProposer:
    """Production vLLM V1 Proposer for Zero-KV-Cache ``KvN-DSpark`` drafting.

    IMPORTANT: Stores the draft ``SpeculatorModel`` on ``self.drafter_model`` (NOT
    ``self.model``) so ``vllm.v1.worker.gpu_model_runner.GPUModelRunner.load_model``
    does not mis-wrap the recurrent proposer as a standard Attention-only draft model.
    """

    def __init__(
        self,
        vllm_config_or_drafter: Any,
        *,
        max_num_seqs: int = 256,
        device: torch.device | str | None = None,
        enable_apc: bool = True,
        apc_block_size: int = 16,
        apc_max_blocks: int = 4096,
        scheduler: PurityLoadAwareAmdahlScheduler | None = None,
        roofline: HardwareRooflineProfile | None = None,
        use_kraus_tree: bool = False,
        tree_budget: int = 24,
        branch_width: int = 3,
    ) -> None:
        self.drafter_model: KvNDSparkDraftModel | None = None
        self._checkpoint_path: str | Path | None = None
        self.vllm_config: Any = None

        if hasattr(vllm_config_or_drafter, "speculative_config"):
            self._init_from_vllm_config(
                vllm_config_or_drafter,
                default_max_seqs=max_num_seqs,
                override_device=device,
            )
        elif isinstance(vllm_config_or_drafter, KvNDSparkDraftModel):
            self.drafter_model = vllm_config_or_drafter
            self.max_num_seqs = int(max_num_seqs)
            inferred_dev = next(vllm_config_or_drafter.parameters()).device
            self.device = torch.device(device) if device is not None else inferred_dev
        elif isinstance(vllm_config_or_drafter, (str, Path)):
            self._checkpoint_path = Path(vllm_config_or_drafter)
            self.max_num_seqs = int(max_num_seqs)
            self.device = torch.device(device or "cpu")
        else:
            raise TypeError(
                "Expected VllmConfig, KvNDSparkDraftModel, or checkpoint path, got "
                f"{type(vllm_config_or_drafter).__name__}."
            )

        self.enable_apc = bool(enable_apc)
        self.apc_block_size = int(apc_block_size)
        self.apc_max_blocks = int(apc_max_blocks)
        self.use_kraus_tree = bool(use_kraus_tree)
        self.tree_budget = int(tree_budget)
        self.branch_width = int(branch_width)
        self._custom_scheduler = scheduler
        self._roofline = roofline

        self.slot_pool: KvNStateSlotPool | None = None
        self.apc_cache: ZeroReplayAPCStateCache | None = (
            ZeroReplayAPCStateCache(
                block_size=self.apc_block_size,
                max_cached_blocks=self.apc_max_blocks,
                snapshot_device="cpu",
            )
            if self.enable_apc
            else None
        )
        self.scheduler: PurityLoadAwareAmdahlScheduler | None = scheduler
        self._req_APC_parent_hash: dict[str, int] = {}
        self._model_loaded: bool = False

        if self.drafter_model is not None:
            self.drafter_model.to(self.device)
            self._initialize_runtime_buffers()
            self._model_loaded = True

    def _init_from_vllm_config(
        self,
        vllm_config: Any,
        default_max_seqs: int,
        override_device: torch.device | str | None,
    ) -> None:
        """Extract checkpoint path, max_num_seqs, and device from ``VllmConfig``."""
        self.vllm_config = vllm_config
        spec_cfg = vllm_config.speculative_config
        sched_cfg = getattr(vllm_config, "scheduler_config", None)
        self.max_num_seqs = int(
            getattr(sched_cfg, "max_num_seqs", default_max_seqs) or default_max_seqs
        )
        dev_cfg = getattr(vllm_config, "device_config", None)
        raw_dev = override_device or getattr(dev_cfg, "device", None) or "cuda"
        self.device = torch.device(raw_dev)

        proposer_import_path = (
            "speculators.vllm_integration.kvn_proposer.KvNDSparkVllmProposer"
        )
        ckpt = getattr(spec_cfg, "kvn_checkpoint_path", None)
        if not ckpt:
            env_ckpt = os.environ.get("KVN_DSPARK_CHECKPOINT_PATH")
            if env_ckpt:
                ckpt = env_ckpt
        if not ckpt:
            raw_model = getattr(spec_cfg, "model", None)
            if (
                isinstance(raw_model, str)
                and raw_model
                and raw_model != proposer_import_path
                and Path(raw_model).exists()
            ):
                ckpt = raw_model
        if not ckpt:
            draft_cfg = getattr(spec_cfg, "draft_model_config", None)
            target_cfg = getattr(spec_cfg, "target_model_config", None) or getattr(
                vllm_config, "model_config", None
            )
            if (
                draft_cfg is not None
                and getattr(spec_cfg, "method", None) != "custom_class"
                and draft_cfg is not target_cfg
            ):
                draft_model_ref = getattr(draft_cfg, "model", None)
                target_model_ref = (
                    getattr(target_cfg, "model", None)
                    if target_cfg is not None
                    else None
                )
                if (
                    isinstance(draft_model_ref, str)
                    and draft_model_ref
                    and draft_model_ref != target_model_ref
                ):
                    ckpt = draft_model_ref
        if ckpt:
            self._checkpoint_path = ckpt

    @property
    def aux_hidden_state_layer_ids(self) -> list[int]:
        """Return target verifier auxiliary hidden-state layer indices."""
        drafter = self._require_drafter()
        aux = getattr(drafter.config, "aux_hidden_state_layer_ids", None)
        return [int(i) for i in aux] if aux else []

    def _require_drafter(self) -> KvNDSparkDraftModel:
        if self.drafter_model is None:
            raise RuntimeError(
                "KvNDSparkVllmProposer.load_model() must be called before proposing."
            )
        return self.drafter_model

    def _require_slot_pool(self) -> KvNStateSlotPool:
        if self.slot_pool is None:
            raise RuntimeError("KvNStateSlotPool has not been initialized.")
        return self.slot_pool

    def _initialize_runtime_buffers(self) -> None:
        """Allocate persistent FP32 ``KvNStateSlotPool`` and Amdahl scheduler."""
        drafter = self._require_drafter()
        drafter.eval()
        cfg = drafter.config
        num_layers = max(1, len(drafter.kvn_layers))
        n_modes = int(cfg.kvn_n_modes)
        d_model = int(cfg.transformer_layer_config.hidden_size)
        use_lindblad = bool(getattr(cfg, "use_lindblad_purity", False))
        fc_dtype = drafter.fc.weight.dtype

        self.slot_pool = KvNStateSlotPool(
            max_num_seqs=self.max_num_seqs,
            num_layers=num_layers,
            n_modes=n_modes,
            d_model=d_model,
            use_lindblad_purity=use_lindblad,
            state_dtype=torch.float32,
            fc_dtype=fc_dtype,
            device=self.device,
        )
        if self.scheduler is None:
            max_k = (
                int(drafter.block_size)
                if drafter.config.sample_from_anchor
                else max(1, int(drafter.block_size) - 1)
            )
            self.scheduler = PurityLoadAwareAmdahlScheduler(
                roofline=self._roofline or HardwareRooflineProfile.h200_sxm(),
                max_horizon=max_k,
            )

    def load_model(self, target_model: nn.Module | None = None) -> None:
        """Load ``KvNDSparkDraftModel`` onto ``self.device`` and set aux layers.

        Idempotent: repeated invocations from ``GPUModelRunner.load_model`` and plugin
        hooks do not reload safetensors or overwrite initialized ``slot_pool`` state.
        """
        if self.drafter_model is None:
            if self._checkpoint_path is None:
                raise ValueError(
                    "No KvNDSparkDraftModel or checkpoint_path configured on proposer."
                )
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            loaded = KvNDSparkDraftModel.from_pretrained(
                self._checkpoint_path,
                torch_dtype=dtype,
            )
            if not isinstance(loaded, KvNDSparkDraftModel):
                raise TypeError(
                    f"Expected KvNDSparkDraftModel, got {type(loaded).__name__}."
                )
            self.drafter_model = loaded.to(self.device)

        self.drafter_model.to(self.device)
        self.drafter_model.eval()
        if not self._model_loaded or self.slot_pool is None:
            self._initialize_runtime_buffers()
            self._model_loaded = True

        if target_model is not None and hasattr(
            target_model, "set_aux_hidden_state_layers"
        ):
            aux_ids = self.aux_hidden_state_layer_ids
            if aux_ids:
                target_model.set_aux_hidden_state_layers(tuple(aux_ids))

    @torch.no_grad()
    def extract_fc_hidden(
        self,
        aux_or_concat_hidden: Tensor | Sequence[Tensor],
        already_projected: bool = False,
    ) -> Tensor:
        """Project verifier auxiliary hidden states via ``hidden_norm(fc(concat))``."""
        drafter = self._require_drafter()
        if isinstance(aux_or_concat_hidden, (list, tuple)):
            raw = torch.cat([h.to(self.device) for h in aux_or_concat_hidden], dim=-1)
        else:
            raw = aux_or_concat_hidden.to(self.device)

        if already_projected:
            return raw.to(drafter.fc.weight.dtype)
        return drafter.hidden_norm(drafter.fc(raw.to(drafter.fc.weight.dtype)))

    def _scan_zero_init_chunk(
        self,
        chunk_ids: Tensor,  # [1, T]
        shifted_fc: Tensor,  # [1, T, D]
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run ``O(T log T)`` parallel scan over ``kvn_layers`` from zero state.

        Returns per-position ``(q_all, p_all, nu_all)`` of shape ``[L, T, M]`` in FP32.
        """
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        tok_emb = drafter.embed_tokens(chunk_ids).to(shifted_fc.dtype)
        x_cur = shifted_fc + tok_emb
        reset_mask = torch.zeros(
            (1, chunk_ids.size(1)), dtype=torch.bool, device=self.device
        )
        reset_mask[:, 0] = True

        fwd_sig = inspect.signature(drafter.kvn_layers[0].forward)
        fwd_kwargs: dict[str, Any] = {"reset_mask": reset_mask}
        if "state_dtype" in fwd_sig.parameters:
            fwd_kwargs["state_dtype"] = torch.float32
        if "return_nu" in fwd_sig.parameters and pool.use_lindblad_purity:
            fwd_kwargs["return_nu"] = True

        q_layers: list[Tensor] = []
        p_layers: list[Tensor] = []
        nu_layers: list[Tensor] = []

        for kvn_layer in drafter.kvn_layers:
            x_cur, q_seq, p_seq, _, nu_seq = _unpack_layer_forward_outputs(
                kvn_layer(x_cur, **fwd_kwargs)
            )
            q_layers.append(q_seq[0].to(torch.float32))
            p_layers.append(p_seq[0].to(torch.float32))
            if nu_seq is not None:
                nu_layers.append(nu_seq[0].to(torch.float32))

        q_all = torch.stack(q_layers, dim=0)
        p_all = torch.stack(p_layers, dim=0)
        nu_all = torch.stack(nu_layers, dim=0) if nu_layers else None
        return q_all, p_all, nu_all

    def _scan_continuation_parallel(
        self,
        slot_idx: int,
        x_cur: Tensor,
        fwd_sig: inspect.Signature,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run ``O(T log T)`` continuation scan when ``initial_state`` is supported."""
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        reset_mask = torch.zeros(
            (1, x_cur.size(1)), dtype=torch.bool, device=self.device
        )
        q_layers: list[Tensor] = []
        p_layers: list[Tensor] = []
        nu_layers: list[Tensor] = []
        for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
            q0 = pool.q_pool[slot_idx, l_idx].unsqueeze(0)
            p0 = pool.p_pool[slot_idx, l_idx].unsqueeze(0)
            nu0 = (
                pool.nu_pool[slot_idx, l_idx].unsqueeze(0)
                if pool.nu_pool is not None
                else None
            )
            l_init = (q0, p0, nu0) if nu0 is not None else (q0, p0)
            fwd_kw: dict[str, Any] = {
                "reset_mask": reset_mask,
                "initial_state": l_init,
            }
            if "state_dtype" in fwd_sig.parameters:
                fwd_kw["state_dtype"] = torch.float32
            if "return_nu" in fwd_sig.parameters and pool.use_lindblad_purity:
                fwd_kw["return_nu"] = True
            x_cur, q_seq, p_seq, _, nu_seq = _unpack_layer_forward_outputs(
                kvn_layer(x_cur, **fwd_kw)
            )
            q_layers.append(q_seq[0].to(torch.float32))
            p_layers.append(p_seq[0].to(torch.float32))
            if nu_seq is not None:
                nu_layers.append(nu_seq[0].to(torch.float32))
        return (
            torch.stack(q_layers, dim=0),
            torch.stack(p_layers, dim=0),
            torch.stack(nu_layers, dim=0) if nu_layers else None,
        )

    def _step_continuation_chunk(
        self,
        slot_idx: int,
        chunk_ids: Tensor,  # [1, T]
        shifted_fc: Tensor,  # [1, T, D]
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Advance recurrent state across a continuation chunk on ``slot_idx``."""
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        tok_emb = drafter.embed_tokens(chunk_ids).to(shifted_fc.dtype)
        x_cur = shifted_fc + tok_emb
        seq_len = chunk_ids.size(1)

        fwd_sig = inspect.signature(drafter.kvn_layers[0].forward)
        if "initial_state" in fwd_sig.parameters:
            return self._scan_continuation_parallel(slot_idx, x_cur, fwd_sig)

        q_layers: list[Tensor] = []
        p_layers: list[Tensor] = []
        nu_layers: list[Tensor] = []
        rec_sig = inspect.signature(drafter.kvn_layers[0].recurrent_step)
        supports_state_dtype = "state_dtype" in rec_sig.parameters

        for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
            q_s = pool.q_pool[slot_idx, l_idx].view(1, 1, -1)
            p_s = pool.p_pool[slot_idx, l_idx].view(1, 1, -1)
            nu_s = (
                pool.nu_pool[slot_idx, l_idx].view(1, 1, -1)
                if pool.nu_pool is not None
                else None
            )

            h_list: list[Tensor] = []
            q_hist: list[Tensor] = []
            p_hist: list[Tensor] = []
            nu_hist: list[Tensor] = []

            for t_idx in range(seq_len):
                state_in: tuple[Tensor, ...] = (
                    (q_s, p_s, nu_s) if nu_s is not None else (q_s, p_s)
                )
                step_kwargs: dict[str, Any] = {"state": state_in}
                if supports_state_dtype:
                    step_kwargs["state_dtype"] = torch.float32
                h_t, q_s, p_s, nu_next, _ = _unpack_recurrent_step_outputs(
                    kvn_layer.recurrent_step(
                        x_cur[:, t_idx : t_idx + 1, :], **step_kwargs
                    )
                )
                q_s = q_s.to(torch.float32)
                p_s = p_s.to(torch.float32)
                if nu_next is not None:
                    nu_s = nu_next.to(torch.float32)
                    nu_hist.append(nu_s.squeeze(0).squeeze(0))
                h_list.append(h_t)
                q_hist.append(q_s.squeeze(0).squeeze(0))
                p_hist.append(p_s.squeeze(0).squeeze(0))

            x_cur = torch.cat(h_list, dim=1)
            q_layers.append(torch.stack(q_hist, dim=0))
            p_layers.append(torch.stack(p_hist, dim=0))
            if nu_hist:
                nu_layers.append(torch.stack(nu_hist, dim=0))

        q_all = torch.stack(q_layers, dim=0)
        p_all = torch.stack(p_layers, dim=0)
        nu_all = torch.stack(nu_layers, dim=0) if nu_layers else None
        return q_all, p_all, nu_all

    @torch.no_grad()
    def prefill_request(
        self,
        req_id: str,
        token_ids: Sequence[int] | Tensor,
        aux_or_fc_hidden: Tensor | Sequence[Tensor],
        *,
        already_projected_fc: bool = False,
        use_apc: bool = True,
    ) -> tuple[int, int]:
        """Prefill or continue a chunked prompt for ``req_id``.

        Args:
            req_id: Unique vLLM request identifier.
            token_ids: Full prompt tokens (on fresh request) OR current chunk tokens.
            aux_or_fc_hidden: Verifier auxiliary hidden states ``[T_fc, ...]`` (either
                matching ``len(token_ids)`` or already sliced to the uncached suffix).
            already_projected_fc: Set ``True`` if ``aux_or_fc_hidden`` is already
                ``drafter.hidden_norm(drafter.fc(...))`` of shape ``[T, D]``.
            use_apc: Whether to attempt 0-FLOP APC restoration on fresh requests.

        Returns:
            ``(slot_idx, restored_apc_tokens)``.
        """
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        slot_idx, is_new = pool.get_or_allocate_slot(req_id)

        tok_list = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, Tensor)
            else [int(t) for t in token_ids]
        )
        fc_seq = self.extract_fc_hidden(
            aux_or_fc_hidden, already_projected=already_projected_fc
        )
        if fc_seq.ndim == _RANK_3D and fc_seq.size(0) == 1:
            fc_seq = fc_seq.squeeze(0)

        restored_tokens = 0
        parent_hash = self._req_APC_parent_hash.get(req_id, 0)

        if is_new and use_apc and self.apc_cache is not None:
            restored_tokens, parent_hash, suffix_toks = (
                self.apc_cache.match_and_restore_prefix(
                    tok_list, slot_pool=pool, slot_idx=slot_idx
                )
            )
            if restored_tokens > 0 and fc_seq.size(0) == len(tok_list):
                fc_seq = fc_seq[restored_tokens:]
            tok_list = suffix_toks

        if not tok_list:
            self._req_APC_parent_hash[req_id] = parent_hash
            return slot_idx, restored_tokens

        if fc_seq.size(0) != len(tok_list):
            raise ValueError(
                f"fc_seq len ({fc_seq.size(0)}) != uncached tokens ({len(tok_list)})."
            )

        start_offset = int(pool.seq_lens[slot_idx].item())
        had_prior = bool(pool.has_last_fc[slot_idx].item())
        shifted_fc = pool.build_causal_shifted_fc(slot_idx, fc_seq).unsqueeze(0)
        chunk_tensor = torch.tensor([tok_list], dtype=torch.long, device=self.device)

        if not had_prior:
            if self.apc_cache is None and hasattr(drafter, "prefill_kvn_state"):
                state_out = drafter.prefill_kvn_state(chunk_tensor, fc_seq.unsqueeze(0))
                self._commit_prefill_contract_output(
                    slot_idx, state_out, fc_seq[-1], len(tok_list)
                )
                return slot_idx, restored_tokens
            q_all, p_all, nu_all = self._scan_zero_init_chunk(chunk_tensor, shifted_fc)
        else:
            q_all, p_all, nu_all = self._step_continuation_chunk(
                slot_idx, chunk_tensor, shifted_fc
            )

        pool.commit_slot_state(
            slot_idx=slot_idx,
            q_state=q_all[:, -1, :],
            p_state=p_all[:, -1, :],
            last_fc=fc_seq[-1],
            tokens_advanced=len(tok_list),
            nu_state=nu_all[:, -1, :] if nu_all is not None else None,
        )

        if self.apc_cache is not None and start_offset % self.apc_block_size == 0:
            new_hashes = self.apc_cache.record_sequence_blocks(
                token_ids=tok_list,
                q_per_step=q_all,
                p_per_step=p_all,
                fc_per_step=fc_seq,
                nu_per_step=nu_all,
                start_offset=start_offset,
                initial_parent_hash=parent_hash,
            )
            if new_hashes:
                parent_hash = new_hashes[-1]
        self._req_APC_parent_hash[req_id] = parent_hash
        return slot_idx, restored_tokens

    def _commit_prefill_contract_output(
        self,
        slot_idx: int,
        state_out: Any,
        fallback_last_fc: Tensor,
        tokens_advanced: int,
    ) -> None:
        """Commit return value of ``drafter.prefill_kvn_state`` into ``slot_pool``."""
        pool = self._require_slot_pool()
        if isinstance(state_out, dict):
            last_fc = state_out.get("last_fc", fallback_last_fc).reshape(-1)
            if "state_cache" in state_out:
                pool.scatter_drafter_state_cache([slot_idx], state_out["state_cache"])
                pool.last_fc_pool[slot_idx].copy_(
                    last_fc.to(self.device, pool.fc_dtype)
                )
                pool.has_last_fc[slot_idx] = True
                pool.seq_lens[slot_idx] += int(tokens_advanced)
                return
            q_s = (
                state_out["q"][:, 0, :]
                if state_out["q"].ndim == _RANK_3D
                else state_out["q"]
            )
            p_s = (
                state_out["p"][:, 0, :]
                if state_out["p"].ndim == _RANK_3D
                else state_out["p"]
            )
            nu_raw = state_out.get("nu")
            nu_s = (
                nu_raw[:, 0, :]
                if (nu_raw is not None and nu_raw.ndim == _RANK_3D)
                else nu_raw
            )
            pool.commit_slot_state(
                slot_idx, q_s, p_s, last_fc, tokens_advanced, nu_state=nu_s
            )
            return
        pool.scatter_drafter_state_cache([slot_idx], state_out)
        pool.last_fc_pool[slot_idx].copy_(
            fallback_last_fc.reshape(-1).to(self.device, pool.fc_dtype)
        )
        pool.has_last_fc[slot_idx] = True
        pool.seq_lens[slot_idx] += int(tokens_advanced)

    def _build_decode_step_tensors(
        self,
        slot_indices: list[int],
        confirmed_token_ids: Sequence[Sequence[int]],
        confirmed_fc_list: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        """Pack variable ``1..block_size+1`` confirmed steps into ``[B, S_max]``."""
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        bsz = len(slot_indices)
        s_max = int(drafter.block_size) + 1
        d_model = pool.d_model
        fc_dtype = drafter.fc.weight.dtype

        prev_fc_block = torch.zeros(
            (bsz, s_max, d_model), dtype=fc_dtype, device=self.device
        )
        token_ids_block = torch.zeros(
            (bsz, s_max), dtype=torch.long, device=self.device
        )
        valid_lens = torch.zeros((bsz,), dtype=torch.long, device=self.device)
        new_last_fcs: list[Tensor] = []

        for b_idx, (slot_i, toks_b, fc_b) in enumerate(
            zip(slot_indices, confirmed_token_ids, confirmed_fc_list, strict=True)
        ):
            v_len = len(toks_b)
            if v_len <= 0 or v_len > s_max:
                raise ValueError(
                    f"confirmed_token_ids length must be in [1, {s_max}], got {v_len}."
                )
            fc_2d = fc_b.reshape(v_len, d_model).to(self.device, fc_dtype)
            shifted_b = pool.build_causal_shifted_fc(slot_i, fc_2d)
            prev_fc_block[b_idx, :v_len, :] = shifted_b
            token_ids_block[b_idx, :v_len] = torch.tensor(
                toks_b, dtype=torch.long, device=self.device
            )
            valid_lens[b_idx] = v_len
            new_last_fcs.append(fc_2d[-1])

        return prev_fc_block, token_ids_block, valid_lens, new_last_fcs

    def _invoke_advance_and_draft(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor,
        temperature: float,
    ) -> tuple[Tensor, Any, dict[str, Tensor]]:
        """Call ``drafter.advance_and_draft_step`` across both v2 and v3 signatures."""
        drafter = self._require_drafter()
        sig = inspect.signature(drafter.advance_and_draft_step)
        kwargs: dict[str, Any] = {}
        if "temperature" in sig.parameters:
            kwargs["temperature"] = float(temperature)
        if "return_details" in sig.parameters:
            kwargs["return_details"] = True

        out = drafter.advance_and_draft_step(
            prev_fc_block, token_ids_block, valid_lens, state_cache, **kwargs
        )
        if len(out) == _ADVANCE_V3_RETURN_LEN:
            draft_ids, updated_cache, details = out
            return draft_ids, updated_cache, dict(details)

        draft_ids, conf_probs = out
        return draft_ids, state_cache, {"confidences": conf_probs}

    def _fallback_kraus_tree_from_chain(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor,
        tree_budget: int,
        branch_width: int,
        temperature: float,
    ) -> tuple[dict[str, Tensor], Any]:
        """Fallback top-w tree builder when ``draft_kraus_tree_step`` is absent."""
        drafter = self._require_drafter()
        draft_ids, updated_cache, details = self._invoke_advance_and_draft(
            prev_fc_block, token_ids_block, valid_lens, state_cache, temperature
        )
        bsz, k_chain = draft_ids.shape
        conf = details.get(
            "confidences",
            torch.ones((bsz, k_chain), dtype=torch.float32, device=self.device),
        )
        budget = max(1, int(tree_budget))
        width = max(1, int(branch_width))

        # Depth-1 root top-w candidates from anchor readout + primary Markov chain
        n_nodes = min(budget, k_chain + max(0, width - 1))
        tree_ids = torch.zeros((bsz, n_nodes), dtype=torch.long, device=self.device)
        tree_parents = torch.full(
            (bsz, n_nodes), -1, dtype=torch.long, device=self.device
        )
        tree_surv = torch.ones((bsz, n_nodes), dtype=torch.float32, device=self.device)

        chain_len = min(k_chain, n_nodes)
        tree_ids[:, :chain_len] = draft_ids[:, :chain_len]
        if chain_len > 1:
            tree_parents[:, 1:chain_len] = torch.arange(
                chain_len - 1, device=self.device, dtype=torch.long
            ).unsqueeze(0)
        chain_surv = torch.cumprod(conf[:, :chain_len].float().clamp(0.0, 1.0), dim=-1)
        tree_surv[:, :chain_len] = chain_surv

        extra = n_nodes - chain_len
        if extra > 0:
            # Populate alternative depth-1 siblings from distinct shortlist token IDs
            alt_base = torch.arange(1, extra + 1, device=self.device, dtype=torch.long)
            alt_draft = (draft_ids[:, 0:1] + alt_base.unsqueeze(0)) % int(
                drafter.verifier_vocab_size
            )
            if drafter.use_draft_vocab and drafter.d2t is not None:
                alt_draft = alt_draft % drafter.d2t.numel()
                alt_target = alt_draft + drafter.d2t[alt_draft]
            else:
                alt_target = alt_draft
            tree_ids[:, chain_len:] = alt_target
            tree_parents[:, chain_len:] = -1
            tree_surv[:, chain_len:] = (conf[:, 0:1].float() * 0.5).expand(bsz, extra)

        _, depths = build_tree_attention_mask(tree_parents, prefix_len=0)
        return (
            {
                "tree_draft_ids": tree_ids,
                "tree_parent_idx": tree_parents,
                "tree_depths": depths,
                "tree_survival": tree_surv,
            },
            updated_cache,
        )

    def _propose_kraus_tree_or_fallback(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor,
        temperature: float,
    ) -> tuple[dict[str, Tensor], Any, Tensor, Tensor, dict[str, Tensor]]:
        """Invoke ``draft_kraus_tree_step`` (or fallback) and collect quantum stats."""
        drafter = self._require_drafter()
        if hasattr(drafter, "draft_kraus_tree_step"):
            tree_dict, updated_cache = drafter.draft_kraus_tree_step(
                prev_fc_block,
                token_ids_block,
                valid_lens,
                state_cache,
                tree_budget=self.tree_budget,
                branch_width=self.branch_width,
                temperature=float(temperature),
            )
        else:
            tree_dict, updated_cache = self._fallback_kraus_tree_from_chain(
                prev_fc_block,
                token_ids_block,
                valid_lens,
                state_cache,
                tree_budget=self.tree_budget,
                branch_width=self.branch_width,
                temperature=float(temperature),
            )
        draft_ids = tree_dict["tree_draft_ids"]
        conf_probs = tree_dict.get(
            "tree_survival",
            torch.ones_like(draft_ids, dtype=torch.float32),
        )
        details: dict[str, Tensor] = {"confidences": conf_probs}
        for key in ("purity", "p_vac"):
            if key in tree_dict:
                details[key] = tree_dict[key]
        return tree_dict, updated_cache, draft_ids, conf_probs, details

    @staticmethod
    def _derive_per_depth_tree_confidences(
        tree_dict: dict[str, Tensor],
        max_horizon: int,
    ) -> Tensor:
        """Convert cumulative BFS ``tree_survival`` into conditional depth alphas.

        ``tree_dict["tree_survival"]`` stores cumulative path survival ``S(u)`` across
        sibling nodes at depths ``1..D``. Taking ``S_d = max_{u: depth(u)=d} S(u)`` and
        returning conditional step ratios ``alpha_d = S_d / S_{d-1}`` ensures
        ``scheduler.compute_effective_survival`` recovers ``cumprod(alpha) == S_d``
        with zero double-``cumprod`` across siblings.
        """
        surv = tree_dict["tree_survival"].to(torch.float32).clamp(0.0, 1.0)
        bsz, n_nodes = surv.shape
        depths = tree_dict.get("tree_depths")
        if depths is None:
            _, depths = build_tree_attention_mask(
                tree_dict["tree_parent_idx"], prefix_len=0
            )
        max_d = int(depths.max().item()) if n_nodes > 0 else 1
        k_depth = max(1, min(int(max_horizon), max(1, max_d)))

        prev_s = torch.ones((bsz,), dtype=torch.float32, device=surv.device)
        step_alphas: list[Tensor] = []
        for depth_val in range(1, k_depth + 1):
            mask_d = depths == depth_val
            masked_surv = torch.where(mask_d, surv, torch.zeros_like(surv))
            s_d = torch.minimum(masked_surv.max(dim=-1).values, prev_s)
            alpha_d = torch.where(
                prev_s > _PROB_EPS,
                (s_d / prev_s.clamp_min(_PROB_EPS)).clamp(0.0, 1.0),
                torch.zeros_like(s_d),
            )
            step_alphas.append(alpha_d)
            prev_s = s_d
        return torch.stack(step_alphas, dim=1)

    @staticmethod
    def _extract_aligned_quantum_details(
        details: dict[str, Tensor],
        block_size: int,
        start_k: int,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Align ``purity``, ``p_vac``, and normalized entropy to ``start_k``."""

        def _slice_slot(val: Tensor | None) -> Tensor | None:
            if (
                val is not None
                and val.ndim >= _RANK_2D
                and val.size(1) == block_size
                and start_k > 0
            ):
                return val[:, start_k:]
            return val

        pur = _slice_slot(details.get("purity"))
        vac = _slice_slot(details.get("p_vac"))
        ent = _slice_slot(details.get("draft_entropy"))
        raw_dp = details.get("draft_probs")
        if ent is not None and raw_dp is not None and raw_dp.size(-1) > 1:
            ent = ent / math.log(float(raw_dp.size(-1)))
        return pur, vac, ent

    @torch.no_grad()
    def propose_batch(
        self,
        req_ids: Sequence[str],
        confirmed_token_ids: Sequence[Sequence[int]],
        confirmed_aux_or_fc: Sequence[Tensor],
        *,
        already_projected_fc: bool = False,
        temperature: float = 0.0,
        active_batch_size: int | None = None,
        use_kraus_tree: bool | None = None,
    ) -> KvNProposeOutput:
        """Advance recurrent states for ``req_ids`` and propose speculative tokens."""
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        if self.scheduler is None:
            raise RuntimeError("Amdahl scheduler is not initialized.")

        slot_indices = [pool.resolve_slot_or_restore(rid) for rid in req_ids]
        fc_list = [
            self.extract_fc_hidden(x, already_projected=already_projected_fc)
            for x in confirmed_aux_or_fc
        ]
        prev_fc_block, token_ids_block, valid_lens, new_last_fcs = (
            self._build_decode_step_tensors(slot_indices, confirmed_token_ids, fc_list)
        )

        # Match state_cache channels to drafter's allocate_state_cache contract
        proto_cache = drafter.allocate_state_cache(len(slot_indices), self.device)
        want_nu = (
            proto_cache.ndim == _RANK_4D and proto_cache.size(1) >= _STEP_STATE_V3_LEN
        )
        state_cache = pool.gather_drafter_state_cache(
            slot_indices, include_nu=want_nu
        ).to(proto_cache.dtype)

        run_tree = (
            self.use_kraus_tree if use_kraus_tree is None else bool(use_kraus_tree)
        )
        tree_dict: dict[str, Tensor] | None = None
        tree_sched: TreeScheduleDecision | None = None

        if run_tree:
            tree_dict, updated_cache, draft_ids, conf_probs, details = (
                self._propose_kraus_tree_or_fallback(
                    prev_fc_block,
                    token_ids_block,
                    valid_lens,
                    state_cache,
                    temperature,
                )
            )
        else:
            draft_ids, updated_cache, details = self._invoke_advance_and_draft(
                prev_fc_block,
                token_ids_block,
                valid_lens,
                state_cache,
                temperature=temperature,
            )
            conf_probs = details["confidences"]

        pool.scatter_drafter_state_cache(slot_indices, updated_cache)
        for slot_i, toks_b, last_fc_b in zip(
            slot_indices, confirmed_token_ids, new_last_fcs, strict=True
        ):
            pool.last_fc_pool[slot_i].copy_(last_fc_b.to(self.device, pool.fc_dtype))
            pool.has_last_fc[slot_i] = True
            pool.seq_lens[slot_i] += len(toks_b)

        # Slice away anchor slot 0 when sample_from_anchor is False; for Kraus trees,
        # extract conditional per-depth survivals without double-cumprod over siblings.
        start_k = 0 if drafter.config.sample_from_anchor else 1
        prop_ids = draft_ids[:, start_k:] if not run_tree else draft_ids
        prop_conf = (
            conf_probs[:, start_k:].to(torch.float32)
            if not run_tree
            else self._derive_per_depth_tree_confidences(
                tree_dict or {"tree_survival": conf_probs},
                self.scheduler.max_horizon,
            )
        )
        pur_detail, vac_detail, ent_detail = self._extract_aligned_quantum_details(
            details, int(drafter.block_size), start_k
        )

        schedule = self.scheduler.select_horizon(
            confidences=prop_conf,
            active_batch_size=active_batch_size or len(req_ids),
            purity=pur_detail,
            p_vac=vac_detail,
            draft_entropy=ent_detail,
            temperature=temperature,
        )
        if run_tree:
            tree_sched = self.scheduler.select_tree_budget(
                confidences=prop_conf,
                active_batch_size=active_batch_size or len(req_ids),
                purity=pur_detail,
                p_vac=vac_detail,
                draft_entropy=ent_detail,
                temperature=temperature,
            )
            horizons = tree_sched.per_req_tree_budgets.clamp(
                max=prop_ids.size(1)
            ).tolist()
        else:
            horizons = schedule.per_req_horizons.clamp(max=prop_ids.size(1)).tolist()

        prop_cpu = prop_ids.detach().cpu().tolist()
        per_req_tokens = [
            row[: int(k_req)] for row, k_req in zip(prop_cpu, horizons, strict=True)
        ]
        raw_probs = details.get("draft_probs")
        if raw_probs is not None and not run_tree:
            raw_probs = raw_probs[:, start_k:]

        return KvNProposeOutput(
            draft_token_ids=per_req_tokens,
            padded_draft_ids=prop_ids,
            confidences=prop_conf,
            schedule=schedule,
            draft_probs=raw_probs,
            tree_dict=tree_dict,
            tree_schedule=tree_sched,
        )

    def free_finished_requests(self, req_ids: Sequence[str]) -> None:
        """Release slots and metadata for completed/aborted requests."""
        pool = self._require_slot_pool()
        for rid in req_ids:
            pool.free_request(rid)
            self._req_APC_parent_hash.pop(rid, None)

    def preempt_requests(self, req_ids: Sequence[str]) -> None:
        """Offload preempted requests to host RAM so resumption has 0 state loss."""
        pool = self._require_slot_pool()
        for rid in req_ids:
            pool.preempt_request(rid)

    def _route_scheduled_vllm_request(
        self,
        rid: str,
        n_sched: int,
        fc_slice: Tensor,
        sampled_b: list[int],
        total_seq: int,
        tok_row: Any,
    ) -> tuple[list[int], Tensor] | None:
        """Handle prefill/continuation chunks or return ``(commit_toks, fc_slice)``."""
        pool = self._require_slot_pool()
        if not pool.has_request(rid) and not pool.is_preempted(rid):
            prompt_chunk = [int(t) for t in tok_row[:total_seq]]
            if len(prompt_chunk) != n_sched:
                prompt_chunk = prompt_chunk[-n_sched:]
            self.prefill_request(
                rid,
                prompt_chunk,
                fc_slice,
                already_projected_fc=True,
                use_apc=(len(prompt_chunk) == total_seq),
            )
            return None

        if not sampled_b:
            cont_chunk = [int(t) for t in tok_row[total_seq - n_sched : total_seq]]
            self.prefill_request(
                rid,
                cont_chunk,
                fc_slice,
                already_projected_fc=True,
                use_apc=False,
            )
            return None

        n_commit = min(n_sched, fc_slice.size(0))
        commit_toks = [int(t) for t in tok_row[total_seq - n_commit : total_seq]]
        return commit_toks, fc_slice[:n_commit]

    @torch.no_grad()
    def propose_vllm_step(
        self,
        runner: Any,
        scheduler_output: Any,
        sampled_token_ids: Tensor | list[list[int]],
        aux_hidden_states: list[Tensor] | None,
    ) -> list[list[int]]:
        """Execute one live ``vLLM V1`` ``GPUModelRunner`` speculative proposal step."""
        pool = self._require_slot_pool()
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        if finished:
            self.free_finished_requests(list(finished))

        input_batch = runner.input_batch
        req_ids: list[str] = list(input_batch.req_ids)
        if not req_ids or not aux_hidden_states:
            return [[] for _ in req_ids]

        active_set = set(req_ids)
        stale = [rid for rid in pool.active_request_ids() if rid not in active_set]
        if stale:
            self.preempt_requests(stale)

        fc_flat = self.extract_fc_hidden(aux_hidden_states, already_projected=False)
        num_scheduled_map: dict[str, int] = getattr(
            scheduler_output, "num_scheduled_tokens", {}
        )
        sampled_rows = (
            sampled_token_ids.detach().cpu().tolist()
            if isinstance(sampled_token_ids, Tensor)
            else [list(r) for r in sampled_token_ids]
        )

        decode_reqs: list[str] = []
        decode_toks: list[list[int]] = []
        decode_fcs: list[Tensor] = []
        decode_positions: list[int] = []
        cursor = 0

        for b_idx, rid in enumerate(req_ids):
            n_sched = int(num_scheduled_map.get(rid, 1))
            fc_slice = fc_flat[cursor : cursor + n_sched]
            cursor += n_sched
            sampled_b = [int(t) for t in sampled_rows[b_idx] if int(t) >= 0]
            routed = self._route_scheduled_vllm_request(
                rid=rid,
                n_sched=n_sched,
                fc_slice=fc_slice,
                sampled_b=sampled_b,
                total_seq=int(input_batch.num_tokens_no_spec[b_idx]),
                tok_row=input_batch.token_ids_cpu[b_idx],
            )
            if routed is not None:
                commit_toks, commit_fc = routed
                decode_reqs.append(rid)
                decode_toks.append(commit_toks)
                decode_fcs.append(commit_fc)
                decode_positions.append(b_idx)

        proposals: list[list[int]] = [[] for _ in req_ids]
        if not decode_reqs:
            return proposals

        out = self.propose_batch(
            decode_reqs,
            decode_toks,
            decode_fcs,
            already_projected_fc=True,
            active_batch_size=len(req_ids),
        )
        for pos, toks in zip(decode_positions, out.draft_token_ids, strict=True):
            proposals[pos] = toks
        return proposals

    def propose(
        self,
        sampled_token_ids: Tensor,
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
    ) -> Tensor:
        """Fallback ``CustomClassProposer.propose`` returning ``[B, K]`` tensor."""
        del num_tokens_no_spec, token_ids_cpu, slot_mappings
        drafter = self._require_drafter()
        bsz = int(sampled_token_ids.shape[0])
        num_spec = (
            int(drafter.block_size)
            if drafter.config.sample_from_anchor
            else max(1, int(drafter.block_size) - 1)
        )
        return torch.zeros((bsz, num_spec), dtype=torch.long, device=self.device)
