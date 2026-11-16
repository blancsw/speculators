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
_MIN_TEMP_EPS = 1e-5
_PREV2_MIN_POS = 2
_CUDAGRAPH_BATCH_BUCKETS: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
)
_MAX_CUDAGRAPH_BATCH = 128

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
        env_tree = os.environ.get("KVN_VLLM_USE_KRAUS_TREE", "").strip() == "1"
        self.use_kraus_tree = bool(use_kraus_tree or env_tree)
        self.tree_budget = int(tree_budget)
        self.branch_width = int(branch_width)
        self._custom_scheduler = scheduler
        self._roofline = roofline
        self._current_sampling_metadata: Any = None
        self._current_spec_decode_metadata: Any = None
        self._async_step_tensors: tuple[Tensor, Tensor, Tensor] | None = None
        self._max_vllm_spec_tokens: int | None = None

        # Static CUDA-Graph runtime buffers (populated in _initialize_runtime_buffers)
        self._supports_fast_v3_decode: bool = False
        self._cg_enabled: bool = False
        self._cg_mempool: Any = None
        self._cg_graphs: dict[
            tuple[int, int, bool, bool, bool], torch.cuda.CUDAGraph
        ] = {}
        self._cg_tree_graphs: dict[
            tuple[int, int, int, int, bool],
            tuple[torch.cuda.CUDAGraph, dict[str, Tensor]],
        ] = {}
        self._cg_prev_fc: Tensor | None = None
        self._cg_token_ids: Tensor | None = None
        self._cg_valid_lens: Tensor | None = None
        self._cg_prev2_ids: Tensor | None = None
        self._cg_slot_idx: Tensor | None = None
        self._cg_temps: Tensor | None = None
        self._cg_state_buf: Tensor | None = None
        self._cg_out_i64: Tensor | None = None
        self._cg_out_cpu: Tensor | None = None
        self._cg_meta_cpu: Tensor | None = None
        self._cg_meta_gpu: Tensor | None = None
        self._cg_temps_cpu: Tensor | None = None
        self._zero_fc_row: Tensor | None = None
        self._want_nu_state: bool = True
        self._state_cache_dtype: torch.dtype = torch.float32

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
            if self.vllm_config is not None:
                spec_cfg = getattr(self.vllm_config, "speculative_config", None)
                spec_num = getattr(spec_cfg, "num_speculative_tokens", None)
                if spec_num is not None and int(spec_num) > 0:
                    self._max_vllm_spec_tokens = int(spec_num)
                    max_k = min(max_k, int(spec_num))
            min_surv_env = os.environ.get("KVN_VLLM_MIN_SURVIVAL_FLOOR", "").strip()
            min_surv = float(min_surv_env) if min_surv_env else 0.08
            self.scheduler = PurityLoadAwareAmdahlScheduler(
                roofline=self._roofline or HardwareRooflineProfile.h200_sxm(),
                max_horizon=max_k,
                min_survival_floor=min_surv,
            )

        proto_cache = drafter.allocate_state_cache(1, self.device)
        self._want_nu_state = (
            proto_cache.ndim == _RANK_4D and proto_cache.size(1) >= _STEP_STATE_V3_LEN
        )
        self._state_cache_dtype = proto_cache.dtype

        adv_params = inspect.signature(drafter.advance_and_draft_step).parameters
        self._supports_fast_v3_decode = (
            "prev_context_ids" in adv_params
            and "temperatures_tensor" in adv_params
            and "include_draft_probs" in adv_params
        )
        self._cg_graphs.clear()
        self._cg_tree_graphs.clear()
        self._zero_fc_row = torch.zeros(
            (1, d_model), dtype=fc_dtype, device=self.device
        )

        disable_cg = os.environ.get("KVN_VLLM_DISABLE_CUDAGRAPH", "0").strip() == "1"
        self._cg_enabled = (
            self.device.type == "cuda"
            and self._supports_fast_v3_decode
            and not disable_cg
        )
        if self._cg_enabled:
            self._init_cudagraph_workspace(
                num_layers=num_layers,
                n_modes=n_modes,
                d_model=d_model,
                fc_dtype=fc_dtype,
                s_max=int(drafter.block_size) + 1,
                k_out=(
                    int(drafter.block_size)
                    if drafter.config.sample_from_anchor
                    else max(1, int(drafter.block_size) - 1)
                ),
            )
            self._warmup_triton_kernels()

    @torch.no_grad()
    def _warmup_triton_kernels(self) -> None:
        """Eagerly JIT-compile KvN Triton scan/decode kernels before live serving."""
        if (
            self.device.type != "cuda"
            or os.environ.get("KVN_VLLM_SKIP_TRITON_WARMUP", "0").strip() == "1"
        ):
            return
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        d_model = pool.d_model
        fc_dtype = drafter.fc.weight.dtype
        for warm_t in (8, 128):
            dummy_ids = torch.zeros(
                (1, warm_t), dtype=torch.long, device=self.device
            )
            dummy_fc = torch.zeros(
                (1, warm_t, d_model), dtype=fc_dtype, device=self.device
            )
            self._scan_zero_init_chunk(dummy_ids, dummy_fc)
        if self._supports_fast_v3_decode:
            sc = drafter.allocate_state_cache(1, self.device)
            s_max = int(drafter.block_size) + 1
            z_fc = torch.zeros((1, s_max, d_model), dtype=fc_dtype, device=self.device)
            z_tok = torch.zeros((1, s_max), dtype=torch.long, device=self.device)
            z_vlen = torch.ones((1,), dtype=torch.long, device=self.device)
            z_prev2 = torch.zeros((1, 2), dtype=torch.long, device=self.device)
            drafter.advance_and_draft_step(
                z_fc,
                z_tok,
                z_vlen,
                sc.clone(),
                temperature=0.0,
                return_details=True,
                prev_context_ids=z_prev2,
                include_draft_probs=False,
            )
            if getattr(drafter, "kraus_head", None) is not None and hasattr(
                drafter, "draft_kraus_tree_step"
            ):
                drafter.draft_kraus_tree_step(
                    z_fc,
                    z_tok,
                    z_vlen,
                    sc.clone(),
                    tree_budget=min(8, max(2, self.tree_budget)),
                    branch_width=max(2, self.branch_width),
                    temperature=0.0,
                    return_q_probs=False,
                    prev_context_ids=z_prev2,
                )

    def _init_cudagraph_workspace(
        self,
        *,
        num_layers: int,
        n_modes: int,
        d_model: int,
        fc_dtype: torch.dtype,
        s_max: int,
        k_out: int,
    ) -> None:
        """Allocate static GPU & pinned-CPU staging buffers for CUDA-Graph decode."""
        max_b = _MAX_CUDAGRAPH_BATCH
        n_comp = 3 if self._want_nu_state else 2
        self._cg_mempool = torch.cuda.graph_pool_handle()
        self._cg_prev_fc = torch.zeros(
            (max_b, s_max, d_model), dtype=fc_dtype, device=self.device
        )
        self._cg_token_ids = torch.zeros(
            (max_b, s_max), dtype=torch.long, device=self.device
        )
        self._cg_valid_lens = torch.ones(
            (max_b,), dtype=torch.long, device=self.device
        )
        self._cg_prev2_ids = torch.zeros(
            (max_b, 2), dtype=torch.long, device=self.device
        )
        self._cg_slot_idx = torch.zeros(
            (max_b,), dtype=torch.long, device=self.device
        )
        self._cg_temps = torch.zeros(
            (max_b,), dtype=torch.float32, device=self.device
        )
        self._cg_state_buf = torch.zeros(
            (num_layers, n_comp, max_b, n_modes),
            dtype=self._state_cache_dtype,
            device=self.device,
        )
        self._cg_out_i64 = torch.zeros(
            (max_b, 1 + k_out), dtype=torch.long, device=self.device
        )
        self._cg_out_cpu = torch.zeros(
            (max_b, 1 + k_out), dtype=torch.long, device="cpu", pin_memory=True
        )
        # Columns: [slot, valid_len, p0, p1, tok_0..S-1, fc_idx_0..S-1]
        meta_cols = 4 + 2 * s_max
        self._cg_meta_cpu = torch.zeros(
            (max_b, meta_cols), dtype=torch.long, device="cpu", pin_memory=True
        )
        self._cg_meta_gpu = torch.zeros(
            (max_b, meta_cols), dtype=torch.long, device=self.device
        )
        self._cg_temps_cpu = torch.zeros(
            (max_b,), dtype=torch.float32, device="cpu", pin_memory=True
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
                pool.advance_slot_last2_tokens(slot_idx, tok_list)
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
        pool.advance_slot_last2_tokens(slot_idx, tok_list)

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
        born_calibrated = (not run_tree) and (
            getattr(drafter, "born_confidence_head", None) is not None
        )

        schedule = self.scheduler.select_horizon(
            confidences=prop_conf,
            active_batch_size=active_batch_size or len(req_ids),
            purity=None if born_calibrated else pur_detail,
            p_vac=None if born_calibrated else vac_detail,
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
        max_saved = max(512, 4 * self.max_num_seqs)
        for rid in req_ids:
            pool.preempt_request(rid)
        for oldest_rid in pool.evict_excess_preempted(max_saved):
            self._req_APC_parent_hash.pop(oldest_rid, None)

    def _prefill_requests_batched(  # noqa: C901
        self,
        prefill_items: list[tuple[str, int, list[int], Tensor, bool]],
    ) -> None:
        """Execute 1 or more prefill/continuation chunks in one batched forward pass.

        Preserves exact 0-FLOP ``ZeroReplayAPCStateCache`` 16-token block restore and
        snapshot recording across all requests in ``prefill_items``.
        """
        if not prefill_items:
            return
        drafter = self._require_drafter()
        pool = self._require_slot_pool()

        prepared: list[tuple[str, int, list[int], Tensor, int, bool, int]] = []
        for rid, slot_idx, raw_toks, raw_fc, use_apc in prefill_items:
            fc_cur = (
                raw_fc.squeeze(0)
                if raw_fc.ndim == _RANK_3D and raw_fc.size(0) == 1
                else raw_fc
            )
            tok_cur = raw_toks
            p_hash = self._req_APC_parent_hash.get(rid, 0)
            if (
                use_apc
                and self.apc_cache is not None
                and pool.get_seq_len(slot_idx) == 0
            ):
                restored, p_hash, suffix_toks = (
                    self.apc_cache.match_and_restore_prefix(
                        tok_cur, slot_pool=pool, slot_idx=slot_idx
                    )
                )
                if restored > 0 and fc_cur.size(0) == len(tok_cur):
                    fc_cur = fc_cur[restored:]
                tok_cur = suffix_toks
            if not tok_cur:
                self._req_APC_parent_hash[rid] = p_hash
                continue
            n_eff = min(len(tok_cur), int(fc_cur.size(0)))
            if n_eff <= 0:
                self._req_APC_parent_hash[rid] = p_hash
                continue
            tok_eff = tok_cur[-n_eff:]
            fc_eff = fc_cur[:n_eff]
            start_offset = pool.get_seq_len(slot_idx)
            had_prior = pool.get_has_last_fc(slot_idx)
            prepared.append(
                (rid, slot_idx, tok_eff, fc_eff, start_offset, had_prior, p_hash)
            )

        if not prepared:
            return

        fwd_sig = inspect.signature(drafter.kvn_layers[0].forward)
        if len(prepared) == 1 or "initial_state" not in fwd_sig.parameters:
            for (
                rid,
                slot_idx,
                tok_list,
                fc_seq,
                start_offset,
                had_prior,
                parent_hash,
            ) in prepared:
                shifted_fc = pool.build_causal_shifted_fc(slot_idx, fc_seq).unsqueeze(0)
                chunk_tensor = torch.tensor(
                    [tok_list], dtype=torch.long, device=self.device
                )
                if not had_prior:
                    q_all, p_all, nu_all = self._scan_zero_init_chunk(
                        chunk_tensor, shifted_fc
                    )
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
                pool.advance_slot_last2_tokens(slot_idx, tok_list)
                if (
                    self.apc_cache is not None
                    and start_offset % self.apc_block_size == 0
                ):
                    new_hashes = self.apc_cache.record_sequence_blocks(
                        token_ids=tok_list,
                        q_per_step=q_all,
                        p_per_step=p_all,
                        fc_per_step=fc_seq,
                        nu_per_step=nu_all,
                        start_offset=start_offset,
                        initial_parent_hash=parent_hash,
                    )
                    final_hash = new_hashes[-1] if new_hashes else parent_hash
                else:
                    final_hash = parent_hash
                self._req_APC_parent_hash[rid] = final_hash
            return

        # Vectorized batched [B_pf, T_max, D] causal prefill across kvn_layers
        b_pf = len(prepared)
        t_max = max(len(it[2]) for it in prepared)
        d_model = pool.d_model
        fc_dtype = drafter.fc.weight.dtype

        chunk_ids_pad = torch.zeros(
            (b_pf, t_max), dtype=torch.long, device=self.device
        )
        shifted_fc_pad = torch.zeros(
            (b_pf, t_max, d_model), dtype=fc_dtype, device=self.device
        )
        reset_mask_pad = torch.zeros(
            (b_pf, t_max), dtype=torch.bool, device=self.device
        )
        slot_idx_list = [it[1] for it in prepared]
        slot_idx_gpu = torch.tensor(slot_idx_list, dtype=torch.long, device=self.device)

        for i_pf, (_, slot_idx, tok_list, fc_seq, _, had_prior, _) in enumerate(
            prepared
        ):
            t_len = len(tok_list)
            chunk_ids_pad[i_pf, :t_len] = torch.tensor(
                tok_list, dtype=torch.long, device=self.device
            )
            shifted_fc_pad[i_pf, :t_len, :] = pool.build_causal_shifted_fc(
                slot_idx, fc_seq.to(fc_dtype)
            )
            if not had_prior:
                reset_mask_pad[i_pf, 0] = True

        x_cur = shifted_fc_pad + drafter.embed_tokens(chunk_ids_pad).to(fc_dtype)
        q0_all = pool.q_pool.index_select(0, slot_idx_gpu)  # [B_pf, L, M]
        p0_all = pool.p_pool.index_select(0, slot_idx_gpu)
        nu0_all = (
            pool.nu_pool.index_select(0, slot_idx_gpu)
            if pool.nu_pool is not None
            else None
        )

        q_layers: list[Tensor] = []
        p_layers: list[Tensor] = []
        nu_layers: list[Tensor] = []
        for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
            q0 = q0_all[:, l_idx, :]
            p0 = p0_all[:, l_idx, :]
            l_init: tuple[Tensor, ...] = (
                (q0, p0, nu0_all[:, l_idx, :]) if nu0_all is not None else (q0, p0)
            )
            fwd_kw: dict[str, Any] = {
                "reset_mask": reset_mask_pad,
                "initial_state": l_init,
            }
            if "state_dtype" in fwd_sig.parameters:
                fwd_kw["state_dtype"] = torch.float32
            if "return_nu" in fwd_sig.parameters and pool.use_lindblad_purity:
                fwd_kw["return_nu"] = True
            x_cur, q_seq, p_seq, _, nu_seq = _unpack_layer_forward_outputs(
                kvn_layer(x_cur, **fwd_kw)
            )
            q_layers.append(q_seq.to(torch.float32))
            p_layers.append(p_seq.to(torch.float32))
            if nu_seq is not None:
                nu_layers.append(nu_seq.to(torch.float32))

        # q_stack: [L, B_pf, T_max, M]
        q_stack = torch.stack(q_layers, dim=0)
        p_stack = torch.stack(p_layers, dim=0)
        nu_stack = torch.stack(nu_layers, dim=0) if nu_layers else None

        for i_pf, (
            rid,
            slot_idx,
            tok_list,
            fc_seq,
            start_offset,
            _,
            parent_hash,
        ) in enumerate(prepared):
            t_len = len(tok_list)
            q_i = q_stack[:, i_pf, :t_len, :]
            p_i = p_stack[:, i_pf, :t_len, :]
            nu_i = nu_stack[:, i_pf, :t_len, :] if nu_stack is not None else None
            pool.commit_slot_state(
                slot_idx=slot_idx,
                q_state=q_i[:, -1, :],
                p_state=p_i[:, -1, :],
                last_fc=fc_seq[-1],
                tokens_advanced=t_len,
                nu_state=nu_i[:, -1, :] if nu_i is not None else None,
            )
            pool.advance_slot_last2_tokens(slot_idx, tok_list)
            if self.apc_cache is not None and start_offset % self.apc_block_size == 0:
                new_hashes = self.apc_cache.record_sequence_blocks(
                    token_ids=tok_list,
                    q_per_step=q_i,
                    p_per_step=p_i,
                    fc_per_step=fc_seq,
                    nu_per_step=nu_i,
                    start_offset=start_offset,
                    initial_parent_hash=parent_hash,
                )
                final_hash = new_hashes[-1] if new_hashes else parent_hash
            else:
                final_hash = parent_hash
            self._req_APC_parent_hash[rid] = final_hash

    @staticmethod
    def _select_cg_bucket(batch_size: int) -> int:
        """Select smallest CUDA-Graph batch bucket >= ``batch_size``."""
        for b in _CUDAGRAPH_BATCH_BUCKETS:
            if b >= batch_size:
                return b
        return batch_size

    @staticmethod
    def _extract_kraus_tree_spine_static(
        tree_dict: dict[str, Tensor],
        k_out: int,
    ) -> tuple[Tensor, Tensor]:
        """Extract causal root-to-leaf spine ``[B, k_out]`` + confs with 0 syncs."""
        t_ids = tree_dict["tree_draft_ids"]
        t_par = tree_dict["tree_parent_idx"]
        t_dep = tree_dict["tree_depths"]
        t_surv = tree_dict["tree_survival"].float().clamp(0.0, 1.0)
        bsz = t_ids.size(0)
        dev = t_ids.device

        # Choose deepest highest-survival leaf and trace exact parent pointers to root
        leaf_score = t_dep.float() * 10.0 + t_surv
        cur = leaf_score.argmax(dim=-1, keepdim=True)  # [B, 1]
        spine_ids = torch.zeros((bsz, k_out), dtype=torch.long, device=dev)
        for _ in range(k_out):
            safe_idx = cur.clamp_min(0)
            dep_u = t_dep.gather(1, safe_idx)
            tok_u = t_ids.gather(1, safe_idx)
            col_u = (dep_u - 1).clamp(0, max(0, k_out - 1))
            valid_u = (cur >= 0) & (dep_u >= 1) & (dep_u <= k_out)
            cur_val = spine_ids.gather(1, col_u)
            spine_ids.scatter_(1, col_u, torch.where(valid_u, tok_u, cur_val))
            par_u = t_par.gather(1, safe_idx)
            cur = torch.where(cur >= 0, par_u, cur)

        prev_s = torch.ones((bsz,), dtype=torch.float32, device=dev)
        alphas: list[Tensor] = []
        for depth_val in range(1, k_out + 1):
            masked_s = torch.where(
                t_dep == depth_val, t_surv, torch.zeros_like(t_surv)
            )
            s_d = torch.minimum(masked_s.max(dim=-1).values, prev_s)
            alpha_d = torch.where(
                prev_s > _PROB_EPS,
                (s_d / prev_s.clamp_min(_PROB_EPS)).clamp(0.0, 1.0),
                torch.zeros_like(s_d),
            )
            alphas.append(alpha_d)
            prev_s = s_d
        return spine_ids, torch.stack(alphas, dim=1)

    def _require_cg_buffers(
        self,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Return validated CUDA-Graph workspace buffers or raise ``RuntimeError``."""
        if (
            self._cg_slot_idx is None
            or self._cg_prev_fc is None
            or self._cg_token_ids is None
            or self._cg_valid_lens is None
            or self._cg_prev2_ids is None
            or self._cg_temps is None
            or self._cg_state_buf is None
            or self._cg_out_i64 is None
            or self._cg_out_cpu is None
            or self._cg_meta_cpu is None
            or self._cg_meta_gpu is None
            or self._cg_temps_cpu is None
        ):
            raise RuntimeError("CUDA-Graph workspace buffers are not initialized.")
        return (
            self._cg_slot_idx,
            self._cg_prev_fc,
            self._cg_token_ids,
            self._cg_valid_lens,
            self._cg_prev2_ids,
            self._cg_temps,
            self._cg_state_buf,
            self._cg_out_i64,
            self._cg_out_cpu,
            self._cg_meta_cpu,
            self._cg_meta_gpu,
            self._cg_temps_cpu,
        )

    def _run_decode_graph_body(
        self,
        b_bucket: int,
        s_bucket: int,
        is_stochastic: bool,
        run_tree: bool,
        max_spec_cap: int,
        compute_prev2_from_pool: bool = False,
    ) -> None:
        """Zero-sync GPU kernel sequence captured inside ``torch.cuda.CUDAGraph``."""
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        if self.scheduler is None:
            raise RuntimeError("Amdahl scheduler is not initialized.")
        (
            cg_slot_idx,
            cg_prev_fc,
            cg_token_ids,
            cg_valid_lens,
            cg_prev2_ids,
            cg_temps,
            cg_state_buf,
            cg_out_i64,
            _,
            _,
            _,
            _,
        ) = self._require_cg_buffers()

        slot_v = cg_slot_idx[:b_bucket]
        prev_fc_v = cg_prev_fc[:b_bucket, :s_bucket, :]
        tok_v = cg_token_ids[:b_bucket, :s_bucket]
        vlen_v = cg_valid_lens[:b_bucket]
        if compute_prev2_from_pool:
            pool_last2 = pool.last2_tokens_pool.index_select(0, slot_v)
            w_tok = torch.cat([pool_last2, tok_v], dim=1)
            vlen_safe = vlen_v.clamp(min=1, max=s_bucket)
            p0_col = w_tok.gather(1, (vlen_safe - 1).unsqueeze(1))
            p1_col = w_tok.gather(1, vlen_safe.unsqueeze(1))
            prev2_v = torch.cat([p0_col, p1_col], dim=1)
            last_col = w_tok.gather(1, (vlen_safe + 1).unsqueeze(1))
            pool.last2_tokens_pool.index_copy_(
                0, slot_v, torch.cat([p1_col, last_col], dim=1)
            )
        else:
            prev2_v = cg_prev2_ids[:b_bucket, :]
        temps_v = cg_temps[:b_bucket]
        state_v = cg_state_buf[:, :, :b_bucket, :]
        out_v = cg_out_i64[:b_bucket, :]
        k_out = out_v.size(1) - 1
        start_k = 0 if drafter.config.sample_from_anchor else 1

        pool.gather_into_state_buffer(slot_v, state_v)

        if not run_tree:
            draft_ids, updated_cache, details = drafter.advance_and_draft_step(
                prev_fc_v,
                tok_v,
                vlen_v,
                state_v,
                temperature=(1.0 if is_stochastic else 0.0),
                return_details=True,
                prev_context_ids=prev2_v,
                temperatures_tensor=(temps_v if is_stochastic else None),
                include_draft_probs=False,
            )
            pool.scatter_from_state_buffer(slot_v, updated_cache)
            prop_ids = draft_ids[:, start_k:]
            prop_conf = details["confidences"][:, start_k:].to(torch.float32)
            pur_raw = details.get("purity")
            vac_raw = details.get("p_vac")
        else:
            tree_dict, updated_cache = drafter.draft_kraus_tree_step(
                prev_fc_v,
                tok_v,
                vlen_v,
                state_v,
                tree_budget=self.tree_budget,
                branch_width=self.branch_width,
                temperature=(0.6 if is_stochastic else 0.0),
                return_q_probs=False,
                prev_context_ids=prev2_v,
            )
            pool.scatter_from_state_buffer(slot_v, updated_cache)
            prop_ids, prop_conf = self._extract_kraus_tree_spine_static(
                tree_dict, k_out
            )
            pur_raw = tree_dict.get("purity")
            vac_raw = tree_dict.get("p_vac")

        if (
            pur_raw is not None
            and pur_raw.ndim >= _RANK_2D
            and pur_raw.size(1) == int(drafter.block_size)
            and start_k > 0
        ):
            pur_raw = pur_raw[:, start_k:]
        if (
            vac_raw is not None
            and vac_raw.ndim >= _RANK_2D
            and vac_raw.size(1) == int(drafter.block_size)
            and start_k > 0
        ):
            vac_raw = vac_raw[:, start_k:]

        eff_cap = min(max_spec_cap, prop_ids.size(1), k_out)
        born_calibrated = (not run_tree) and (
            getattr(drafter, "born_confidence_head", None) is not None
        )
        horizons = self.scheduler.select_horizon_tensor(
            confidences=prop_conf[:, :eff_cap],
            active_batch_size=b_bucket,
            purity=None
            if born_calibrated
            else (
                pur_raw[:, :eff_cap]
                if (pur_raw is not None and pur_raw.ndim == _RANK_2D)
                else pur_raw
            ),
            p_vac=None
            if born_calibrated
            else (
                vac_raw[:, :eff_cap]
                if (vac_raw is not None and vac_raw.ndim == _RANK_2D)
                else vac_raw
            ),
            temperature=0.0,
            temperatures_tensor=(temps_v if is_stochastic else None),
            valid_mask=(slot_v < pool.scratch_slot_base),
        ).clamp(min=1, max=eff_cap)

        out_v[:, 0] = horizons
        out_v[:, 1 : 1 + eff_cap] = prop_ids[:, :eff_cap]

    def _get_or_capture_decode_graph(
        self,
        b_bucket: int,
        s_bucket: int,
        is_stochastic: bool,
        run_tree: bool,
        max_spec_cap: int,
        compute_prev2_from_pool: bool = False,
    ) -> torch.cuda.CUDAGraph:
        """Lazily capture ``CUDAGraph`` on dedicated scratch slots (0 active drift)."""
        key = (b_bucket, s_bucket, is_stochastic, run_tree, compute_prev2_from_pool)
        cached = self._cg_graphs.get(key)
        if cached is not None:
            return cached

        pool = self._require_slot_pool()
        (
            cg_slot_idx,
            cg_prev_fc,
            cg_token_ids,
            cg_valid_lens,
            cg_prev2_ids,
            cg_temps,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._require_cg_buffers()

        scratch_base = pool.scratch_slot_base
        scratch_slots = torch.arange(
            scratch_base,
            scratch_base + b_bucket,
            dtype=torch.long,
            device=self.device,
        )
        cg_slot_idx[:b_bucket].copy_(scratch_slots)
        cg_prev_fc[:b_bucket, :s_bucket].zero_()
        cg_token_ids[:b_bucket, :s_bucket].zero_()
        cg_valid_lens[:b_bucket].fill_(1)
        cg_prev2_ids[:b_bucket].zero_()
        cg_temps[:b_bucket].fill_(0.6 if is_stochastic else 0.0)

        s_cap = torch.cuda.Stream(device=self.device)
        s_cap.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s_cap):
            for _ in range(2):
                self._run_decode_graph_body(
                    b_bucket,
                    s_bucket,
                    is_stochastic,
                    run_tree,
                    max_spec_cap,
                    compute_prev2_from_pool=compute_prev2_from_pool,
                )
        torch.cuda.current_stream(self.device).wait_stream(s_cap)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._cg_mempool, stream=s_cap):
            self._run_decode_graph_body(
                b_bucket,
                s_bucket,
                is_stochastic,
                run_tree,
                max_spec_cap,
                compute_prev2_from_pool=compute_prev2_from_pool,
            )
        torch.cuda.current_stream(self.device).wait_stream(s_cap)

        # Clean scratch pool rows touched during warmup/capture
        pool.q_pool[scratch_base : scratch_base + b_bucket].zero_()
        pool.p_pool[scratch_base : scratch_base + b_bucket].zero_()
        if pool.nu_pool is not None:
            pool.nu_pool[scratch_base : scratch_base + b_bucket].zero_()
        pool.last2_tokens_pool[scratch_base : scratch_base + b_bucket].zero_()

        self._cg_graphs[key] = graph
        return graph

    @torch.no_grad()
    def propose_tree_cudagraph(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor,
        *,
        tree_budget: int | None = None,
        branch_width: int | None = None,
        temperature: float = 0.0,
        prev_context_ids: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """CUDA-Graph-captured vectorized ``[B * W]`` Kraus POVM tree step."""
        drafter = self._require_drafter()
        budget = int(tree_budget or self.tree_budget)
        width = int(branch_width or self.branch_width)
        bsz, s_len, d_model = prev_fc_block.shape
        is_stoch = float(temperature) > 0.0

        if (
            not self._cg_enabled
            or bsz not in _CUDAGRAPH_BATCH_BUCKETS
            or (
                self.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            out_dict, out_sc = drafter.draft_kraus_tree_step(
                prev_fc_block,
                token_ids_block,
                valid_lens,
                state_cache,
                tree_budget=budget,
                branch_width=width,
                temperature=float(temperature),
                return_q_probs=False,
                prev_context_ids=prev_context_ids,
            )
            if not isinstance(out_sc, Tensor):
                raise TypeError("Expected Tensor state_cache from tree step.")
            return out_dict, out_sc

        t_key = (bsz, s_len, budget, width, is_stoch)
        entry = self._cg_tree_graphs.get(t_key)
        if entry is None:
            st_fc = torch.zeros(
                (bsz, s_len, d_model),
                dtype=prev_fc_block.dtype,
                device=self.device,
            )
            st_tok = torch.zeros((bsz, s_len), dtype=torch.long, device=self.device)
            st_vlen = torch.ones((bsz,), dtype=torch.long, device=self.device)
            st_prev2 = torch.zeros((bsz, 2), dtype=torch.long, device=self.device)
            st_sc = torch.zeros_like(state_cache)

            s_cap = torch.cuda.Stream(device=self.device)
            s_cap.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(s_cap):
                for _ in range(2):
                    drafter.draft_kraus_tree_step(
                        st_fc,
                        st_tok,
                        st_vlen,
                        st_sc,
                        tree_budget=budget,
                        branch_width=width,
                        temperature=float(temperature),
                        return_q_probs=False,
                        prev_context_ids=st_prev2,
                    )
            torch.cuda.current_stream(self.device).wait_stream(s_cap)

            t_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(t_graph, pool=self._cg_mempool, stream=s_cap):
                cap_dict, cap_sc = drafter.draft_kraus_tree_step(
                    st_fc,
                    st_tok,
                    st_vlen,
                    st_sc,
                    tree_budget=budget,
                    branch_width=width,
                    temperature=float(temperature),
                    return_q_probs=False,
                    prev_context_ids=st_prev2,
                )
            torch.cuda.current_stream(self.device).wait_stream(s_cap)
            if not isinstance(cap_sc, Tensor):
                raise TypeError("Expected Tensor state_cache from tree step.")
            bufs: dict[str, Tensor] = {
                "in_fc": st_fc,
                "in_tok": st_tok,
                "in_vlen": st_vlen,
                "in_prev2": st_prev2,
                "in_sc": st_sc,
                "out_sc": cap_sc,
            }
            for k_name, v_ten in cap_dict.items():
                bufs[f"tree::{k_name}"] = v_ten
            entry = (t_graph, bufs)
            self._cg_tree_graphs[t_key] = entry

        t_graph, bufs = entry
        bufs["in_fc"].copy_(prev_fc_block)
        bufs["in_tok"].copy_(token_ids_block)
        bufs["in_vlen"].copy_(valid_lens)
        if prev_context_ids is not None:
            bufs["in_prev2"].copy_(prev_context_ids)
        else:
            bufs["in_prev2"].zero_()
        bufs["in_sc"].copy_(state_cache)
        t_graph.replay()

        out_tree = {
            k_name[6:]: v_ten
            for k_name, v_ten in bufs.items()
            if k_name.startswith("tree::")
        }
        return out_tree, bufs["out_sc"]

    def _resolve_step_temperatures(
        self,
        input_batch: Any,
        req_ids: list[str],
        decode_batch_positions: list[int],
        sampling_metadata: Any,
    ) -> tuple[bool, list[float] | None, Tensor | None]:
        """Resolve per-request decode temperatures with 0 GPU->CPU stream syncs."""
        s_meta = sampling_metadata or self._current_sampling_metadata
        if s_meta is not None and bool(getattr(s_meta, "all_greedy", False)):
            return False, None, None

        greedy_reqs = getattr(input_batch, "greedy_reqs", None)
        random_reqs = getattr(input_batch, "random_reqs", None)
        if (
            greedy_reqs is not None
            and random_reqs is not None
            and len(random_reqs) == 0
        ):
            return False, None, None

        temp_cpu_arr = getattr(input_batch, "temperature_cpu", None)
        if temp_cpu_arr is not None:
            temps_list: list[float] = []
            any_stoch = False
            for b_pos in decode_batch_positions:
                rid = req_ids[b_pos]
                if greedy_reqs is not None and rid in greedy_reqs:
                    t_val = 0.0
                else:
                    t_val = float(temp_cpu_arr[b_pos])
                    if not math.isfinite(t_val) or t_val < _MIN_TEMP_EPS:
                        t_val = 0.0
                if t_val > 0.0:
                    any_stoch = True
                temps_list.append(t_val)
            return any_stoch, (temps_list if any_stoch else None), None

        temp_gpu = getattr(s_meta, "temperature", None) if s_meta is not None else None
        if isinstance(temp_gpu, Tensor):
            pos_gpu = torch.tensor(
                decode_batch_positions, dtype=torch.long, device=temp_gpu.device
            )
            gathered = temp_gpu.index_select(0, pos_gpu).to(
                device=self.device, dtype=torch.float32
            )
            gathered = torch.where(
                torch.isfinite(gathered) & (gathered >= _MIN_TEMP_EPS),
                gathered,
                torch.zeros_like(gathered),
            )
            return True, None, gathered

        return False, None, None

    @torch.no_grad()
    def propose_vllm_step(  # noqa: C901
        self,
        runner: Any,
        scheduler_output: Any,
        sampled_token_ids: Tensor | list[list[int]],
        aux_hidden_states: list[Tensor] | None,
        sampling_metadata: Any = None,
    ) -> list[list[int]]:
        """Execute one live ``vLLM V1`` ``GPUModelRunner`` speculative proposal step.

        Implements exact 3-case inductive token/hidden-state alignment (zero rejected
        draft state corruption across chunked prefill, vLLM prefix cache, step-0 decode,
        preemption, and speculative verification) plus zero-sync static-slot
        ``torch.cuda.CUDAGraph`` decode execution.
        """
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        if self.scheduler is None:
            raise RuntimeError("Amdahl scheduler is not initialized.")

        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        if finished:
            self.free_finished_requests(list(finished))

        input_batch = runner.input_batch
        req_ids: list[str] = list(input_batch.req_ids)
        if not req_ids or not aux_hidden_states:
            return [[] for _ in req_ids]

        active_set = set(req_ids)
        if pool.num_active_requests > 0:
            stale = [rid for rid in pool.active_request_ids() if rid not in active_set]
            if stale:
                self.preempt_requests(stale)

        num_scheduled_map: dict[str, int] = getattr(
            scheduler_output, "num_scheduled_tokens", {}
        )
        total_sched = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        )
        if total_sched <= 0 and num_scheduled_map:
            total_sched = sum(int(num_scheduled_map.get(r, 1)) for r in req_ids)
        sliced_aux: Tensor | list[Tensor]
        if total_sched > 0:
            sliced_aux = (
                [h[:total_sched] for h in aux_hidden_states]
                if isinstance(aux_hidden_states, (list, tuple))
                else aux_hidden_states[:total_sched]
            )
        else:
            sliced_aux = aux_hidden_states

        fc_flat = self.extract_fc_hidden(sliced_aux, already_projected=False)
        n_total_fc = int(fc_flat.size(0))
        pool_base = n_total_fc
        zero_fc_idx = pool_base + int(pool.last_fc_pool.size(0))
        sampled_rows = (
            sampled_token_ids.detach().cpu().tolist()
            if isinstance(sampled_token_ids, Tensor)
            else sampled_token_ids
        )
        num_tokens_no_spec = input_batch.num_tokens_no_spec
        token_ids_cpu = input_batch.token_ids_cpu

        s_max = int(drafter.block_size) + 1
        k_out = (
            int(drafter.block_size)
            if drafter.config.sample_from_anchor
            else max(1, int(drafter.block_size) - 1)
        )
        max_spec_cap = (
            min(k_out, self._max_vllm_spec_tokens)
            if self._max_vllm_spec_tokens is not None
            else k_out
        )

        prefill_queue: list[tuple[str, int, list[int], Tensor, bool]] = []
        # Decode batch staging lists (CPU ints only, 0 CUDA syncs)
        decode_positions: list[int] = []
        decode_slots: list[int] = []
        decode_toks: list[list[int]] = []
        decode_fc_src: list[list[int]] = []
        decode_prev2: list[tuple[int, int]] = []
        last_fc_update_slots: list[int] = []
        last_fc_update_src: list[int] = []

        cursor = 0
        for b_idx, rid in enumerate(req_ids):
            n_sched = int(num_scheduled_map.get(rid, 1))
            c_start = cursor
            cursor += n_sched
            raw_samp = sampled_rows[b_idx]
            sampled_b = [int(t) for t in raw_samp if int(t) >= 0]
            n_samp = len(sampled_b)
            total_seq = int(num_tokens_no_spec[b_idx])
            tok_row = token_ids_cpu[b_idx]
            row_cap = len(tok_row)

            if not pool.has_request(rid) and pool.is_preempted(rid):
                saved = pool.get_preempted_state(rid)
                target_prior = (total_seq - n_samp) if n_samp > 0 else total_seq
                if saved is None or saved.seq_len != target_prior or (
                    target_prior > 1 and n_sched == target_prior
                ):
                    pool.free_request(rid)
                else:
                    pool.get_or_allocate_slot(rid)

            # Case 1: Intermediate chunked prefill (no token sampled yet)
            if n_samp == 0:
                slot_idx, is_new = pool.get_or_allocate_slot(rid)
                c_len = min(n_sched, max(0, n_total_fc - c_start))
                if c_len > 0:
                    chunk_start = max(0, total_seq - c_len)
                    chunk_toks = [int(t) for t in tok_row[chunk_start:total_seq]]
                    prefill_queue.append(
                        (
                            rid,
                            slot_idx,
                            chunk_toks,
                            fc_flat[c_start : c_start + c_len],
                            bool(is_new and chunk_start == 0),
                        )
                    )
                continue

            slot_idx, is_new = pool.get_or_allocate_slot(rid)
            seq_committed = pool.get_seq_len(slot_idx)
            expected_prior = max(0, total_seq - n_samp)

            # Case 3: Steady-state speculative decode verification step
            if not is_new and seq_committed == expected_prior:
                avail_fc = max(0, n_total_fc - c_start)
                v_len = min(n_samp, avail_fc, s_max)
                if v_len <= 0:
                    continue
                commit_toks = sampled_b[-v_len:]
                fc_indices = [c_start + j for j in range(v_len)]
                last_fc_update_slots.append(slot_idx)
                last_fc_update_src.append(c_start + v_len - 1)

                anc_pos = max(0, min(total_seq - 1, row_cap - 1))
                p0 = int(tok_row[anc_pos - 2]) if anc_pos >= _PREV2_MIN_POS else 0
                p1 = int(tok_row[anc_pos - 1]) if anc_pos >= 1 else 0

                pool.set_slot_meta(slot_idx, seq_len=total_seq, has_last_fc=False)
                decode_positions.append(b_idx)
                decode_slots.append(slot_idx)
                decode_toks.append(commit_toks)
                decode_fc_src.append(fc_indices)
                decode_prev2.append((p0, p1))
                continue

            # Case 2: Final prefill / vLLM prefix-cache hit / recompute -> Step-0 decode
            prompt_len = expected_prior
            if seq_committed > prompt_len:
                pool.reset_slot(slot_idx)
                self._req_APC_parent_hash.pop(rid, None)
                seq_committed = 0
                is_new = True

            if prompt_len > 0 and seq_committed < prompt_len:
                avail_fc = max(0, n_total_fc - c_start)
                n_pf_fc = min(n_sched, avail_fc, prompt_len - seq_committed)
                if n_pf_fc > 0:
                    if is_new and n_pf_fc == prompt_len:
                        pf_toks = [int(t) for t in tok_row[:prompt_len]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                pf_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                True,
                            )
                        )
                    elif is_new and n_pf_fc < prompt_len:
                        prefix_skip = prompt_len - n_pf_fc
                        restored = 0
                        if self.apc_cache is not None:
                            prefix_toks = [int(t) for t in tok_row[:prefix_skip]]
                            restored, p_hash, _ = (
                                self.apc_cache.match_and_restore_prefix(
                                    prefix_toks,
                                    slot_pool=pool,
                                    slot_idx=slot_idx,
                                    require_uncomputed_tail=False,
                                )
                            )
                            if restored > 0:
                                self._req_APC_parent_hash[rid] = p_hash
                        if restored < prefix_skip:
                            pool.set_slot_meta(
                                slot_idx, seq_len=prefix_skip, has_last_fc=False
                            )
                        suf_toks = [int(t) for t in tok_row[prefix_skip:prompt_len]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                suf_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                False,
                            )
                        )
                    else:
                        cont_start = prompt_len - n_pf_fc
                        cont_toks = [int(t) for t in tok_row[cont_start:prompt_len]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                cont_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                False,
                            )
                        )

            # Immediately draft from newly sampled Step-0 bonus token(s)
            v_len0 = min(n_samp, s_max)
            commit_toks0 = sampled_b[-v_len0:]
            first_src = (pool_base + slot_idx) if prompt_len > 0 else zero_fc_idx
            fc_indices0 = [first_src] + [zero_fc_idx] * (v_len0 - 1)

            anc_pos = max(0, min(total_seq - 1, row_cap - 1))
            p0 = int(tok_row[anc_pos - 2]) if anc_pos >= _PREV2_MIN_POS else 0
            p1 = int(tok_row[anc_pos - 1]) if anc_pos >= 1 else 0

            decode_positions.append(b_idx)
            decode_slots.append(slot_idx)
            decode_toks.append(commit_toks0)
            decode_fc_src.append(fc_indices0)
            decode_prev2.append((p0, p1))

        # 1. Execute all queued prefill/continuation chunks first so last_fc_pool[slot]
        #    holds exact fc_{P-1} before Step-0 decode gathers it!
        if prefill_queue:
            self._prefill_requests_batched(prefill_queue)
        # Finalize seq_lens for any Case 2 requests whose prefill just ran
        for pos_i, slot_i in zip(decode_positions, decode_slots, strict=True):
            pool.set_slot_meta(
                slot_i, seq_len=int(num_tokens_no_spec[pos_i]), has_last_fc=False
            )

        proposals: list[list[int]] = [[] for _ in req_ids]
        decode_count = len(decode_slots)
        if decode_count == 0:
            return proposals

        # Update last_fc_pool for Case 3 slots in one vectorized GPU copy
        if last_fc_update_slots:
            upd_dst = torch.tensor(
                last_fc_update_slots, dtype=torch.long, device=self.device
            )
            upd_src = torch.tensor(
                last_fc_update_src, dtype=torch.long, device=self.device
            )
            pool.last_fc_pool.index_copy_(0, upd_dst, fc_flat.index_select(0, upd_src))

        if self._zero_fc_row is None:
            raise RuntimeError("Zero fc row is not initialized.")
        fc_sources = torch.cat([fc_flat, pool.last_fc_pool, self._zero_fc_row], dim=0)

        is_stoch, temps_cpu_list, temps_gpu_vec = self._resolve_step_temperatures(
            input_batch=input_batch,
            req_ids=req_ids,
            decode_batch_positions=decode_positions,
            sampling_metadata=sampling_metadata,
        )

        # Preserve vLLM V1 CUDAGraphMode.FULL (which requires uniform query_len ==
        # 1 + num_spec_tokens across all requests) whenever the batch fits inside the
        # verifier's captured FULL cudagraph token ceiling; above that ceiling, use
        # live PurityLoadAwareAmdahlScheduler dynamic horizon throttling.
        force_dyn = (
            os.environ.get("KVN_VLLM_FORCE_DYNAMIC_HORIZON", "0").strip() == "1"
        )
        runner_uniform_q = getattr(runner, "uniform_decode_query_len", None)
        comp_cfg = getattr(runner, "compilation_config", None)
        cg_cap_env = os.environ.get("KVN_VLLM_FULL_CG_MAX_TOKENS", "").strip()
        max_cg_tokens = (
            int(cg_cap_env)
            if cg_cap_env
            else int(getattr(comp_cfg, "max_cudagraph_capture_size", 512) or 512)
        )
        preserve_vllm_full_cg = (
            not force_dyn
            and runner_uniform_q is not None
            and int(runner_uniform_q) == 1 + max_spec_cap
            and decode_count == len(req_ids)
            and (decode_count * int(runner_uniform_q) <= max_cg_tokens)
        )

        # 2. Fast zero-sync CUDA-Graph decode path (B <= 128 on CUDA)
        if (
            self._cg_enabled
            and decode_count <= _MAX_CUDAGRAPH_BATCH
            and not (
                self.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            self._require_cg_buffers()

            b_bucket = self._select_cg_bucket(decode_count)
            max_vlen = max(len(t) for t in decode_toks)
            s_bucket = min(4, s_max) if max_vlen <= min(4, s_max) else s_max

            # Ensure graph is captured BEFORE writing live request slots into buffers!
            graph = self._get_or_capture_decode_graph(
                b_bucket=b_bucket,
                s_bucket=s_bucket,
                is_stochastic=is_stoch,
                run_tree=self.use_kraus_tree,
                max_spec_cap=max_spec_cap,
            )

            meta_cols_used = 4 + 2 * s_bucket
            scratch_base = pool.scratch_slot_base
            # Build flat Python table -> 1 tensor construction -> 1 pinned H2D DMA
            rows_flat: list[list[int]] = []
            for r_idx in range(decode_count):
                toks_r = decode_toks[r_idx]
                fcs_r = decode_fc_src[r_idx]
                v_len = len(toks_r)
                p0, p1 = decode_prev2[r_idx]
                pad_len = s_bucket - v_len
                rows_flat.append(
                    [decode_slots[r_idx], v_len, p0, p1]
                    + toks_r
                    + ([0] * pad_len)
                    + fcs_r
                    + ([zero_fc_idx] * pad_len)
                )
            pad_tok_fc = [0] * s_bucket + [zero_fc_idx] * s_bucket
            for r_idx in range(decode_count, b_bucket):
                rows_flat.append([scratch_base + r_idx, 1, 0, 0] + pad_tok_fc)

            meta_view_cpu = self._cg_meta_cpu[:b_bucket, :meta_cols_used]
            meta_view_cpu.copy_(torch.tensor(rows_flat, dtype=torch.long))
            meta_view_gpu = self._cg_meta_gpu[:b_bucket, :meta_cols_used]
            meta_view_gpu.copy_(meta_view_cpu, non_blocking=True)

            self._cg_slot_idx[:b_bucket].copy_(meta_view_gpu[:, 0])
            self._cg_valid_lens[:b_bucket].copy_(meta_view_gpu[:, 1])
            self._cg_prev2_ids[:b_bucket, :].copy_(meta_view_gpu[:, 2:4])
            self._cg_token_ids[:b_bucket, :s_bucket].copy_(
                meta_view_gpu[:, 4 : 4 + s_bucket]
            )
            fc_idx_flat = meta_view_gpu[:, 4 + s_bucket : meta_cols_used].reshape(-1)
            gathered_fc = fc_sources.index_select(0, fc_idx_flat).view(
                b_bucket, s_bucket, pool.d_model
            )
            self._cg_prev_fc[:b_bucket, :s_bucket, :].copy_(gathered_fc)

            if is_stoch:
                if temps_cpu_list is not None:
                    padded_t = temps_cpu_list + [0.0] * (b_bucket - decode_count)
                    t_cpu = self._cg_temps_cpu[:b_bucket]
                    t_cpu.copy_(torch.tensor(padded_t, dtype=torch.float32))
                    self._cg_temps[:b_bucket].copy_(t_cpu, non_blocking=True)
                elif temps_gpu_vec is not None:
                    self._cg_temps[:decode_count].copy_(temps_gpu_vec)
                    if b_bucket > decode_count:
                        self._cg_temps[decode_count:b_bucket].zero_()

            graph.replay()

            out_cpu_slice = self._cg_out_cpu[:decode_count, :]
            out_cpu_slice.copy_(
                self._cg_out_i64[:decode_count, :], non_blocking=False
            )
            out_rows = out_cpu_slice.tolist()
            for pos_i, row in zip(decode_positions, out_rows, strict=True):
                k_req = (
                    max_spec_cap
                    if preserve_vllm_full_cg
                    else max(1, min(int(row[0]), max_spec_cap))
                )
                proposals[pos_i] = row[1 : 1 + k_req]
            return proposals

        # 3. Vectorized eager path (CPU unit tests, >128 batch, or CUDAGraph disabled)
        max_vlen = max(len(t) for t in decode_toks)
        s_eager = s_max
        tok_pad = [r + [0] * (s_eager - len(r)) for r in decode_toks]
        fc_idx_pad = [r + [zero_fc_idx] * (s_eager - len(r)) for r in decode_fc_src]

        slot_gpu = torch.tensor(decode_slots, dtype=torch.long, device=self.device)
        vlen_gpu = torch.tensor(
            [len(r) for r in decode_toks], dtype=torch.long, device=self.device
        )
        prev2_gpu = torch.tensor(decode_prev2, dtype=torch.long, device=self.device)
        tok_gpu = torch.tensor(tok_pad, dtype=torch.long, device=self.device)
        fc_idx_gpu = torch.tensor(
            fc_idx_pad, dtype=torch.long, device=self.device
        ).reshape(-1)
        prev_fc_gpu = fc_sources.index_select(0, fc_idx_gpu).view(
            decode_count, s_eager, pool.d_model
        )

        state_cache = pool.gather_drafter_state_cache(
            slot_gpu, include_nu=self._want_nu_state
        ).to(self._state_cache_dtype)

        temps_eager: Tensor | None = None
        if is_stoch:
            if temps_cpu_list is not None:
                temps_eager = torch.tensor(
                    temps_cpu_list, dtype=torch.float32, device=self.device
                )
            elif temps_gpu_vec is not None:
                temps_eager = temps_gpu_vec

        start_k = 0 if drafter.config.sample_from_anchor else 1
        if self.use_kraus_tree and hasattr(drafter, "draft_kraus_tree_step"):
            tree_kw: dict[str, Any] = {
                "tree_budget": self.tree_budget,
                "branch_width": self.branch_width,
                "temperature": (0.6 if is_stoch else 0.0),
            }
            tree_params = inspect.signature(drafter.draft_kraus_tree_step).parameters
            if "return_q_probs" in tree_params:
                tree_kw["return_q_probs"] = False
            if "prev_context_ids" in tree_params:
                tree_kw["prev_context_ids"] = prev2_gpu
            tree_dict, updated_cache = drafter.draft_kraus_tree_step(
                prev_fc_gpu, tok_gpu, vlen_gpu, state_cache, **tree_kw
            )
            prop_ids, prop_conf = self._extract_kraus_tree_spine_static(
                tree_dict, k_out
            )
            details = {
                k_d: tree_dict[k_d]
                for k_d in ("purity", "p_vac")
                if k_d in tree_dict
            }
        elif self._supports_fast_v3_decode:
            draft_ids, updated_cache, details = drafter.advance_and_draft_step(
                prev_fc_gpu,
                tok_gpu,
                vlen_gpu,
                state_cache,
                temperature=(1.0 if is_stoch else 0.0),
                return_details=True,
                prev_context_ids=prev2_gpu,
                temperatures_tensor=temps_eager,
                include_draft_probs=False,
            )
            prop_ids = draft_ids[:, start_k:]
            prop_conf = details["confidences"][:, start_k:].to(torch.float32)
        else:
            draft_ids, updated_cache, details = self._invoke_advance_and_draft(
                prev_fc_gpu, tok_gpu, vlen_gpu, state_cache, temperature=0.0
            )
            prop_ids = draft_ids[:, start_k:]
            prop_conf = details["confidences"][:, start_k:].to(torch.float32)

        pool.scatter_drafter_state_cache(slot_gpu, updated_cache)
        pur_detail, vac_detail, _ = self._extract_aligned_quantum_details(
            details, int(drafter.block_size), start_k
        )
        born_calibrated = (not self.use_kraus_tree) and (
            getattr(drafter, "born_confidence_head", None) is not None
        )
        eff_cap = min(max_spec_cap, prop_ids.size(1))
        eager_eff_batch = (
            self._select_cg_bucket(decode_count)
            if decode_count <= _MAX_CUDAGRAPH_BATCH
            else len(req_ids)
        )
        horizons_t = self.scheduler.select_horizon_tensor(
            confidences=prop_conf[:, :eff_cap],
            active_batch_size=eager_eff_batch,
            purity=None
            if born_calibrated
            else (
                pur_detail[:, :eff_cap]
                if (pur_detail is not None and pur_detail.ndim == _RANK_2D)
                else pur_detail
            ),
            p_vac=None
            if born_calibrated
            else (
                vac_detail[:, :eff_cap]
                if (vac_detail is not None and vac_detail.ndim == _RANK_2D)
                else vac_detail
            ),
            temperature=0.0,
            temperatures_tensor=temps_eager,
        ).clamp(min=1, max=eff_cap)

        horizons_list = horizons_t.detach().cpu().tolist()
        prop_rows = prop_ids[:, :eff_cap].detach().cpu().tolist()
        for pos_i, row, k_req in zip(
            decode_positions, prop_rows, horizons_list, strict=True
        ):
            k_final = eff_cap if preserve_vllm_full_cg else int(k_req)
            proposals[pos_i] = row[:k_final]
        return proposals

    @torch.no_grad()
    def prepare_async_sampled_tokens(
        self,
        runner: Any,
        sampled_token_ids_gpu: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute ``(next_token_ids, valid_sampled_tokens_count)`` on GPU (0 syncs).

        Satisfies ``GPUModelRunner._copy_valid_sampled_token_count`` for
        ``use_async_scheduling=True`` speculative execution without blocking the CPU.
        """
        input_batch = runner.input_batch
        num_reqs, num_cols = sampled_token_ids_gpu.shape
        dev = sampled_token_ids_gpu.device
        vocab_size = int(getattr(input_batch, "vocab_size", 1 << 30))

        discard_mask_obj = getattr(runner, "discard_request_mask", None)
        discard_gpu = getattr(discard_mask_obj, "gpu", None)
        discard_np = getattr(discard_mask_obj, "np", None)

        is_valid = (sampled_token_ids_gpu >= 0) & (sampled_token_ids_gpu < vocab_size)
        if isinstance(discard_gpu, Tensor):
            keep_row = ~discard_gpu[:num_reqs].to(device=dev, dtype=torch.bool)
            is_valid = is_valid & keep_row.unsqueeze(1)

        valid_counts = is_valid.sum(dim=1, dtype=torch.int32)
        col_offs = torch.arange(num_cols, device=dev, dtype=torch.long).unsqueeze(0)
        last_col = torch.where(is_valid, col_offs, -1).max(dim=1).values.clamp_min(0)
        next_tok = (
            sampled_token_ids_gpu.gather(1, last_col.unsqueeze(1))
            .squeeze(1)
            .to(torch.int32)
        )

        any_discarded = (
            bool(discard_np[:num_reqs].any())
            if (discard_np is not None and num_reqs > 0)
            else False
        )
        if any_discarded:
            requests_map = getattr(runner, "requests", None) or {}
            req_ids = list(input_batch.req_ids)
            no_spec = getattr(input_batch, "num_tokens_no_spec", None)
            backups: list[int] = []
            for idx_r in range(num_reqs):
                rid = req_ids[idx_r] if idx_r < len(req_ids) else ""
                req_st = requests_map.get(rid)
                pos_prev = (
                    max(0, int(no_spec[idx_r]) - 1) if no_spec is not None else 0
                )
                if req_st is not None and hasattr(req_st, "get_token_id"):
                    backups.append(int(req_st.get_token_id(pos_prev)))
                else:
                    backups.append(0)
            backup_gpu = torch.tensor(backups, dtype=torch.int32, device=dev)
            next_tok = torch.where(valid_counts > 0, next_tok, backup_gpu)

        return next_tok, valid_counts

    @torch.no_grad()
    def propose_vllm_async_gpu_step(  # noqa: C901
        self,
        runner: Any,
        scheduler_output: Any,
        sampled_token_ids_gpu: Tensor,
        next_token_ids_gpu: Tensor,
        valid_counts_gpu: Tensor,
        aux_hidden_states: list[Tensor] | None,
        sampling_metadata: Any = None,
    ) -> Tensor:
        """Zero-CPU-sync GPU proposal step for ``vLLM V1`` ``async_scheduling=True``.

        Consumes ``sampler_output.sampled_token_ids`` and ``valid_counts_gpu`` directly
        on GPU and returns ``[num_reqs, k_sched]`` ``int64`` on ``self.device`` so
        ``GPUModelRunner`` overlaps EngineCore scheduling/IPC with GPU execution.
        """
        del next_token_ids_gpu
        drafter = self._require_drafter()
        pool = self._require_slot_pool()
        if self.scheduler is None:
            raise RuntimeError("Amdahl scheduler is not initialized.")

        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        if finished:
            self.free_finished_requests(list(finished))

        input_batch = runner.input_batch
        req_ids: list[str] = list(input_batch.req_ids)
        num_reqs = len(req_ids)
        k_out = (
            int(drafter.block_size)
            if drafter.config.sample_from_anchor
            else max(1, int(drafter.block_size) - 1)
        )
        max_spec_cap = (
            min(k_out, self._max_vllm_spec_tokens)
            if self._max_vllm_spec_tokens is not None
            else k_out
        )
        sched_k_raw = getattr(scheduler_output, "num_spec_tokens_to_schedule", None)
        k_sched = (
            min(max_spec_cap, int(sched_k_raw))
            if sched_k_raw is not None
            else max_spec_cap
        )
        if num_reqs == 0 or not aux_hidden_states or k_sched < 0:
            return torch.zeros(
                (num_reqs, max(0, k_sched)), dtype=torch.long, device=self.device
            )

        active_set = set(req_ids)
        if pool.num_active_requests > 0:
            stale = [rid for rid in pool.active_request_ids() if rid not in active_set]
            if stale:
                self.preempt_requests(stale)

        num_scheduled_map: dict[str, int] = getattr(
            scheduler_output, "num_scheduled_tokens", {}
        )
        spec_sched_map: dict[str, Any] = (
            getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        )
        total_sched = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        )
        if total_sched <= 0 and num_scheduled_map:
            total_sched = sum(int(num_scheduled_map.get(r, 1)) for r in req_ids)

        sliced_aux: Tensor | list[Tensor]
        if total_sched > 0:
            sliced_aux = (
                [h[:total_sched] for h in aux_hidden_states]
                if isinstance(aux_hidden_states, (list, tuple))
                else aux_hidden_states[:total_sched]
            )
        else:
            sliced_aux = aux_hidden_states

        fc_flat = self.extract_fc_hidden(sliced_aux, already_projected=False)
        n_total_fc = int(fc_flat.size(0))
        pool_base = n_total_fc
        zero_fc_idx = pool_base + int(pool.last_fc_pool.size(0))

        token_ids_cpu = input_batch.token_ids_cpu
        num_prompt_arr = getattr(input_batch, "num_prompt_tokens", None)
        num_comp_arr = getattr(input_batch, "num_computed_tokens_cpu", None)
        discard_mask_obj = getattr(runner, "discard_request_mask", None)
        discard_np = getattr(discard_mask_obj, "np", None)

        s_max = int(drafter.block_size) + 1
        prefill_queue: list[tuple[str, int, list[int], Tensor, bool]] = []
        case2_last2_updates: list[tuple[int, int, int, int]] = []

        decode_positions: list[int] = []
        decode_slots: list[int] = []
        decode_is_case3: list[bool] = []
        decode_c_starts: list[int] = []
        decode_n_scheds: list[int] = []
        decode_case2_first_src: list[int] = []
        max_possible_vlen = 1

        cursor = 0
        for b_idx, rid in enumerate(req_ids):
            n_sched = int(num_scheduled_map.get(rid, 1))
            c_start = cursor
            cursor += n_sched
            is_discarded = (
                bool(discard_np[b_idx]) if discard_np is not None else False
            )
            n_spec_r = len(spec_sched_map.get(rid, ()))
            tok_row = token_ids_cpu[b_idx]
            row_cap = len(tok_row)

            num_comp = (
                int(num_comp_arr[b_idx])
                if num_comp_arr is not None
                else (
                    pool.get_seq_len(pool.get_slot(rid))
                    if pool.has_request(rid)
                    else 0
                )
            )
            prompt_len = (
                int(num_prompt_arr[b_idx])
                if num_prompt_arr is not None
                else (num_comp + n_sched)
            )

            if not pool.has_request(rid) and pool.is_preempted(rid):
                saved = pool.get_preempted_state(rid)
                recomp_prior = num_comp + n_sched
                if (
                    saved is not None
                    and not is_discarded
                    and n_sched == 1
                    and recomp_prior == saved.seq_len
                    and saved.seq_len > 1
                    and saved.prompt_committed
                ):
                    pool.get_or_allocate_slot(rid)
                else:
                    pool.free_request(rid)

            # Case 1: Intermediate chunked prefill (no token sampled on this step)
            if is_discarded:
                slot_idx, is_new = pool.get_or_allocate_slot(rid)
                c_len = min(n_sched, max(0, n_total_fc - c_start))
                chunk_end = min(row_cap, num_comp + c_len)
                chunk_start = max(0, chunk_end - c_len)
                if is_new and chunk_start > 0:
                    restored_c1 = 0
                    if self.apc_cache is not None:
                        p_toks = [int(t) for t in tok_row[:chunk_start]]
                        restored_c1, p_hash, _ = (
                            self.apc_cache.match_and_restore_prefix(
                                p_toks,
                                slot_pool=pool,
                                slot_idx=slot_idx,
                                require_uncomputed_tail=False,
                            )
                        )
                        if restored_c1 > 0:
                            self._req_APC_parent_hash[rid] = p_hash
                    if restored_c1 < chunk_start:
                        pool.set_slot_meta(
                            slot_idx, seq_len=chunk_start, has_last_fc=False
                        )
                        p0_s = (
                            int(tok_row[chunk_start - 2])
                            if chunk_start >= _PREV2_MIN_POS
                            else 0
                        )
                        p1_s = int(tok_row[chunk_start - 1]) if chunk_start >= 1 else 0
                        pool.advance_slot_last2_tokens(slot_idx, [p0_s, p1_s])
                if c_len > 0 and chunk_end > chunk_start:
                    chunk_toks = [int(t) for t in tok_row[chunk_start:chunk_end]]
                    prefill_queue.append(
                        (
                            rid,
                            slot_idx,
                            chunk_toks,
                            fc_flat[c_start : c_start + len(chunk_toks)],
                            bool(is_new and chunk_start == 0),
                        )
                    )
                continue

            slot_idx, is_new = pool.get_or_allocate_slot(rid)

            # Case 3: Steady-state speculative decode verification step
            if (
                not is_new
                and pool.is_prompt_committed(slot_idx)
                and (num_comp >= prompt_len or prompt_len == 0)
            ):
                avail_fc = max(0, n_total_fc - c_start)
                if avail_fc <= 0:
                    continue
                upper_v = min(n_sched, avail_fc, s_max)
                max_possible_vlen = max(max_possible_vlen, upper_v)
                next_seq = (num_comp + 2) if n_sched == 1 else -1
                pool.set_slot_meta(slot_idx, seq_len=next_seq, has_last_fc=False)
                decode_positions.append(b_idx)
                decode_slots.append(slot_idx)
                decode_is_case3.append(True)
                decode_c_starts.append(c_start)
                decode_n_scheds.append(min(n_sched, avail_fc))
                decode_case2_first_src.append(zero_fc_idx)
                continue

            # Case 2: Final prefill / prefix-cache hit / RECOMPUTE -> Step-0 decode
            eff_p = (
                max(prompt_len, num_comp + n_sched)
                if n_spec_r == 0
                else prompt_len
            )
            eff_p = min(eff_p, row_cap)
            seq_committed = pool.get_seq_len(slot_idx)
            if seq_committed > eff_p:
                pool.reset_slot(slot_idx)
                self._req_APC_parent_hash.pop(rid, None)
                seq_committed = 0
                is_new = True

            if eff_p > 0 and seq_committed < eff_p:
                avail_fc = max(0, n_total_fc - c_start)
                n_pf_fc = min(n_sched, avail_fc, eff_p - seq_committed)
                if n_pf_fc > 0:
                    if is_new and n_pf_fc == eff_p:
                        pf_toks = [int(t) for t in tok_row[:eff_p]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                pf_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                True,
                            )
                        )
                    elif is_new and n_pf_fc < eff_p:
                        prefix_skip = eff_p - n_pf_fc
                        restored = 0
                        if self.apc_cache is not None:
                            prefix_toks = [int(t) for t in tok_row[:prefix_skip]]
                            restored, p_hash, _ = (
                                self.apc_cache.match_and_restore_prefix(
                                    prefix_toks,
                                    slot_pool=pool,
                                    slot_idx=slot_idx,
                                    require_uncomputed_tail=False,
                                )
                            )
                            if restored > 0:
                                self._req_APC_parent_hash[rid] = p_hash
                        if restored < prefix_skip:
                            pool.set_slot_meta(
                                slot_idx, seq_len=prefix_skip, has_last_fc=False
                            )
                        suf_toks = [int(t) for t in tok_row[prefix_skip:eff_p]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                suf_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                False,
                            )
                        )
                    else:
                        cont_start = eff_p - n_pf_fc
                        cont_toks = [int(t) for t in tok_row[cont_start:eff_p]]
                        prefill_queue.append(
                            (
                                rid,
                                slot_idx,
                                cont_toks,
                                fc_flat[c_start : c_start + n_pf_fc],
                                False,
                            )
                        )

            p0_pf = (
                max(0, int(tok_row[eff_p - 2])) if eff_p >= _PREV2_MIN_POS else 0
            )
            p1_pf = max(0, int(tok_row[eff_p - 1])) if eff_p >= 1 else 0
            case2_last2_updates.append((slot_idx, p0_pf, p1_pf, eff_p + 1))

            upper_v0 = min(max(1, 1 + n_spec_r), s_max)
            max_possible_vlen = max(max_possible_vlen, upper_v0)
            first_src = (pool_base + slot_idx) if eff_p > 0 else zero_fc_idx
            decode_positions.append(b_idx)
            decode_slots.append(slot_idx)
            decode_is_case3.append(False)
            decode_c_starts.append(0)
            decode_n_scheds.append(1)
            decode_case2_first_src.append(first_src)

        if prefill_queue:
            self._prefill_requests_batched(prefill_queue)

        if case2_last2_updates:
            c2_slots = torch.tensor(
                [u[0] for u in case2_last2_updates],
                dtype=torch.long,
                device=self.device,
            )
            c2_pairs = torch.tensor(
                [[u[1], u[2]] for u in case2_last2_updates],
                dtype=torch.int64,
                device=self.device,
            )
            pool.last2_tokens_pool.index_copy_(0, c2_slots, c2_pairs)
            for slot_i, _, _, post_len in case2_last2_updates:
                pool.set_slot_meta(
                    slot_i,
                    seq_len=post_len,
                    has_last_fc=False,
                )
                pool.set_prompt_committed(slot_i, True)

        decode_count = len(decode_slots)
        if decode_count == 0 or k_sched == 0:
            return torch.zeros(
                (num_reqs, max(0, k_sched)), dtype=torch.long, device=self.device
            )

        if self._zero_fc_row is None:
            raise RuntimeError("Zero fc row is not initialized.")
        fc_sources = torch.cat([fc_flat, pool.last_fc_pool, self._zero_fc_row], dim=0)

        is_stoch, temps_cpu_list, temps_gpu_vec = self._resolve_step_temperatures(
            input_batch=input_batch,
            req_ids=req_ids,
            decode_batch_positions=decode_positions,
            sampling_metadata=sampling_metadata,
        )
        decode_is_identity = (
            decode_count == num_reqs
            and decode_positions == list(range(num_reqs))
        )

        # Fast zero-sync CUDA-Graph GPU decode path
        if (
            self._cg_enabled
            and decode_count <= _MAX_CUDAGRAPH_BATCH
            and not (
                self.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            (
                cg_slot_idx,
                cg_prev_fc,
                cg_token_ids,
                cg_valid_lens,
                _,
                cg_temps,
                _,
                cg_out_i64,
                _,
                cg_meta_cpu,
                cg_meta_gpu,
                cg_temps_cpu,
            ) = self._require_cg_buffers()

            b_bucket = self._select_cg_bucket(decode_count)
            s_bucket = min(4, s_max) if max_possible_vlen <= min(4, s_max) else s_max

            graph = self._get_or_capture_decode_graph(
                b_bucket=b_bucket,
                s_bucket=s_bucket,
                is_stochastic=is_stoch,
                run_tree=self.use_kraus_tree,
                max_spec_cap=max_spec_cap,
                compute_prev2_from_pool=True,
            )

            meta_cols_used = 4 + s_bucket
            scratch_base = pool.scratch_slot_base
            rows_async: list[list[int]] = []
            case3_local_rows: list[int] = []
            for r_idx in range(decode_count):
                slot_r = decode_slots[r_idx]
                pos_r = decode_positions[r_idx]
                if decode_is_case3[r_idx]:
                    case3_local_rows.append(r_idx)
                    cs_r = decode_c_starts[r_idx]
                    ns_r = decode_n_scheds[r_idx]
                    fcs_r = [
                        (cs_r + j) if j < ns_r else zero_fc_idx
                        for j in range(s_bucket)
                    ]
                    rows_async.append([slot_r, pos_r, 1, cs_r, *fcs_r])
                else:
                    f0 = decode_case2_first_src[r_idx]
                    fcs_r = [f0] + [zero_fc_idx] * (s_bucket - 1)
                    rows_async.append([slot_r, pos_r, 0, 0, *fcs_r])

            pad_fc_cols = [zero_fc_idx] * s_bucket
            for r_idx in range(decode_count, b_bucket):
                rows_async.append([scratch_base + r_idx, 0, 0, 0, *pad_fc_cols])

            meta_view_cpu = cg_meta_cpu[:b_bucket, :meta_cols_used]
            meta_view_cpu.copy_(torch.tensor(rows_async, dtype=torch.long))
            meta_view_gpu = cg_meta_gpu[:b_bucket, :meta_cols_used]
            meta_view_gpu.copy_(meta_view_cpu, non_blocking=True)

            slot_col = meta_view_gpu[:, 0]
            pos_col = meta_view_gpu[:decode_count, 1]
            cg_slot_idx[:b_bucket].copy_(slot_col)

            vlen_dec = (
                valid_counts_gpu.index_select(0, pos_col)
                .to(torch.long)
                .clamp(1, s_bucket)
            )
            cg_valid_lens[:decode_count].copy_(vlen_dec)
            if b_bucket > decode_count:
                cg_valid_lens[decode_count:b_bucket].fill_(1)

            samp_dec = sampled_token_ids_gpu.index_select(0, pos_col)
            n_copy_cols = min(s_bucket, int(samp_dec.size(1)))
            cg_token_ids[:b_bucket, :s_bucket].zero_()
            cg_token_ids[:decode_count, :n_copy_cols].copy_(
                samp_dec[:, :n_copy_cols].clamp_min(0).to(torch.long)
            )

            fc_idx_flat = meta_view_gpu[:, 4:meta_cols_used].reshape(-1)
            gathered_fc = fc_sources.index_select(0, fc_idx_flat).view(
                b_bucket, s_bucket, pool.d_model
            )
            cg_prev_fc[:b_bucket, :s_bucket, :].copy_(gathered_fc)

            if case3_local_rows and n_total_fc > 0:
                if len(case3_local_rows) == decode_count:
                    cstart_c3 = meta_view_gpu[:decode_count, 3]
                    src_last = (cstart_c3 + vlen_dec - 1).clamp(0, n_total_fc - 1)
                    pool.last_fc_pool.index_copy_(
                        0, slot_col[:decode_count], fc_flat.index_select(0, src_last)
                    )
                else:
                    c3_idx = torch.tensor(
                        case3_local_rows, dtype=torch.long, device=self.device
                    )
                    cstart_c3 = meta_view_gpu[:decode_count, 3].index_select(0, c3_idx)
                    vlen_c3 = vlen_dec.index_select(0, c3_idx)
                    slots_c3 = slot_col[:decode_count].index_select(0, c3_idx)
                    src_last = (cstart_c3 + vlen_c3 - 1).clamp(0, n_total_fc - 1)
                    pool.last_fc_pool.index_copy_(
                        0, slots_c3, fc_flat.index_select(0, src_last)
                    )

            if is_stoch:
                if temps_cpu_list is not None:
                    padded_t = temps_cpu_list + [0.0] * (b_bucket - decode_count)
                    t_cpu = cg_temps_cpu[:b_bucket]
                    t_cpu.copy_(torch.tensor(padded_t, dtype=torch.float32))
                    cg_temps[:b_bucket].copy_(t_cpu, non_blocking=True)
                elif temps_gpu_vec is not None:
                    cg_temps[:decode_count].copy_(temps_gpu_vec)
                    if b_bucket > decode_count:
                        cg_temps[decode_count:b_bucket].zero_()

            graph.replay()

            draft_slice = cg_out_i64[:decode_count, 1 : 1 + k_sched]
            if decode_is_identity:
                return draft_slice.clone()
            out_full = torch.zeros(
                (num_reqs, k_sched), dtype=torch.long, device=self.device
            )
            out_full.index_copy_(0, pos_col, draft_slice)
            return out_full

        # Vectorized eager async fallback (CPU tests, >128 batch, or CUDAGraph disabled)
        s_eager = s_max
        pos_gpu = torch.tensor(decode_positions, dtype=torch.long, device=self.device)
        slot_gpu = torch.tensor(decode_slots, dtype=torch.long, device=self.device)
        vlen_gpu = (
            valid_counts_gpu.index_select(0, pos_gpu)
            .to(torch.long)
            .clamp(1, s_eager)
        )
        samp_dec_e = sampled_token_ids_gpu.index_select(0, pos_gpu)
        tok_gpu = torch.zeros(
            (decode_count, s_eager), dtype=torch.long, device=self.device
        )
        n_cols_e = min(s_eager, int(samp_dec_e.size(1)))
        tok_gpu[:, :n_cols_e] = samp_dec_e[:, :n_cols_e].clamp_min(0).to(torch.long)

        fc_rows_e: list[list[int]] = []
        c3_rows_e: list[int] = []
        for r_idx in range(decode_count):
            if decode_is_case3[r_idx]:
                c3_rows_e.append(r_idx)
                cs_r = decode_c_starts[r_idx]
                ns_r = decode_n_scheds[r_idx]
                fc_rows_e.append(
                    [(cs_r + j) if j < ns_r else zero_fc_idx for j in range(s_eager)]
                )
            else:
                f0 = decode_case2_first_src[r_idx]
                fc_rows_e.append([f0] + [zero_fc_idx] * (s_eager - 1))

        fc_idx_gpu = torch.tensor(
            fc_rows_e, dtype=torch.long, device=self.device
        ).reshape(-1)
        prev_fc_gpu = fc_sources.index_select(0, fc_idx_gpu).view(
            decode_count, s_eager, pool.d_model
        )
        if c3_rows_e and n_total_fc > 0:
            c3_t = torch.tensor(c3_rows_e, dtype=torch.long, device=self.device)
            cs_t = torch.tensor(
                [decode_c_starts[i] for i in c3_rows_e],
                dtype=torch.long,
                device=self.device,
            )
            src_last_e = (cs_t + vlen_gpu.index_select(0, c3_t) - 1).clamp(
                0, n_total_fc - 1
            )
            pool.last_fc_pool.index_copy_(
                0, slot_gpu.index_select(0, c3_t), fc_flat.index_select(0, src_last_e)
            )

        pool_last2_e = pool.last2_tokens_pool.index_select(0, slot_gpu)
        w_tok_e = torch.cat([pool_last2_e, tok_gpu], dim=1)
        p0_e = w_tok_e.gather(1, (vlen_gpu - 1).unsqueeze(1))
        p1_e = w_tok_e.gather(1, vlen_gpu.unsqueeze(1))
        prev2_gpu = torch.cat([p0_e, p1_e], dim=1)
        last_e = w_tok_e.gather(1, (vlen_gpu + 1).unsqueeze(1))
        pool.last2_tokens_pool.index_copy_(
            0, slot_gpu, torch.cat([p1_e, last_e], dim=1)
        )

        state_cache = pool.gather_drafter_state_cache(
            slot_gpu, include_nu=self._want_nu_state
        ).to(self._state_cache_dtype)
        temps_eager: Tensor | None = None
        if is_stoch:
            if temps_cpu_list is not None:
                temps_eager = torch.tensor(
                    temps_cpu_list, dtype=torch.float32, device=self.device
                )
            elif temps_gpu_vec is not None:
                temps_eager = temps_gpu_vec

        start_k = 0 if drafter.config.sample_from_anchor else 1
        if self.use_kraus_tree and hasattr(drafter, "draft_kraus_tree_step"):
            tree_kw: dict[str, Any] = {
                "tree_budget": self.tree_budget,
                "branch_width": self.branch_width,
                "temperature": (0.6 if is_stoch else 0.0),
            }
            tree_params = inspect.signature(drafter.draft_kraus_tree_step).parameters
            if "return_q_probs" in tree_params:
                tree_kw["return_q_probs"] = False
            if "prev_context_ids" in tree_params:
                tree_kw["prev_context_ids"] = prev2_gpu
            tree_dict, updated_cache = drafter.draft_kraus_tree_step(
                prev_fc_gpu, tok_gpu, vlen_gpu, state_cache, **tree_kw
            )
            prop_ids, _ = self._extract_kraus_tree_spine_static(tree_dict, k_out)
        elif self._supports_fast_v3_decode:
            draft_ids, updated_cache, _ = drafter.advance_and_draft_step(
                prev_fc_gpu,
                tok_gpu,
                vlen_gpu,
                state_cache,
                temperature=(1.0 if is_stoch else 0.0),
                return_details=True,
                prev_context_ids=prev2_gpu,
                temperatures_tensor=temps_eager,
                include_draft_probs=False,
            )
            prop_ids = draft_ids[:, start_k:]
        else:
            draft_ids, updated_cache, _ = self._invoke_advance_and_draft(
                prev_fc_gpu, tok_gpu, vlen_gpu, state_cache, temperature=0.0
            )
            prop_ids = draft_ids[:, start_k:]

        pool.scatter_drafter_state_cache(slot_gpu, updated_cache)
        draft_eager = prop_ids[:, :k_sched].to(torch.long)
        if decode_is_identity:
            return draft_eager.contiguous()
        out_full_e = torch.zeros(
            (num_reqs, k_sched), dtype=torch.long, device=self.device
        )
        out_full_e.index_copy_(0, pos_gpu, draft_eager)
        return out_full_e

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
