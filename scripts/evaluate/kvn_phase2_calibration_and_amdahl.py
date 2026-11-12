#!/usr/bin/env python3
"""Confidence Calibration & End-to-End Amdahl Speculative Evaluator.

Implements scientifically fair, lossless evaluation for ``KvN-DSpark`` (v2 & v3),
``hybrid_1swa_2kvn``, and ``swa`` (DSpark SWA-128 / SWA-512):

1. ``compute_ece`` & ``compute_binary_auroc``: 15-bin ECE and tie-aware rank AUROC.
2. ``compute_dynamic_stopping_pareto``: Sweeps ``gamma_conf`` over **cumulative
   prefix survival** ``S_k = prod_{j <= k} c_j >= gamma_conf``
   (``deepseekv4.1.pdf`` Sec. 2.4.3).
3. ``evaluate_checkpoint_calibration``: Computes ground-truth greedy targets from
   the **full target verifier vocabulary** (e.g. ``151,936`` tokens, never the
   sliced 32k shortlist!), masks coincidental post-rejection matches via
   ``prefix_alive`` conditioning, and reports ``oov_target_rate_pct`` +
   ``prefix_survival_ece_15bin``.
4. ``evaluate_checkpoint_e2e`` (``benchmark-e2e``):
   - Full ``O(P log P)`` parallel prompt prefill emitting step-0 token ``y_P``
     from verifier prefill + ``drafter.prefill_kvn_state``;
   - Clean ``eos_token_id`` stopping on both baseline and speculative paths;
   - True batched incremental ``DynamicCache`` SDPA inference for ``swa`` and
     persistent recurrent + rolling SWA state for ``hybrid_1swa_2kvn`` (zero
     serial ``B=1`` loops, zero recurrent amnesia);
   - ``>= 5`` CUDA warmup iterations and multi-step ``torch.cuda.Event`` timing;
   - Optional ``--use-kraus-tree`` lossless multi-branch ``SpecInfer`` verification
     at ``T = 0.0`` (bit-exact greedy match) and ``T > 0`` (exact target sampling).
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as fn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.train.data import ArrowDataset
from speculators.vllm_integration.kvn_proposer import (
    build_tree_attention_mask,
    compact_tree_kv_cache,
    specinfer_verify_tree_step,
)

try:
    import datasets
except ImportError:
    datasets = None  # type: ignore[assignment]

try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None  # type: ignore[assignment]

_RANK_2D = 2
_RANK_4D = 4
_MIN_PROMPT_TOKENS = 2
_NU_MIN_COMPONENTS = 3
_LAYER_OUT_V3_LEN = 5
_PROB_EPS = 1e-12

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8)

E2E_BENCHMARK_SPECS: dict[str, tuple[str, str | None, str, str]] = {
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", None, "train", "prompt"),
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test", "prompt"),
}


def compute_ece(
    probs: Tensor,
    labels: Tensor,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) over ``num_bins`` equal-width bins."""
    p = probs.detach().to(torch.float64).reshape(-1).clamp(0.0, 1.0)
    y = labels.detach().to(torch.float64).reshape(-1)
    if p.numel() == 0:
        return 0.0

    boundaries = torch.linspace(
        0.0, 1.0, num_bins + 1, dtype=torch.float64, device=p.device
    )
    ece = torch.zeros((), dtype=torch.float64, device=p.device)
    n_total = float(p.numel())

    for idx in range(num_bins):
        low = boundaries[idx]
        high = boundaries[idx + 1]
        in_bin = (p >= low) & (p <= high) if idx == 0 else (p > low) & (p <= high)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        avg_conf = p[in_bin].mean()
        avg_acc = y[in_bin].mean()
        ece = ece + (count / n_total) * torch.abs(avg_acc - avg_conf)

    return float(ece.item())


def compute_binary_auroc(
    scores: Tensor,
    labels: Tensor,
) -> float:
    """Exact rank-based AUROC (handles ties properly) without requiring scikit-learn."""
    s = scores.detach().to(torch.float64).reshape(-1).cpu()
    y = labels.detach().to(torch.long).reshape(-1).cpu()
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(s, stable=True)
    sorted_s = s[order]
    sorted_y = y[order]

    n = sorted_s.numel()
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[i:j] = 0.5 * ((i + 1) + j)
        i = j

    pos_rank_sum = ranks[sorted_y == 1].sum().item()
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def compute_dynamic_stopping_pareto(
    confidences: Tensor,
    accepted_flags: Tensor,
    thresholds: list[float] | tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    """Evaluate block truncation on prefix survival ``prod_{j<=k} c_j >= gamma``."""
    conf = confidences.detach().to(torch.float32).clamp(0.0, 1.0)
    acc = accepted_flags.detach().to(torch.bool)
    if conf.ndim != _RANK_2D or conf.shape != acc.shape:
        raise ValueError(
            f"Expected matching 2D shapes [N, K], got {tuple(conf.shape)} vs "
            f"{tuple(acc.shape)}"
        )
    _, block_size = conf.shape
    prefix_survival = torch.cumprod(conf, dim=-1)
    contiguous_truth = torch.cumprod(acc.to(torch.long), dim=-1).to(torch.bool)
    results: list[dict[str, float]] = []

    for gamma in thresholds:
        if gamma <= 0.0:
            keep_mask = torch.ones_like(acc, dtype=torch.bool)
        else:
            keep_mask = torch.cumprod(
                (prefix_survival >= gamma).to(torch.long), dim=-1
            ).to(torch.bool)

        effective_draft_lens = keep_mask.sum(dim=-1).to(torch.float32)
        contiguous_accept = torch.cumprod(
            (keep_mask & contiguous_truth).to(torch.long), dim=-1
        )
        accepted_spec_tokens = contiguous_accept.sum(dim=-1).to(torch.float32)
        mean_tau = float((1.0 + accepted_spec_tokens.mean()).item())
        mean_draft = float(effective_draft_lens.mean().item())
        saved_pct = float(100.0 * (1.0 - mean_draft / float(block_size)))

        results.append(
            {
                "gamma_conf": float(gamma),
                "mean_drafted_tokens": round(mean_draft, 4),
                "mean_accept_length_tau": round(mean_tau, 4),
                "verifier_slot_flops_saved_pct": round(saved_pct, 2),
            }
        )

    return results


def compute_amdahl_e2e_metrics(
    baseline_tps: float,
    spec_tps: float,
    draft_latency_ms: float,
    verify_latency_ms: float,
    mean_accept_len: float,
) -> dict[str, float]:
    """Compute honest end-to-end speculative speedup and Amdahl's Law decomposition."""
    if baseline_tps <= 0.0 or verify_latency_ms <= 0.0:
        raise ValueError(
            "baseline_tps and verify_latency_ms must be strictly positive."
        )
    step_ms = draft_latency_ms + verify_latency_ms
    e2e_speedup = spec_tps / baseline_tps
    draft_fraction = draft_latency_ms / step_ms if step_ms > 0.0 else 0.0
    amdahl_ceiling = mean_accept_len / (1.0 + draft_latency_ms / verify_latency_ms)
    return {
        "baseline_tps": round(float(baseline_tps), 2),
        "spec_tps": round(float(spec_tps), 2),
        "e2e_speedup": round(float(e2e_speedup), 4),
        "draft_latency_ms": round(float(draft_latency_ms), 4),
        "verify_latency_ms": round(float(verify_latency_ms), 4),
        "draft_overhead_fraction": round(float(draft_fraction), 4),
        "mean_accept_len_tau": round(float(mean_accept_len), 4),
        "amdahl_theoretical_ceiling": round(float(amdahl_ceiling), 4),
    }


def _sync_if_cuda(device: str | torch.device) -> None:
    dev_str = str(device)
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _DeviceStepTimer:
    """CUDA Event timer on GPU with ``time.perf_counter()`` fallback on CPU."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = str(device)
        self.use_cuda_events = (
            self.device.startswith("cuda") and torch.cuda.is_available()
        )
        if self.use_cuda_events:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
        self._t0 = 0.0

    def start(self) -> None:
        if self.use_cuda_events:
            self._start_evt.record()
        else:
            self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self.use_cuda_events:
            self._end_evt.record()
            torch.cuda.synchronize(self.device)
            return float(self._start_evt.elapsed_time(self._end_evt))
        return (time.perf_counter() - self._t0) * 1000.0


def _get_backbone_layers(verifier: nn.Module) -> nn.ModuleList:
    inner = getattr(verifier, "model", verifier)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers  # type: ignore[return-value]


def _get_verifier_norm_and_lm_head(verifier: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Extract final RMSNorm and full-vocab ``lm_head`` from a HuggingFace CausalLM."""
    inner = getattr(verifier, "model", verifier)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    norm_mod = getattr(inner, "norm", nn.Identity())
    lm_head = getattr(verifier, "lm_head", None)
    if lm_head is None:
        raise ValueError(
            f"Verifier {type(verifier).__name__} has no lm_head attribute."
        )
    return norm_mod, lm_head


def _resolve_eos_token_ids(verifier: nn.Module) -> set[int]:
    """Collect valid ``eos_token_id`` ints from ``generation_config`` / ``config``."""
    eos_set: set[int] = set()
    for cfg_obj in (
        getattr(verifier, "generation_config", None),
        getattr(verifier, "config", None),
    ):
        if cfg_obj is None:
            continue
        raw = getattr(cfg_obj, "eos_token_id", None)
        if isinstance(raw, int) and raw >= 0:
            eos_set.add(raw)
        elif isinstance(raw, (list, tuple, set)):
            eos_set.update(int(x) for x in raw if int(x) >= 0)
    return eos_set


@torch.no_grad()
def _forward_verifier_with_speculator_hs(
    verifier: nn.Module,
    input_ids: Tensor,
    aux_layer_ids: list[int],
    past_key_values: Any = None,
    use_cache: bool = False,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Any]:
    """Run ``verifier`` and capture ``(aux_hs, pre_norm_final_hs, logits, pkv)``."""
    layers = _get_backbone_layers(verifier)
    num_layers = len(layers)
    raw_last_holder: list[Tensor] = []

    def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        raw_last_holder.append(h)

    # Custom 4D float tree masks require sdpa/eager rather than FlashAttention-2
    orig_attn_impl: str | None = None
    v_cfg = getattr(verifier, "config", None)
    if (
        attention_mask is not None
        and attention_mask.ndim == _RANK_4D
        and v_cfg is not None
        and getattr(v_cfg, "_attn_implementation", None) == "flash_attention_2"
    ):
        orig_attn_impl = v_cfg._attn_implementation
        v_cfg._attn_implementation = "sdpa"

    call_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "output_hidden_states": True,
    }
    if attention_mask is not None:
        call_kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        call_kwargs["position_ids"] = position_ids

    handle = layers[-1].register_forward_hook(_hook)
    try:
        out = verifier(**call_kwargs)
    finally:
        handle.remove()
        if orig_attn_impl is not None and v_cfg is not None:
            v_cfg._attn_implementation = orig_attn_impl

    pre_norm_last = raw_last_holder[-1]
    aux_parts: list[Tensor] = []
    for lid in aux_layer_ids:
        if lid == num_layers:
            aux_parts.append(pre_norm_last)
        else:
            aux_parts.append(out.hidden_states[lid])
    aux_hs = torch.cat(aux_parts, dim=-1)
    return aux_hs, pre_norm_last, out.logits, out.past_key_values


def _load_drafter_from_checkpoint(
    checkpoint_dir: Path,
    verifier_model: str | None,
    data_dir: Path | None,
    draft_vocab_size: int | None,
    device: str,
    eval_dtype: torch.dtype,
) -> KvNDSparkDraftModel:
    t2d = None
    d2t = None
    if data_dir is not None and draft_vocab_size is not None:
        t2d_path = data_dir / f"t2d-{draft_vocab_size}.npy"
        d2t_path = data_dir / f"d2t-{draft_vocab_size}.npy"
        if t2d_path.exists() and d2t_path.exists():
            t2d = torch.from_numpy(np.load(t2d_path))
            d2t = torch.from_numpy(np.load(d2t_path))

    drafter = KvNDSparkDraftModel.from_pretrained(
        checkpoint_dir,
        t2d=t2d,
        d2t=d2t,
        verifier=verifier_model,
    ).to(device=device, dtype=eval_dtype)
    drafter.eval()
    return drafter


def _compute_full_vocab_greedy_targets(
    final_hs: Tensor,
    anchored_block_indices: Tensor,
    sample_from_anchor: bool,
    full_norm: nn.Module,
    full_lm_head: nn.Module,
) -> Tensor:
    """Compute exact full-vocabulary (e.g. 151,936) verifier greedy target IDs.

    Fixes ``[C4]``: never uses ``drafter.verifier_lm_head`` when sliced to the 32k
    shortlist, so out-of-shortlist verifier argmax tokens are faithfully scored as
    drafter rejections instead of hallucinated shortlist matches.
    """
    total_seq_len = final_hs.size(1)
    target_indices = (
        anchored_block_indices
        if sample_from_anchor
        else (anchored_block_indices - 1) % total_seq_len
    )
    selected_hs = final_hs[:, target_indices].to(full_lm_head.weight.dtype)
    full_logits = full_lm_head(full_norm(selected_hs))
    return full_logits.argmax(dim=-1)


def _collect_calibration_samples(
    drafter: KvNDSparkDraftModel,
    resolved_verifier: str | None,
    data_dir: Path,
    aux_layer_ids: list[int],
    max_samples: int,
    train_data_ratio: float,
    device: str,
    eval_dtype: torch.dtype,
) -> tuple[
    list[tuple[Tensor, Tensor, Tensor, Tensor]],
    nn.Module,
    nn.Module,
    str,
]:
    """Load offline hidden-state files or online Arrow val samples + full lm_head."""
    hs_dir = data_dir / "hidden_states"
    hs_files = (
        sorted(hs_dir.glob("hs_*.safetensors"))[:max_samples] if hs_dir.is_dir() else []
    )
    head_is_sliced = int(drafter.verifier_lm_head.weight.size(0)) < int(
        drafter.verifier_vocab_size
    )
    full_norm: nn.Module = drafter.verifier_norm
    full_lm_head: nn.Module = drafter.verifier_lm_head
    eval_samples: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    if hs_files:
        if head_is_sliced and resolved_verifier:
            verifier_tmp = (
                AutoModelForCausalLM.from_pretrained(
                    resolved_verifier, dtype=eval_dtype
                )
                .to(device)
                .eval()
            )
            v_norm, v_head = _get_verifier_norm_and_lm_head(verifier_tmp)
            full_norm = v_norm.to(device=device, dtype=eval_dtype)
            full_lm_head = v_head.to(device=device, dtype=eval_dtype)
            del verifier_tmp
        for hs_path in hs_files:
            tensors = load_file(str(hs_path))
            hs_all = tensors["hidden_states"].to(device=device, dtype=eval_dtype)
            toks = tensors["token_ids"].to(device=device, dtype=torch.long)
            seq_len = toks.shape[0]
            aux_hs = hs_all[:, :-1, :].reshape(1, seq_len, -1)
            final_hs = hs_all[:, -1, :].unsqueeze(0)
            lmask = torch.ones((1, seq_len), dtype=eval_dtype, device=device)
            lmask[:, :4] = 0
            eval_samples.append((aux_hs, final_hs, toks, lmask))
        return eval_samples, full_norm, full_lm_head, "offline_hidden_states"

    if not resolved_verifier:
        raise ValueError(
            "verifier_model is required when data_dir/hidden_states is absent."
        )
    val_ds = ArrowDataset(
        datapath=data_dir,
        train_ratio=train_data_ratio,
        split="val" if train_data_ratio < 1.0 else "train",
    )
    verifier = (
        AutoModelForCausalLM.from_pretrained(resolved_verifier, dtype=eval_dtype)
        .to(device)
        .eval()
    )
    full_norm, full_lm_head = _get_verifier_norm_and_lm_head(verifier)
    n_take = min(max_samples, len(val_ds))
    for idx in range(n_take):
        rec = val_ds[idx]
        toks = rec.input_ids.to(device=device, dtype=torch.long)
        aux_hs, final_hs, _, _ = _forward_verifier_with_speculator_hs(
            verifier, toks.unsqueeze(0), aux_layer_ids, use_cache=False
        )
        lmask = rec.loss_mask.to(device=device, dtype=eval_dtype).unsqueeze(0)
        eval_samples.append(
            (aux_hs.to(eval_dtype), final_hs.to(eval_dtype), toks, lmask)
        )
    del verifier
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return eval_samples, full_norm, full_lm_head, "online_arrow_val"


def _build_per_position_calibration(
    confs_cat: Tensor,
    raw_accs_cat: Tensor,
    contiguous_accs: Tensor,
    prefix_alive: Tensor,
) -> dict[str, dict[str, float]]:
    """Compute per-slot unconditional and prefix-conditioned calibration metrics."""
    per_pos: dict[str, dict[str, float]] = {}
    for k in range(confs_cat.size(1)):
        alive_k = prefix_alive[:, k]
        has_alive = bool(alive_k.any().item())
        c_k = confs_cat[alive_k, k] if has_alive else confs_cat[:, k]
        a_k = raw_accs_cat[alive_k, k] if has_alive else raw_accs_cat[:, k]
        r_scores_k = 1.0 - c_k
        r_labels_k = (~a_k).to(torch.long)
        per_pos[f"pos_{k}"] = {
            "accept_rate_pct": round(
                100.0 * float(contiguous_accs[:, k].to(torch.float32).mean().item()), 2
            ),
            "conditional_accept_rate_pct": round(
                100.0 * float(a_k.to(torch.float32).mean().item()), 2
            ),
            "ece": round(compute_ece(c_k, a_k.to(torch.float32)), 4),
            "rejection_auroc": round(compute_binary_auroc(r_scores_k, r_labels_k), 4),
        }
    return per_pos


@torch.no_grad()
def evaluate_checkpoint_calibration(
    checkpoint_dir: Path,
    verifier_model: str | None,
    data_dir: Path,
    draft_vocab_size: int,
    max_samples: int = 64,
    max_anchors_per_seq: int = 32,
    train_data_ratio: float = 0.95,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run full-vocab, prefix-conditioned calibration audit on HS or Arrow val."""
    eval_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    drafter = _load_drafter_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        verifier_model=verifier_model,
        data_dir=data_dir,
        draft_vocab_size=draft_vocab_size,
        device=device,
        eval_dtype=eval_dtype,
    )
    aux_layer_ids = list(drafter.config.aux_hidden_state_layer_ids or [2, 18, 33])
    resolved_verifier = verifier_model or (
        drafter.config.speculators_config.verifier.name_or_path
        if drafter.config.speculators_config
        else None
    )
    eval_samples, full_norm, full_lm_head, source_kind = _collect_calibration_samples(
        drafter=drafter,
        resolved_verifier=resolved_verifier,
        data_dir=data_dir,
        aux_layer_ids=aux_layer_ids,
        max_samples=max_samples,
        train_data_ratio=train_data_ratio,
        device=device,
        eval_dtype=eval_dtype,
    )

    all_confs: list[Tensor] = []
    all_raw_accs: list[Tensor] = []
    all_oov_flags: list[Tensor] = []
    block_size = int(drafter.config.block_size)
    sample_from_anchor = bool(drafter.config.sample_from_anchor)
    expected_valid_slots = block_size if sample_from_anchor else (block_size - 1)
    start_slot = 0 if sample_from_anchor else 1

    for aux_hs, final_hs, toks, raw_loss_mask in eval_samples:
        seq_len = toks.shape[0]
        if seq_len <= block_size + 4:
            continue
        cur_mask = raw_loss_mask.clone()
        cur_mask[:, seq_len - block_size :] = 0
        n_valid_mask = int((cur_mask > 0).sum().item())
        if n_valid_mask == 0:
            continue
        doc_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
        n_anc = min(max_anchors_per_seq, max(1, n_valid_mask // block_size))
        (
            hidden,
            base_logits,
            _shortlist_targets,
            aligned_loss_mask,
            anchored_block_indices,
            born_features,
        ) = drafter._kvn_backbone_forward(
            hidden_states=aux_hs,
            input_ids=toks.unsqueeze(0),
            loss_mask=cur_mask,
            verifier_last_hidden_states=final_hs,
            document_ids=doc_ids,
            max_anchors=n_anc,
        )

        draft_target_ids, conf_probs = drafter._decode_draft_block_greedy(
            hidden_blocks=hidden.reshape(n_anc, block_size, -1),
            base_logits=base_logits.reshape(n_anc, block_size, -1),
            anchor_token_ids=toks[anchored_block_indices[::block_size]],
            born_features=born_features,
        )
        expected_ids = _compute_full_vocab_greedy_targets(
            final_hs=final_hs,
            anchored_block_indices=anchored_block_indices,
            sample_from_anchor=sample_from_anchor,
            full_norm=full_norm,
            full_lm_head=full_lm_head,
        ).reshape(n_anc, block_size)

        valid_blocks = (aligned_loss_mask.reshape(n_anc, block_size) > 0).sum(
            dim=-1
        ) == expected_valid_slots
        if not bool(valid_blocks.any().item()):
            continue

        d_ids_v = draft_target_ids[valid_blocks, start_slot:]
        e_ids_v = expected_ids[valid_blocks, start_slot:]
        c_probs_v = conf_probs[valid_blocks, start_slot:].to(torch.float32)
        raw_matched = d_ids_v == e_ids_v
        oov_flags = (
            ~drafter.t2d.to(e_ids_v.device)[e_ids_v]
            if (drafter.use_draft_vocab and drafter.t2d is not None)
            else torch.zeros_like(raw_matched)
        )
        all_confs.append(c_probs_v.cpu())
        all_raw_accs.append(raw_matched.cpu())
        all_oov_flags.append(oov_flags.cpu())

    if not all_confs:
        raise RuntimeError(f"No valid evaluation blocks found in {data_dir}")

    confs_cat = torch.cat(all_confs, dim=0)
    raw_accs_cat = torch.cat(all_raw_accs, dim=0)
    oov_cat = torch.cat(all_oov_flags, dim=0)

    # Mask coincidental post-rejection matches: slot k is alive iff slots 0..k-1 matched
    contiguous_accs = torch.cumprod(raw_accs_cat.to(torch.long), dim=-1).to(torch.bool)
    prefix_alive = torch.ones_like(contiguous_accs, dtype=torch.bool)
    if contiguous_accs.size(1) > 1:
        prefix_alive[:, 1:] = contiguous_accs[:, :-1]

    alive_confs = confs_cat[prefix_alive]
    alive_accs = raw_accs_cat[prefix_alive]
    prefix_survival = torch.cumprod(confs_cat.clamp(0.0, 1.0), dim=-1)
    prefix_surv_ece = compute_ece(
        prefix_survival, contiguous_accs.to(torch.float32), num_bins=15
    )

    return {
        "checkpoint": str(checkpoint_dir),
        "source": source_kind,
        "full_vocab_size_evaluated": int(full_lm_head.weight.size(0)),
        "oov_target_rate_pct": round(
            100.0 * float(oov_cat.to(torch.float32).mean().item()), 2
        ),
        "state_bytes_per_seq": drafter.config.estimated_state_bytes_per_seq(),
        "num_evaluated_blocks": int(confs_cat.shape[0]),
        "overall_ece_15bin": round(
            compute_ece(alive_confs, alive_accs.to(torch.float32), num_bins=15), 4
        ),
        "prefix_survival_ece_15bin": round(prefix_surv_ece, 4),
        "overall_rejection_auroc": round(
            compute_binary_auroc(1.0 - alive_confs, (~alive_accs).to(torch.long)), 4
        ),
        "per_position": _build_per_position_calibration(
            confs_cat, raw_accs_cat, contiguous_accs, prefix_alive
        ),
        "dynamic_stopping_pareto": compute_dynamic_stopping_pareto(
            confs_cat, contiguous_accs
        ),
    }


def _compute_markov_step_bias(
    drafter: Any,
    cur_prev_id: Tensor,
    h_k: Tensor,
    rnn_state: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Compute single-step ``MarkovHead`` logit bias across vanilla/gated/rnn heads."""
    p_emb_k = drafter.markov_head.prev_embeddings(cur_prev_id)
    w_dtype = drafter.markov_head.markov_w2.weight.dtype
    p_emb_cast = p_emb_k.to(w_dtype)
    h_cast = h_k.to(w_dtype)
    if drafter.markov_head.head_type == "vanilla":
        return drafter.markov_head.markov_w2(p_emb_cast), rnn_state
    if drafter.markov_head.head_type == "gated":
        gate = torch.sigmoid(
            drafter.markov_head.gate_proj(torch.cat([h_cast, p_emb_cast], dim=-1))
        )
        return drafter.markov_head.markov_w2(gate * p_emb_cast), rnn_state
    if rnn_state is None:
        raise RuntimeError("MarkovHead RNN state unexpectedly None.")
    z = torch.cat([rnn_state.to(w_dtype), p_emb_cast, h_cast], dim=-1)
    g_raw, c_raw, o_raw = drafter.markov_head.joint_proj(z).chunk(3, dim=-1)
    gate = torch.sigmoid(g_raw)
    next_rnn = gate * rnn_state.to(w_dtype) + (1.0 - gate) * torch.tanh(c_raw)
    return drafter.markov_head.markov_w2(torch.tanh(o_raw)), next_rnn


def _map_draft_choice_to_target(
    drafter: Any,
    draft_idx_k: Tensor,
    q_draft_k: Tensor,
    temperature: float,
    v_target: int,
) -> tuple[Tensor, Tensor]:
    """Map draft shortlist indices/probs onto full verifier vocabulary space."""
    bsz = draft_idx_k.size(0)
    if drafter.use_draft_vocab and drafter.d2t is not None:
        target_id_k = draft_idx_k + drafter.d2t[draft_idx_k]
        if temperature <= 0.0:
            return target_id_k, torch.empty(0, device=draft_idx_k.device)
        draft_all_idx = torch.arange(q_draft_k.size(-1), device=q_draft_k.device)
        mapped_all = draft_all_idx + drafter.d2t[draft_all_idx]
        q_target_k = torch.zeros(
            (bsz, v_target), dtype=torch.float32, device=q_draft_k.device
        )
        q_target_k.scatter_(1, mapped_all.unsqueeze(0).expand(bsz, -1), q_draft_k)
        return target_id_k, q_target_k
    q_target = (
        q_draft_k if temperature > 0.0 else torch.empty(0, device=q_draft_k.device)
    )
    return draft_idx_k, q_target


def _sample_draft_block_with_probs(
    drafter: Any,
    hidden_blocks: Tensor,  # [B, K, D]
    base_logits: Tensor,  # [B, K, V_draft]
    anchor_token_ids: Tensor,  # [B]
    temperature: float,
    prev_context_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample ``K`` draft tokens via ``_decode_draft_block`` (Engram + MarkovHead)."""
    if hasattr(drafter, "_decode_draft_block"):
        draft_tokens, _, draft_probs = drafter._decode_draft_block(
            hidden_blocks=hidden_blocks,
            base_logits=base_logits,
            anchor_token_ids=anchor_token_ids,
            temperature=float(temperature),
            prev_context_ids=prev_context_ids,
        )
        if temperature <= 0.0:
            return draft_tokens, torch.empty(0, device=draft_tokens.device)
        return draft_tokens, _lift_shortlist_probs_batch(draft_probs, drafter)

    bsz, block_size, _ = hidden_blocks.shape
    cur_prev_id = anchor_token_ids.long()
    rnn_state = (
        hidden_blocks.new_zeros(bsz, drafter.markov_head.markov_rank)
        if (drafter.markov_head is not None and drafter.markov_head.head_type == "rnn")
        else None
    )

    drafted_ids: list[Tensor] = []
    target_probs_per_k: list[Tensor] = []
    v_target = int(drafter.verifier_vocab_size)

    for k in range(block_size):
        logits_k = base_logits[:, k, :].float()
        if drafter.markov_head is not None:
            bias_k, rnn_state = _compute_markov_step_bias(
                drafter, cur_prev_id, hidden_blocks[:, k, :], rnn_state
            )
            logits_k = logits_k + bias_k.float()

        if temperature <= 0.0:
            draft_idx_k = logits_k.argmax(dim=-1)
            q_draft_k = fn.one_hot(draft_idx_k, num_classes=logits_k.size(-1)).float()
        else:
            q_draft_k = fn.softmax(logits_k / temperature, dim=-1)
            draft_idx_k = torch.multinomial(q_draft_k, num_samples=1).squeeze(-1)

        target_id_k, q_target_k = _map_draft_choice_to_target(
            drafter, draft_idx_k, q_draft_k, temperature, v_target
        )
        drafted_ids.append(target_id_k)
        if temperature > 0.0:
            target_probs_per_k.append(q_target_k)
        cur_prev_id = (
            target_id_k
            if drafter.config.sample_from_anchor or k > 0
            else anchor_token_ids.long()
        )

    stacked_probs = (
        torch.stack(target_probs_per_k, dim=1)
        if target_probs_per_k
        else torch.empty(0, device=hidden_blocks.device)
    )
    return torch.stack(drafted_ids, dim=1), stacked_probs


def _lift_shortlist_probs_batch(
    q_probs: Tensor,
    drafter: Any,
) -> Tensor:
    """Lift ``[B, K, V_draft]`` probabilities to ``[B, K, V_verifier]`` via ``d2t``."""
    v_full = int(drafter.verifier_vocab_size)
    if q_probs.size(-1) == v_full:
        return q_probs.to(torch.float32)
    bsz, k_len, v_draft = q_probs.shape
    out = torch.zeros((bsz, k_len, v_full), dtype=torch.float32, device=q_probs.device)
    d_idx = torch.arange(v_draft, device=q_probs.device, dtype=torch.long)
    t_idx = d_idx + drafter.d2t.to(device=q_probs.device, dtype=torch.long)
    out.scatter_(
        2, t_idx.view(1, 1, v_draft).expand(bsz, k_len, v_draft), q_probs.float()
    )
    return out


@dataclass
class _SwaIncrementalCacheState:
    """True batched incremental KV cache state for ``Qwen3DFlashDecoderLayer``."""

    cache: Any  # DynamicCache
    cached_positions: Tensor  # [B, L_cached] int64
    sliding_window: int | None
    uses_full_attn: bool


def _init_swa_cache_state(
    drafter: Any,
    bsz: int,
    device: str | torch.device,
) -> _SwaIncrementalCacheState:
    tl_cfg = drafter.config.transformer_layer_config
    win = getattr(tl_cfg, "sliding_window", None)
    return _SwaIncrementalCacheState(
        cache=DynamicCache(),
        cached_positions=torch.empty((bsz, 0), dtype=torch.long, device=device),
        sliding_window=int(win) if win is not None else None,
        uses_full_attn=bool(getattr(drafter, "uses_full_attn", False)),
    )


def _trim_swa_cache_to_window(
    swa_state: _SwaIncrementalCacheState,
    block_size: int,
) -> None:
    """Bound SWA KV cache length to ``sliding_window + block_size`` when pure SWA."""
    if swa_state.uses_full_attn or swa_state.sliding_window is None:
        return
    max_keep = swa_state.sliding_window + block_size + 4
    cur_len = swa_state.cached_positions.size(1)
    if cur_len <= max_keep:
        return

    swa_state.cached_positions = swa_state.cached_positions[:, -max_keep:].contiguous()
    cache = swa_state.cache
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[:, :, -max_keep:, :].contiguous()
                layer.values = layer.values[:, :, -max_keep:, :].contiguous()
    elif hasattr(cache, "key_cache"):
        for idx in range(len(cache.key_cache)):
            cache.key_cache[idx] = cache.key_cache[idx][
                :, :, -max_keep:, :
            ].contiguous()
            cache.value_cache[idx] = cache.value_cache[idx][
                :, :, -max_keep:, :
            ].contiguous()


def _build_swa_batched_sdpa_mask(
    drafter: Any,
    cached_and_new_pos: Tensor,  # [B, L_base]
    anchor_pos: Tensor,  # [B]
    block_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
) -> Tensor:
    """Build exact ``[B, 1, K, L_base + K]`` additive SDPA mask for anchor blocks."""
    bsz, l_base = cached_and_new_pos.shape
    device = anchor_pos.device
    k_offs = torch.arange(block_size, device=device, dtype=torch.long)
    q_pos = anchor_pos.unsqueeze(1) + k_offs.unsqueeze(0)  # [B, K]

    # Base prefix: kv_pos < anchor_pos AND within sliding_window of query_position
    base_pos_exp = cached_and_new_pos.unsqueeze(1)  # [B, 1, L_base]
    allow_base = (base_pos_exp >= 0) & (
        base_pos_exp < anchor_pos.view(bsz, 1, 1)
    ).expand(bsz, block_size, l_base)
    if sliding_window is not None:
        allow_base = allow_base & (base_pos_exp > (q_pos.unsqueeze(2) - sliding_window))

    # Intra-block: causal or bidirectional matching DFlash/DSpark config
    is_causal = getattr(drafter.config.transformer_layer_config, "is_causal", None)
    if is_causal is None:
        is_causal = (
            sliding_window is not None and not drafter.config.sliding_window_non_causal
        )
    if is_causal:
        block_allow_2d = k_offs.unsqueeze(1) >= k_offs.unsqueeze(0)
    else:
        block_allow_2d = torch.ones(
            (block_size, block_size), dtype=torch.bool, device=device
        )
    allow_block = block_allow_2d.unsqueeze(0).expand(bsz, block_size, block_size)

    allow = torch.cat([allow_base, allow_block], dim=-1)  # [B, K, L_base + K]
    min_val = torch.finfo(dtype).min
    mask = torch.full(
        (bsz, 1, block_size, l_base + block_size),
        min_val,
        dtype=dtype,
        device=device,
    )
    return mask.masked_fill(allow.unsqueeze(1), 0.0)


def _run_swa_layers_batched(
    drafter: Any,
    query_blocks: Tensor,  # [B, K, D]
    new_target_fc: Tensor,  # [B, L_new, D]
    new_target_pos: Tensor,  # [B, L_new] int64
    anchor_pos: Tensor,  # [B] int64
    swa_state: _SwaIncrementalCacheState,
) -> Tensor:
    """Execute ``drafter.layers`` across ``B`` items via SDPA + ``DynamicCache``."""
    bsz, block_size, _ = query_blocks.shape
    device = query_blocks.device
    dtype = query_blocks.dtype

    k_offs = torch.arange(block_size, device=device, dtype=torch.long).unsqueeze(0)
    q_pos = anchor_pos.unsqueeze(1) + k_offs  # [B, K]
    step_pos = torch.cat([new_target_pos, q_pos], dim=1)  # [B, L_new + K]
    position_embeddings = drafter.rotary_emb(query_blocks, step_pos)

    all_base_pos = torch.cat([swa_state.cached_positions, new_target_pos], dim=1)
    l_base_after = all_base_pos.size(1)

    full_mask = (
        _build_swa_batched_sdpa_mask(
            drafter, all_base_pos, anchor_pos, block_size, None, dtype
        )
        if swa_state.uses_full_attn
        else None
    )
    swa_mask = (
        _build_swa_batched_sdpa_mask(
            drafter,
            all_base_pos,
            anchor_pos,
            block_size,
            swa_state.sliding_window,
            dtype,
        )
        if getattr(drafter, "uses_sliding_window_attn", False)
        else full_mask
    )

    saved_impls: list[tuple[Any, Any]] = []
    for layer in drafter.layers:
        cfg_obj = layer.self_attn.config
        saved_impls.append((cfg_obj, getattr(cfg_obj, "_attn_implementation", None)))
        cfg_obj._attn_implementation = "sdpa"

    hidden = query_blocks
    try:
        for layer_idx, layer in enumerate(drafter.layers):
            mask_l = (
                swa_mask
                if layer_idx in getattr(drafter, "sliding_window_indices", ())
                else full_mask
            )
            hidden = layer(
                hidden_states=hidden,
                target_hidden=new_target_fc,
                attention_mask=mask_l,
                position_ids=step_pos,
                past_key_value=swa_state.cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
    finally:
        for cfg_obj, orig_impl in saved_impls:
            cfg_obj._attn_implementation = orig_impl

    # Strip temporary synthetic noise entries while retaining confirmed base K/Vs
    remove_noise = int(swa_state.cache.get_seq_length()) - l_base_after
    if remove_noise > 0:
        swa_state.cache.crop(-remove_noise)
    swa_state.cached_positions = all_base_pos
    _trim_swa_cache_to_window(swa_state, block_size=block_size)
    return hidden


def _draft_step_swa_batched(
    drafter: Any,
    new_target_fc: Tensor,  # [B, L_new, D]
    new_target_pos: Tensor,  # [B, L_new]
    anchor_token_ids: Tensor,  # [B]
    anchor_pos: Tensor,  # [B]
    swa_state: _SwaIncrementalCacheState,
    temperature: float,
    prev_context_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """True batched incremental SWA block proposal (no ``for _ in range(bs)`` loop)."""
    bsz = anchor_token_ids.size(0)
    block_size = int(drafter.config.block_size)
    mask_token_ids = torch.full(
        (bsz, block_size),
        int(drafter.mask_token_id),
        dtype=torch.long,
        device=anchor_token_ids.device,
    )
    mask_token_ids[:, 0] = anchor_token_ids
    noise_emb = drafter.embed_tokens(mask_token_ids).to(new_target_fc.dtype)

    raw_blocks = _run_swa_layers_batched(
        drafter=drafter,
        query_blocks=noise_emb,
        new_target_fc=new_target_fc,
        new_target_pos=new_target_pos,
        anchor_pos=anchor_pos,
        swa_state=swa_state,
    )
    if getattr(drafter, "intrablock_mixer", None) is not None:
        raw_blocks = drafter.intrablock_mixer(raw_blocks)
    hidden_blocks = drafter.norm(raw_blocks)
    base_logits = drafter.lm_head(hidden_blocks)
    return _sample_draft_block_with_probs(
        drafter=drafter,
        hidden_blocks=hidden_blocks,
        base_logits=base_logits,
        anchor_token_ids=anchor_token_ids,
        temperature=temperature,
        prev_context_ids=prev_context_ids,
    )


def _prefill_kvn_layers_parallel(
    drafter: Any,
    prompt_ids: Tensor,  # [B, P]
    fc_prefill: Tensor,  # [B, P, D]
    state_cache: Tensor,  # [L_kvn, C, B, M]
) -> Tensor:
    """Prefill ``drafter.kvn_layers`` across ``P`` prompt tokens in ``O(P log P)``.

    Updates ``state_cache`` in-place and returns ``last_fc`` ``[B, 1, D]``.
    """
    if hasattr(drafter, "prefill_kvn_state"):
        out = drafter.prefill_kvn_state(prompt_ids, fc_prefill)
        if isinstance(out, dict):
            if "state_cache" in out and isinstance(out["state_cache"], Tensor):
                state_cache.copy_(out["state_cache"].to(state_cache.dtype))
            elif "q" in out and "p" in out:
                state_cache[:, 0].copy_(out["q"].to(state_cache.dtype))
                state_cache[:, 1].copy_(out["p"].to(state_cache.dtype))
                if (
                    state_cache.size(1) >= _NU_MIN_COMPONENTS
                    and out.get("nu") is not None
                ):
                    state_cache[:, 2].copy_(out["nu"].to(state_cache.dtype))
            last_fc = out.get("last_fc", fc_prefill[:, -1:, :])
            return last_fc.reshape(prompt_ids.size(0), 1, -1).to(fc_prefill.dtype)

    tok_emb = drafter.embed_tokens(prompt_ids).to(fc_prefill.dtype)
    x_cur, reset_mask = drafter._prepare_causal_kvn_inputs(
        fc_output=fc_prefill,
        tok_emb=tok_emb,
        document_ids=None,
    )
    want_nu = state_cache.size(1) >= _NU_MIN_COMPONENTS
    fwd_sig = inspect.signature(drafter.kvn_layers[0].forward)
    fwd_kw: dict[str, Any] = {"reset_mask": reset_mask}
    if "return_nu" in fwd_sig.parameters and want_nu:
        fwd_kw["return_nu"] = True
    for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
        layer_out = kvn_layer(x_cur, **fwd_kw)
        x_cur = layer_out[0]
        state_cache[l_idx, 0].copy_(layer_out[1][:, -1, :].to(state_cache.dtype))
        state_cache[l_idx, 1].copy_(layer_out[2][:, -1, :].to(state_cache.dtype))
        if want_nu and len(layer_out) >= _LAYER_OUT_V3_LEN and layer_out[4] is not None:
            state_cache[l_idx, 2].copy_(layer_out[4][:, -1, :].to(state_cache.dtype))

    return fc_prefill[:, -1:, :].contiguous()


def _advance_kvn_rollout_blocks(
    drafter: Any,
    prev_fc_chunk: Tensor,  # [B, S_max, D]
    token_ids_chunk: Tensor,  # [B, S_max]
    valid_lens: Tensor,  # [B]
    state_cache: Tensor,  # [L_kvn, C, B, M]
    *,
    apply_mixer: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance ``kvn_layers`` in FP32 and return ``(raw_blocks, anchor_tok, x_seq)``."""
    bsz, s_max = token_ids_chunk.shape
    tok_emb = drafter.embed_tokens(token_ids_chunk)
    x_cur = prev_fc_chunk.to(tok_emb.dtype) + tok_emb
    step_idx = torch.arange(s_max, device=x_cur.device).unsqueeze(0)
    active_mask = step_idx < valid_lens.to(x_cur.device).unsqueeze(1)

    has_nu = state_cache.size(1) >= _NU_MIN_COMPONENTS
    rec_sig = inspect.signature(drafter.kvn_layers[0].recurrent_step)
    pass_state_dtype = "state_dtype" in rec_sig.parameters

    new_states: list[Tensor] = []
    q_anc_layers: list[Tensor] = []
    p_anc_layers: list[Tensor] = []
    dt_anc_layers: list[Tensor] = []
    nu_anc_layers: list[Tensor] = []

    for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
        q_s = state_cache[l_idx, 0].float()
        p_s = state_cache[l_idx, 1].float()
        nu_s = state_cache[l_idx, 2].float() if has_nu else None
        dtheta_anc = torch.zeros(
            (bsz, kvn_layer.n_modes), dtype=x_cur.dtype, device=x_cur.device
        )
        h_steps: list[Tensor] = []

        for s in range(s_max):
            is_val = active_mask[:, s].unsqueeze(-1)
            is_anc = (valid_lens.to(x_cur.device) == (s + 1)).unsqueeze(-1)
            state_in: tuple[Tensor, ...] = (
                (q_s.unsqueeze(1), p_s.unsqueeze(1), nu_s.unsqueeze(1))
                if nu_s is not None
                else (q_s.unsqueeze(1), p_s.unsqueeze(1))
            )
            step_kw: dict[str, Any] = {"state": state_in}
            if pass_state_dtype:
                step_kw["state_dtype"] = torch.float32
            h_s, state_out, dt_s = kvn_layer.recurrent_step(
                x_cur[:, s : s + 1, :], **step_kw
            )
            q_s = torch.where(is_val, state_out[0].squeeze(1).float(), q_s)
            p_s = torch.where(is_val, state_out[1].squeeze(1).float(), p_s)
            if (
                nu_s is not None
                and len(state_out) >= _NU_MIN_COMPONENTS
                and state_out[2] is not None
            ):
                nu_s = torch.where(is_val, state_out[2].squeeze(1).float(), nu_s)
            dtheta_anc = torch.where(
                is_anc, dt_s.squeeze(1).to(dtheta_anc.dtype), dtheta_anc
            )
            h_steps.append(h_s)

        if nu_s is not None:
            new_states.append(
                torch.stack([q_s, p_s, nu_s], dim=0).to(state_cache.dtype)
            )
            nu_anc_layers.append(nu_s.to(x_cur.dtype))
        else:
            new_states.append(torch.stack([q_s, p_s], dim=0).to(state_cache.dtype))
        q_anc_layers.append(q_s.to(x_cur.dtype))
        p_anc_layers.append(p_s.to(x_cur.dtype))
        dt_anc_layers.append(dtheta_anc)
        x_cur = torch.cat(h_steps, dim=1)

    state_cache.copy_(torch.stack(new_states, dim=0))
    last_step = (valid_lens.to(x_cur.device) - 1).clamp(min=0)
    anchor_h = x_cur.gather(
        1, last_step.view(bsz, 1, 1).expand(bsz, 1, x_cur.size(-1))
    ).squeeze(1)
    anchor_tok = token_ids_chunk.gather(1, last_step.view(bsz, 1)).squeeze(1)

    anc_q, anc_p, anc_dt, anc_nu = _aggregate_rollout_anchor_states(
        drafter, q_anc_layers, p_anc_layers, dt_anc_layers, nu_anc_layers
    )
    roll_sig = inspect.signature(drafter.rollout.forward)
    roll_kw: dict[str, Any] = {}
    if "anchor_nu" in roll_sig.parameters and anc_nu is not None:
        roll_kw["anchor_nu"] = anc_nu
    if "return_quantum_stats" in roll_sig.parameters:
        roll_kw["return_quantum_stats"] = True
    roll_out = drafter.rollout(anchor_h, anc_q, anc_p, anc_dt, **roll_kw)
    raw_blocks = roll_out[0]
    if apply_mixer and drafter.intrablock_mixer is not None:
        raw_blocks = drafter.intrablock_mixer(raw_blocks)
    return raw_blocks, anchor_tok, roll_out


_ROLLOUT_QOBS_TUPLE_LEN = 6


def _aggregate_rollout_anchor_states(
    drafter: Any,
    q_anc_layers: list[Tensor],
    p_anc_layers: list[Tensor],
    dt_anc_layers: list[Tensor],
    nu_anc_layers: list[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Aggregate multi-layer ``(q, p, dtheta[, nu])`` anchor states via ``nu_list``."""
    agg_sig = inspect.signature(drafter.rollout.aggregate_multilayer_states)
    if "nu_list" in agg_sig.parameters and nu_anc_layers:
        agg_out = drafter.rollout.aggregate_multilayer_states(
            q_anc_layers, p_anc_layers, dt_anc_layers, nu_list=nu_anc_layers
        )
        return agg_out[0], agg_out[1], agg_out[2], agg_out[3]
    if "nu_per_layer" in agg_sig.parameters and nu_anc_layers:
        agg_out = drafter.rollout.aggregate_multilayer_states(
            q_anc_layers, p_anc_layers, dt_anc_layers, nu_per_layer=nu_anc_layers
        )
        return agg_out[0], agg_out[1], agg_out[2], agg_out[3]
    agg_out = drafter.rollout.aggregate_multilayer_states(
        q_anc_layers, p_anc_layers, dt_anc_layers
    )
    return agg_out[0], agg_out[1], agg_out[2], None


def _maybe_apply_kraus_logit_delta(
    drafter: Any,
    base_logits: Tensor,
    roll_out: Any,
) -> Tensor:
    """Apply ``kraus_head.forward_topc_logit_delta`` when ``q_obs, p_obs`` exist."""
    if (
        getattr(drafter, "kraus_head", None) is not None
        and isinstance(roll_out, tuple)
        and len(roll_out) >= _ROLLOUT_QOBS_TUPLE_LEN
    ):
        q_obs, p_obs = roll_out[4], roll_out[5]
        return base_logits + drafter.kraus_head.forward_topc_logit_delta(
            logits=base_logits,
            q_obs=q_obs,
            p_obs=p_obs,
            lm_head_weight=drafter.lm_head.weight,
        )
    return base_logits


def _draft_step_kvn_recurrent(
    drafter: Any,
    prev_fc_chunk: Tensor,  # [B, S_chunk, D]
    token_ids_chunk: Tensor,  # [B, S_chunk]
    valid_lens: Tensor,  # [B]
    state_cache: Tensor,  # [L_kvn, C, B, M]
    temperature: float,
    s_pad_to: int,
    prev_context_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run O(1)-state KvN block proposal at greedy or stochastic temperature."""
    bsz, cur_s, d_model = prev_fc_chunk.shape
    if cur_s < s_pad_to:
        pad_fc = prev_fc_chunk.new_zeros((bsz, s_pad_to - cur_s, d_model))
        pad_id = token_ids_chunk.new_zeros((bsz, s_pad_to - cur_s))
        prev_fc_chunk = torch.cat([prev_fc_chunk, pad_fc], dim=1)
        token_ids_chunk = torch.cat([token_ids_chunk, pad_id], dim=1)

    adv_sig = inspect.signature(drafter.advance_and_draft_step)
    if "return_details" in adv_sig.parameters and "temperature" in adv_sig.parameters:
        adv_kw: dict[str, Any] = {
            "temperature": float(temperature),
            "return_details": True,
        }
        if "prev_context_ids" in adv_sig.parameters and prev_context_ids is not None:
            adv_kw["prev_context_ids"] = prev_context_ids
        draft_ids, updated_cache, details = drafter.advance_and_draft_step(
            prev_fc_chunk,
            token_ids_chunk,
            valid_lens,
            state_cache,
            **adv_kw,
        )
        if (
            isinstance(updated_cache, Tensor)
            and updated_cache.data_ptr() != state_cache.data_ptr()
        ):
            state_cache.copy_(updated_cache)
        if temperature <= 0.0:
            return draft_ids, torch.empty(0, device=draft_ids.device)
        if "draft_probs" in details and details["draft_probs"] is not None:
            return draft_ids, _lift_shortlist_probs_batch(
                details["draft_probs"], drafter
            )

    if temperature <= 0.0:
        out = drafter.advance_and_draft_step(
            prev_fc_block=prev_fc_chunk,
            token_ids_block=token_ids_chunk,
            valid_lens=valid_lens,
            state_cache=state_cache,
        )
        draft_ids = out[0]
        return draft_ids, torch.empty(0, device=draft_ids.device)

    raw_blocks, anchor_tok, roll_out = _advance_kvn_rollout_blocks(
        drafter, prev_fc_chunk, token_ids_chunk, valid_lens, state_cache
    )
    hidden_blocks = drafter.norm(raw_blocks)
    base_logits = _maybe_apply_kraus_logit_delta(
        drafter, drafter.lm_head(hidden_blocks), roll_out
    )
    return _sample_draft_block_with_probs(
        drafter,
        hidden_blocks,
        base_logits,
        anchor_tok,
        temperature,
        prev_context_ids=prev_context_ids,
    )


def _draft_step_hybrid_batched(
    drafter: Any,
    prev_fc_chunk: Tensor,  # [B, S_chunk, D]
    token_ids_chunk: Tensor,  # [B, S_chunk]
    valid_lens: Tensor,  # [B]
    state_cache: Tensor,  # [L_kvn, C, B, M] persistent across decode!
    new_swa_fc: Tensor,  # [B, L_new, D]
    new_swa_pos: Tensor,  # [B, L_new]
    anchor_pos: Tensor,  # [B]
    swa_state: _SwaIncrementalCacheState,
    temperature: float,
    s_pad_to: int,
    prev_context_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Batched sequential ``B3_Hybrid`` step with persistent KvN + SWA cache."""
    bsz, cur_s, d_model = prev_fc_chunk.shape
    if cur_s < s_pad_to:
        pad_fc = prev_fc_chunk.new_zeros((bsz, s_pad_to - cur_s, d_model))
        pad_id = token_ids_chunk.new_zeros((bsz, s_pad_to - cur_s))
        prev_fc_chunk = torch.cat([prev_fc_chunk, pad_fc], dim=1)
        token_ids_chunk = torch.cat([token_ids_chunk, pad_id], dim=1)

    kvn_blocks, anchor_tok, roll_out = _advance_kvn_rollout_blocks(
        drafter,
        prev_fc_chunk,
        token_ids_chunk,
        valid_lens,
        state_cache,
        apply_mixer=False,
    )
    # Sequential B3 (KvN -> rollout -> SWA); residual fallback only if legacy flag set
    if getattr(drafter.config, "hybrid_parallel_sum", False):
        block_size = int(drafter.config.block_size)
        mask_ids = torch.full(
            (bsz, block_size),
            int(drafter.mask_token_id),
            dtype=torch.long,
            device=anchor_tok.device,
        )
        mask_ids[:, 0] = anchor_tok
        noise_emb = drafter.embed_tokens(mask_ids).to(kvn_blocks.dtype)
        swa_out = _run_swa_layers_batched(
            drafter, noise_emb, new_swa_fc, new_swa_pos, anchor_pos, swa_state
        )
        combined = kvn_blocks + swa_out
    else:
        combined = _run_swa_layers_batched(
            drafter, kvn_blocks, new_swa_fc, new_swa_pos, anchor_pos, swa_state
        )
    if drafter.intrablock_mixer is not None:
        combined = drafter.intrablock_mixer(combined)

    hidden_blocks = drafter.norm(combined)
    base_logits = _maybe_apply_kraus_logit_delta(
        drafter, drafter.lm_head(hidden_blocks), roll_out
    )
    return _sample_draft_block_with_probs(
        drafter,
        hidden_blocks,
        base_logits,
        anchor_tok,
        temperature,
        prev_context_ids=prev_context_ids,
    )


def _draft_step_kraus_tree(
    drafter: Any,
    prev_fc_chunk: Tensor,
    token_ids_chunk: Tensor,
    valid_lens: Tensor,
    state_cache: Tensor,
    tree_budget: int,
    branch_width: int,
    temperature: float,
    s_pad_to: int,
) -> dict[str, Tensor]:
    """Run ``drafter.draft_kraus_tree_step`` (or top-w fallback) with static padding."""
    bsz, cur_s, d_model = prev_fc_chunk.shape
    if cur_s < s_pad_to:
        pad_fc = prev_fc_chunk.new_zeros((bsz, s_pad_to - cur_s, d_model))
        pad_id = token_ids_chunk.new_zeros((bsz, s_pad_to - cur_s))
        prev_fc_chunk = torch.cat([prev_fc_chunk, pad_fc], dim=1)
        token_ids_chunk = torch.cat([token_ids_chunk, pad_id], dim=1)

    if hasattr(drafter, "draft_kraus_tree_step"):
        tree_dict, updated_cache = drafter.draft_kraus_tree_step(
            prev_fc_chunk,
            token_ids_chunk,
            valid_lens,
            state_cache,
            tree_budget=int(tree_budget),
            branch_width=int(branch_width),
            temperature=float(temperature),
        )
        if (
            isinstance(updated_cache, Tensor)
            and updated_cache.data_ptr() != state_cache.data_ptr()
        ):
            state_cache.copy_(updated_cache)
        return dict(tree_dict)

    # Fallback top-w fork from anchor slot 0 + chain continuation on v2 checkpoints
    raw_blocks, anchor_tok, roll_out = _advance_kvn_rollout_blocks(
        drafter, prev_fc_chunk, token_ids_chunk, valid_lens, state_cache
    )
    hidden_blocks = drafter.norm(raw_blocks)
    base_logits = _maybe_apply_kraus_logit_delta(
        drafter, drafter.lm_head(hidden_blocks), roll_out
    )
    chain_ids, _ = _sample_draft_block_with_probs(
        drafter, hidden_blocks, base_logits, anchor_tok, temperature=0.0
    )
    start_k = 0 if drafter.config.sample_from_anchor else 1
    chain_spec = chain_ids[:, start_k:]
    k_chain = chain_spec.size(1)

    if hasattr(drafter, "_step_markov_and_engram_logits"):
        init_rnn = (
            hidden_blocks.new_zeros(bsz, drafter.markov_head.markov_rank)
            if (
                drafter.markov_head is not None
                and drafter.markov_head.head_type == "rnn"
            )
            else None
        )
        zero_id = torch.zeros_like(anchor_tok.long())
        logits_0_raw, _, _ = drafter._step_markov_and_engram_logits(
            base_logits_k=base_logits[:, start_k, :],
            h_k=hidden_blocks[:, start_k, :],
            cur_prev_id=anchor_tok.long(),
            cur_prev_prev_id=zero_id,
            cur_prev3_id=zero_id,
            rnn_state=init_rnn,
        )
        logits_0 = logits_0_raw.float()
    else:
        logits_0 = base_logits[:, start_k, :].float()
        if drafter.markov_head is not None:
            init_rnn = (
                hidden_blocks.new_zeros(bsz, drafter.markov_head.markov_rank)
                if drafter.markov_head.head_type == "rnn"
                else None
            )
            bias_0, _ = _compute_markov_step_bias(
                drafter, anchor_tok.long(), hidden_blocks[:, start_k, :], init_rnn
            )
            logits_0 = logits_0 + bias_0.float()

    n_extra = min(max(0, int(branch_width) - 1), max(0, int(tree_budget) - k_chain))
    top_w = torch.topk(logits_0, k=min(1 + n_extra, logits_0.size(-1)), dim=-1).indices
    if drafter.use_draft_vocab and drafter.d2t is not None:
        top_w_target = top_w + drafter.d2t[top_w]
    else:
        top_w_target = top_w

    alt_siblings = top_w_target[:, 1 : 1 + n_extra]
    n_chain_keep = min(k_chain, max(1, int(tree_budget) - alt_siblings.size(1)))
    tree_ids = torch.cat([chain_spec[:, :n_chain_keep], alt_siblings], dim=1)
    n_total = tree_ids.size(1)
    parents = torch.full((bsz, n_total), -1, dtype=torch.long, device=tree_ids.device)
    if n_chain_keep > 1:
        parents[:, 1:n_chain_keep] = torch.arange(
            n_chain_keep - 1, device=tree_ids.device, dtype=torch.long
        ).unsqueeze(0)
    return {"tree_draft_ids": tree_ids, "tree_parent_idx": parents}


def _load_benchmark_prompts(
    benchmark: str, tokenizer: Any, num_prompts: int, max_prompt_tokens: int = 1024
) -> list[list[int]]:
    if benchmark not in E2E_BENCHMARK_SPECS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}; expected {list(E2E_BENCHMARK_SPECS)}"
        )
    if datasets is None or huggingface_hub is None:
        raise ImportError("`datasets` and `huggingface_hub` are required for prompts.")
    hf_path, subset, split, col = E2E_BENCHMARK_SPECS[benchmark]
    try:
        ds: Any = (
            datasets.load_dataset(hf_path, subset, split=split)
            if subset is not None
            else datasets.load_dataset(hf_path, split=split)
        )
    except (RuntimeError, ValueError):
        files = sorted(
            f
            for f in huggingface_hub.list_repo_files(hf_path, repo_type="dataset")
            if f.endswith(".jsonl")
        )
        local = huggingface_hub.hf_hub_download(hf_path, files[0], repo_type="dataset")
        ds = [
            json.loads(line)
            for line in Path(local).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    prompts: list[list[int]] = []
    for row in ds:
        raw_val = row.get(col)
        text = (
            raw_val[0] if isinstance(raw_val, list) and raw_val else str(raw_val or "")
        )
        if not text.strip():
            continue
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
            )
            if not isinstance(ids, list):
                ids = list(ids["input_ids"])
        else:
            ids = tokenizer.encode(text)
        ids = ids[:max_prompt_tokens]
        if len(ids) >= _MIN_PROMPT_TOKENS:
            prompts.append(ids)
        if len(prompts) >= num_prompts:
            break
    return prompts


def _sample_single_token(logits_1d: Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(logits_1d.argmax(dim=-1).item())
    probs = fn.softmax(logits_1d.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _run_warmup_passes(
    verifier: nn.Module,
    drafter: Any,
    sample_prompt: list[int],
    aux_layer_ids: list[int],
    eval_dtype: torch.dtype,
    device: str,
    warmup_iters: int,
) -> None:
    """Execute ``>= 5`` CUDA warmup iterations before steady-state latency timing."""
    inp = torch.tensor([sample_prompt[:16]], dtype=torch.long, device=device)
    block_size = int(drafter.config.block_size)
    backbone = drafter.config.kvn_backbone_type

    for _ in range(warmup_iters):
        pkv = DynamicCache()
        aux, _, _, _ = _forward_verifier_with_speculator_hs(
            verifier, inp, aux_layer_ids, past_key_values=pkv, use_cache=True
        )
        fc = drafter.hidden_norm(drafter.fc(aux.to(eval_dtype)))
        if backbone == "kvn":
            sc = drafter.allocate_state_cache(batch_size=1, device=device)
            last_fc = _prefill_kvn_layers_parallel(drafter, inp, fc, sc)
            _ = _draft_step_kvn_recurrent(
                drafter,
                last_fc,
                inp[:, -1:],
                torch.ones((1,), dtype=torch.long, device=device),
                sc,
                temperature=0.0,
                s_pad_to=block_size + 1,
            )
        elif backbone == "hybrid_1swa_2kvn":
            sc = drafter.allocate_state_cache(batch_size=1, device=device)
            last_fc = _prefill_kvn_layers_parallel(drafter, inp, fc, sc)
            swa_st = _init_swa_cache_state(drafter, bsz=1, device=device)
            pos = torch.arange(inp.size(1), device=device, dtype=torch.long).unsqueeze(
                0
            )
            anc_p = torch.tensor([inp.size(1)], device=device, dtype=torch.long)
            _ = _draft_step_hybrid_batched(
                drafter,
                last_fc,
                inp[:, -1:],
                torch.ones((1,), dtype=torch.long, device=device),
                sc,
                fc,
                pos,
                anc_p,
                swa_st,
                temperature=0.0,
                s_pad_to=block_size + 1,
            )
        else:
            swa_st = _init_swa_cache_state(drafter, bsz=1, device=device)
            pos = torch.arange(inp.size(1), device=device, dtype=torch.long).unsqueeze(
                0
            )
            anc_p = torch.tensor([inp.size(1)], device=device, dtype=torch.long)
            _ = _draft_step_swa_batched(
                drafter, fc, pos, inp[:, -1], anc_p, swa_st, temperature=0.0
            )
    _sync_if_cuda(device)


def _time_batched_kvn_drafter_tier(
    drafter: Any,
    batch_prompt: Tensor,
    fc_pref: Tensor,
    fc_blk: Tensor,
    tok_blk: Tensor,
    vl: Tensor,
    bs: int,
    s_max: int,
    use_kraus_tree: bool,
    tree_budget: int,
    branch_width: int,
    warmup_steps: int,
    timed_steps: int,
    timer: _DeviceStepTimer,
    device: str,
) -> list[float]:
    """Time steady-state batched ``kvn`` proposer steps (chain or Kraus tree)."""
    sc = drafter.allocate_state_cache(batch_size=bs, device=device)
    _ = _prefill_kvn_layers_parallel(drafter, batch_prompt, fc_pref, sc)
    draft_ms_list: list[float] = []
    for step_i in range(warmup_steps + timed_steps):
        if step_i >= warmup_steps:
            timer.start()
        if use_kraus_tree:
            _ = _draft_step_kraus_tree(
                drafter,
                fc_blk,
                tok_blk,
                vl,
                sc,
                tree_budget=tree_budget,
                branch_width=branch_width,
                temperature=0.0,
                s_pad_to=s_max,
            )
        else:
            _ = _draft_step_kvn_recurrent(
                drafter, fc_blk, tok_blk, vl, sc, temperature=0.0, s_pad_to=s_max
            )
        if step_i >= warmup_steps:
            draft_ms_list.append(timer.stop_ms())
    return draft_ms_list


def _time_batched_swa_or_hybrid_tier(
    drafter: Any,
    backbone: str,
    batch_prompt: Tensor,
    fc_pref: Tensor,
    fc_blk: Tensor,
    tok_blk: Tensor,
    vl: Tensor,
    bs: int,
    p_len: int,
    adv_len: int,
    s_max: int,
    warmup_steps: int,
    timed_steps: int,
    timer: _DeviceStepTimer,
    device: str,
) -> list[float]:
    """Time steady-state batched ``swa`` or ``hybrid_1swa_2kvn`` proposer steps."""
    swa_st = _init_swa_cache_state(drafter, bsz=bs, device=device)
    init_pos = (
        torch.arange(p_len, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
    )
    anc_pos = torch.full((bs,), p_len, dtype=torch.long, device=device)
    sc: Tensor | None = None
    if backbone == "hybrid_1swa_2kvn":
        sc = drafter.allocate_state_cache(batch_size=bs, device=device)
        _ = _prefill_kvn_layers_parallel(drafter, batch_prompt, fc_pref, sc)
        _ = _draft_step_hybrid_batched(
            drafter,
            fc_blk,
            tok_blk,
            vl,
            sc,
            fc_pref,
            init_pos,
            anc_pos,
            swa_st,
            temperature=0.0,
            s_pad_to=s_max,
        )
    else:
        _ = _draft_step_swa_batched(
            drafter, fc_pref, init_pos, tok_blk[:, 0], anc_pos, swa_st, temperature=0.0
        )

    cur_pos = p_len
    step_fc = fc_blk[:, :adv_len, :]
    draft_ms_list: list[float] = []
    for step_i in range(warmup_steps + timed_steps):
        new_pos = (
            torch.arange(cur_pos, cur_pos + adv_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bs, adv_len)
        )
        cur_pos += adv_len
        anc_p = torch.full((bs,), cur_pos, dtype=torch.long, device=device)
        if step_i >= warmup_steps:
            timer.start()
        if sc is not None:
            _ = _draft_step_hybrid_batched(
                drafter,
                fc_blk,
                tok_blk,
                vl,
                sc,
                step_fc,
                new_pos,
                anc_p,
                swa_st,
                temperature=0.0,
                s_pad_to=s_max,
            )
        else:
            _ = _draft_step_swa_batched(
                drafter,
                step_fc,
                new_pos,
                tok_blk[:, 0],
                anc_p,
                swa_st,
                temperature=0.0,
            )
        if step_i >= warmup_steps:
            draft_ms_list.append(timer.stop_ms())
    return draft_ms_list


def _profile_batched_amdahl_tier(
    verifier: nn.Module,
    drafter: Any,
    sample_prompt: list[int],
    bs: int,
    mean_tau: float,
    aux_layer_ids: list[int],
    eval_dtype: torch.dtype,
    device: str,
    use_kraus_tree: bool,
    tree_budget: int,
    branch_width: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float]:
    """Profile batched baseline, verifier, and drafter latency at batch ``bs``."""
    block_size = int(drafter.config.block_size)
    sample_from_anchor = bool(drafter.config.sample_from_anchor)
    n_spec = (
        int(tree_budget)
        if use_kraus_tree
        else (block_size if sample_from_anchor else max(1, block_size - 1))
    )
    backbone = drafter.config.kvn_backbone_type
    p_len = len(sample_prompt)
    batch_prompt = torch.tensor([sample_prompt] * bs, dtype=torch.long, device=device)
    timer = _DeviceStepTimer(device)

    # 1. Steady-state batched 1-token autoregressive baseline
    pkv_base = DynamicCache()
    _ = verifier(input_ids=batch_prompt, past_key_values=pkv_base, use_cache=True)
    one_tok = batch_prompt[:, -1:]
    for _ in range(warmup_steps):
        _ = verifier(input_ids=one_tok, past_key_values=pkv_base, use_cache=True)
        pkv_base.crop(-1)
    base_ms_list: list[float] = []
    for _ in range(timed_steps):
        timer.start()
        _ = verifier(input_ids=one_tok, past_key_values=pkv_base, use_cache=True)
        base_ms_list.append(timer.stop_ms())
        pkv_base.crop(-1)
    base_step_ms = max(float(np.mean(base_ms_list)), 1e-4)
    bs_base_tps = (1000.0 * float(bs)) / base_step_ms

    # 2. Steady-state batched (1 + n_spec)-token verifier pass
    pkv_spec = DynamicCache()
    aux_pref, _, _, pkv_spec = _forward_verifier_with_speculator_hs(
        verifier, batch_prompt, aux_layer_ids, past_key_values=pkv_spec, use_cache=True
    )
    fc_pref = drafter.hidden_norm(drafter.fc(aux_pref.to(eval_dtype)))
    v_size = int(drafter.verifier_vocab_size)
    spec_in = torch.randint(0, v_size, (bs, 1 + n_spec), device=device)

    for _ in range(warmup_steps):
        _ = _forward_verifier_with_speculator_hs(
            verifier, spec_in, aux_layer_ids, past_key_values=pkv_spec, use_cache=True
        )
        pkv_spec.crop(-(1 + n_spec))
    ver_ms_list: list[float] = []
    for _ in range(timed_steps):
        timer.start()
        _ = _forward_verifier_with_speculator_hs(
            verifier, spec_in, aux_layer_ids, past_key_values=pkv_spec, use_cache=True
        )
        ver_ms_list.append(timer.stop_ms())
        pkv_spec.crop(-(1 + n_spec))
    verify_ms = max(float(np.mean(ver_ms_list)), 1e-4)

    # 3. True batched steady-state drafter pass
    s_max = block_size + 1
    adv_len = max(1, min(s_max, round(mean_tau)))
    vl = torch.full((bs,), adv_len, dtype=torch.long, device=device)
    fc_blk = torch.zeros(
        (bs, s_max, drafter.hidden_size), dtype=eval_dtype, device=device
    )
    tok_blk = torch.randint(0, v_size, (bs, s_max), device=device)

    if backbone == "kvn":
        draft_ms_list = _time_batched_kvn_drafter_tier(
            drafter=drafter,
            batch_prompt=batch_prompt,
            fc_pref=fc_pref,
            fc_blk=fc_blk,
            tok_blk=tok_blk,
            vl=vl,
            bs=bs,
            s_max=s_max,
            use_kraus_tree=use_kraus_tree,
            tree_budget=tree_budget,
            branch_width=branch_width,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            timer=timer,
            device=device,
        )
    else:
        draft_ms_list = _time_batched_swa_or_hybrid_tier(
            drafter=drafter,
            backbone=backbone,
            batch_prompt=batch_prompt,
            fc_pref=fc_pref,
            fc_blk=fc_blk,
            tok_blk=tok_blk,
            vl=vl,
            bs=bs,
            p_len=p_len,
            adv_len=adv_len,
            s_max=s_max,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            timer=timer,
            device=device,
        )

    draft_ms = max(float(np.mean(draft_ms_list)), 1e-4)
    step_s = max((draft_ms + verify_ms) / 1000.0, 1e-6)
    bs_spec_tps = (float(bs) * mean_tau) / step_s
    return compute_amdahl_e2e_metrics(
        baseline_tps=bs_base_tps,
        spec_tps=bs_spec_tps,
        draft_latency_ms=draft_ms,
        verify_latency_ms=verify_ms,
        mean_accept_len=mean_tau,
    )


def _generate_baseline_outputs(
    verifier: nn.Module,
    prompt_token_ids_list: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    eos_ids: set[int],
    device: str,
) -> tuple[list[list[int]], int, float]:
    """Run standard 1-token autoregressive target decoding with EOS stopping."""
    _sync_if_cuda(device)
    t_start = time.perf_counter()
    outputs: list[list[int]] = []
    total_tokens = 0
    for p_ids in prompt_token_ids_list:
        inp = torch.tensor([p_ids], dtype=torch.long, device=device)
        pkv = DynamicCache()
        out = verifier(input_ids=inp, past_key_values=pkv, use_cache=True)
        gen: list[int] = []
        next_logits = out.logits[:, -1, :]
        for _ in range(max_new_tokens):
            nt = _sample_single_token(next_logits[0], temperature)
            gen.append(nt)
            if nt in eos_ids or len(gen) >= max_new_tokens:
                break
            out = verifier(
                input_ids=torch.tensor([[nt]], dtype=torch.long, device=device),
                past_key_values=pkv,
                use_cache=True,
            )
            next_logits = out.logits[:, -1, :]
        outputs.append(gen)
        total_tokens += len(gen)
    _sync_if_cuda(device)
    elapsed = max(time.perf_counter() - t_start, 1e-6)
    return outputs, total_tokens, total_tokens / elapsed


def _verify_chain_logits(
    v_logits_0: Tensor,
    draft_ids_0: Tensor,
    draft_q_probs: Tensor,
    n_chain_spec: int,
    temperature: float,
    device: str,
) -> tuple[int, int]:
    """Verify linear speculative chain at ``T=0`` (greedy) or ``T>0`` (lossless)."""
    if temperature <= 0.0:
        target_greedy = v_logits_0.argmax(dim=-1)
        n_acc = 0
        for k in range(n_chain_spec):
            if int(draft_ids_0[k].item()) == int(target_greedy[k].item()):
                n_acc += 1
            else:
                break
        return n_acc, int(target_greedy[n_acc].item())

    p_probs = fn.softmax(v_logits_0.float() / temperature, dim=-1)
    n_acc = 0
    for k in range(n_chain_spec):
        d_tok = int(draft_ids_0[k].item())
        p_d = float(p_probs[k, d_tok].item())
        q_d = float(draft_q_probs[0, k, d_tok].item())
        acc_p = min(1.0, p_d / max(q_d, _PROB_EPS)) if q_d > 0.0 else 0.0
        if float(torch.rand((), device=device).item()) < acc_p:
            n_acc += 1
            continue
        resid = (p_probs[k] - draft_q_probs[0, k]).clamp(min=0.0)
        resid_sum = float(resid.sum().item())
        resid = resid / resid_sum if resid_sum > _PROB_EPS else p_probs[k]
        return n_acc, int(torch.multinomial(resid, num_samples=1).item())

    bonus = int(torch.multinomial(p_probs[n_chain_spec], num_samples=1).item())
    return n_acc, bonus


def _execute_kraus_tree_decode_step(
    verifier: nn.Module,
    drafter: Any,
    pkv: Any,
    cur_prev_fc: Tensor,
    cur_tok_chunk: Tensor,
    vlen: Tensor,
    state_cache: Tensor,
    anchor_tok: Tensor,
    kv_len_before: int,
    aux_layer_ids: list[int],
    eval_dtype: torch.dtype,
    device: str,
    tree_budget: int,
    branch_width: int,
    temperature: float,
    s_pad: int,
    step_timer: _DeviceStepTimer,
) -> tuple[list[int], int, Tensor, float, float]:
    """Draft + SpecInfer-verify + compact KV cache for one Kraus POVM tree step."""
    step_timer.start()
    tree_dict = _draft_step_kraus_tree(
        drafter=drafter,
        prev_fc_chunk=cur_prev_fc,
        token_ids_chunk=cur_tok_chunk,
        valid_lens=vlen,
        state_cache=state_cache,
        tree_budget=tree_budget,
        branch_width=branch_width,
        temperature=temperature,
        s_pad_to=s_pad,
    )
    draft_ms = step_timer.stop_ms()

    tree_ids = tree_dict["tree_draft_ids"]
    tree_parents = tree_dict["tree_parent_idx"]
    tree_mask, tree_depths = build_tree_attention_mask(
        tree_parent_idx=tree_parents,
        prefix_len=kv_len_before,
        dtype=eval_dtype,
        tree_attn_mask=tree_dict.get("tree_attn_mask"),
    )
    verify_in = torch.cat([anchor_tok, tree_ids], dim=1)
    root_pos = torch.tensor([[kv_len_before]], dtype=torch.long, device=device)
    pos_ids = torch.cat([root_pos, kv_len_before + tree_depths], dim=1)

    step_timer.start()
    v_aux, _, v_logits, _ = _forward_verifier_with_speculator_hs(
        verifier,
        verify_in,
        aux_layer_ids,
        past_key_values=pkv,
        use_cache=True,
        attention_mask=tree_mask,
        position_ids=pos_ids,
    )
    verify_ms = step_timer.stop_ms()

    has_stoch_q = bool(
        tree_dict.get(
            "stochastic_proposals",
            temperature > 0.0 and tree_dict.get("tree_q_probs") is not None,
        )
    )
    tree_res = specinfer_verify_tree_step(
        verifier_logits=v_logits,
        tree_draft_ids=tree_ids,
        tree_parent_idx=tree_parents,
        temperature=temperature,
        tree_q_probs=tree_dict.get("tree_q_probs"),
        d2t=drafter.d2t if drafter.use_draft_vocab else None,
        stochastic_proposals=has_stoch_q,
    )
    accepted_slice = tree_res.accepted_tokens[0]
    acc_node_idx = tree_res.accepted_node_indices[0]
    bonus_tok = int(tree_res.bonus_tokens[0].item())
    _ = compact_tree_kv_cache(pkv, kv_len_before, [acc_node_idx])
    gather_aux = torch.tensor(
        [0, *(1 + i for i in acc_node_idx)], dtype=torch.long, device=device
    )
    confirmed_v_aux = v_aux.index_select(dim=1, index=gather_aux).to(eval_dtype)
    return accepted_slice, bonus_tok, confirmed_v_aux, draft_ms, verify_ms


def _execute_chain_decode_step(
    verifier: nn.Module,
    drafter: Any,
    backbone: str,
    pkv: Any,
    cur_prev_fc: Tensor,
    cur_tok_chunk: Tensor,
    vlen: Tensor,
    state_cache: Tensor | None,
    swa_state: _SwaIncrementalCacheState | None,
    pending_swa_fc: Tensor,
    pending_swa_pos: Tensor,
    cur_anchor_pos: int,
    anchor_tok: Tensor,
    start_k: int,
    n_chain_spec: int,
    aux_layer_ids: list[int],
    eval_dtype: torch.dtype,
    device: str,
    temperature: float,
    s_pad: int,
    step_timer: _DeviceStepTimer,
    prev_context_ids: Tensor | None = None,
) -> tuple[list[int], int, Tensor, float, float]:
    """Draft + lossless-verify + crop KV cache for one linear chain step."""
    step_timer.start()
    anc_pos_t = torch.tensor([cur_anchor_pos], dtype=torch.long, device=device)
    if backbone == "kvn" and state_cache is not None:
        raw_draft_ids, raw_q_probs = _draft_step_kvn_recurrent(
            drafter=drafter,
            prev_fc_chunk=cur_prev_fc,
            token_ids_chunk=cur_tok_chunk,
            valid_lens=vlen,
            state_cache=state_cache,
            temperature=temperature,
            s_pad_to=s_pad,
            prev_context_ids=prev_context_ids,
        )
    elif (
        backbone == "hybrid_1swa_2kvn"
        and state_cache is not None
        and swa_state is not None
    ):
        raw_draft_ids, raw_q_probs = _draft_step_hybrid_batched(
            drafter=drafter,
            prev_fc_chunk=cur_prev_fc,
            token_ids_chunk=cur_tok_chunk,
            valid_lens=vlen,
            state_cache=state_cache,
            new_swa_fc=pending_swa_fc,
            new_swa_pos=pending_swa_pos,
            anchor_pos=anc_pos_t,
            swa_state=swa_state,
            temperature=temperature,
            s_pad_to=s_pad,
            prev_context_ids=prev_context_ids,
        )
    else:
        if swa_state is None:
            raise RuntimeError("SWA cache state not initialized.")
        raw_draft_ids, raw_q_probs = _draft_step_swa_batched(
            drafter=drafter,
            new_target_fc=pending_swa_fc,
            new_target_pos=pending_swa_pos,
            anchor_token_ids=anchor_tok.squeeze(1),
            anchor_pos=anc_pos_t,
            swa_state=swa_state,
            temperature=temperature,
            prev_context_ids=prev_context_ids,
        )
    draft_ms = step_timer.stop_ms()

    draft_ids = raw_draft_ids[:, start_k : start_k + n_chain_spec]
    draft_q_probs = (
        raw_q_probs[:, start_k : start_k + n_chain_spec]
        if raw_q_probs.numel() > 0
        else raw_q_probs
    )
    verify_in = torch.cat([anchor_tok, draft_ids], dim=1)

    step_timer.start()
    v_aux, _, v_logits, _ = _forward_verifier_with_speculator_hs(
        verifier, verify_in, aux_layer_ids, past_key_values=pkv, use_cache=True
    )
    verify_ms = step_timer.stop_ms()

    n_acc, bonus_tok = _verify_chain_logits(
        v_logits[0], draft_ids[0], draft_q_probs, n_chain_spec, temperature, device
    )
    accepted_slice = draft_ids[0, :n_acc].tolist()
    rejected_count = n_chain_spec - n_acc
    if rejected_count > 0:
        pkv.crop(-rejected_count)
    confirmed_v_aux = v_aux[:, : 1 + n_acc, :].to(eval_dtype)
    return accepted_slice, bonus_tok, confirmed_v_aux, draft_ms, verify_ms


def _run_speculative_prompt(
    verifier: nn.Module,
    drafter: Any,
    p_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    eos_ids: set[int],
    aux_layer_ids: list[int],
    eval_dtype: torch.dtype,
    device: str,
    use_kraus_tree: bool,
    tree_budget: int,
    branch_width: int,
    step_timer: _DeviceStepTimer,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Run lossless speculative decoding on one prompt with O(P log P) prefill & EOS."""
    block_size = int(drafter.config.block_size)
    sample_from_anchor = bool(drafter.config.sample_from_anchor)
    start_k = 0 if sample_from_anchor else 1
    n_chain_spec = block_size if sample_from_anchor else max(1, block_size - 1)
    backbone = drafter.config.kvn_backbone_type

    prompt_t = torch.tensor([p_ids], dtype=torch.long, device=device)
    p_len = prompt_t.size(1)
    pkv = DynamicCache()

    aux_prefill, _, prefill_logits, pkv = _forward_verifier_with_speculator_hs(
        verifier, prompt_t, aux_layer_ids, past_key_values=pkv, use_cache=True
    )
    fc_prefill = drafter.hidden_norm(drafter.fc(aux_prefill.to(eval_dtype)))
    y_first = _sample_single_token(prefill_logits[0, -1, :], temperature)
    spec_gen: list[int] = [y_first]
    if y_first in eos_ids or max_new_tokens <= 1:
        return spec_gen, [], [], []

    state_cache: Tensor | None = None
    swa_state: _SwaIncrementalCacheState | None = None
    cur_prev_fc = fc_prefill[:, -1:, :]
    cur_tok_chunk = torch.tensor([[y_first]], dtype=torch.long, device=device)
    pending_swa_fc = fc_prefill
    pending_swa_pos = torch.arange(p_len, device=device, dtype=torch.long).unsqueeze(0)
    cur_anchor_pos = p_len

    if backbone in ("kvn", "hybrid_1swa_2kvn"):
        state_cache = drafter.allocate_state_cache(batch_size=1, device=device)
        cur_prev_fc = _prefill_kvn_layers_parallel(
            drafter, prompt_t, fc_prefill, state_cache
        )
    if backbone in ("swa", "hybrid_1swa_2kvn"):
        swa_state = _init_swa_cache_state(drafter, bsz=1, device=device)

    taus: list[float] = []
    draft_times: list[float] = []
    verify_times: list[float] = []
    hit_eos = False

    while len(spec_gen) < max_new_tokens and not hit_eos:
        anchor_tok = cur_tok_chunk[:, -1:]
        kv_len_before = pkv.get_seq_length()
        vlen = torch.tensor([cur_tok_chunk.size(1)], dtype=torch.long, device=device)
        s_pad = max(cur_tok_chunk.size(1), block_size + 1)

        if use_kraus_tree and backbone == "kvn" and state_cache is not None:
            acc_slice, bonus_tok, confirmed_v_aux, d_ms, v_ms = (
                _execute_kraus_tree_decode_step(
                    verifier=verifier,
                    drafter=drafter,
                    pkv=pkv,
                    cur_prev_fc=cur_prev_fc,
                    cur_tok_chunk=cur_tok_chunk,
                    vlen=vlen,
                    state_cache=state_cache,
                    anchor_tok=anchor_tok,
                    kv_len_before=kv_len_before,
                    aux_layer_ids=aux_layer_ids,
                    eval_dtype=eval_dtype,
                    device=device,
                    tree_budget=tree_budget,
                    branch_width=branch_width,
                    temperature=temperature,
                    s_pad=s_pad,
                    step_timer=step_timer,
                )
            )
        else:
            prior_ctx = (p_ids + spec_gen[:-1])[-2:]
            prev_ctx_t = (
                torch.tensor([prior_ctx], dtype=torch.long, device=device)
                if prior_ctx
                else None
            )
            acc_slice, bonus_tok, confirmed_v_aux, d_ms, v_ms = (
                _execute_chain_decode_step(
                    verifier=verifier,
                    drafter=drafter,
                    backbone=backbone,
                    pkv=pkv,
                    cur_prev_fc=cur_prev_fc,
                    cur_tok_chunk=cur_tok_chunk,
                    vlen=vlen,
                    state_cache=state_cache,
                    swa_state=swa_state,
                    pending_swa_fc=pending_swa_fc,
                    pending_swa_pos=pending_swa_pos,
                    cur_anchor_pos=cur_anchor_pos,
                    anchor_tok=anchor_tok,
                    start_k=start_k,
                    n_chain_spec=n_chain_spec,
                    aux_layer_ids=aux_layer_ids,
                    eval_dtype=eval_dtype,
                    device=device,
                    temperature=temperature,
                    s_pad=s_pad,
                    step_timer=step_timer,
                    prev_context_ids=prev_ctx_t,
                )
            )

        draft_times.append(d_ms)
        verify_times.append(v_ms)
        taus.append(float(1 + len(acc_slice)))
        emitted = [*acc_slice, bonus_tok]

        for tok in emitted:
            if len(spec_gen) >= max_new_tokens:
                break
            spec_gen.append(tok)
            if tok in eos_ids:
                hit_eos = True
                break
        if hit_eos or len(spec_gen) >= max_new_tokens:
            break

        confirmed_fc = drafter.hidden_norm(drafter.fc(confirmed_v_aux))
        cur_prev_fc = confirmed_fc
        cur_tok_chunk = torch.tensor([emitted], dtype=torch.long, device=device)
        pending_swa_fc = confirmed_fc
        pending_swa_pos = torch.arange(
            cur_anchor_pos,
            cur_anchor_pos + len(emitted),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        cur_anchor_pos += len(emitted)

    return spec_gen, taus, draft_times, verify_times


@torch.no_grad()
def evaluate_checkpoint_e2e(
    checkpoint_dir: Path,
    verifier_model: str,
    prompt_token_ids_list: list[list[int]] | None = None,
    benchmark: str = "gsm8k",
    num_prompts: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    batch_sizes: list[int] | None = None,
    data_dir: Path | None = None,
    draft_vocab_size: int | None = None,
    device: str = "cuda",
    *,
    use_kraus_tree: bool = False,
    tree_budget: int = 24,
    branch_width: int = 3,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    """Run lossless speculative decoding + batched Amdahl speedup profiling."""
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    eval_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    drafter = _load_drafter_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        verifier_model=verifier_model,
        data_dir=data_dir,
        draft_vocab_size=draft_vocab_size,
        device=device,
        eval_dtype=eval_dtype,
    )
    verifier = (
        AutoModelForCausalLM.from_pretrained(verifier_model, dtype=eval_dtype)
        .to(device)
        .eval()
    )
    aux_layer_ids = list(drafter.config.aux_hidden_state_layer_ids or [2, 18, 33])
    backbone = drafter.config.kvn_backbone_type
    eos_ids = set() if ignore_eos else _resolve_eos_token_ids(verifier)

    if prompt_token_ids_list is None:
        tokenizer = AutoTokenizer.from_pretrained(verifier_model)
        prompt_token_ids_list = _load_benchmark_prompts(
            benchmark=benchmark, tokenizer=tokenizer, num_prompts=num_prompts
        )

    _run_warmup_passes(
        verifier=verifier,
        drafter=drafter,
        sample_prompt=prompt_token_ids_list[0],
        aux_layer_ids=aux_layer_ids,
        eval_dtype=eval_dtype,
        device=device,
        warmup_iters=5 if is_cuda else 1,
    )

    baseline_outputs, _, base_tps_b1 = _generate_baseline_outputs(
        verifier=verifier,
        prompt_token_ids_list=prompt_token_ids_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_ids=eos_ids,
        device=device,
    )

    all_step_taus: list[float] = []
    exact_greedy_matches: list[bool] = []
    draft_step_times_ms: list[float] = []
    verify_step_times_ms: list[float] = []
    total_spec_tokens = 0
    step_timer = _DeviceStepTimer(device)

    _sync_if_cuda(device)
    t_spec_start = time.perf_counter()
    for p_idx, p_ids in enumerate(prompt_token_ids_list):
        spec_gen, taus, d_times, v_times = _run_speculative_prompt(
            verifier=verifier,
            drafter=drafter,
            p_ids=p_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_ids=eos_ids,
            aux_layer_ids=aux_layer_ids,
            eval_dtype=eval_dtype,
            device=device,
            use_kraus_tree=use_kraus_tree,
            tree_budget=tree_budget,
            branch_width=branch_width,
            step_timer=step_timer,
        )
        total_spec_tokens += len(spec_gen)
        all_step_taus.extend(taus)
        draft_step_times_ms.extend(d_times)
        verify_step_times_ms.extend(v_times)
        if temperature <= 0.0:
            exact_greedy_matches.append(spec_gen == baseline_outputs[p_idx])

    _sync_if_cuda(device)
    t_spec_elapsed = max(time.perf_counter() - t_spec_start, 1e-6)
    spec_tps_b1 = total_spec_tokens / t_spec_elapsed
    mean_tau = float(np.mean(all_step_taus)) if all_step_taus else 1.0
    mean_draft_ms_b1 = (
        float(np.mean(draft_step_times_ms)) if draft_step_times_ms else 0.01
    )
    mean_verify_ms_b1 = max(
        float(np.mean(verify_step_times_ms)) if verify_step_times_ms else 0.1, 1e-4
    )

    # Steady-state batched Amdahl throughput sweep (true batched kernels, >=5 warmups)
    eval_batches = batch_sizes or [1]
    amdahl_by_bs: dict[str, dict[str, float]] = {}
    sample_prompt = prompt_token_ids_list[0]
    tier_warmup = 5 if is_cuda else 1
    tier_steps = 8 if is_cuda else 2

    for bs in eval_batches:
        if bs == 1:
            amdahl_by_bs["1"] = compute_amdahl_e2e_metrics(
                baseline_tps=base_tps_b1,
                spec_tps=spec_tps_b1,
                draft_latency_ms=mean_draft_ms_b1,
                verify_latency_ms=mean_verify_ms_b1,
                mean_accept_len=mean_tau,
            )
            continue

        amdahl_by_bs[str(bs)] = _profile_batched_amdahl_tier(
            verifier=verifier,
            drafter=drafter,
            sample_prompt=sample_prompt,
            bs=int(bs),
            mean_tau=mean_tau,
            aux_layer_ids=aux_layer_ids,
            eval_dtype=eval_dtype,
            device=device,
            use_kraus_tree=use_kraus_tree,
            tree_budget=tree_budget,
            branch_width=branch_width,
            warmup_steps=tier_warmup,
            timed_steps=tier_steps,
        )

    return {
        "checkpoint": str(checkpoint_dir),
        "backbone_type": backbone,
        "use_kraus_tree": bool(use_kraus_tree),
        "tree_budget": int(tree_budget) if use_kraus_tree else None,
        "state_bytes_per_seq": drafter.config.estimated_state_bytes_per_seq(),
        "benchmark": benchmark,
        "temperature": float(temperature),
        "num_prompts": len(prompt_token_ids_list),
        "max_new_tokens": max_new_tokens,
        "mean_accept_len_tau": round(mean_tau, 4),
        "exact_greedy_match_all_prompts": (
            all(exact_greedy_matches) if temperature <= 0.0 else None
        ),
        "amdahl_by_batch_size": amdahl_by_bs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="KvN-DSpark Confidence Calibration & End-to-End Amdahl Evaluator."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser(
        "calibrate",
        help="Evaluate full-vocab ECE, AUROC & prefix-survival Pareto curve.",
    )
    cal.add_argument("--checkpoint", type=Path, required=True)
    cal.add_argument("--verifier-model", type=str, required=True)
    cal.add_argument("--data-dir", type=Path, required=True)
    cal.add_argument("--draft-vocab-size", type=int, default=32768)
    cal.add_argument("--max-samples", type=int, default=64)
    cal.add_argument("--train-data-ratio", type=float, default=0.95)
    cal.add_argument("--output-json", type=Path, required=True)

    e2e = sub.add_parser(
        "benchmark-e2e",
        help="Run lossless E2E speculative decoding & batched Amdahl evaluation.",
    )
    e2e.add_argument("--checkpoint", type=Path, required=True)
    e2e.add_argument("--verifier-model", type=str, required=True)
    e2e.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        choices=list(E2E_BENCHMARK_SPECS),
    )
    e2e.add_argument("--num-prompts", type=int, default=50)
    e2e.add_argument("--max-new-tokens", type=int, default=256)
    e2e.add_argument("--temperature", type=float, default=0.0)
    e2e.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes for Amdahl throughput sweep (e.g. 1 4 8 16 32 64 128).",
    )
    e2e.add_argument("--data-dir", type=Path, default=None)
    e2e.add_argument("--draft-vocab-size", type=int, default=32768)
    e2e.add_argument(
        "--use-kraus-tree",
        action="store_true",
        help="Enable Axis 5 Kraus POVM multi-branch tree + lossless SpecInfer.",
    )
    e2e.add_argument("--tree-budget", type=int, default=24)
    e2e.add_argument("--branch-width", type=int, default=3)
    e2e.add_argument("--ignore-eos", action="store_true")
    e2e.add_argument("--output-json", type=Path, required=True)

    amd = sub.add_parser("amdahl", help="Compute end-to-end Amdahl speedup summary.")
    amd.add_argument("--baseline-tps", type=float, required=True)
    amd.add_argument("--spec-tps", type=float, required=True)
    amd.add_argument("--draft-latency-ms", type=float, required=True)
    amd.add_argument("--verify-latency-ms", type=float, required=True)
    amd.add_argument("--mean-accept-len", type=float, required=True)
    amd.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "calibrate":
        report = evaluate_checkpoint_calibration(
            checkpoint_dir=args.checkpoint,
            verifier_model=args.verifier_model,
            data_dir=args.data_dir,
            draft_vocab_size=args.draft_vocab_size,
            max_samples=args.max_samples,
            train_data_ratio=args.train_data_ratio,
            device=dev,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark-e2e":
        report = evaluate_checkpoint_e2e(
            checkpoint_dir=args.checkpoint,
            verifier_model=args.verifier_model,
            benchmark=args.benchmark,
            num_prompts=args.num_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_sizes=list(args.batch_sizes),
            data_dir=args.data_dir,
            draft_vocab_size=args.draft_vocab_size,
            device=dev,
            use_kraus_tree=bool(args.use_kraus_tree),
            tree_budget=int(args.tree_budget),
            branch_width=int(args.branch_width),
            ignore_eos=bool(args.ignore_eos),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    else:
        summary = compute_amdahl_e2e_metrics(
            baseline_tps=args.baseline_tps,
            spec_tps=args.spec_tps,
            draft_latency_ms=args.draft_latency_ms,
            verify_latency_ms=args.verify_latency_ms,
            mean_accept_len=args.mean_accept_len,
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
