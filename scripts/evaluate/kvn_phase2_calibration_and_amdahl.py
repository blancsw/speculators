#!/usr/bin/env python3
"""Confidence Calibration (ECE, AUROC, Dynamic Stopping Pareto) & End-to-End Amdahl Speculative Evaluator.

Implements Phase D.1, D.2 & D.3 of ``kvn_rff_rnn_mtp_train_phase_2.md``:
1. ``compute_ece``: 15-bin Expected Calibration Error on speculative token acceptance probabilities.
2. ``compute_binary_auroc``: Exact rank-based AUROC for rejection prediction.
3. ``compute_dynamic_stopping_pareto``: Sweeps confidence stopping threshold
   ``gamma_conf in {0.0, 0.2, 0.35, 0.5, 0.65, 0.8}`` to measure expected acceptance length ``tau(gamma)``
   vs verifier speculative slot FLOPs saved.
4. ``evaluate_checkpoint_calibration``: Offline OR online-Arrow validation audit (automatically
   extracts exact pre-final-norm verifier hidden states from ``ArrowDataset(split="val")`` when
   ``data_dir/hidden_states`` is absent after ``--on-missing generate`` online training).
5. ``evaluate_checkpoint_e2e`` (``benchmark-e2e``): Lossless autoregressive speculative decoding
   (greedy ``T=0.0`` + exact Leviathan/Chen rejection sampling ``T>0``) and batched Amdahl throughput
   profiler across ``B in {1, 4, 8, 16, 32, 64, 128}`` for ``kvn``, ``hybrid_1swa_2kvn``, and ``swa``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8)

E2E_BENCHMARK_SPECS: dict[str, tuple[str, str | None, str, str]] = {
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", None, "train", "prompt"),
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test", "prompt"),
}


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) over ``num_bins`` equal-width probability bins."""
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
        if idx == 0:
            in_bin = (p >= low) & (p <= high)
        else:
            in_bin = (p > low) & (p <= high)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        avg_conf = p[in_bin].mean()
        avg_acc = y[in_bin].mean()
        ece = ece + (count / n_total) * torch.abs(avg_acc - avg_conf)

    return float(ece.item())


def compute_binary_auroc(
    scores: torch.Tensor,
    labels: torch.Tensor,
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
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg_rank
        i = j

    pos_rank_sum = ranks[sorted_y == 1].sum().item()
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def compute_dynamic_stopping_pareto(
    confidences: torch.Tensor,
    accepted_flags: torch.Tensor,
    thresholds: list[float] | tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    """Evaluate dynamic block truncation at ``c_{t,k} < gamma_conf``."""
    conf = confidences.detach().to(torch.float32)
    acc = accepted_flags.detach().to(torch.bool)
    if conf.ndim != 2 or conf.shape != acc.shape:
        raise ValueError(
            f"Expected matching 2D shapes [N_blocks, K], got {tuple(conf.shape)} vs {tuple(acc.shape)}"
        )
    _, block_size = conf.shape
    results: list[dict[str, float]] = []

    for gamma in thresholds:
        if gamma <= 0.0:
            keep_mask = torch.ones_like(acc, dtype=torch.bool)
        else:
            keep_mask = torch.cumprod((conf >= gamma).to(torch.long), dim=-1).to(
                torch.bool
            )

        effective_draft_lens = keep_mask.sum(dim=-1).to(torch.float32)
        contiguous_accept = torch.cumprod((keep_mask & acc).to(torch.long), dim=-1)
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
        raise ValueError("baseline_tps and verify_latency_ms must be strictly positive.")
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


def _get_backbone_layers(verifier: torch.nn.Module) -> torch.nn.ModuleList:
    inner = getattr(verifier, "model", verifier)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers  # type: ignore[return-value]


@torch.no_grad()
def _forward_verifier_with_speculator_hs(
    verifier: torch.nn.Module,
    input_ids: torch.Tensor,
    aux_layer_ids: list[int],
    past_key_values: Any = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Run ``verifier`` and capture exact ``vLLM``-convention ``(aux_hs, pre_norm_final_hs, logits, pkv)``.

    In ``speculators`` / ``vLLM``:
    - ``layer_id == 0`` is ``embed_tokens`` output (``out.hidden_states[0]``).
    - ``1 <= layer_id < num_layers`` is ``layers[layer_id - 1]`` output (``out.hidden_states[layer_id]``, pre-norm).
    - ``layer_id == num_layers`` (and ``verifier_last_hidden_states``) is ``layers[-1]`` output **before**
      ``verifier.model.norm`` (captured via forward hook because HF ``out.hidden_states[-1]`` is post-norm).
    """
    layers = _get_backbone_layers(verifier)
    num_layers = len(layers)
    raw_last_holder: list[torch.Tensor] = []

    def _hook(_mod: torch.nn.Module, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        raw_last_holder.append(h)

    handle = layers[-1].register_forward_hook(_hook)
    try:
        out = verifier(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
    finally:
        handle.remove()

    pre_norm_last = raw_last_holder[-1]
    aux_parts: list[torch.Tensor] = []
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
):
    from speculators.models.kvn_dspark import KvNDSparkDraftModel

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
) -> dict:
    """Run calibration audit on offline ``hidden_states/*.safetensors`` OR online ``ArrowDataset``."""
    from transformers import AutoModelForCausalLM

    from speculators.train.data import ArrowDataset

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

    hs_dir = data_dir / "hidden_states"
    hs_files = (
        sorted(hs_dir.glob("hs_*.safetensors"))[:max_samples]
        if hs_dir.is_dir()
        else []
    )

    eval_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if hs_files:
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
    else:
        resolved_verifier = verifier_model or (
            drafter.config.speculators_config.verifier.name_or_path
            if drafter.config.speculators_config
            else None
        )
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

    all_confs: list[torch.Tensor] = []
    all_accs: list[torch.Tensor] = []
    block_size = drafter.config.block_size

    for aux_hs, final_hs, toks, loss_mask in eval_samples:
        seq_len = toks.shape[0]
        if seq_len <= block_size + 4:
            continue
        loss_mask = loss_mask.clone()
        loss_mask[:, seq_len - block_size :] = 0
        if int((loss_mask > 0).sum().item()) == 0:
            continue
        doc_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

        n_anc = min(
            max_anchors_per_seq,
            max(1, int((loss_mask > 0).sum().item()) // block_size),
        )
        (
            hidden,
            base_logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            born_features,
        ) = drafter._kvn_backbone_forward(  # noqa: SLF001
            hidden_states=aux_hs,
            input_ids=toks.unsqueeze(0),
            loss_mask=loss_mask,
            verifier_last_hidden_states=final_hs,
            document_ids=doc_ids,
            max_anchors=n_anc,
        )

        anchor_positions = anchored_block_indices[::block_size]
        anchor_token_ids = toks[anchor_positions]
        draft_target_ids, conf_probs = drafter._decode_draft_block_greedy(  # noqa: SLF001
            hidden_blocks=hidden.reshape(n_anc, block_size, -1),
            base_logits=base_logits.reshape(n_anc, block_size, -1),
            anchor_token_ids=anchor_token_ids,
            born_features=born_features,
        )
        expected_ids = targets.argmax(dim=-1).reshape(n_anc, block_size)
        if drafter.use_draft_vocab and drafter.d2t is not None:
            expected_ids = expected_ids + drafter.d2t[expected_ids]

        valid_blocks = (
            aligned_loss_mask.reshape(n_anc, block_size).sum(dim=-1) == block_size
        )
        if not bool(valid_blocks.any().item()):
            continue

        matched = (draft_target_ids == expected_ids)[valid_blocks]
        all_confs.append(conf_probs[valid_blocks].to(torch.float32).cpu())
        all_accs.append(matched.cpu())

    if not all_confs:
        raise RuntimeError(f"No valid evaluation blocks found in {data_dir}")

    confs_cat = torch.cat(all_confs, dim=0)
    accs_cat = torch.cat(all_accs, dim=0)
    rej_scores = 1.0 - confs_cat
    rej_labels = (~accs_cat).to(torch.long)

    per_pos = {}
    for k in range(block_size):
        per_pos[f"pos_{k}"] = {
            "accept_rate_pct": round(
                100.0 * float(accs_cat[:, k].to(torch.float32).mean().item()), 2
            ),
            "ece": round(
                compute_ece(confs_cat[:, k], accs_cat[:, k].to(torch.float32)), 4
            ),
            "rejection_auroc": round(
                compute_binary_auroc(rej_scores[:, k], rej_labels[:, k]), 4
            ),
        }

    return {
        "checkpoint": str(checkpoint_dir),
        "source": "offline_hidden_states" if hs_files else "online_arrow_val",
        "state_bytes_per_seq": drafter.config.estimated_state_bytes_per_seq(),
        "num_evaluated_blocks": int(confs_cat.shape[0]),
        "overall_ece_15bin": round(
            compute_ece(confs_cat, accs_cat.to(torch.float32), num_bins=15), 4
        ),
        "overall_rejection_auroc": round(
            compute_binary_auroc(rej_scores, rej_labels), 4
        ),
        "per_position": per_pos,
        "dynamic_stopping_pareto": compute_dynamic_stopping_pareto(confs_cat, accs_cat),
    }


def _sample_draft_block_with_probs(
    drafter: Any,
    hidden_blocks: torch.Tensor,  # [B, K, D]
    base_logits: torch.Tensor,  # [B, K, V_draft]
    anchor_token_ids: torch.Tensor,  # [B]
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``K`` draft tokens autoregressively via ``MarkovHead`` and return ``(target_ids, draft_probs_target_vocab)``."""
    bsz, block_size, _ = hidden_blocks.shape
    cur_prev_id = anchor_token_ids.long()
    rnn_state = (
        hidden_blocks.new_zeros(bsz, drafter.markov_head.markov_rank)
        if (drafter.markov_head is not None and drafter.markov_head.head_type == "rnn")
        else None
    )

    drafted_ids: list[torch.Tensor] = []
    target_probs_per_k: list[torch.Tensor] = []
    v_target = int(drafter.verifier_vocab_size)

    for k in range(block_size):
        logits_k = base_logits[:, k, :].float()
        if drafter.markov_head is not None:
            p_emb_k = drafter.markov_head.prev_embeddings(cur_prev_id)
            w_dtype = drafter.markov_head.markov_w2.weight.dtype
            p_emb_cast = p_emb_k.to(w_dtype)
            h_k = hidden_blocks[:, k, :].to(w_dtype)
            if drafter.markov_head.head_type == "vanilla":
                bias_k = drafter.markov_head.markov_w2(p_emb_cast)
            elif drafter.markov_head.head_type == "gated":
                gate = torch.sigmoid(
                    drafter.markov_head.gate_proj(torch.cat([h_k, p_emb_cast], dim=-1))
                )
                bias_k = drafter.markov_head.markov_w2(gate * p_emb_cast)
            else:
                assert rnn_state is not None
                z = torch.cat([rnn_state.to(w_dtype), p_emb_cast, h_k], dim=-1)
                g_raw, c_raw, o_raw = drafter.markov_head.joint_proj(z).chunk(3, dim=-1)
                gate = torch.sigmoid(g_raw)
                rnn_state = gate * rnn_state.to(w_dtype) + (1.0 - gate) * torch.tanh(
                    c_raw
                )
                bias_k = drafter.markov_head.markov_w2(torch.tanh(o_raw))
            logits_k = logits_k + bias_k.float()

        if temperature <= 0.0:
            draft_idx_k = logits_k.argmax(dim=-1)
            q_draft_k = F.one_hot(draft_idx_k, num_classes=logits_k.size(-1)).float()
        else:
            q_draft_k = F.softmax(logits_k / temperature, dim=-1)
            draft_idx_k = torch.multinomial(q_draft_k, num_samples=1).squeeze(-1)

        if drafter.use_draft_vocab and drafter.d2t is not None:
            target_id_k = draft_idx_k + drafter.d2t[draft_idx_k]
            draft_all_idx = torch.arange(logits_k.size(-1), device=logits_k.device)
            mapped_all = draft_all_idx + drafter.d2t[draft_all_idx]
            q_target_k = torch.zeros(
                (bsz, v_target), dtype=torch.float32, device=logits_k.device
            )
            q_target_k.scatter_(
                1, mapped_all.unsqueeze(0).expand(bsz, -1), q_draft_k
            )
        else:
            target_id_k = draft_idx_k
            q_target_k = q_draft_k

        drafted_ids.append(target_id_k)
        target_probs_per_k.append(q_target_k)
        if drafter.config.sample_from_anchor:
            cur_prev_id = target_id_k
        else:
            cur_prev_id = anchor_token_ids.long() if k == 0 else target_id_k

    return torch.stack(drafted_ids, dim=1), torch.stack(target_probs_per_k, dim=1)


def _draft_step_non_kvn_window(
    drafter: Any,
    all_aux_hs: torch.Tensor,  # [1, a, n_aux * D] (strictly before anchor a)
    all_tokens: torch.Tensor,  # [1, a + 1] (ending at anchor token a)
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal SWA / Hybrid block proposal at the final position ``a`` of ``all_tokens``."""
    block_size = drafter.config.block_size
    win = int(getattr(drafter.config.transformer_layer_config, "sliding_window", 128) or 128)
    # Pad trailing K-1 dummy positions so get_base_indices_for_anchored_blocks stays in bounds;
    # causal mask kv_base_pos < q_anchor strictly forbids reading position >= a!
    pad_aux = torch.zeros(
        (1, block_size, all_aux_hs.size(-1)),
        dtype=all_aux_hs.dtype,
        device=all_aux_hs.device,
    )
    full_aux = torch.cat([all_aux_hs, pad_aux], dim=1)  # [1, a + K, n_aux * D]
    pad_toks = torch.zeros(
        (1, block_size - 1), dtype=torch.long, device=all_tokens.device
    )
    full_toks = torch.cat([all_tokens, pad_toks], dim=1)  # [1, a + K]

    start = max(0, all_tokens.size(1) - 1 - (win + 4))
    win_aux = full_aux[:, start:, :]
    win_toks = full_toks[:, start:]
    anchor_local = all_tokens.size(1) - 1 - start
    seq_len = win_toks.size(1)

    loss_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=win_toks.device)
    loss_mask[0, anchor_local] = True
    dummy_final = torch.zeros(
        (1, seq_len, drafter.hidden_size),
        dtype=win_aux.dtype,
        device=win_aux.device,
    )
    doc_ids = torch.zeros((1, seq_len), dtype=torch.long, device=win_toks.device)

    hidden, base_logits, _, _, _, _ = drafter._kvn_backbone_forward(  # noqa: SLF001
        hidden_states=win_aux,
        input_ids=win_toks,
        loss_mask=loss_mask,
        verifier_last_hidden_states=dummy_final,
        document_ids=doc_ids,
        max_anchors=1,
    )
    return _sample_draft_block_with_probs(
        drafter=drafter,
        hidden_blocks=hidden.view(1, block_size, -1),
        base_logits=base_logits.view(1, block_size, -1),
        anchor_token_ids=all_tokens[:, -1],
        temperature=temperature,
    )


def _draft_step_kvn_recurrent(
    drafter: Any,
    prev_fc_chunk: torch.Tensor,  # [B, S_chunk, D]
    token_ids_chunk: torch.Tensor,  # [B, S_chunk]
    valid_lens: torch.Tensor,  # [B]
    state_cache: torch.Tensor,  # [L_kvn, 2, B, M]
    reset_seq_mask: torch.Tensor | None,
    temperature: float,
    s_pad_to: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run O(1)-state KvN block proposal at greedy (T=0) or stochastic (T>0) temperature."""
    bsz, cur_s, d_model = prev_fc_chunk.shape
    if cur_s < s_pad_to:
        pad_fc = prev_fc_chunk.new_zeros((bsz, s_pad_to - cur_s, d_model))
        pad_id = token_ids_chunk.new_zeros((bsz, s_pad_to - cur_s))
        prev_fc_chunk = torch.cat([prev_fc_chunk, pad_fc], dim=1)
        token_ids_chunk = torch.cat([token_ids_chunk, pad_id], dim=1)

    if temperature <= 0.0:
        draft_ids, _ = drafter.advance_and_draft_step(
            prev_fc_block=prev_fc_chunk,
            token_ids_block=token_ids_chunk,
            valid_lens=valid_lens,
            state_cache=state_cache,
            reset_seq_mask=reset_seq_mask,
        )
        return draft_ids, torch.empty(0, device=draft_ids.device)

    # Stochastic T > 0 path: update state identically and sample with exact q_k probabilities
    tok_emb = drafter.embed_tokens(token_ids_chunk)
    x_cur = prev_fc_chunk.to(tok_emb.dtype) + tok_emb
    s_max = token_ids_chunk.size(1)
    step_idx = torch.arange(s_max, device=x_cur.device).unsqueeze(0)
    active_mask = step_idx < valid_lens.to(x_cur.device).unsqueeze(1)

    new_states: list[torch.Tensor] = []
    q_anc_layers: list[torch.Tensor] = []
    p_anc_layers: list[torch.Tensor] = []
    dt_anc_layers: list[torch.Tensor] = []

    for l_idx, kvn_layer in enumerate(drafter.kvn_layers):
        q_s = state_cache[l_idx, 0]
        p_s = state_cache[l_idx, 1]
        if reset_seq_mask is not None:
            keep = (~reset_seq_mask.to(x_cur.device)).to(q_s.dtype).unsqueeze(-1)
            q_s = q_s * keep
            p_s = p_s * keep
        dtheta_anc = torch.zeros(
            (bsz, kvn_layer.n_modes), dtype=x_cur.dtype, device=x_cur.device
        )
        h_steps: list[torch.Tensor] = []
        for s in range(s_max):
            is_val = active_mask[:, s].unsqueeze(-1)
            is_anc = (valid_lens.to(x_cur.device) == (s + 1)).unsqueeze(-1)
            h_s, (q_c, p_c), dt_s = kvn_layer.recurrent_step(
                x_cur[:, s : s + 1, :], state=(q_s.unsqueeze(1), p_s.unsqueeze(1))
            )
            q_s = torch.where(is_val, q_c.squeeze(1).to(q_s.dtype), q_s)
            p_s = torch.where(is_val, p_c.squeeze(1).to(p_s.dtype), p_s)
            dtheta_anc = torch.where(is_anc, dt_s.squeeze(1).to(dtheta_anc.dtype), dtheta_anc)
            h_steps.append(h_s)
        new_states.append(torch.stack([q_s, p_s], dim=0))
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
    anc_q, anc_p, anc_dt = drafter.rollout.aggregate_multilayer_states(
        q_anc_layers, p_anc_layers, dt_anc_layers
    )
    raw_blocks, _ = drafter.rollout(anchor_h, anc_q, anc_p, anc_dt)
    if drafter.intrablock_mixer is not None:
        raw_blocks = drafter.intrablock_mixer(raw_blocks)
    hidden_blocks = drafter.norm(raw_blocks)
    base_logits = drafter.lm_head(hidden_blocks)
    return _sample_draft_block_with_probs(
        drafter, hidden_blocks, base_logits, anchor_tok, temperature
    )


def _load_benchmark_prompts(
    benchmark: str, tokenizer: Any, num_prompts: int, max_prompt_tokens: int = 1024
) -> list[list[int]]:
    import datasets
    import huggingface_hub

    if benchmark not in E2E_BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {list(E2E_BENCHMARK_SPECS)}")
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
        text = raw_val[0] if isinstance(raw_val, list) and raw_val else str(raw_val or "")
        if not text.strip():
            continue
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
            )
            if not isinstance(ids, list):
                ids = list(ids["input_ids"])
        else:
            ids = tokenizer.encode(text)
        ids = ids[:max_prompt_tokens]
        if len(ids) >= 2:
            prompts.append(ids)
        if len(prompts) >= num_prompts:
            break
    return prompts


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
) -> dict[str, Any]:
    """Run lossless autoregressive speculative decoding + batched Amdahl speedup profiling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
    block_size = drafter.config.block_size
    is_pure_kvn = drafter.config.kvn_backbone_type == "kvn"

    if prompt_token_ids_list is None:
        tokenizer = AutoTokenizer.from_pretrained(verifier_model)
        prompt_token_ids_list = _load_benchmark_prompts(
            benchmark=benchmark, tokenizer=tokenizer, num_prompts=num_prompts
        )

    all_step_taus: list[float] = []
    exact_greedy_matches: list[bool] = []
    draft_step_times_ms: list[float] = []
    verify_step_times_ms: list[float] = []
    total_spec_tokens = 0
    total_base_tokens = 0

    _sync_if_cuda(device)
    t_base_start = time.perf_counter()
    baseline_outputs: list[list[int]] = []
    for p_ids in prompt_token_ids_list:
        inp = torch.tensor([p_ids], dtype=torch.long, device=device)
        pkv = DynamicCache()
        out = verifier(input_ids=inp, past_key_values=pkv, use_cache=True)
        gen: list[int] = []
        next_logits = out.logits[:, -1, :]
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                nt = int(next_logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(next_logits.float() / temperature, dim=-1)
                nt = int(torch.multinomial(probs, num_samples=1).item())
            gen.append(nt)
            if len(gen) >= max_new_tokens:
                break
            out = verifier(
                input_ids=torch.tensor([[nt]], dtype=torch.long, device=device),
                past_key_values=pkv,
                use_cache=True,
            )
            next_logits = out.logits[:, -1, :]
        baseline_outputs.append(gen)
        total_base_tokens += len(gen)
    _sync_if_cuda(device)
    t_base_elapsed = max(time.perf_counter() - t_base_start, 1e-6)
    base_tps_b1 = total_base_tokens / t_base_elapsed

    _sync_if_cuda(device)
    t_spec_start = time.perf_counter()
    for p_idx, p_ids in enumerate(prompt_token_ids_list):
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=device)
        pkv = DynamicCache()
        # Prefill verifier on prompt[:-1] so prompt[-1] is the initial unverified anchor token
        prefill_ids = prompt_t[:, :-1]
        aux_prefill, _, _, pkv = _forward_verifier_with_speculator_hs(
            verifier, prefill_ids, aux_layer_ids, past_key_values=pkv, use_cache=True
        )
        fc_prefill = drafter.hidden_norm(drafter.fc(aux_prefill.to(eval_dtype)))

        if is_pure_kvn:
            state_cache = drafter.allocate_state_cache(batch_size=1, device=device)
            # Chunk 0: shifted fc across prompt_t (fc_shifted[0] = 0, fc_shifted[1..P-1] = fc_prefill[0..P-2])
            fc_init = torch.cat(
                [fc_prefill.new_zeros((1, 1, fc_prefill.size(-1))), fc_prefill], dim=1
            )
            cur_prev_fc = fc_init
            cur_tok_chunk = prompt_t
            cur_reset: torch.Tensor | None = torch.tensor([True], device=device)
        else:
            all_aux_confirmed = aux_prefill.to(eval_dtype)
            all_toks_confirmed = prompt_t

        spec_gen: list[int] = []
        while len(spec_gen) < max_new_tokens:
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            if is_pure_kvn:
                vlen = torch.tensor([cur_tok_chunk.size(1)], dtype=torch.long, device=device)
                draft_ids, draft_q_probs = _draft_step_kvn_recurrent(
                    drafter=drafter,
                    prev_fc_chunk=cur_prev_fc,
                    token_ids_chunk=cur_tok_chunk,
                    valid_lens=vlen,
                    state_cache=state_cache,
                    reset_seq_mask=cur_reset,
                    temperature=temperature,
                    s_pad_to=max(cur_tok_chunk.size(1), block_size + 1),
                )
                anchor_tok = cur_tok_chunk[:, -1:]
                cur_reset = None
            else:
                draft_ids, draft_q_probs = _draft_step_non_kvn_window(
                    drafter=drafter,
                    all_aux_hs=all_aux_confirmed,
                    all_tokens=all_toks_confirmed,
                    temperature=temperature,
                )
                anchor_tok = all_toks_confirmed[:, -1:]
            _sync_if_cuda(device)
            draft_step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            verify_in = torch.cat([anchor_tok, draft_ids], dim=1)  # [1, K + 1]
            kv_len_before = pkv.get_seq_length()

            _sync_if_cuda(device)
            t1 = time.perf_counter()
            v_aux, _, v_logits, pkv = _forward_verifier_with_speculator_hs(
                verifier, verify_in, aux_layer_ids, past_key_values=pkv, use_cache=True
            )
            _sync_if_cuda(device)
            verify_step_times_ms.append((time.perf_counter() - t1) * 1000.0)

            n_acc = 0
            bonus_tok = 0
            if temperature <= 0.0:
                target_greedy = v_logits[0].argmax(dim=-1)  # [K + 1]
                for k in range(block_size):
                    if int(draft_ids[0, k].item()) == int(target_greedy[k].item()):
                        n_acc += 1
                    else:
                        break
                bonus_tok = int(target_greedy[n_acc].item())
            else:
                p_probs = F.softmax(v_logits[0].float() / temperature, dim=-1)  # [K+1, V]
                rejected = False
                for k in range(block_size):
                    d_tok = int(draft_ids[0, k].item())
                    p_d = float(p_probs[k, d_tok].item())
                    q_d = max(float(draft_q_probs[0, k, d_tok].item()), 1e-12)
                    if float(torch.rand((), device=device).item()) < min(1.0, p_d / q_d):
                        n_acc += 1
                    else:
                        resid = (p_probs[k] - draft_q_probs[0, k]).clamp(min=0.0)
                        resid_sum = float(resid.sum().item())
                        resid = resid / resid_sum if resid_sum > 1e-12 else p_probs[k]
                        bonus_tok = int(torch.multinomial(resid, num_samples=1).item())
                        rejected = True
                        break
                if not rejected:
                    bonus_tok = int(
                        torch.multinomial(p_probs[block_size], num_samples=1).item()
                    )

            accepted_slice = draft_ids[0, :n_acc].tolist()
            emitted = [*accepted_slice, bonus_tok]
            all_step_taus.append(float(1 + n_acc))

            # Crop rejected draft tokens from verifier KV cache (compatible with transformers 4.x & 5.x)
            rejected_count = block_size - n_acc
            if rejected_count > 0:
                pkv.crop(-rejected_count)
            confirmed_v_aux = v_aux[:, : 1 + n_acc, :].to(eval_dtype)
            emitted_t = torch.tensor([emitted], dtype=torch.long, device=device)

            if is_pure_kvn:
                cur_prev_fc = drafter.hidden_norm(drafter.fc(confirmed_v_aux))
                cur_tok_chunk = emitted_t
            else:
                all_aux_confirmed = torch.cat([all_aux_confirmed, confirmed_v_aux], dim=1)
                all_toks_confirmed = torch.cat([all_toks_confirmed, emitted_t], dim=1)

            for tok in emitted:
                if len(spec_gen) < max_new_tokens:
                    spec_gen.append(tok)

        total_spec_tokens += len(spec_gen)
        if temperature <= 0.0:
            exact_greedy_matches.append(spec_gen == baseline_outputs[p_idx])

    _sync_if_cuda(device)
    t_spec_elapsed = max(time.perf_counter() - t_spec_start, 1e-6)
    spec_tps_b1 = total_spec_tokens / t_spec_elapsed
    mean_tau = float(np.mean(all_step_taus)) if all_step_taus else 1.0
    mean_draft_ms_b1 = float(np.mean(draft_step_times_ms)) if draft_step_times_ms else 0.01
    mean_verify_ms_b1 = max(
        float(np.mean(verify_step_times_ms)) if verify_step_times_ms else 0.1, 1e-4
    )

    # Batched Amdahl step-latency & throughput sweep across requested batch sizes
    eval_batches = batch_sizes or [1]
    amdahl_by_bs: dict[str, dict[str, float]] = {}
    sample_prompt = prompt_token_ids_list[0]
    p_len = len(sample_prompt)

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

        batch_prompt = torch.tensor(
            [sample_prompt] * bs, dtype=torch.long, device=device
        )
        pkv_base = DynamicCache()
        _ = verifier(input_ids=batch_prompt, past_key_values=pkv_base, use_cache=True)
        one_tok = batch_prompt[:, -1:]
        _sync_if_cuda(device)
        tb0 = time.perf_counter()
        _ = verifier(input_ids=one_tok, past_key_values=pkv_base, use_cache=True)
        _sync_if_cuda(device)
        base_step_s = max(time.perf_counter() - tb0, 1e-6)
        bs_base_tps = float(bs) / base_step_s

        pkv_spec = DynamicCache()
        _ = verifier(
            input_ids=batch_prompt[:, :-1], past_key_values=pkv_spec, use_cache=True
        )
        spec_chunk = torch.randint(
            0, drafter.draft_vocab_size, (bs, block_size + 1), device=device
        )
        _sync_if_cuda(device)
        tv0 = time.perf_counter()
        _ = _forward_verifier_with_speculator_hs(
            verifier, spec_chunk, aux_layer_ids, past_key_values=pkv_spec, use_cache=True
        )
        _sync_if_cuda(device)
        verify_ms = max((time.perf_counter() - tv0) * 1000.0, 1e-4)

        if is_pure_kvn:
            sc = drafter.allocate_state_cache(batch_size=bs, device=device)
            fc_blk = torch.zeros(
                (bs, block_size + 1, drafter.hidden_size),
                dtype=eval_dtype,
                device=device,
            )
            vl = torch.full((bs,), max(1, int(round(mean_tau))), dtype=torch.long, device=device)
            _sync_if_cuda(device)
            td0 = time.perf_counter()
            _ = drafter.advance_and_draft_step(fc_blk, spec_chunk, vl, sc)
            _sync_if_cuda(device)
            draft_ms = (time.perf_counter() - td0) * 1000.0
        else:
            aux_win = torch.zeros(
                (1, p_len, len(aux_layer_ids) * drafter.hidden_size),
                dtype=eval_dtype,
                device=device,
            )
            _sync_if_cuda(device)
            td0 = time.perf_counter()
            for _ in range(bs):
                _ = _draft_step_non_kvn_window(
                    drafter, aux_win[:, :-1, :], batch_prompt[:1], 0.0
                )
            _sync_if_cuda(device)
            draft_ms = (time.perf_counter() - td0) * 1000.0

        step_s = max((draft_ms + verify_ms) / 1000.0, 1e-6)
        bs_spec_tps = (float(bs) * mean_tau) / step_s
        amdahl_by_bs[str(bs)] = compute_amdahl_e2e_metrics(
            baseline_tps=bs_base_tps,
            spec_tps=bs_spec_tps,
            draft_latency_ms=draft_ms,
            verify_latency_ms=verify_ms,
            mean_accept_len=mean_tau,
        )

    return {
        "checkpoint": str(checkpoint_dir),
        "backbone_type": drafter.config.kvn_backbone_type,
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
        description="KvN-DSpark Phase 2 Confidence Calibration & End-to-End Amdahl Evaluator."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser(
        "calibrate", help="Evaluate ECE, AUROC & dynamic stopping Pareto curve."
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
        help="Run lossless E2E speculative decoding & batched Amdahl throughput evaluation.",
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
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
