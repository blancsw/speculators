#!/usr/bin/env python3
"""Confidence Calibration (ECE, AUROC, Dynamic Stopping Pareto) & Amdahl End-to-End Speedup Evaluator.

Implements Phase D.2 & D.3 of ``kvn_rff_rnn_mtp_train_phase_2.md``:
1. ``compute_ece``: 15-bin Expected Calibration Error on speculative token acceptance probabilities.
2. ``compute_binary_auroc``: Exact Mann-Whitney / trapezoidal AUROC for rejection prediction.
3. ``compute_dynamic_stopping_pareto``: Sweeps confidence stopping threshold
   ``gamma_conf in {0.0, 0.2, 0.35, 0.5, 0.65, 0.8}`` to measure expected acceptance length ``tau(gamma)``
   vs verifier speculative slot FLOPs saved.
4. ``compute_amdahl_e2e_metrics``: Computes true end-to-end speculative speedup ``S_e2e(B)``
   including target verifier time, proposer overhead fraction, and Amdahl's theoretical ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8)


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

    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64, device=p.device)
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

    # Pairwise concordance (vectorized when small, sorted rank-sum when large)
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
    """Evaluate dynamic block truncation at ``c_{t,k} < gamma_conf``.

    Args:
        confidences: ``[N_blocks, K]`` predicted acceptance probabilities in ``[0, 1]``.
        accepted_flags: ``[N_blocks, K]`` boolean exact token matches against target greedy/sampled tokens.
        thresholds: Confidence thresholds ``gamma_conf`` to sweep.
    """
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
            # Truncate block starting at the first speculative position k where conf < gamma
            keep_mask = torch.cumprod((conf >= gamma).to(torch.long), dim=-1).to(torch.bool)

        effective_draft_lens = keep_mask.sum(dim=-1).to(torch.float32)
        # Contiguous prefix of tokens that were BOTH drafted (not truncated) AND accepted by verifier
        contiguous_accept = torch.cumprod((keep_mask & acc).to(torch.long), dim=-1)
        accepted_spec_tokens = contiguous_accept.sum(dim=-1).to(torch.float32)
        # Total tokens produced per verifier pass = 1 (target bonus/correction token) + accepted_spec_tokens
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


@torch.no_grad()
def evaluate_checkpoint_calibration(
    checkpoint_dir: Path,
    verifier_model: str | None,
    data_dir: Path,
    draft_vocab_size: int,
    max_samples: int = 64,
    max_anchors_per_seq: int = 32,
    device: str = "cuda",
) -> dict:
    """Run offline calibration audit on a saved ``KvNDSparkDraftModel`` checkpoint.

    Uses ``drafter._kvn_backbone_forward`` and ``drafter._decode_draft_block_greedy`` directly
    (which automatically leverage ``drafter.verifier_norm`` and ``drafter.verifier_lm_head``
    loaded by ``from_pretrained`` without duplicating the full verifier LLM in VRAM).
    """
    from speculators.models.kvn_dspark import KvNDSparkDraftModel

    del verifier_model  # Verifier norm/lm_head are loaded directly via SpeculatorConfig
    t2d = torch.from_numpy(np.load(data_dir / f"t2d-{draft_vocab_size}.npy"))
    d2t = torch.from_numpy(np.load(data_dir / f"d2t-{draft_vocab_size}.npy"))
    eval_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    drafter = KvNDSparkDraftModel.from_pretrained(checkpoint_dir, t2d=t2d, d2t=d2t).to(
        device=device, dtype=eval_dtype
    )
    drafter.eval()

    hs_files = sorted((data_dir / "hidden_states").glob("hs_*.safetensors"))[:max_samples]
    all_confs: list[torch.Tensor] = []
    all_accs: list[torch.Tensor] = []

    block_size = drafter.config.block_size
    for hs_path in hs_files:
        tensors = load_file(str(hs_path))
        hs_all = tensors["hidden_states"].to(device=device, dtype=eval_dtype)
        toks = tensors["token_ids"].to(device=device, dtype=torch.long)
        seq_len = toks.shape[0]
        if seq_len <= block_size + 4:
            continue

        aux_hs = hs_all[:, :-1, :].reshape(1, seq_len, -1)
        final_hs = hs_all[:, -1, :].unsqueeze(0)
        loss_mask = torch.ones((1, seq_len), dtype=eval_dtype, device=device)
        loss_mask[:, :4] = 0
        loss_mask[:, seq_len - block_size :] = 0
        doc_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

        n_anc = min(max_anchors_per_seq, max(1, (seq_len - block_size - 4) // block_size))
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
        valid_blocks = aligned_loss_mask.reshape(n_anc, block_size).sum(dim=-1) == block_size
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
            "accept_rate_pct": round(100.0 * float(accs_cat[:, k].to(torch.float32).mean().item()), 2),
            "ece": round(compute_ece(confs_cat[:, k], accs_cat[:, k].to(torch.float32)), 4),
            "rejection_auroc": round(compute_binary_auroc(rej_scores[:, k], rej_labels[:, k]), 4),
        }

    return {
        "checkpoint": str(checkpoint_dir),
        "state_bytes_per_seq": drafter.config.estimated_state_bytes_per_seq(),
        "num_evaluated_blocks": int(confs_cat.shape[0]),
        "overall_ece_15bin": round(compute_ece(confs_cat, accs_cat.to(torch.float32), num_bins=15), 4),
        "overall_rejection_auroc": round(compute_binary_auroc(rej_scores, rej_labels), 4),
        "per_position": per_pos,
        "dynamic_stopping_pareto": compute_dynamic_stopping_pareto(confs_cat, accs_cat),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="KvN-DSpark Phase 2 Confidence Calibration & Amdahl Speedup Evaluator."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Evaluate ECE, AUROC & dynamic stopping Pareto curve.")
    cal.add_argument("--checkpoint", type=Path, required=True)
    cal.add_argument("--verifier-model", type=str, required=True)
    cal.add_argument("--data-dir", type=Path, required=True)
    cal.add_argument("--draft-vocab-size", type=int, default=32768)
    cal.add_argument("--max-samples", type=int, default=64)
    cal.add_argument("--output-json", type=Path, required=True)

    amd = sub.add_parser("amdahl", help="Compute end-to-end Amdahl speedup summary.")
    amd.add_argument("--baseline-tps", type=float, required=True)
    amd.add_argument("--spec-tps", type=float, required=True)
    amd.add_argument("--draft-latency-ms", type=float, required=True)
    amd.add_argument("--verify-latency-ms", type=float, required=True)
    amd.add_argument("--mean-accept-len", type=float, required=True)
    amd.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        report = evaluate_checkpoint_calibration(
            checkpoint_dir=args.checkpoint,
            verifier_model=args.verifier_model,
            data_dir=args.data_dir,
            draft_vocab_size=args.draft_vocab_size,
            max_samples=args.max_samples,
            device="cuda" if torch.cuda.is_available() else "cpu",
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
            args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
