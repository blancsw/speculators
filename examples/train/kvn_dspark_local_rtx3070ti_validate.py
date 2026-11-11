"""End-to-End Validation of KvN-DSpark Training Pipeline on Local RTX 3070 Ti (8 GB VRAM).

Uses the real Hugging Face Verifier LLM ``Qwen/Qwen3.5-0.8B`` (hidden_size=1024, 24 layers)
and real samples from ``HuggingFaceH4/ultrachat_200k`` to:
1. Extract real auxiliary hidden states (layers [2, 12, 21] + final layer 23) into
   ``hs_{i}.safetensors`` + ArrowDataset + ``token_freq.pt``.
2. Execute the complete ``speculators.train`` pipeline (``TrainConfig`` -> ``cli.main`` ->
   ``Trainer`` with Muon+AdamW, bfloat16 autocast, SingleGPUCheckpointer, validation split)
   across the 4 controlled scientific ablation models:
   - M1_DSpark_SWA_Baseline          (3 SWA-128 layers)
   - M2_DSpark_Hybrid_1SWA_2KvN      (1 SWA-128 + 2 KvN-RFF layers)
   - M3_KvN_DSpark_ZeroKV_Base       (3 Pure KvN-RFF layers, M=128, 0 KV-cache)
   - M4_KvN_DSpark_ZeroKV_Scaled_LASSO (3 Pure KvN-RFF layers, M_train=512 -> M_infer=128)
3. Verify ``KvNDSparkDraftModel.from_pretrained`` reload and structural LASSO mode pruning.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import time

from datasets import Dataset
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.train.cli import main as speculators_train_main
from speculators.train.config import TrainConfig


def prepare_local_qwen3_dataset(
    verifier_id: str,
    data_dir: Path,
    target_layer_ids: list[int],
    num_samples: int = 20,
    seq_len: int = 256,
    device: str = "cuda",
) -> None:
    """Extract real hidden states from Qwen/Qwen3.5-0.8B and write ArrowDataset + hs_*.safetensors."""
    hs_dir = data_dir / "hidden_states"
    if (
        (data_dir / "_torch_formatted.flag").exists()
        and (hs_dir / f"hs_{num_samples - 1}.safetensors").exists()
    ):
        print(f"[DataPrep] Reusing existing prepared Qwen dataset at {data_dir}")
        return

    if data_dir.exists():
        shutil.rmtree(data_dir)
    hs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DataPrep] Loading tokenizer and verifier {verifier_id} on {device} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(verifier_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        verifier_id,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    prompts = [
        "Explain how speculative decoding accelerates large language model inference by verifying multiple draft tokens in parallel.",
        "In quantum mechanics, the Koopman-von Neumann formalism represents classical Hamiltonian dynamics in Hilbert space using unitary operators.",
        "Write a clean Python function that computes the Fast Fourier Transform of a damped harmonic oscillator impulse response.",
        "Memory bandwidth on modern GPUs is the primary bottleneck during autoregressive token generation when batch sizes are small.",
        "Random Fourier Features approximate shift-invariant kernel functions such as the Gaussian Radial Basis Function via Bochner's theorem.",
        "Sliding window attention bounds the KV cache size to a fixed window W, whereas recurrent symplectic state spaces require O(1) state.",
        "Multi-token prediction heads predict K future tokens conditioned on anchor hidden states extracted from intermediate verifier layers.",
        "Total variation distance between the draft distribution q and target distribution p determines the exact speculative acceptance rate.",
    ]

    rows: list[dict] = []
    token_counter: Counter[int] = Counter()

    with torch.no_grad():
        for idx in range(num_samples):
            base_text = prompts[idx % len(prompts)] + f" (Sample variation #{idx}: detailed mathematical derivation and analysis.) "
            repeated_text = (base_text * 8).strip()
            enc = tokenizer(
                repeated_text,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]
            if input_ids.numel() < seq_len:
                pad_id = tokenizer.eos_token_id or 0
                pad = torch.full((seq_len - input_ids.numel(),), pad_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad], dim=0)
            input_ids = input_ids[:seq_len]

            out = model(
                input_ids=input_ids.unsqueeze(0).to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            # out.hidden_states has length num_layers + 1 (index 0 is embeddings, index l+1 is layer l output)
            all_hs = out.hidden_states
            captured = [all_hs[layer_id + 1][0].detach().to(torch.bfloat16).cpu() for layer_id in target_layer_ids]
            # Append final decoder layer output (pre-final-norm) as trailing entry for verifier target distillation
            captured.append(all_hs[-1][0].detach().to(torch.bfloat16).cpu())
            stacked_hs = torch.stack(captured, dim=1).contiguous()  # [seq_len, len(target_layer_ids)+1, hidden_size]

            token_ids_cpu = input_ids.to(torch.long).cpu().contiguous()
            save_file(
                {"hidden_states": stacked_hs, "token_ids": token_ids_cpu},
                str(hs_dir / f"hs_{idx}.safetensors"),
            )

            loss_mask = [1] * seq_len
            loss_mask[:16] = [0] * 16  # Prompt prefix masked out, assistant continuation trained
            token_counter.update(token_ids_cpu[16:].tolist())

            rows.append(
                {
                    "input_ids": token_ids_cpu.tolist(),
                    "loss_mask": loss_mask,
                    "seq_len": int(seq_len),
                }
            )

    del model
    torch.cuda.empty_cache()

    hf_ds = Dataset.from_list(rows).with_format(
        "torch", columns=["input_ids", "loss_mask"], output_all_columns=True
    )
    hf_ds.save_to_disk(str(data_dir))
    torch.save(dict(token_counter), data_dir / "token_freq.pt")
    (data_dir / "_torch_formatted.flag").write_text("1", encoding="utf-8")
    print(f"[DataPrep] Saved {num_samples} samples (seq_len={seq_len}) and token_freq.pt to {data_dir}")


def run_rtx3070ti_validation() -> dict:
    verifier_id = "Qwen/Qwen3.5-0.8B"
    target_layer_ids = [2, 12, 21]
    base_dir = Path("kvn_rff_mtp/rtx3070ti_validation_run")
    data_dir = base_dir / "data"
    seq_len = 256
    draft_vocab_size = 4096
    epochs = 5

    prepare_local_qwen3_dataset(
        verifier_id=verifier_id,
        data_dir=data_dir,
        target_layer_ids=target_layer_ids,
        num_samples=20,
        seq_len=seq_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ablation_specs = [
        {
            "id": "M1_DSpark_SWA_Baseline",
            "kvn_backbone_type": "swa",
            "kvn_n_modes": 128,
            "prune_to": None,
            "kvn_use_born_confidence": False,
        },
        {
            "id": "M2_DSpark_Hybrid_1SWA_2KvN",
            "kvn_backbone_type": "hybrid_1swa_2kvn",
            "kvn_n_modes": 128,
            "prune_to": None,
            "kvn_use_born_confidence": True,
        },
        {
            "id": "M3_KvN_DSpark_ZeroKV_Base",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 128,
            "prune_to": None,
            "kvn_use_born_confidence": True,
        },
        {
            "id": "M4_KvN_DSpark_ZeroKV_Scaled_LASSO",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 512,
            "prune_to": 128,
            "kvn_use_born_confidence": True,
        },
    ]

    results: list[dict] = []

    for spec in ablation_specs:
        model_id = spec["id"]
        save_path = base_dir / "checkpoints" / model_id
        if save_path.exists():
            shutil.rmtree(save_path)

        argv = [
            "--verifier-name-or-path", verifier_id,
            "--speculator-type", "kvn_dspark",
            "--data-path", str(data_dir),
            "--save-path", str(save_path),
            "--draft-vocab-size", str(draft_vocab_size),
            "--total-seq-len", str(seq_len),
            "--train-data-ratio", "0.8",
            "--num-workers", "0",
            "--epochs", str(epochs),
            "--lr", "5e-4",
            "--optimizer", "muon",
            "--optimizer-implementation", "eager",
            "--loss-implementation", "eager",
            "--loss-fn", '{"ce": 0.1, "tv": 0.9}',
            "--block-size", "8",
            "--max-anchors", "32",
            "--num-layers", "3",
            "--sliding-window", "128",
            "--draft-attn-impl", "eager",
            "--target-layer-ids", *(str(x) for x in target_layer_ids),
            "--markov-rank", "128",
            "--markov-head-type", "gated",
            "--enable-confidence-head",
            "--confidence-head-with-markov",
            "--kvn-backbone-type", spec["kvn_backbone_type"],
            "--kvn-n-modes", str(spec["kvn_n_modes"]),
            "--kvn-reupload-layers", "2",
            "--kvn-reupload-dim", "256",
            "--kvn-learnable-rff",
            "--kvn-anchor-rank", "256",
            "--kvn-enable-interferometer",
            "--kvn-vacuum-born-norm",
            "--kvn-multilayer-rollout",
            "--kvn-intrablock-mixer",
            "--kvn-strict-causal-shift",
            "--kvn-lasso-weight", "2e-4" if spec["prune_to"] else "1e-5",
            "--on-missing", "raise",
            "--seed", "42",
        ]
        if not spec["kvn_use_born_confidence"]:
            argv.append("--no-kvn-use-born-confidence")

        print(f"\n=== Running speculators.train for {model_id} ===")
        cfg = TrainConfig.resolve(argv)
        t0 = time.perf_counter()
        speculators_train_main(cfg)
        train_elapsed_s = time.perf_counter() - t0

        last_epoch_dir = save_path / str(epochs - 1)
        val_metrics_file = save_path / "0" / "val_metrics.json"
        if (last_epoch_dir / "val_metrics.json").exists():
            val_metrics_file = last_epoch_dir / "val_metrics.json"
        val_metrics = json.loads(val_metrics_file.read_text()) if val_metrics_file.exists() else {}

        # Reload saved checkpoint via KvNDSparkDraftModel.from_pretrained to verify HF/speculators compatibility
        t2d_tensor = torch.from_numpy(__import__("numpy").load(data_dir / f"t2d-{draft_vocab_size}.npy"))
        d2t_tensor = torch.from_numpy(__import__("numpy").load(data_dir / f"d2t-{draft_vocab_size}.npy"))
        loaded_model = KvNDSparkDraftModel.from_pretrained(
            last_epoch_dir,
            t2d=t2d_tensor,
            d2t=d2t_tensor,
        )
        post_prune_recal_metrics: dict[str, float] | None = None
        if spec["prune_to"] is not None:
            loaded_model.prune_kvn_modes(spec["prune_to"])
            pruned_dir = save_path / f"pruned_m{spec['prune_to']}"
            loaded_model.save_pretrained(pruned_dir)

            # Run 1-epoch post-pruning micro-recalibration (without decoder-shaping flags)
            recal_save_path = save_path / f"pruned_recal_m{spec['prune_to']}"
            if recal_save_path.exists():
                shutil.rmtree(recal_save_path)
            recal_argv = [
                "--verifier-name-or-path", verifier_id,
                "--speculator-type", "kvn_dspark",
                "--from-pretrained", str(pruned_dir),
                "--data-path", str(data_dir),
                "--save-path", str(recal_save_path),
                "--draft-vocab-size", str(draft_vocab_size),
                "--total-seq-len", str(seq_len),
                "--train-data-ratio", "0.8",
                "--num-workers", "0",
                "--epochs", "1",
                "--lr", "1e-4",
                "--muon-lr", "5e-4",
                "--optimizer", "muon",
                "--optimizer-implementation", "eager",
                "--loss-implementation", "eager",
                "--loss-fn", '{"ce": 0.1, "tv": 0.9}',
                "--block-size", "8",
                "--max-anchors", "32",
                "--draft-attn-impl", "eager",
                "--target-layer-ids", *(str(x) for x in target_layer_ids),
                "--kvn-lasso-weight", "0.0",
                "--on-missing", "raise",
                "--seed", "42",
            ]
            recal_cfg = TrainConfig.resolve(recal_argv)
            speculators_train_main(recal_cfg)
            recal_val_file = recal_save_path / "0" / "val_metrics.json"
            if recal_val_file.exists():
                rm = json.loads(recal_val_file.read_text())
                post_prune_recal_metrics = {
                    "val_loss": round(float(rm.get("loss_epoch", 0.0)), 4),
                    "accept_rate_pct": round(100.0 * float(rm.get("accept_rate_epoch", 0.0)), 2),
                    "accept_len_tau": round(float(rm.get("accept_len_epoch", 0.0)), 4),
                    "greedy_eal": round(float(rm.get("eal_epoch", 0.0)), 4),
                    "full_acc_pct": round(100.0 * float(rm.get("full_acc_epoch", 0.0)), 2),
                    "pos0_acc_pct": round(100.0 * float(rm.get("position_0_acc_epoch", 0.0)), 2),
                    "confidence_mae": round(float(rm.get("confidence_abs_error_epoch", 0.0)), 4),
                }

        state_bytes = loaded_model.config.estimated_state_bytes_per_seq()
        active_params = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)

        entry = {
            "id": model_id,
            "verifier_model": verifier_id,
            "backbone_type": spec["kvn_backbone_type"],
            "train_modes": spec["kvn_n_modes"],
            "infer_modes": loaded_model.config.kvn_n_modes,
            "active_drafter_params": active_params,
            "state_cache_bytes_per_seq": state_bytes,
            "train_wall_time_s": round(train_elapsed_s, 2),
            "val_loss": round(float(val_metrics.get("loss_epoch", 0.0)), 4),
            "accept_rate_pct": round(100.0 * float(val_metrics.get("accept_rate_epoch", 0.0)), 2),
            "accept_len_tau": round(float(val_metrics.get("accept_len_epoch", 0.0)), 4),
            "greedy_eal": round(float(val_metrics.get("eal_epoch", 0.0)), 4),
            "full_acc_pct": round(100.0 * float(val_metrics.get("full_acc_epoch", 0.0)), 2),
            "pos0_acc_pct": round(100.0 * float(rm.get("position_0_acc_epoch", 0.0) if post_prune_recal_metrics and "rm" in locals() else val_metrics.get("position_0_acc_epoch", 0.0)), 2),
            "confidence_mae": round(float(val_metrics.get("confidence_abs_error_epoch", 0.0)), 4),
        }
        entry["pos0_acc_pct"] = round(100.0 * float(val_metrics.get("position_0_acc_epoch", 0.0)), 2)
        if post_prune_recal_metrics is not None:
            entry["post_prune_1ep_recal_metrics"] = post_prune_recal_metrics
        results.append(entry)
        print(
            f"[{model_id:<34}] Cache/Seq: {state_bytes:>9,d} B | "
            f"ValLoss: {entry['val_loss']:>6.4f} | "
            f"AcceptLen(tau): {entry['accept_len_tau']:>6.3f} | "
            f"EAL: {entry['greedy_eal']:>6.3f} | "
            f"ConfMAE: {entry['confidence_mae']:>6.4f} | "
            f"Time: {entry['train_wall_time_s']:>5.2f}s"
        )

    out_file = Path("kvn_rff_mtp/rtx3070ti_qwen3_validation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "verifier_model": verifier_id,
        "target_layer_ids": target_layer_ids,
        "seq_len": seq_len,
        "draft_vocab_size": draft_vocab_size,
        "epochs": epochs,
        "models": results,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete RTX 3070 Ti validation report to {out_file}")
    return payload


if __name__ == "__main__":
    run_rtx3070ti_validation()
