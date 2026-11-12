# ruff: noqa: BLE001, C901, E402, PLR2004, S101, S603, S607, SLF001, T201
"""Live RTX 3070 Ti (8 GB) End-to-End GPU Validation & Benchmark for KvN-DSpark v3.

Executes a controlled, strictly iso-compute (6 total epochs per model) GPU training and
lossless speculative decoding benchmark on ``Qwen/Qwen3.5-0.8B`` (hidden_size=1024,
24 layers, vocab_size=248,320, target_layer_ids=[2, 12, 21]):

Models compared under strict iso-compute (6 epochs total):
1. ``B1_SWA``                : Clean SWA-128 baseline (no LASSO, RNN Markov head).
2. ``M_v2_Fixed``            : KvN-DSpark v2 w/ [C1]-[C5] fixes (FP32 state, 0 LASSO).
3. ``M_v3_Full``             : KvN-DSpark v3 (Multiplicative 2^l Dyadic RFF + Engram +
                               sp(4,R) Symplectic Butterfly Scan + Lindblad Purity +
                               DEAO survival loss + 2-pass self-conditioned rollout +
                               Kraus POVM speculative tree).
4. ``M_v3_ProxLASSO_Pruned`` : Over-parameterized M=512 -> 128 under joint cross-layer
                               proximal dyadic Group-LASSO (5 epochs LASSO + zero-shot
                               prune + 1 epoch warm recal = 6 iso-compute epochs).

Outputs complete empirical GPU results to:
``speculators/kvn_rff_mtp/rtx3070ti_kvn_v3_validation_results.json``
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.cache_utils import DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src_p in (REPO_ROOT / "src", REPO_ROOT / "hs_connectors" / "src"):
    if str(_src_p) not in sys.path:
        sys.path.insert(0, str(_src_p))

from hs_connectors import HiddenStatesBackend
from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.train.checkpointer import convert_float_dtype, patch_config_dtype
from speculators.train.cli import main as speculators_train_main
from speculators.train.config import TrainConfig
from speculators.train.config.resolution import build_parser
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.utils import normalize_counted_metrics
from speculators.vllm_integration import (
    HardwareRooflineProfile,
    KvNDSparkVllmProposer,
    PurityLoadAwareAmdahlScheduler,
    build_tree_attention_mask,
    compact_tree_kv_cache,
    specinfer_verify_tree_step,
)

FIXED_CAUSAL_PAD = 192


CURATED_TECHNICAL_PROMPTS: list[str] = [
    (
        "Write a Python function `fft_convolve_1d(x, h)` using `numpy.fft.rfft` and "
        "explain why zero-padding to 2N-1 prevents circular aliasing."
    ),
    (
        "Derive the eigenvalues and contraction modulus of a damped harmonic "
        "oscillator dz/dt = (-eta + i*omega)*z driven by discrete impulses."
    ),
    (
        "Explain lossless speculative rejection sampling: prove why accepting with "
        "min(1, p(x)/q(x)) and resampling from norm(max(0, p - q)) yields p(x)."
    ),
    (
        "Implement a thread-safe LRU prefix cache in Python using "
        "`collections.OrderedDict` with O(1) get and put operations."
    ),
    (
        "In Hamiltonian mechanics, prove that any 2x2 real matrix M satisfying "
        "M^T J M = J has determinant +1 and preserves phase-space area."
    ),
    (
        "Compare GPU memory bandwidth roofline limits of Sliding Window Attention "
        "KV-caches against O(1) recurrent state-space models at batch size 64."
    ),
    (
        "Solve the initial value problem y'' + 4y' + 13y = 0 with y(0) = 1, "
        "y'(0) = 0, showing every algebraic step cleanly."
    ),
    (
        "Write an iterative binary search and lower_bound implementation in Python "
        "with unit tests for duplicate elements and empty arrays."
    ),
]

HELD_OUT_EVAL_PROMPTS: list[str] = [
    (
        "Write a clean Python implementation of topological sort (Kahn's algorithm) "
        "on a directed graph represented as an adjacency list."
    ),
    (
        "Derive the closed-form geometric series sum_{k=0}^{n-1} r^k and analyze "
        "convergence when |r| < 1 in the complex plane."
    ),
    (
        "Explain how Automatic Prefix Caching (APC) in vLLM hashes 16-token KV "
        "blocks and why recurrent models require boundary state snapshots."
    ),
    (
        "Implement `merge_k_sorted_lists` in Python using `heapq` and state its "
        "exact time and auxiliary space complexity."
    ),
    (
        "In open quantum systems, explain how the Lindblad master equation evolves "
        "pure states into mixed Gaussian states with purity P < 1."
    ),
    (
        "Given a list of intervals, write a Python function "
        "`merge_intervals(intervals)` that merges all overlapping intervals in "
        "O(n log n) time."
    ),
    (
        "Prove by mathematical induction that 1^2 + 2^2 + ... + n^2 = "
        "n(n + 1)(2n + 1) / 6 for all positive integers n."
    ),
    (
        "Compare Newton-Schulz matrix orthogonalization in the Muon optimizer "
        "against coordinate-wise adaptive second moments in AdamW."
    ),
]


def _load_eval_script_module():
    eval_path = (
        REPO_ROOT / "scripts" / "evaluate" / "kvn_phase2_calibration_and_amdahl.py"
    )
    spec = importlib.util.spec_from_file_location("kvn_eval_harness", eval_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kvn_eval_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect_local_ultrachat_train_prompts(max_needed: int) -> list[str]:
    """Read prompts from cached UltraChat `train_sft` parquet (never `test_sft`)."""
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    uc_root = hf_hub / "datasets--HuggingFaceH4--ultrachat_200k" / "snapshots"
    prompts: list[str] = []
    if uc_root.exists():
        parquet_files = sorted(uc_root.rglob("train_sft-*.parquet"))
        for pf in parquet_files:
            table = pq.read_table(pf, columns=["messages"])
            msgs_col = table["messages"].to_pylist()
            for msgs in msgs_col:
                if isinstance(msgs, list) and msgs:
                    first = msgs[0]
                    if isinstance(first, dict) and first.get("role") == "user":
                        text = str(first.get("content", "")).strip()
                        if 40 <= len(text) <= 420:
                            prompts.append(text)
                            if len(prompts) >= max_needed:
                                return prompts
    return prompts


def prepare_onpolicy_qwen_dataset(
    verifier_id: str,
    data_dir: Path,
    target_layer_ids: list[int],
    num_samples: int = 40,
    train_ratio: float = 0.80,
    seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate on-policy Qwen3.5-0.8B sequences + hidden states (0% conv leak)."""
    hs_dir = data_dir / "hidden_states"
    meta_file = data_dir / "prep_metadata.json"
    if (
        (data_dir / "_torch_formatted.flag").exists()
        and (hs_dir / f"hs_{num_samples - 1}.safetensors").exists()
        and meta_file.exists()
    ):
        print(f"[DataPrep] Reusing verified on-policy dataset at {data_dir}")
        return json.loads(meta_file.read_text(encoding="utf-8"))

    if data_dir.exists():
        shutil.rmtree(data_dir)
    hs_dir.mkdir(parents=True, exist_ok=True)

    uc_prompts = _collect_local_ultrachat_train_prompts(
        max(0, num_samples - len(CURATED_TECHNICAL_PROMPTS))
    )
    all_prompts = (CURATED_TECHNICAL_PROMPTS + uc_prompts)[:num_samples]
    while len(all_prompts) < num_samples:
        idx = len(all_prompts)
        base_prompt = CURATED_TECHNICAL_PROMPTS[idx % len(CURATED_TECHNICAL_PROMPTS)]
        all_prompts.append(f"{base_prompt} (Provide concise derivation #{idx}.)")

    n_train = int(round(num_samples * train_ratio))
    print(
        f"[DataPrep] Loading {verifier_id} on {device} (bfloat16) for {num_samples} "
        f"on-policy sequences ({n_train} train / {num_samples - n_train} val, "
        f"seq_len={seq_len})..."
    )
    tokenizer = AutoTokenizer.from_pretrained(verifier_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        verifier_id,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Cap prompt <= 96 tokens so >= 160 assistant tokens fit inside seq_len=256
    tokenized_prompts: list[list[int]] = []
    for user_prompt in all_prompts:
        messages = [{"role": "user", "content": user_prompt}]
        try:
            p_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            p_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        if not isinstance(p_ids, list):
            p_ids = list(p_ids["input_ids"])
        tokenized_prompts.append(p_ids[:96])

    rows: list[dict[str, Any]] = []
    train_token_counter: Counter[int] = Counter()
    gen_batch_size = 8

    t0 = time.perf_counter()
    with torch.no_grad():
        for b_start in range(0, num_samples, gen_batch_size):
            b_prompts = tokenized_prompts[b_start : b_start + gen_batch_size]
            bsz = len(b_prompts)
            max_p = max(len(p) for p in b_prompts)
            min_p = min(len(p) for p in b_prompts)
            max_new = max(32, seq_len - min_p)

            # Left-pad batch for parallel autoregressive generation on Qwen3.5-0.8B
            batch_in = torch.full((bsz, max_p), pad_id, dtype=torch.long, device=device)
            batch_mask = torch.zeros((bsz, max_p), dtype=torch.long, device=device)
            for row_i, p_ids in enumerate(b_prompts):
                pad_offset = max_p - len(p_ids)
                batch_in[row_i, pad_offset:] = torch.tensor(
                    p_ids, dtype=torch.long, device=device
                )
                batch_mask[row_i, pad_offset:] = 1

            use_stoch = (b_start // gen_batch_size) % 2 == 1
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": use_stoch,
                "pad_token_id": pad_id,
                "use_cache": True,
            }
            if use_stoch:
                gen_kwargs["temperature"] = 0.6
                gen_kwargs["top_p"] = 0.95

            gen_out = model.generate(
                input_ids=batch_in, attention_mask=batch_mask, **gen_kwargs
            )

            # Extract exact causal hidden states per sequence (bsz=1, <2 GB VRAM)
            for row_i, p_ids in enumerate(b_prompts):
                idx = b_start + row_i
                conv_id = f"conv_{idx:04d}"
                p_len = len(p_ids)
                comp_ids = gen_out[row_i, max_p:].tolist()
                if eos_id in comp_ids:
                    eos_pos = comp_ids.index(eos_id)
                    valid_comp = comp_ids[: eos_pos + 1]
                else:
                    valid_comp = comp_ids[: seq_len - p_len]

                full_ids = (p_ids + valid_comp)[:seq_len]
                valid_total = len(full_ids)
                loss_mask = [0] * p_len + [1] * (valid_total - p_len)
                doc_ids = [0] * valid_total
                if valid_total < seq_len:
                    rem = seq_len - valid_total
                    full_ids = full_ids + [pad_id] * rem
                    loss_mask = loss_mask + [0] * rem
                    doc_ids = (
                        doc_ids + [1] * rem
                    )  # Separate pad segment tests [C3] doc boundary!

                input_ids_t = torch.tensor(full_ids, dtype=torch.long, device=device)
                attn_t = torch.ones((1, seq_len), dtype=torch.long, device=device)
                if valid_total < seq_len:
                    attn_t[0, valid_total:] = 0
                out = model(
                    input_ids=input_ids_t.unsqueeze(0),
                    attention_mask=attn_t,
                    output_hidden_states=True,
                    use_cache=False,
                )
                all_hs = out.hidden_states
                captured = [
                    all_hs[layer_id + 1][0].detach().to(torch.bfloat16).cpu()
                    for layer_id in target_layer_ids
                ]
                captured.append(all_hs[-1][0].detach().to(torch.bfloat16).cpu())
                stacked_hs = torch.stack(captured, dim=1).contiguous()

                token_ids_cpu = input_ids_t.detach().cpu().contiguous()
                save_file(
                    {"hidden_states": stacked_hs, "token_ids": token_ids_cpu},
                    str(hs_dir / f"hs_{idx}.safetensors"),
                )

                # [C2] Strictly accumulate token_freq.pt on TRAIN conversations only!
                if idx < n_train:
                    train_assistant_tokens = [
                        tok
                        for tok, m in zip(full_ids, loss_mask, strict=True)
                        if m == 1
                    ]
                    train_token_counter.update(train_assistant_tokens)

                rows.append(
                    {
                        "conversation_id": conv_id,
                        "input_ids": full_ids,
                        "loss_mask": loss_mask,
                        "document_ids": doc_ids,
                        "seq_len": int(seq_len),
                    }
                )
            torch.cuda.empty_cache()

    prep_time_s = time.perf_counter() - t0
    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_conv_ids = {r["conversation_id"] for r in rows[:n_train]}
    val_conv_ids = {r["conversation_id"] for r in rows[n_train:]}
    assert len(train_conv_ids & val_conv_ids) == 0, "Conversation leak detected!"

    hf_ds = Dataset.from_list(rows).with_format(
        "torch",
        columns=["input_ids", "loss_mask", "document_ids"],
        output_all_columns=True,
    )
    hf_ds.save_to_disk(str(data_dir))
    torch.save(dict(train_token_counter), data_dir / "token_freq.pt")
    (data_dir / "_torch_formatted.flag").write_text("1", encoding="utf-8")

    meta = {
        "num_samples": num_samples,
        "num_train_conversations": n_train,
        "num_val_conversations": num_samples - n_train,
        "sibling_conversation_overlap": len(train_conv_ids & val_conv_ids),
        "token_freq_source": "train_only",
        "unique_train_assistant_tokens": len(train_token_counter),
        "prep_wall_time_s": round(prep_time_s, 2),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DataPrep] Completed in {prep_time_s:.2f}s: {meta}")
    return meta


def _supported_cli_flags() -> set[str]:
    """Introspect exact CLI option strings supported by `TrainConfig.resolve`."""
    parser = build_parser()
    flags: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            flags.add(opt)
    return flags


def _append_if_supported(
    argv: list[str], supported: set[str], flag: str, *vals: str
) -> None:
    if flag in supported:
        argv.append(flag)
        argv.extend(vals)


def _build_model_argv(
    spec: dict[str, Any],
    verifier_id: str,
    data_dir: Path,
    save_path: Path,
    target_layer_ids: list[int],
    seq_len: int,
    draft_vocab_size: int,
    train_ratio: float,
    supported: set[str],
) -> list[str]:
    argv = [
        "--verifier-name-or-path",
        verifier_id,
        "--speculator-type",
        "kvn_dspark",
        "--data-path",
        str(data_dir),
        "--save-path",
        str(save_path),
        "--draft-vocab-size",
        str(draft_vocab_size),
        "--total-seq-len",
        str(seq_len),
        "--train-data-ratio",
        str(train_ratio),
        "--num-workers",
        "0",
        "--epochs",
        str(spec["epochs"]),
        "--lr",
        str(spec.get("lr", "4e-4")),
        "--muon-lr",
        str(spec.get("muon_lr", "1.5e-3")),
        "--optimizer",
        "muon",
        "--optimizer-implementation",
        "eager",
        "--scheduler-type",
        "cosine",
        "--loss-implementation",
        "eager",
        "--loss-fn",
        '{"ce": 0.1, "tv": 0.9}',
        "--block-size",
        "8",
        "--max-anchors",
        "48",
        "--num-layers",
        "3",
        "--sliding-window",
        "128",
        "--draft-attn-impl",
        "eager",
        "--target-layer-ids",
        *(str(x) for x in target_layer_ids),
        "--markov-rank",
        "128",
        "--markov-head-type",
        spec["markov_head_type"],
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        "--kvn-backbone-type",
        spec["kvn_backbone_type"],
        "--kvn-n-modes",
        str(spec["kvn_n_modes"]),
        "--kvn-reupload-layers",
        "2",
        "--kvn-reupload-dim",
        "256",
        "--kvn-learnable-rff",
        "--kvn-anchor-rank",
        "256",
        "--kvn-enable-interferometer",
        "--kvn-vacuum-born-norm",
        "--kvn-multilayer-rollout",
        "--kvn-intrablock-mixer",
        "--kvn-strict-causal-shift",
        "--kvn-state-dtype",
        "float32",
        "--kvn-lasso-weight",
        str(spec["kvn_lasso_weight"]),
        "--on-missing",
        "raise",
        "--seed",
        "42",
    ]
    if not spec.get("kvn_use_born_confidence", True):
        argv.append("--no-kvn-use-born-confidence")

    if spec.get("is_v3", False):
        _append_if_supported(argv, supported, "--kvn-dyadic-rff")
        _append_if_supported(argv, supported, "--kvn-use-engram")
        _append_if_supported(argv, supported, "--kvn-engram-buckets", "16384")
        _append_if_supported(argv, supported, "--kvn-engram-dim", "128")
        _append_if_supported(argv, supported, "--kvn-symplectic-butterfly")
        _append_if_supported(argv, supported, "--kvn-lindblad-purity")
        _append_if_supported(argv, supported, "--kvn-use-kraus-tree")
        _append_if_supported(argv, supported, "--kvn-self-cond-prob", "0.5")
        _append_if_supported(argv, supported, "--per-position-loss-weight", "deao")
    else:
        _append_if_supported(argv, supported, "--no-kvn-dyadic-rff")
        _append_if_supported(argv, supported, "--no-kvn-use-engram")
        _append_if_supported(argv, supported, "--no-kvn-symplectic-butterfly")
        _append_if_supported(argv, supported, "--no-kvn-lindblad-purity")
        _append_if_supported(argv, supported, "--no-kvn-use-kraus-tree")
        _append_if_supported(argv, supported, "--kvn-self-cond-prob", "0.0")

    return argv


@torch.no_grad()
def _forward_qwen35_fixed_causal(
    verifier: torch.nn.Module,
    seq_ids: list[int],
    aux_layer_ids: list[int],
    device: str,
    pad_to: int = FIXED_CAUSAL_PAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Qwen3.5-0.8B at fixed 64-multiple pad for 0.0-diff causal parity."""
    cur_len = len(seq_ids)
    eff_pad = max(pad_to, ((cur_len + 63) // 64) * 64)
    in_ids = torch.zeros((1, eff_pad), dtype=torch.long, device=device)
    in_ids[0, :cur_len] = torch.tensor(seq_ids, dtype=torch.long, device=device)
    attn_mask = torch.zeros((1, eff_pad), dtype=torch.long, device=device)
    attn_mask[0, :cur_len] = 1
    out = verifier(
        input_ids=in_ids,
        attention_mask=attn_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    hs = out.hidden_states
    aux = torch.cat([hs[idx + 1][:, :cur_len, :] for idx in aux_layer_ids], dim=-1)
    return aux, out.logits[:, :cur_len, :]


@torch.no_grad()
def _eval_kraus_tree_exact_qwen35(
    verifier: torch.nn.Module,
    confirmed_seq: list[int],
    anchor_tok_id: int,
    tree_ids: torch.Tensor,
    tree_parents: torch.Tensor,
    aux_layer_ids: list[int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Kraus POVM tree nodes on Qwen3.5 via root-to-leaf causal chains."""
    parents = tree_parents[0].tolist()
    toks = tree_ids[0].tolist()
    n_nodes = len(toks)

    node_chains: list[list[int]] = []
    for i in range(n_nodes):
        chain_rev = [toks[i]]
        p = parents[i]
        while p >= 0:
            chain_rev.append(toks[p])
            p = parents[p]
        chain_rev.reverse()
        node_chains.append(chain_rev)

    parent_set = {p for p in parents if p >= 0}
    leaves = [i for i in range(n_nodes) if i not in parent_set]
    node_to_leaf: dict[int, tuple[int, int]] = {}
    for l_pos, leaf_node in enumerate(leaves):
        cur = leaf_node
        ancestors_rev = []
        while cur >= 0:
            ancestors_rev.append(cur)
            cur = parents[cur]
        ancestors = list(reversed(ancestors_rev))
        for depth_idx, nid in enumerate(ancestors):
            if nid not in node_to_leaf:
                node_to_leaf[nid] = (l_pos, depth_idx + 1)

    prefix_len = len(confirmed_seq)
    leaf_aux: list[torch.Tensor] = []
    leaf_logits: list[torch.Tensor] = []
    for leaf_node in leaves:
        seq_leaf = [*confirmed_seq, anchor_tok_id, *node_chains[leaf_node]]
        aux_l, logits_l = _forward_qwen35_fixed_causal(
            verifier, seq_leaf, aux_layer_ids, device=device
        )
        leaf_aux.append(aux_l[0, prefix_len:, :])
        leaf_logits.append(logits_l[0, prefix_len:, :])

    vocab_sz = leaf_logits[0].size(-1)
    aux_dim = leaf_aux[0].size(-1)
    v_logits = torch.empty(
        (1, 1 + n_nodes, vocab_sz), dtype=leaf_logits[0].dtype, device=device
    )
    v_aux = torch.empty(
        (1, 1 + n_nodes, aux_dim), dtype=leaf_aux[0].dtype, device=device
    )
    v_logits[0, 0] = leaf_logits[0][0]
    v_aux[0, 0] = leaf_aux[0][0]
    for nid in range(n_nodes):
        l_pos, offset = node_to_leaf[nid]
        v_logits[0, 1 + nid] = leaf_logits[l_pos][offset]
        v_aux[0, 1 + nid] = leaf_aux[l_pos][offset]
    return v_aux, v_logits


@torch.no_grad()
def _evaluate_qwen35_hybrid_e2e(
    eval_mod: Any,
    verifier: torch.nn.Module,
    drafter: KvNDSparkDraftModel,
    prompt_ids_list: list[list[int]],
    base_greedy_tokens_list: list[list[int]] | None,
    eos_ids: set[int],
    aux_layer_ids: list[int],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    use_kraus_tree: bool = False,
    tree_budget: int = 16,
    branch_width: int = 3,
    device: str = "cuda",
) -> dict[str, Any]:
    """Lossless E2E speculative decoding on Qwen3.5-0.8B (chain & Kraus POVM tree)."""
    eval_dtype = torch.bfloat16
    drafter = drafter.to(device=device, dtype=eval_dtype).eval()
    block_size = int(drafter.config.block_size)
    sample_from_anchor = bool(drafter.config.sample_from_anchor)
    start_k = 0 if sample_from_anchor else 1
    n_chain_spec = block_size if sample_from_anchor else max(1, block_size - 1)
    backbone = drafter.config.kvn_backbone_type

    all_taus: list[float] = []
    draft_ms_list: list[float] = []
    exact_matches: list[bool] = []

    if temperature > 0.0:
        torch.manual_seed(2026)

    for p_idx, p_ids in enumerate(prompt_ids_list):
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=device)
        p_len = len(p_ids)
        aux_prefill, prefill_logits = _forward_qwen35_fixed_causal(
            verifier, p_ids, aux_layer_ids, device=device
        )
        fc_prefill = drafter.hidden_norm(drafter.fc(aux_prefill.to(eval_dtype)))
        y_first = eval_mod._sample_single_token(prefill_logits[0, -1, :], temperature)
        spec_gen: list[int] = [y_first]

        if y_first not in eos_ids and max_new_tokens > 1:
            state_cache = None
            swa_state = None
            cur_prev_fc = fc_prefill[:, -1:, :]
            cur_tok_chunk = torch.tensor([[y_first]], dtype=torch.long, device=device)
            pending_swa_fc = fc_prefill
            pending_swa_pos = torch.arange(
                p_len, device=device, dtype=torch.long
            ).unsqueeze(0)
            cur_anchor_pos = p_len
            confirmed_seq = list(p_ids)

            if backbone in ("kvn", "hybrid_1swa_2kvn"):
                state_cache = drafter.allocate_state_cache(batch_size=1, device=device)
                cur_prev_fc = eval_mod._prefill_kvn_layers_parallel(
                    drafter, prompt_t, fc_prefill, state_cache
                )
            if backbone in ("swa", "hybrid_1swa_2kvn"):
                swa_state = eval_mod._init_swa_cache_state(
                    drafter, bsz=1, device=device
                )

            hit_eos = False
            while len(spec_gen) < max_new_tokens and not hit_eos:
                anchor_tok = cur_tok_chunk[:, -1:]
                anchor_id = int(anchor_tok.item())
                prefix_len = len(confirmed_seq)
                vlen = torch.tensor(
                    [cur_tok_chunk.size(1)], dtype=torch.long, device=device
                )
                s_pad = max(cur_tok_chunk.size(1), block_size + 1)

                torch.cuda.synchronize(device)
                t_d0 = time.perf_counter()

                if use_kraus_tree and backbone == "kvn" and state_cache is not None:
                    tree_dict = eval_mod._draft_step_kraus_tree(
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
                    torch.cuda.synchronize(device)
                    draft_ms_list.append((time.perf_counter() - t_d0) * 1000.0)

                    tree_ids = tree_dict["tree_draft_ids"]
                    tree_parents = tree_dict["tree_parent_idx"]
                    v_aux, v_logits = _eval_kraus_tree_exact_qwen35(
                        verifier=verifier,
                        confirmed_seq=confirmed_seq,
                        anchor_tok_id=anchor_id,
                        tree_ids=tree_ids,
                        tree_parents=tree_parents,
                        aux_layer_ids=aux_layer_ids,
                        device=device,
                    )
                    has_stoch_q = bool(
                        tree_dict.get(
                            "stochastic_proposals",
                            temperature > 0.0
                            and tree_dict.get("tree_q_probs") is not None,
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
                    acc_slice = tree_res.accepted_tokens[0]
                    acc_node_idx = tree_res.accepted_node_indices[0]
                    bonus_tok = int(tree_res.bonus_tokens[0].item())
                    gather_aux = torch.tensor(
                        [0, *(1 + i for i in acc_node_idx)],
                        dtype=torch.long,
                        device=device,
                    )
                    confirmed_v_aux = v_aux.index_select(dim=1, index=gather_aux).to(
                        eval_dtype
                    )
                else:
                    prior_ctx = (p_ids + spec_gen[:-1])[-2:]
                    prev_ctx_t = (
                        torch.tensor([prior_ctx], dtype=torch.long, device=device)
                        if prior_ctx
                        else None
                    )
                    anc_pos_t = torch.tensor(
                        [cur_anchor_pos], dtype=torch.long, device=device
                    )
                    if backbone == "kvn" and state_cache is not None:
                        raw_draft_ids, raw_q_probs = eval_mod._draft_step_kvn_recurrent(
                            drafter=drafter,
                            prev_fc_chunk=cur_prev_fc,
                            token_ids_chunk=cur_tok_chunk,
                            valid_lens=vlen,
                            state_cache=state_cache,
                            temperature=temperature,
                            s_pad_to=s_pad,
                            prev_context_ids=prev_ctx_t,
                        )
                    else:
                        raw_draft_ids, raw_q_probs = eval_mod._draft_step_swa_batched(
                            drafter=drafter,
                            new_target_fc=pending_swa_fc,
                            new_target_pos=pending_swa_pos,
                            anchor_token_ids=anchor_tok.squeeze(1),
                            anchor_pos=anc_pos_t,
                            swa_state=swa_state,
                            temperature=temperature,
                            prev_context_ids=prev_ctx_t,
                        )
                    torch.cuda.synchronize(device)
                    draft_ms_list.append((time.perf_counter() - t_d0) * 1000.0)

                    draft_ids = raw_draft_ids[:, start_k : start_k + n_chain_spec]
                    draft_q_probs = (
                        raw_q_probs[:, start_k : start_k + n_chain_spec]
                        if raw_q_probs.numel() > 0
                        else raw_q_probs
                    )
                    verify_seq = [*confirmed_seq, anchor_id, *draft_ids[0].tolist()]
                    aux_full, logits_full = _forward_qwen35_fixed_causal(
                        verifier, verify_seq, aux_layer_ids, device=device
                    )
                    v_logits_step = logits_full[
                        0, prefix_len : prefix_len + 1 + n_chain_spec, :
                    ]
                    v_aux_step = aux_full[
                        :, prefix_len : prefix_len + 1 + n_chain_spec, :
                    ]
                    n_acc, bonus_tok = eval_mod._verify_chain_logits(
                        v_logits_step,
                        draft_ids[0],
                        draft_q_probs,
                        n_chain_spec,
                        temperature,
                        device,
                    )
                    acc_slice = draft_ids[0, :n_acc].tolist()
                    confirmed_v_aux = v_aux_step[:, : 1 + n_acc, :].to(eval_dtype)

                confirmed_seq.extend([anchor_id, *acc_slice])
                all_taus.append(float(1 + len(acc_slice)))
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

        if temperature <= 0.0 and base_greedy_tokens_list is not None:
            exact_matches.append(spec_gen == base_greedy_tokens_list[p_idx])

    drafter.cpu()
    torch.cuda.empty_cache()
    return {
        "mean_accept_len_tau": float(np.mean(all_taus)) if all_taus else 1.0,
        "std_accept_len_tau": float(np.std(all_taus)) if all_taus else 0.0,
        "mean_draft_step_ms": float(np.mean(draft_ms_list)) if draft_ms_list else 0.0,
        "exact_greedy_match_all_prompts": bool(all(exact_matches))
        if exact_matches
        else True,
    }


@torch.no_grad()
def _build_uniform_loaders_and_kwargs(
    verifier_id: str,
    data_dir: Path,
    save_probe_dir: Path,
    target_layer_ids: list[int],
    seq_len: int,
    draft_vocab_size: int,
    train_ratio: float,
    hidden_size: int = 1024,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build identical train/val loaders and uniform eval kwargs for parity."""
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            verifier_id,
            "--speculator-type",
            "kvn_dspark",
            "--data-path",
            str(data_dir),
            "--save-path",
            str(save_probe_dir),
            "--draft-vocab-size",
            str(draft_vocab_size),
            "--total-seq-len",
            str(seq_len),
            "--train-data-ratio",
            str(train_ratio),
            "--num-workers",
            "0",
            "--loss-implementation",
            "eager",
            "--loss-fn",
            '{"ce": 0.1, "tv": 0.9}',
            "--per-position-loss-weight",
            "fixed-exp-decay",
            "--block-size",
            "8",
            "--max-anchors",
            "48",
            "--target-layer-ids",
            *(str(x) for x in target_layer_ids),
            "--on-missing",
            "raise",
            "--seed",
            "42",
        ]
    )
    args = argparse.Namespace(**cfg.flatten())
    backend_cls = HiddenStatesBackend.registry[args.hidden_states_backend]
    transfer = backend_cls.from_train_args(args, args.data_path)
    train_loader, val_loader = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=torch.bfloat16,
        transfer=transfer,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        generation_validation_retries=args.generation_validation_retries,
        max_consecutive_generation_failures=args.max_consecutive_generation_failures,
        hidden_size=hidden_size,
        num_target_layers=len(target_layer_ids),
        num_workers=0,
        prefetch_factor=args.prefetch_factor,
        max_transfer_workers=args.max_transfer_workers,
        preprocess=None,
        train_data_ratio=args.train_data_ratio,
        seed=args.seed,
    )
    _, uniform_val_kwargs = KvNDSparkDraftModel.get_trainer_kwargs(**vars(args))
    return train_loader, val_loader, uniform_val_kwargs


@torch.no_grad()
def _evaluate_loader_metrics_uniform(
    model: KvNDSparkDraftModel,
    loader: Any,
    val_call_kwargs: dict[str, Any],
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate uniform fixed-exp-decay teacher-forced metrics across a DataLoader."""
    if loader is None:
        return {}
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)
    accumulated: dict[str, torch.Tensor] = {}
    num_batches = len(loader)
    for batch in loader:
        gpu_batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _draft_tokens, _loss, metrics = model(**gpu_batch, **val_call_kwargs)
        for k, v in metrics.items():
            acc = accumulated.get(k)
            accumulated[k] = v.float() if acc is None else acc + v.float()
    model.cpu()
    torch.cuda.empty_cache()
    if not accumulated or num_batches <= 0:
        return {}
    raw = {k: v.item() / num_batches for k, v in accumulated.items()}
    normed = normalize_counted_metrics(raw, world_size=1)
    return {f"{k}_epoch": v for k, v in normed.items()}


@torch.no_grad()
def _measure_batched_drafter_latency_and_vram(
    model: KvNDSparkDraftModel,
    eval_mod: Any,
    batch_sizes: list[int],
    device: str = "cuda",
    warmup_steps: int = 5,
    timed_steps: int = 20,
) -> dict[str, dict[str, float]]:
    """Measure batched steady-state drafter latency and peak VRAM across B=[1,8,32]."""
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    d_model = model.config.transformer_layer_config.hidden_size
    vocab_lim = int(model.config.draft_vocab_size or 4096)
    backbone = model.config.kvn_backbone_type

    profile: dict[str, dict[str, float]] = {}
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        prefill_len = 128

        if backbone == "kvn":
            init_ids = torch.randint(1, vocab_lim, (bs, prefill_len), device=device)
            init_fc = torch.randn(
                bs, prefill_len, d_model, device=device, dtype=torch.bfloat16
            )
            st = model.prefill_kvn_state(init_ids, init_fc)
            state_cache = st["state_cache"]

            step_fc = torch.randn(bs, 4, d_model, device=device, dtype=torch.bfloat16)
            step_ids = torch.randint(1, vocab_lim, (bs, 4), device=device)
            valid_lens = torch.full((bs,), 3, dtype=torch.long, device=device)

            # Note: state_cache is updated in-place inside advance_and_draft_step!
            for _ in range(warmup_steps):
                _ = model.advance_and_draft_step(
                    step_fc, step_ids, valid_lens, state_cache
                )

            torch.cuda.synchronize(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(timed_steps):
                _ = model.advance_and_draft_step(
                    step_fc, step_ids, valid_lens, state_cache
                )
            end_ev.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_ev.elapsed_time(end_ev)) / timed_steps
        else:
            # Incremental batched SWA DynamicCache pass at full W=128 steady state
            swa_state = eval_mod._init_swa_cache_state(model, bsz=bs, device=device)
            init_fc = torch.randn(
                bs, prefill_len, d_model, device=device, dtype=torch.bfloat16
            )
            init_pos = (
                torch.arange(prefill_len, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(bs, -1)
            )
            init_anc_ids = torch.randint(1, vocab_lim, (bs,), device=device)
            init_anc_pos = torch.full(
                (bs,), prefill_len, device=device, dtype=torch.long
            )
            _ = eval_mod._draft_step_swa_batched(
                drafter=model,
                new_target_fc=init_fc,
                new_target_pos=init_pos,
                anchor_token_ids=init_anc_ids,
                anchor_pos=init_anc_pos,
                swa_state=swa_state,
                temperature=0.0,
            )

            cur_len = prefill_len
            step_fc = torch.randn(bs, 3, d_model, device=device, dtype=torch.bfloat16)
            step_anc_ids = torch.randint(1, vocab_lim, (bs,), device=device)

            for _ in range(warmup_steps):
                step_pos = (
                    torch.arange(cur_len, cur_len + 3, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(bs, -1)
                )
                anc_pos = torch.full(
                    (bs,), cur_len + 3, device=device, dtype=torch.long
                )
                _ = eval_mod._draft_step_swa_batched(
                    drafter=model,
                    new_target_fc=step_fc,
                    new_target_pos=step_pos,
                    anchor_token_ids=step_anc_ids,
                    anchor_pos=anc_pos,
                    swa_state=swa_state,
                    temperature=0.0,
                )
                cur_len += 3

            torch.cuda.synchronize(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(timed_steps):
                step_pos = (
                    torch.arange(cur_len, cur_len + 3, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(bs, -1)
                )
                anc_pos = torch.full(
                    (bs,), cur_len + 3, device=device, dtype=torch.long
                )
                _ = eval_mod._draft_step_swa_batched(
                    drafter=model,
                    new_target_fc=step_fc,
                    new_target_pos=step_pos,
                    anchor_token_ids=step_anc_ids,
                    anchor_pos=anc_pos,
                    swa_state=swa_state,
                    temperature=0.0,
                )
                cur_len += 3
            end_ev.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_ev.elapsed_time(end_ev)) / timed_steps

        peak_vram_mb = float(torch.cuda.max_memory_allocated(device)) / (
            1024.0 * 1024.0
        )
        draft_tok_per_s = (bs * model.block_size) / max(1e-6, elapsed_ms * 1e-3)
        profile[str(bs)] = {
            "step_latency_ms": round(elapsed_ms, 4),
            "draft_tokens_per_s": round(draft_tok_per_s, 1),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }

    model.cpu()
    torch.cuda.empty_cache()
    return profile


@torch.no_grad()
def _run_vllm_proposer_and_apc_smoke_benchmark(
    ckpt_dir: Path,
    t2d: torch.Tensor,
    d2t: torch.Tensor,
    device: str = "cuda",
) -> dict[str, Any]:
    """Exercise KvNDSparkVllmProposer, APC, preemption, and AmdahlScheduler on GPU."""
    drafter = KvNDSparkDraftModel.from_pretrained(ckpt_dir, t2d=t2d, d2t=d2t).to(
        device=device, dtype=torch.bfloat16
    )
    drafter.eval()
    d_model = drafter.config.transformer_layer_config.hidden_size
    roofline = HardwareRooflineProfile.rtx_3070_ti_laptop()

    proposer = KvNDSparkVllmProposer(
        drafter,
        max_num_seqs=16,
        device=device,
        enable_apc=True,
        apc_block_size=16,
        use_kraus_tree=True,
        tree_budget=16,
        branch_width=3,
        roofline=roofline,
    )
    assert proposer.slot_pool is not None
    assert proposer.apc_cache is not None

    # 1. Single-chunk 64-token prefill vs chunked 32 + 32 prefill continuation parity
    torch.manual_seed(1337)
    toks_64 = torch.randint(10, 2000, (64,), device=device).tolist()
    fc_64 = torch.randn(64, d_model, device=device, dtype=torch.bfloat16) * 0.25

    slot_full, _ = proposer.prefill_request(
        "req_full_64", toks_64, fc_64, already_projected_fc=True, use_apc=True
    )
    slot_chunk, _ = proposer.prefill_request(
        "req_chunk_64",
        toks_64[:32],
        fc_64[:32],
        already_projected_fc=True,
        use_apc=False,
    )
    proposer.prefill_request(
        "req_chunk_64",
        toks_64[32:],
        fc_64[32:],
        already_projected_fc=True,
        use_apc=False,
    )

    chunk_q_err = float(
        (proposer.slot_pool.q_pool[slot_full] - proposer.slot_pool.q_pool[slot_chunk])
        .abs()
        .max()
        .item()
    )
    chunk_p_err = float(
        (proposer.slot_pool.p_pool[slot_full] - proposer.slot_pool.p_pool[slot_chunk])
        .abs()
        .max()
        .item()
    )
    chunk_fc_err = float(
        (
            proposer.slot_pool.last_fc_pool[slot_full].float()
            - proposer.slot_pool.last_fc_pool[slot_chunk].float()
        )
        .abs()
        .max()
        .item()
    )
    proposer.free_finished_requests(["req_full_64", "req_chunk_64"])

    # 2. Multi-turn Zero-Replay APC 0-FLOP prefix restore on 80-token prompt (4x16)
    toks_80 = toks_64 + torch.randint(10, 2000, (16,), device=device).tolist()
    fc_80 = torch.cat(
        [fc_64, torch.randn(16, d_model, device=device, dtype=torch.bfloat16) * 0.25],
        dim=0,
    )

    slot_apc, restored_toks = proposer.prefill_request(
        "req_apc_hit", toks_80, fc_80, already_projected_fc=True, use_apc=True
    )
    slot_ref, _ = proposer.prefill_request(
        "req_apc_ref", toks_80, fc_80, already_projected_fc=True, use_apc=False
    )
    apc_restore_max_err = float(
        max(
            (proposer.slot_pool.q_pool[slot_apc] - proposer.slot_pool.q_pool[slot_ref])
            .abs()
            .max()
            .item(),
            (proposer.slot_pool.p_pool[slot_apc] - proposer.slot_pool.p_pool[slot_ref])
            .abs()
            .max()
            .item(),
        )
    )

    # 3. Preemption to CPU host memory + zero-loss resume + chain & Kraus-tree propose
    q_before_preempt = proposer.slot_pool.q_pool[slot_apc].clone()
    proposer.preempt_requests(["req_apc_hit"])
    slot_resumed = proposer.slot_pool.resolve_slot_or_restore("req_apc_hit")
    preempt_restore_err = float(
        (proposer.slot_pool.q_pool[slot_resumed] - q_before_preempt).abs().max().item()
    )

    step_fc = torch.randn(1, d_model, device=device, dtype=torch.bfloat16) * 0.25
    out_chain = proposer.propose_batch(
        ["req_apc_hit"],
        [[101]],
        [step_fc],
        already_projected_fc=True,
        temperature=0.0,
        use_kraus_tree=False,
    )
    out_tree = proposer.propose_batch(
        ["req_apc_ref"],
        [[101]],
        [step_fc],
        already_projected_fc=True,
        temperature=0.0,
        use_kraus_tree=True,
    )
    apc_mem = proposer.apc_cache.memory_stats()
    slot_bytes_breakdown = drafter.config.estimated_state_bytes_breakdown()
    proposer.free_finished_requests(["req_apc_hit", "req_apc_ref"])

    # 4. Pure-Attention Qwen3ForCausalLM 4D Tree Mask + compact_tree_kv_cache check
    cfg_q3 = Qwen3Config(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    pure_q3 = Qwen3ForCausalLM(cfg_q3).to(device=device, dtype=torch.float32).eval()
    pref_q3 = torch.randint(5, 1000, (1, 16), device=device)
    kv_q3 = DynamicCache()
    _ = pure_q3(input_ids=pref_q3, past_key_values=kv_q3, use_cache=True)
    t_parents = torch.tensor([[-1, 0, 0, 1, 2]], device=device)
    t_mask, t_depths = build_tree_attention_mask(
        tree_parent_idx=t_parents, prefix_len=16, dtype=torch.float32
    )
    v_in = torch.tensor([[17, 21, 22, 23, 24, 25]], device=device)
    p_ids_q3 = torch.cat([torch.tensor([[16]], device=device), 16 + t_depths], dim=1)
    _ = pure_q3(
        input_ids=v_in,
        past_key_values=kv_q3,
        use_cache=True,
        attention_mask=t_mask,
        position_ids=p_ids_q3,
    )
    _ = compact_tree_kv_cache(kv_q3, 16, [[0, 2, 4]])
    nxt_q3 = pure_q3(
        input_ids=torch.tensor([[99]], device=device),
        past_key_values=kv_q3,
        use_cache=True,
    )
    ref_q3 = pure_q3(
        input_ids=torch.cat(
            [pref_q3, torch.tensor([[17, 21, 23, 25, 99]], device=device)], dim=1
        ),
        use_cache=False,
    )
    tree_kv_compact_err = float(
        (nxt_q3.logits[0, -1] - ref_q3.logits[0, -1]).abs().max().item()
    )
    del pure_q3

    # 5. PurityLoadAwareAmdahlScheduler horizon & tree budget across B in {1, 8, 32, 64}
    sched = PurityLoadAwareAmdahlScheduler(roofline=roofline, max_horizon=7)
    sched_horizons: dict[str, Any] = {}
    sample_conf = torch.tensor(
        [[0.91, 0.84, 0.77, 0.70, 0.62, 0.54, 0.46]], device=device
    )
    sample_purity = torch.tensor(
        [[0.96, 0.93, 0.90, 0.87, 0.83, 0.79, 0.75]], device=device
    )
    for b_load in (1, 8, 32, 64):
        dec_h = sched.select_horizon(
            confidences=sample_conf,
            active_batch_size=b_load,
            purity=sample_purity,
            temperature=0.0,
        )
        dec_t = sched.select_tree_budget(
            confidences=sample_conf,
            active_batch_size=b_load,
            purity=sample_purity,
            temperature=0.0,
        )
        sched_horizons[f"B={b_load}"] = {
            "linear_horizon_K": int(dec_h.batch_horizon),
            "kraus_tree_budget_N": int(dec_t.batch_tree_budget),
            "predicted_speedup": round(float(dec_h.predicted_speedup), 4),
        }

    drafter.cpu()
    torch.cuda.empty_cache()
    return {
        "proposer_class": KvNDSparkVllmProposer.__name__,
        "instantiated": True,
        "chunked_prefill_32_plus_32_max_abs_err": round(
            max(chunk_q_err, chunk_p_err, chunk_fc_err), 8
        ),
        "preemption_cpu_swap_restore_max_abs_err": round(preempt_restore_err, 8),
        "apc_restored_prefix_tokens": int(restored_toks),
        "apc_zero_replay_restore_max_abs_err": round(apc_restore_max_err, 8),
        "apc_zero_replay_flops": 0,
        "apc_dynamical_bytes_per_token": apc_mem["recurrent_bytes_per_token"],
        "apc_total_bytes_per_token_with_last_fc": apc_mem["total_bytes_per_token"],
        "slot_state_bytes_breakdown": slot_bytes_breakdown,
        "propose_chain_len": len(out_chain.draft_token_ids[0]),
        "propose_kraus_tree_nodes": int(out_tree.padded_draft_ids.size(1)),
        "pure_attention_tree_kv_compaction_max_abs_err": round(tree_kv_compact_err, 7),
        "amdahl_scheduler_decisions": sched_horizons,
    }


def _run_wsl_triton_and_vllm_probe(
    ckpt_dir: Path, data_dir: Path, draft_vocab_size: int
) -> dict[str, Any]:
    """Execute live Linux Triton 3.7.1 Mega-KvN + vLLM V1 E2E benchmark in WSL2."""
    wsl_py = "/home/blancsw/vllm-infomaniak/.venv/bin/python"
    repo_wsl = "/mnt/c/Users/swanb/PycharmProjects/KvN-RFF-RNN/speculators"
    bench_wsl = repo_wsl + "/kvn_rff_mtp/_wsl_triton_vllm_v3_bench.py"
    ckpt_wsl = repo_wsl + "/" + ckpt_dir.relative_to(REPO_ROOT).as_posix()
    data_wsl = repo_wsl + "/" + data_dir.relative_to(REPO_ROOT).as_posix()

    print("\n=== [WSL2 CUDA] Running Live Triton 3.7.1 + vLLM V1 E2E GPU Benchmark ===")
    try:
        proc = subprocess.run(
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                "env",
                "HF_HOME=/mnt/c/Users/swanb/.cache/huggingface",
                f"PYTHONPATH={repo_wsl}/src:{repo_wsl}/hs_connectors/src",
                wsl_py,
                bench_wsl,
                "--checkpoint",
                ckpt_wsl,
                "--verifier-model",
                "Qwen/Qwen3.5-0.8B",
                "--data-dir",
                data_wsl,
                "--draft-vocab-size",
                str(draft_vocab_size),
            ],
            capture_output=True,
            text=True,
            timeout=360,
            check=False,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("WSL_JSON:"):
                parsed = json.loads(line[len("WSL_JSON:") :])
                chain_ok = parsed["live_wsl_e2e_linear_chain"][
                    "exact_greedy_parity_all_prompts"
                ]
                kraus_ok = parsed["live_wsl_e2e_kraus_tree_specinfer"][
                    "exact_greedy_parity_all_prompts"
                ]
                print(
                    f"[WSL2 Linux CUDA] Triton={parsed.get('triton_version')} | "
                    f"vLLM={parsed.get('vllm_version')} | "
                    f"Chain Exact={chain_ok} | Kraus Exact={kraus_ok}"
                )
                return parsed
        return {
            "wsl_available": False,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-800:],
        }
    except Exception as exc:
        return {"wsl_available": False, "error": str(exc)}


def run_v3_rtx3070ti_benchmark() -> dict[str, Any]:
    verifier_id = "Qwen/Qwen3.5-0.8B"
    target_layer_ids = [2, 12, 21]
    base_dir = REPO_ROOT / "kvn_rff_mtp" / "rtx3070ti_validation_run" / "v3_run"
    data_dir = base_dir / "data"
    seq_len = 256
    draft_vocab_size = 4096
    train_ratio = 0.80
    total_iso_epochs = 6
    max_eval_new_tokens = 24
    kraus_tree_budget = 9
    kraus_branch_width = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prep_meta = prepare_onpolicy_qwen_dataset(
        verifier_id=verifier_id,
        data_dir=data_dir,
        target_layer_ids=target_layer_ids,
        num_samples=40,
        train_ratio=train_ratio,
        seq_len=seq_len,
        device=device,
    )

    supported_flags = _supported_cli_flags()
    eval_mod = _load_eval_script_module()
    train_loader_uni, val_loader_uni, uniform_val_kwargs = (
        _build_uniform_loaders_and_kwargs(
            verifier_id=verifier_id,
            data_dir=data_dir,
            save_probe_dir=base_dir / "_loader_probe",
            target_layer_ids=target_layer_ids,
            seq_len=seq_len,
            draft_vocab_size=draft_vocab_size,
            train_ratio=train_ratio,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(verifier_id, local_files_only=True)
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    eval_prompt_ids: list[list[int]] = []
    for p_txt in HELD_OUT_EVAL_PROMPTS:
        msgs = [{"role": "user", "content": p_txt}]
        try:
            pids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            pids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            )
        if not isinstance(pids, list):
            pids = list(pids["input_ids"])
        eval_prompt_ids.append(pids[:88])

    # Strict 6-epoch iso-compute comparison across all 4 models!
    ablation_specs: list[dict[str, Any]] = [
        {
            "id": "B1_SWA",
            "kvn_backbone_type": "swa",
            "kvn_n_modes": 128,
            "prune_to": None,
            "epochs": 6,
            "kvn_lasso_weight": 0.0,
            "kvn_use_born_confidence": False,
            "markov_head_type": "rnn",
            "is_v3": False,
        },
        {
            "id": "M_v2_Fixed",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 128,
            "prune_to": None,
            "epochs": 6,
            "kvn_lasso_weight": 0.0,
            "kvn_use_born_confidence": True,
            "markov_head_type": "gated",
            "is_v3": False,
        },
        {
            "id": "M_v3_Full",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 128,
            "prune_to": None,
            "epochs": 6,
            "kvn_lasso_weight": 0.0,
            "kvn_use_born_confidence": True,
            "markov_head_type": "rnn",
            "is_v3": True,
        },
        {
            "id": "M_v3_ProxLASSO_Pruned",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 512,
            "prune_to": 128,
            "epochs": 5,  # Strictly iso-compute: 5 LASSO + 1 warm recal = 6 total!
            "kvn_lasso_weight": 0.592,  # Calibrated proximal threshold
            "kvn_use_born_confidence": True,
            "markov_head_type": "rnn",
            "is_v3": True,
        },
    ]

    t2d_tensor = torch.from_numpy(np.load(data_dir / f"t2d-{draft_vocab_size}.npy"))
    d2t_tensor = torch.from_numpy(np.load(data_dir / f"d2t-{draft_vocab_size}.npy"))

    # Phase 1: Train all 4 models first while verifier is NOT occupying GPU VRAM!
    trained_records: list[dict[str, Any]] = []
    for spec in ablation_specs:
        model_id = spec["id"]
        save_path = base_dir / "checkpoints" / model_id
        last_ep_dir = save_path / str(spec["epochs"] - 1)

        if (
            spec["prune_to"] is None
            and (last_ep_dir / "model.safetensors").exists()
            and (save_path / "run.yaml").exists()
        ):
            train_wall_s = max(
                1.0,
                float(
                    (last_ep_dir / "model.safetensors").stat().st_mtime
                    - (save_path / "run.yaml").stat().st_mtime
                ),
            )
            print(
                f"\n=== [RTX 3070 Ti] Reusing {spec['epochs']}-epoch checkpoint "
                f"for {model_id} (train_wall_s={train_wall_s:.2f}s) ==="
            )
            loaded_model = KvNDSparkDraftModel.from_pretrained(
                last_ep_dir, t2d=t2d_tensor, d2t=d2t_tensor
            )
        elif spec["prune_to"] is None:
            if save_path.exists():
                shutil.rmtree(save_path)
            argv = _build_model_argv(
                spec=spec,
                verifier_id=verifier_id,
                data_dir=data_dir,
                save_path=save_path,
                target_layer_ids=target_layer_ids,
                seq_len=seq_len,
                draft_vocab_size=draft_vocab_size,
                train_ratio=train_ratio,
                supported=supported_flags,
            )
            print(
                f"\n=== [RTX 3070 Ti] Training {model_id} ({spec['epochs']} epochs) ==="
            )
            cfg = TrainConfig.resolve(argv)
            t0 = time.perf_counter()
            speculators_train_main(cfg)
            train_wall_s = time.perf_counter() - t0
            torch.cuda.empty_cache()
            loaded_model = KvNDSparkDraftModel.from_pretrained(
                last_ep_dir, t2d=t2d_tensor, d2t=d2t_tensor
            )
        else:
            # Proximal Group-LASSO run with automatic bracket guard in [0.52, 0.72]
            lw_low, lw_high = 0.52, 0.72
            lw_cur = float(spec["kvn_lasso_weight"])
            train_wall_s = 0.0
            loaded_model = None
            for _attempt in range(3):
                spec["kvn_lasso_weight"] = round(lw_cur, 4)
                if save_path.exists():
                    shutil.rmtree(save_path)
                argv = _build_model_argv(
                    spec=spec,
                    verifier_id=verifier_id,
                    data_dir=data_dir,
                    save_path=save_path,
                    target_layer_ids=target_layer_ids,
                    seq_len=seq_len,
                    draft_vocab_size=draft_vocab_size,
                    train_ratio=train_ratio,
                    supported=supported_flags,
                )
                print(
                    f"\n=== [RTX 3070 Ti] Training {model_id} ({spec['epochs']} "
                    f"epochs, kvn_lasso_weight={spec['kvn_lasso_weight']}) ==="
                )
                cfg = TrainConfig.resolve(argv)
                t0 = time.perf_counter()
                speculators_train_main(cfg)
                train_wall_s = time.perf_counter() - t0
                torch.cuda.empty_cache()

                loaded_model = KvNDSparkDraftModel.from_pretrained(
                    last_ep_dir, t2d=t2d_tensor, d2t=d2t_tensor
                )
                with torch.no_grad():
                    rms_probe = loaded_model._compute_joint_dyad_rms_norms()
                    act_probe = int((rms_probe > 1e-8).sum().item())
                print(
                    f"[ProxLASSO Check] lw={spec['kvn_lasso_weight']} -> "
                    f"active_dyads={act_probe}/256"
                )
                if act_probe == 0:
                    lw_high = lw_cur
                    lw_cur = 0.5 * (lw_low + lw_high)
                elif act_probe == 256:
                    lw_low = lw_cur
                    lw_cur = 0.5 * (lw_low + lw_high)
                else:
                    break
            assert loaded_model is not None

        eval_ckpt_dir = last_ep_dir
        prox_meta: dict[str, Any] | None = None

        if spec["prune_to"] is not None:
            target_m = int(spec["prune_to"])
            unpruned_val_m = _evaluate_loader_metrics_uniform(
                loaded_model, val_loader_uni, uniform_val_kwargs, device=device
            )
            unpruned_val_loss = float(unpruned_val_m.get("loss_epoch", 0.0))

            zero_dyad_frac = 0.0
            active_dyads_remaining = 0
            if loaded_model.kvn_layers:
                with torch.no_grad():
                    rms_norms = loaded_model._compute_joint_dyad_rms_norms()
                    zero_dyad_frac = float((rms_norms <= 1e-8).float().mean().item())
                    active_dyads_remaining = int((rms_norms > 1e-8).sum().item())
            trainer_logged_zero_frac = zero_dyad_frac

            with torch.no_grad():
                probe_x = torch.randn(1, 16, 1024, dtype=torch.float32)
                probe_r = torch.zeros(1, 16, dtype=torch.bool)
                probe_r[:, 0] = True
                loaded_model.float().eval()
                pre_out = loaded_model.kvn_layers[0](probe_x, reset_mask=probe_r)[0]

            loaded_model.prune_kvn_modes(target_m)

            with torch.no_grad():
                post_out = loaded_model.kvn_layers[0](probe_x, reset_mask=probe_r)[0]
                zeroshot_layer0_max_drift = float(
                    (pre_out - post_out).abs().max().item()
                )

            pruned_dir = save_path / f"pruned_zeroshot_m{target_m}"
            sd = convert_float_dtype(loaded_model.state_dict(), torch.bfloat16)
            loaded_model.save_pretrained(pruned_dir, state_dict=sd)
            patch_config_dtype(pruned_dir / "config.json", torch.bfloat16)

            recal_dir = save_path / f"pruned_recal_m{target_m}"
            if recal_dir.exists():
                shutil.rmtree(recal_dir)
            recal_argv = [
                "--verifier-name-or-path",
                verifier_id,
                "--speculator-type",
                "kvn_dspark",
                "--from-pretrained",
                str(pruned_dir),
                "--data-path",
                str(data_dir),
                "--save-path",
                str(recal_dir),
                "--draft-vocab-size",
                str(draft_vocab_size),
                "--total-seq-len",
                str(seq_len),
                "--train-data-ratio",
                str(train_ratio),
                "--num-workers",
                "0",
                "--epochs",
                "1",
                "--lr",
                "1e-4",
                "--muon-lr",
                "4e-4",
                "--optimizer",
                "muon",
                "--optimizer-implementation",
                "eager",
                "--scheduler-type",
                "cosine",
                "--loss-implementation",
                "eager",
                "--loss-fn",
                '{"ce": 0.1, "tv": 0.9}',
                "--per-position-loss-weight",
                "deao",
                "--block-size",
                "8",
                "--max-anchors",
                "48",
                "--draft-attn-impl",
                "eager",
                "--target-layer-ids",
                *(str(x) for x in target_layer_ids),
                "--kvn-lasso-weight",
                "0.0",
                "--on-missing",
                "raise",
                "--seed",
                "42",
            ]
            recal_cfg = TrainConfig.resolve(recal_argv)
            t_rec = time.perf_counter()
            speculators_train_main(recal_cfg)
            train_wall_s += time.perf_counter() - t_rec
            torch.cuda.empty_cache()

            eval_ckpt_dir = recal_dir / "0"
            loaded_model = KvNDSparkDraftModel.from_pretrained(
                eval_ckpt_dir, t2d=t2d_tensor, d2t=d2t_tensor
            )

            prox_meta = {
                "unpruned_ckpt_dir": last_ep_dir,
                "zeroshot_ckpt_dir": pruned_dir,
                "trainer_logged_zero_frac": trainer_logged_zero_frac,
                "zero_dyad_frac": zero_dyad_frac,
                "active_dyads_remaining": active_dyads_remaining,
                "unpruned_val_loss": unpruned_val_loss,
                "zeroshot_layer0_max_drift": zeroshot_layer0_max_drift,
            }
            print(
                f"[ProxLASSO] zero_dyad_frac={zero_dyad_frac:.4f} "
                f"(active={active_dyads_remaining}/256) | "
                f"layer0_drift={zeroshot_layer0_max_drift:.6f}"
            )

        # Evaluate uniform fixed-exp-decay train & val metrics on final checkpoint
        train_m = _evaluate_loader_metrics_uniform(
            loaded_model, train_loader_uni, uniform_val_kwargs, device=device
        )
        val_m = _evaluate_loader_metrics_uniform(
            loaded_model, val_loader_uni, uniform_val_kwargs, device=device
        )

        trained_records.append(
            {
                "spec": spec,
                "eval_ckpt_dir": eval_ckpt_dir,
                "train_wall_s": train_wall_s,
                "train_m": train_m,
                "val_m": val_m,
                "prox_meta": prox_meta,
            }
        )

    del train_loader_uni, val_loader_uni
    if (base_dir / "_loader_probe").exists():
        shutil.rmtree(base_dir / "_loader_probe")

    # Phase 2: Load Qwen3.5-0.8B verifier ONCE and precompute greedy baseline tokens
    print(f"\n=== [RTX 3070 Ti] Loading {verifier_id} for Lossless E2E & Calib ===")
    verifier_model = (
        AutoModelForCausalLM.from_pretrained(
            verifier_id, dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
    )

    base_greedy_tokens: list[list[int]] = []
    with torch.no_grad():
        for p_ids in eval_prompt_ids:
            seq = list(p_ids)
            gen: list[int] = []
            for _ in range(max_eval_new_tokens):
                _, logits = _forward_qwen35_fixed_causal(
                    verifier_model, seq, target_layer_ids, device=device
                )
                nxt = int(logits[0, -1].argmax(dim=-1).item())
                gen.append(nxt)
                seq.append(nxt)
                if nxt in eos_ids:
                    break
            base_greedy_tokens.append(gen)

    model_results: list[dict[str, Any]] = []
    final_v3_ckpt_dir: Path | None = None

    for rec in trained_records:
        spec = rec["spec"]
        model_id = spec["id"]
        eval_ckpt_dir = rec["eval_ckpt_dir"]
        train_m = rec["train_m"]
        val_m = rec["val_m"]
        prox_meta = rec["prox_meta"]

        if model_id == "M_v3_Full":
            final_v3_ckpt_dir = eval_ckpt_dir

        loaded_model = KvNDSparkDraftModel.from_pretrained(
            eval_ckpt_dir, t2d=t2d_tensor, d2t=d2t_tensor
        )

        prox_diagnostics: dict[str, Any] | None = None
        if prox_meta is not None:
            unpruned_model = KvNDSparkDraftModel.from_pretrained(
                prox_meta["unpruned_ckpt_dir"], t2d=t2d_tensor, d2t=d2t_tensor
            )
            zs_model = KvNDSparkDraftModel.from_pretrained(
                prox_meta["zeroshot_ckpt_dir"], t2d=t2d_tensor, d2t=d2t_tensor
            )
            unpruned_e2e = _evaluate_qwen35_hybrid_e2e(
                eval_mod=eval_mod,
                verifier=verifier_model,
                drafter=unpruned_model,
                prompt_ids_list=eval_prompt_ids,
                base_greedy_tokens_list=base_greedy_tokens,
                eos_ids=eos_ids,
                aux_layer_ids=target_layer_ids,
                max_new_tokens=max_eval_new_tokens,
                temperature=0.0,
                use_kraus_tree=False,
                device=device,
            )
            zs_e2e = _evaluate_qwen35_hybrid_e2e(
                eval_mod=eval_mod,
                verifier=verifier_model,
                drafter=zs_model,
                prompt_ids_list=eval_prompt_ids,
                base_greedy_tokens_list=base_greedy_tokens,
                eos_ids=eos_ids,
                aux_layer_ids=target_layer_ids,
                max_new_tokens=max_eval_new_tokens,
                temperature=0.0,
                use_kraus_tree=False,
                device=device,
            )
            del unpruned_model, zs_model
            prox_diagnostics = {
                "iso_compute_schedule": "5_epochs_prox_lasso_plus_1_epoch_warm_recal",
                "total_epochs": 6,
                "kvn_lasso_weight": spec["kvn_lasso_weight"],
                "trainer_logged_zero_dyad_fraction": round(
                    prox_meta["trainer_logged_zero_frac"], 4
                ),
                "pre_prune_zero_dyad_fraction": round(prox_meta["zero_dyad_frac"], 4),
                "active_dyads_before_prune_out_of_256": prox_meta[
                    "active_dyads_remaining"
                ],
                "unpruned_m512_val_loss": round(prox_meta["unpruned_val_loss"], 4),
                "unpruned_m512_greedy_tau": round(
                    float(unpruned_e2e["mean_accept_len_tau"]), 4
                ),
                "zeroshot_layer0_max_abs_drift": round(
                    prox_meta["zeroshot_layer0_max_drift"], 6
                ),
                "zeroshot_greedy_accept_len_tau": round(
                    float(zs_e2e["mean_accept_len_tau"]), 4
                ),
                "zeroshot_exact_greedy_parity": bool(
                    zs_e2e["exact_greedy_match_all_prompts"]
                ),
            }

        print(
            f"=== [RTX 3070 Ti] Evaluating E2E Speculative Decoding for {model_id} ==="
        )
        e2e_t0 = _evaluate_qwen35_hybrid_e2e(
            eval_mod=eval_mod,
            verifier=verifier_model,
            drafter=loaded_model,
            prompt_ids_list=eval_prompt_ids,
            base_greedy_tokens_list=base_greedy_tokens,
            eos_ids=eos_ids,
            aux_layer_ids=target_layer_ids,
            max_new_tokens=max_eval_new_tokens,
            temperature=0.0,
            use_kraus_tree=False,
            device=device,
        )
        e2e_t06 = _evaluate_qwen35_hybrid_e2e(
            eval_mod=eval_mod,
            verifier=verifier_model,
            drafter=loaded_model,
            prompt_ids_list=eval_prompt_ids,
            base_greedy_tokens_list=None,
            eos_ids=eos_ids,
            aux_layer_ids=target_layer_ids,
            max_new_tokens=max_eval_new_tokens,
            temperature=0.6,
            use_kraus_tree=False,
            device=device,
        )

        kraus_tree_metrics: dict[str, Any] | None = None
        if spec.get("is_v3", False):
            kt_t0 = _evaluate_qwen35_hybrid_e2e(
                eval_mod=eval_mod,
                verifier=verifier_model,
                drafter=loaded_model,
                prompt_ids_list=eval_prompt_ids,
                base_greedy_tokens_list=base_greedy_tokens,
                eos_ids=eos_ids,
                aux_layer_ids=target_layer_ids,
                max_new_tokens=max_eval_new_tokens,
                temperature=0.0,
                use_kraus_tree=True,
                tree_budget=kraus_tree_budget,
                branch_width=kraus_branch_width,
                device=device,
            )
            kt_t06 = _evaluate_qwen35_hybrid_e2e(
                eval_mod=eval_mod,
                verifier=verifier_model,
                drafter=loaded_model,
                prompt_ids_list=eval_prompt_ids,
                base_greedy_tokens_list=None,
                eos_ids=eos_ids,
                aux_layer_ids=target_layer_ids,
                max_new_tokens=max_eval_new_tokens,
                temperature=0.6,
                use_kraus_tree=True,
                tree_budget=kraus_tree_budget,
                branch_width=kraus_branch_width,
                device=device,
            )
            kraus_tree_metrics = {
                "tree_budget": kraus_tree_budget,
                "branch_width": kraus_branch_width,
                "greedy_T0.0_tau": round(float(kt_t0["mean_accept_len_tau"]), 4),
                "greedy_T0.0_exact_parity": bool(
                    kt_t0["exact_greedy_match_all_prompts"]
                ),
                "stochastic_T0.6_tau": round(float(kt_t06["mean_accept_len_tau"]), 4),
                "delta_tau_vs_linear_T0.0": round(
                    float(kt_t0["mean_accept_len_tau"])
                    - float(e2e_t0["mean_accept_len_tau"]),
                    4,
                ),
                "delta_tau_vs_linear_T0.6": round(
                    float(kt_t06["mean_accept_len_tau"])
                    - float(e2e_t06["mean_accept_len_tau"]),
                    4,
                ),
            }

        model_results.append(
            {
                "_ckpt_dir": eval_ckpt_dir,
                "_loaded_model": loaded_model,
                "id": model_id,
                "backbone_type": spec["kvn_backbone_type"],
                "is_v3_architecture": bool(spec["is_v3"]),
                "total_iso_compute_epochs": total_iso_epochs,
                "train_modes": spec["kvn_n_modes"],
                "infer_modes": loaded_model.config.kvn_n_modes,
                "train_wall_time_s": round(rec["train_wall_s"], 2),
                "train_loss": round(float(train_m.get("loss_epoch", 0.0)), 4),
                "val_loss": round(float(val_m.get("loss_epoch", 0.0)), 4),
                "val_pos0_shortlist_acc_pct": round(
                    100.0 * float(val_m.get("position_0_acc_epoch", 0.0)), 2
                ),
                "val_accept_len_tau": round(
                    float(val_m.get("accept_len_epoch", 0.0)), 4
                ),
                "val_greedy_eal": round(float(val_m.get("eal_epoch", 0.0)), 4),
                "e2e_linear_T0.0_tau": round(float(e2e_t0["mean_accept_len_tau"]), 4),
                "e2e_linear_T0.0_exact_greedy_parity": bool(
                    e2e_t0["exact_greedy_match_all_prompts"]
                ),
                "e2e_linear_T0.6_tau": round(float(e2e_t06["mean_accept_len_tau"]), 4),
                "kraus_tree_specinfer": kraus_tree_metrics,
                "proximal_lasso_pruning": prox_diagnostics,
            }
        )

    # Free verifier before calibration & batched VRAM benchmarks!
    del verifier_model
    gc.collect()
    torch.cuda.empty_cache()

    final_entries: list[dict[str, Any]] = []
    for partial in model_results:
        eval_ckpt_dir = partial.pop("_ckpt_dir")
        loaded_model: KvNDSparkDraftModel = partial.pop("_loaded_model")
        model_id = partial["id"]

        cal_report = eval_mod.evaluate_checkpoint_calibration(
            checkpoint_dir=eval_ckpt_dir,
            verifier_model=verifier_id,
            data_dir=data_dir,
            draft_vocab_size=draft_vocab_size,
            max_samples=8,
            train_data_ratio=train_ratio,
            device=device,
        )
        torch.cuda.empty_cache()

        batched_hw = _measure_batched_drafter_latency_and_vram(
            loaded_model, eval_mod, batch_sizes=[1, 8, 32], device=device
        )

        trainable_params = sum(
            p.numel() for p in loaded_model.parameters() if p.requires_grad
        )
        lm_head_params = loaded_model.lm_head.weight.numel()
        dyn_state_bytes = loaded_model.config.estimated_state_bytes_per_seq()
        boundary_fc_bytes = (
            loaded_model.config.transformer_layer_config.hidden_size * 2
            if partial["backbone_type"] != "swa"
            else 0
        )
        pos0_full_vocab_acc = float(
            cal_report.get("per_position", {})
            .get("pos_0", {})
            .get(
                "accept_rate_pct",
                partial["val_pos0_shortlist_acc_pct"],
            )
        )

        entry = {
            "id": model_id,
            "backbone_type": partial["backbone_type"],
            "is_v3_architecture": partial["is_v3_architecture"],
            "total_iso_compute_epochs": partial["total_iso_compute_epochs"],
            "train_modes": partial["train_modes"],
            "infer_modes": partial["infer_modes"],
            "trainable_params": trainable_params,
            "active_params_with_lm_head": trainable_params + lm_head_params,
            "dynamical_state_bytes_per_seq": dyn_state_bytes,
            "chunk_boundary_fc_bytes_per_seq": boundary_fc_bytes,
            "total_slot_bytes_per_seq": dyn_state_bytes + boundary_fc_bytes,
            "train_wall_time_s": partial["train_wall_time_s"],
            "train_loss": partial["train_loss"],
            "val_loss": partial["val_loss"],
            "val_pos0_shortlist_acc_pct": partial["val_pos0_shortlist_acc_pct"],
            "exact_full_vocab_pos0_acc_pct": round(pos0_full_vocab_acc, 2),
            "val_accept_len_tau": partial["val_accept_len_tau"],
            "val_greedy_eal": partial["val_greedy_eal"],
            "e2e_linear_T0.0_tau": partial["e2e_linear_T0.0_tau"],
            "e2e_linear_T0.0_exact_greedy_parity": partial[
                "e2e_linear_T0.0_exact_greedy_parity"
            ],
            "e2e_linear_T0.6_tau": partial["e2e_linear_T0.6_tau"],
            "kraus_tree_specinfer": partial["kraus_tree_specinfer"],
            "calibration_15bin_ece": round(float(cal_report["overall_ece_15bin"]), 4),
            "calibration_rejection_auroc": round(
                float(cal_report["overall_rejection_auroc"]), 4
            ),
            "dynamic_stopping_pareto": cal_report.get("dynamic_stopping_pareto", []),
            "proximal_lasso_pruning": partial["proximal_lasso_pruning"],
            "batched_drafter_latency_and_vram": batched_hw,
        }
        final_entries.append(entry)
        is_exact = entry["e2e_linear_T0.0_exact_greedy_parity"]
        print(
            f"[{model_id:<24}] ValLoss={entry['val_loss']:.4f} | "
            f"FullVocabPos0={entry['exact_full_vocab_pos0_acc_pct']:.2f}% | "
            f"E2E_tau(T=0)={entry['e2e_linear_T0.0_tau']:.4f} (exact={is_exact}) | "
            f"E2E_tau(T=0.6)={entry['e2e_linear_T0.6_tau']:.4f} | "
            f"ECE={entry['calibration_15bin_ece']:.4f} | "
            f"AUROC={entry['calibration_rejection_auroc']:.4f}"
        )

    # Phase 3: Live KvNDSparkVllmProposer + APC + WSL2 Triton 3.7.1 / vLLM V1 Benchmark
    vllm_apc_report: dict[str, Any] = {}
    wsl_triton_vllm_report: dict[str, Any] = {}
    if final_v3_ckpt_dir is not None:
        vllm_apc_report = _run_vllm_proposer_and_apc_smoke_benchmark(
            final_v3_ckpt_dir, t2d_tensor, d2t_tensor, device=device
        )
        wsl_triton_vllm_report = _run_wsl_triton_and_vllm_probe(
            final_v3_ckpt_dir, data_dir, draft_vocab_size
        )

    out_json = REPO_ROOT / "kvn_rff_mtp" / "rtx3070ti_kvn_v3_validation_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_suite": (
            "KvN-DSpark v3 Local RTX 3070 Ti Pre-H200 Go-Live Validation"
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU",
            "vram_total_mib": round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            )
            if torch.cuda.is_available()
            else 0,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "verifier_model": verifier_id,
        "verifier_vocab_size": 248320,
        "draft_vocab_size": draft_vocab_size,
        "target_layer_ids": target_layer_ids,
        "seq_len": seq_len,
        "iso_compute_epochs_per_model": total_iso_epochs,
        "dataset_integrity": prep_meta,
        "num_held_out_eval_prompts": len(eval_prompt_ids),
        "max_eval_new_tokens": max_eval_new_tokens,
        "models": final_entries,
        "vllm_proposer_and_zero_replay_apc": vllm_apc_report,
        "wsl_linux_triton_and_vllm_v1_probe": wsl_triton_vllm_report,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[RTX 3070 Ti] Saved complete v3 validation results to {out_json}")
    return payload


if __name__ == "__main__":
    run_v3_rtx3070ti_benchmark()
