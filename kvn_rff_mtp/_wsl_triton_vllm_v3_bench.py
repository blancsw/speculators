"""Live WSL2 CUDA + Triton 3.7.1 + vLLM V1 End-to-End GPU Benchmark for KvN-DSpark v3.

Invoked automatically by ``validate_rtx3070ti_kvn_v3.py`` inside WSL2 Ubuntu using
``/home/blancsw/vllm-infomaniak/.venv/bin/python`` with ``Triton 3.7.1`` and editable
``vllm 0.29.1`` on the local NVIDIA GeForce RTX 3070 Ti Laptop GPU.

Validates on real Linux CUDA hardware:
1. ``Mega-KvN`` Triton ``sp4_butterfly`` + Lindblad associative scan parity and latency
   speedup vs PyTorch eager scan across ``T in {512, 2048, 8192}``.
2. ``vllm.general_plugins`` registration (``register_kvn_vllm_plugin``), verifying
   ``SUPPORTED_SPECULATORS_TYPES['kvn_dspark']`` and ``GPUModelRunner`` hooks.
3. Live ``KvNDSparkVllmProposer`` + ``ZeroReplayAPCStateCache`` +
   ``specinfer_verify_tree_step`` end-to-end speculative decoding with
   ``Qwen/Qwen3.5-0.8B`` (both linear chain and Kraus POVM tree), asserting 100%
   bit-exact greedy parity at ``T=0.0`` and 0-FLOP APC prefix restoration, plus
   standard pure-attention ``Qwen3ForCausalLM`` GQA ``build_tree_attention_mask`` +
   ``compact_tree_kv_cache`` verification.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
import vllm
import vllm.v1.spec_decode.custom_class_proposer as ccp
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.cache_utils import DynamicCache

try:
    import datasets as _datasets  # noqa: F401
except ImportError:
    _ds_stub = types.ModuleType("datasets")
    _ds_stub.load_from_disk = lambda *_a, **_kw: None  # type: ignore[attr-defined]
    _ds_stub.Dataset = object  # type: ignore[attr-defined]
    sys.modules["datasets"] = _ds_stub

from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.models.kvn_dspark import triton_kernels as tk
from speculators.vllm_integration import (
    PROPOSER_IMPORT_PATH,
    HardwareRooflineProfile,
    KvNDSparkVllmProposer,
    build_tree_attention_mask,
    compact_tree_kv_cache,
    configure_kvn_speculator_dict,
    is_kvn_vllm_plugin_registered,
    register_kvn_vllm_plugin,
    specinfer_verify_tree_step,
)

WSL_EVAL_PROMPTS: list[str] = [
    (
        "Write a Python function `binary_search(arr, target)` and state its loop "
        "invariant."
    ),
    "Derive the determinant of a 2x2 rotation-squeezing symplectic matrix in sp(2, R).",
    (
        "Explain why O(1) recurrent state-space drafters eliminate PagedAttention "
        "KV-cache bloat."
    ),
    "Implement Kahn's topological sort algorithm in Python using `collections.deque`.",
]

FIXED_CAUSAL_PAD = 192
_MEDIUM_SEQ_LEN = 2048
_PARITY_ATOL = 1e-4


@torch.no_grad()
def _benchmark_triton_sp4_scan(
    drafter: KvNDSparkDraftModel, device: str
) -> dict[str, Any]:
    """Compare Triton Mega-KvN sp4_butterfly scan vs PyTorch reference across T."""
    layer_triton = (
        copy.deepcopy(drafter.kvn_layers[0])
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    layer_eager = (
        copy.deepcopy(drafter.kvn_layers[0])
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    layer_triton.use_triton_scan = True
    layer_eager.use_triton_scan = False
    d_model = drafter.config.transformer_layer_config.hidden_size

    length_metrics: dict[str, Any] = {}
    for seq_t in (512, 2048, 8192):
        bsz = 2 if seq_t <= _MEDIUM_SEQ_LEN else 1
        torch.manual_seed(2026 + seq_t)
        x = torch.randn(bsz, seq_t, d_model, device=device, dtype=torch.bfloat16) * 0.25
        reset = torch.zeros(bsz, seq_t, device=device, dtype=torch.bool)
        reset[:, 0] = True
        reset[:, seq_t // 2] = True

        out_tri = layer_triton(
            x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
        )
        out_ref = layer_eager(
            x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
        )
        max_h_diff = float((out_tri[0].float() - out_ref[0].float()).abs().max().item())
        max_q_diff = float((out_tri[1].float() - out_ref[1].float()).abs().max().item())
        max_p_diff = float((out_tri[2].float() - out_ref[2].float()).abs().max().item())
        max_nu_diff = float(
            (out_tri[4].float() - out_ref[4].float()).abs().max().item()
        )

        for _ in range(5):
            _ = layer_triton(
                x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
            )
            _ = layer_eager(
                x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
            )
        torch.cuda.synchronize()

        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        iters = 15 if seq_t <= _MEDIUM_SEQ_LEN else 8

        s_ev.record()
        for _ in range(iters):
            _ = layer_triton(
                x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
            )
        e_ev.record()
        torch.cuda.synchronize()
        tri_ms = float(s_ev.elapsed_time(e_ev)) / iters

        s_ev.record()
        for _ in range(iters):
            _ = layer_eager(
                x, reset_mask=reset, state_dtype=torch.float32, return_nu=True
            )
        e_ev.record()
        torch.cuda.synchronize()
        ref_ms = float(s_ev.elapsed_time(e_ev)) / iters

        length_metrics[f"T={seq_t}"] = {
            "batch_size": bsz,
            "triton_ms": round(tri_ms, 4),
            "pytorch_eager_ms": round(ref_ms, 4),
            "triton_speedup_x": round(ref_ms / max(tri_ms, 1e-6), 3),
            "max_out_abs_diff": round(max_h_diff, 6),
            "max_q_fp32_abs_diff": round(max_q_diff, 6),
            "max_p_fp32_abs_diff": round(max_p_diff, 6),
            "max_nu_fp32_abs_diff": round(max_nu_diff, 6),
            "all_finite": bool(torch.isfinite(out_tri[0]).all().item()),
        }

    del layer_triton, layer_eager
    torch.cuda.empty_cache()
    return length_metrics


def _verify_vllm_plugin_registration(
    drafter: KvNDSparkDraftModel, ckpt_path: Path
) -> dict[str, Any]:
    """Register vLLM V1 plugin and verify SpeculatorsConfig + GPUModelRunner wiring."""
    ok = register_kvn_vllm_plugin()
    algos_mod = importlib.import_module(
        "vllm.transformers_utils.configs.speculators.algos"
    )
    supported = getattr(algos_mod, "SUPPORTED_SPECULATORS_TYPES", {})

    spec_dict: dict[str, Any] = {"model": str(ckpt_path)}
    configure_kvn_speculator_dict(drafter.config.to_dict(), spec_dict)
    return {
        "plugin_registered": bool(ok and is_kvn_vllm_plugin_registered()),
        "kvn_dspark_in_vllm_supported_types": "kvn_dspark" in supported,
        "has_vllm_custom_class_proposer": hasattr(ccp, "CustomClassProposer"),
        "resolved_vllm_speculative_config": spec_dict,
        "routes_to_proposer_class": spec_dict.get("model") == PROPOSER_IMPORT_PATH,
    }


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

    # Reconstruct unique root-to-node chain for each node i
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
    # Map each node i -> (leaf_idx_in_leaves, depth_offset_1based)
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
def _verify_pure_attention_tree_kv_compaction(device: str) -> dict[str, Any]:
    """Verify 4D tree attention mask and KV compaction on pure-attention Qwen3 GQA."""
    torch.manual_seed(42)
    cfg = Qwen3Config(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    pure_qwen = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()
    prefix = torch.randint(5, 1000, (1, 16), device=device)
    kv = DynamicCache()
    _ = pure_qwen(input_ids=prefix, past_key_values=kv, use_cache=True)

    # 5-node tree: root(-1)->0, 0->1, 0->2, 1->3, 2->4
    anchor = torch.tensor([[17]], device=device)
    tree_ids = torch.tensor([[21, 22, 23, 24, 25]], device=device)
    tree_parents = torch.tensor([[-1, 0, 0, 1, 2]], device=device)
    t_mask, t_depths = build_tree_attention_mask(
        tree_parent_idx=tree_parents, prefix_len=16, dtype=torch.float32
    )
    verify_in = torch.cat([anchor, tree_ids], dim=1)
    pos_ids = torch.cat([torch.tensor([[16]], device=device), 16 + t_depths], dim=1)
    tree_out = pure_qwen(
        input_ids=verify_in,
        past_key_values=kv,
        use_cache=True,
        attention_mask=t_mask,
        position_ids=pos_ids,
    )
    # Branch 0 -> 2 -> 4 must match exact sequential prefix [prefix, 17, 21, 23, 25]
    seq_branch = torch.cat(
        [prefix, torch.tensor([[17, 21, 23, 25]], device=device)], dim=1
    )
    ref_out = pure_qwen(input_ids=seq_branch, use_cache=False)
    branch_logit_err = float(
        (tree_out.logits[0, [0, 1, 3, 5]] - ref_out.logits[0, 16:20]).abs().max().item()
    )

    # Compact KV cache to branch [0, 2, 4] and advance one token [99]
    _ = compact_tree_kv_cache(kv, 16, [[0, 2, 4]])
    next_out = pure_qwen(
        input_ids=torch.tensor([[99]], device=device),
        past_key_values=kv,
        use_cache=True,
    )
    ref_next = pure_qwen(
        input_ids=torch.cat([seq_branch, torch.tensor([[99]], device=device)], dim=1),
        use_cache=False,
    )
    post_compact_err = float(
        (next_out.logits[0, -1] - ref_next.logits[0, -1]).abs().max().item()
    )
    del pure_qwen
    torch.cuda.empty_cache()
    return {
        "tree_4d_mask_branch_logit_max_abs_err": round(branch_logit_err, 7),
        "post_compact_kv_next_step_max_abs_err": round(post_compact_err, 7),
        "passed": branch_logit_err < _PARITY_ATOL and post_compact_err < _PARITY_ATOL,
    }


@torch.no_grad()
def _run_live_vllm_proposer_e2e(
    verifier: torch.nn.Module,
    drafter: KvNDSparkDraftModel,
    prompt_ids_list: list[list[int]],
    base_tokens_list: list[list[int]],
    aux_layer_ids: list[int],
    max_new_tokens: int,
    use_kraus_tree: bool,
    device: str,
) -> dict[str, Any]:
    """Drive KvNDSparkVllmProposer + SpecInfer verification end-to-end on CUDA."""
    drafter.use_triton_scan = True
    for lyr in drafter.kvn_layers:
        lyr.use_triton_scan = True

    proposer = KvNDSparkVllmProposer(
        drafter,
        max_num_seqs=16,
        device=device,
        enable_apc=True,
        apc_block_size=16,
        use_kraus_tree=use_kraus_tree,
        tree_budget=9,
        branch_width=2,
        roofline=HardwareRooflineProfile.rtx_3070_ti_laptop(),
    )

    exact_matches: list[bool] = []
    step_taus: list[float] = []
    propose_ms_list: list[float] = []

    for p_idx, (p_ids, base_tokens) in enumerate(
        zip(prompt_ids_list, base_tokens_list, strict=True)
    ):
        req_id = f"wsl_req_{use_kraus_tree}_{p_idx}"
        aux_pref, pref_logits = _forward_qwen35_fixed_causal(
            verifier, p_ids, aux_layer_ids, device=device
        )
        y0 = int(pref_logits[0, -1].argmax(dim=-1).item())
        spec_tokens = [y0]
        confirmed_seq = list(p_ids)
        cur_prev_fc = proposer.extract_fc_hidden(
            aux_pref[0, -1:], already_projected=False
        )
        cur_tok_chunk = [y0]

        proposer.prefill_request(req_id, p_ids, aux_pref[0], use_apc=True)
        slot_idx = proposer.slot_pool.get_slot(req_id)

        while len(spec_tokens) < max_new_tokens:
            v_len = len(cur_tok_chunk)
            s_max = int(drafter.block_size) + 1
            d_model = proposer.slot_pool.d_model
            fc_dtype = drafter.fc.weight.dtype

            prev_fc_blk = torch.zeros(
                (1, s_max, d_model), dtype=fc_dtype, device=device
            )
            prev_fc_blk[0, :v_len, :] = cur_prev_fc.reshape(v_len, d_model)
            tok_blk = torch.zeros((1, s_max), dtype=torch.long, device=device)
            tok_blk[0, :v_len] = torch.tensor(
                cur_tok_chunk, dtype=torch.long, device=device
            )
            valid_lens = torch.tensor([v_len], dtype=torch.long, device=device)

            state_cache = proposer.slot_pool.gather_drafter_state_cache(
                [slot_idx], include_nu=True
            )
            anchor_id = cur_tok_chunk[-1]
            prefix_len = len(confirmed_seq)
            torch.cuda.synchronize()
            t_p0 = time.perf_counter()

            if use_kraus_tree:
                tree_dict, updated_cache = drafter.draft_kraus_tree_step(
                    prev_fc_blk,
                    tok_blk,
                    valid_lens,
                    state_cache,
                    tree_budget=9,
                    branch_width=2,
                    temperature=0.0,
                )
                torch.cuda.synchronize()
                propose_ms_list.append((time.perf_counter() - t_p0) * 1000.0)
                proposer.slot_pool.scatter_drafter_state_cache(
                    [slot_idx], updated_cache
                )

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
                res = specinfer_verify_tree_step(
                    verifier_logits=v_logits,
                    tree_draft_ids=tree_ids,
                    tree_parent_idx=tree_parents,
                    temperature=0.0,
                    d2t=drafter.d2t if drafter.use_draft_vocab else None,
                )
                acc_toks = res.accepted_tokens[0]
                acc_nodes = res.accepted_node_indices[0]
                bonus_tok = int(res.bonus_tokens[0].item())
                gather_idx = torch.tensor(
                    [0, *(1 + i for i in acc_nodes)], dtype=torch.long, device=device
                )
                confirmed_aux = v_aux[0].index_select(0, gather_idx)
            else:
                draft_all, updated_cache, _ = drafter.advance_and_draft_step(
                    prev_fc_blk,
                    tok_blk,
                    valid_lens,
                    state_cache,
                    temperature=0.0,
                    return_details=True,
                )
                torch.cuda.synchronize()
                propose_ms_list.append((time.perf_counter() - t_p0) * 1000.0)
                proposer.slot_pool.scatter_drafter_state_cache(
                    [slot_idx], updated_cache
                )

                start_k = 0 if drafter.config.sample_from_anchor else 1
                draft_ids_target = draft_all[:, start_k:]
                k_spec = draft_ids_target.size(1)

                verify_seq = [*confirmed_seq, anchor_id, *draft_ids_target[0].tolist()]
                aux_full, logits_full = _forward_qwen35_fixed_causal(
                    verifier, verify_seq, aux_layer_ids, device=device
                )
                v_logits = logits_full[:, prefix_len : prefix_len + 1 + k_spec, :]
                v_aux = aux_full[:, prefix_len : prefix_len + 1 + k_spec, :]
                chain_parents = torch.arange(
                    -1, k_spec - 1, dtype=torch.long, device=device
                ).unsqueeze(0)
                res = specinfer_verify_tree_step(
                    verifier_logits=v_logits,
                    tree_draft_ids=draft_ids_target,
                    tree_parent_idx=chain_parents,
                    temperature=0.0,
                )
                acc_toks = res.accepted_tokens[0]
                bonus_tok = int(res.bonus_tokens[0].item())
                confirmed_aux = v_aux[0, : 1 + len(acc_toks), :]

            confirmed_seq.extend([anchor_id, *acc_toks])
            step_taus.append(float(1 + len(acc_toks)))
            emitted = [*acc_toks, bonus_tok]
            for t_id in emitted:
                if len(spec_tokens) >= max_new_tokens:
                    break
                spec_tokens.append(t_id)
            if len(spec_tokens) >= max_new_tokens:
                break

            cur_prev_fc = proposer.extract_fc_hidden(
                confirmed_aux, already_projected=False
            )
            cur_tok_chunk = emitted

        exact_matches.append(
            spec_tokens[:max_new_tokens] == base_tokens[:max_new_tokens]
        )
        proposer.free_finished_requests([req_id])

    # Multi-turn Zero-Replay APC 0-FLOP prefix hit check on 64-token shared prefix
    shared_64 = (prompt_ids_list[0] * ((80 // len(prompt_ids_list[0])) + 2))[:64]
    turn2_80 = shared_64 + prompt_ids_list[1][:16]
    aux64, _ = _forward_qwen35_fixed_causal(
        verifier, shared_64, aux_layer_ids, device=device
    )
    aux80, _ = _forward_qwen35_fixed_causal(
        verifier, turn2_80, aux_layer_ids, device=device
    )

    proposer.prefill_request("apc_turn1", shared_64, aux64[0], use_apc=True)
    proposer.free_finished_requests(["apc_turn1"])

    slot_hit, restored_toks = proposer.prefill_request(
        "apc_turn2_hit", turn2_80, aux80[0], use_apc=True
    )
    slot_ref, _ = proposer.prefill_request(
        "apc_turn2_ref", turn2_80, aux80[0], use_apc=False
    )
    q_err = float(
        (proposer.slot_pool.q_pool[slot_hit] - proposer.slot_pool.q_pool[slot_ref])
        .abs()
        .max()
        .item()
    )
    p_err = float(
        (proposer.slot_pool.p_pool[slot_hit] - proposer.slot_pool.p_pool[slot_ref])
        .abs()
        .max()
        .item()
    )
    apc_stats = (
        proposer.apc_cache.memory_stats() if proposer.apc_cache is not None else {}
    )
    proposer.free_finished_requests(["apc_turn2_hit", "apc_turn2_ref"])

    return {
        "mode": "kraus_tree_specinfer" if use_kraus_tree else "linear_chain",
        "num_prompts": len(prompt_ids_list),
        "max_new_tokens": max_new_tokens,
        "mean_accept_len_tau": round(float(np.mean(step_taus)), 4),
        "mean_propose_latency_ms": round(float(np.mean(propose_ms_list)), 4),
        "exact_greedy_parity_all_prompts": all(exact_matches),
        "apc_restored_prefix_tokens": int(restored_toks),
        "apc_restore_max_abs_err": round(max(q_err, p_err), 8),
        "apc_memory_stats": apc_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verifier-model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--draft-vocab-size", type=int, default=4096)
    args = parser.parse_args()

    device = "cuda"
    t2d = torch.from_numpy(np.load(args.data_dir / f"t2d-{args.draft_vocab_size}.npy"))
    d2t = torch.from_numpy(np.load(args.data_dir / f"d2t-{args.draft_vocab_size}.npy"))
    drafter = (
        KvNDSparkDraftModel.from_pretrained(args.checkpoint, t2d=t2d, d2t=d2t)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    triton_scan_bench = _benchmark_triton_sp4_scan(drafter, device=device)
    plugin_report = _verify_vllm_plugin_registration(drafter, args.checkpoint)
    pure_attn_kv_report = _verify_pure_attention_tree_kv_compaction(device=device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, local_files_only=True
    )
    verifier = (
        AutoModelForCausalLM.from_pretrained(
            args.verifier_model, dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
    )

    prompt_ids_list: list[list[int]] = []
    for txt in WSL_EVAL_PROMPTS:
        msgs = [{"role": "user", "content": txt}]
        try:
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            )
        if not isinstance(ids, list):
            ids = list(ids["input_ids"])
        prompt_ids_list.append(ids[:80])

    max_new_tokens = 24
    aux_ids = list(drafter.config.aux_hidden_state_layer_ids or [2, 12, 21])

    # Precompute exact greedy baseline once across all 4 prompts
    base_tokens_list: list[list[int]] = []
    with torch.no_grad():
        for p_ids in prompt_ids_list:
            seq = list(p_ids)
            gen: list[int] = []
            for _ in range(max_new_tokens):
                _, logits = _forward_qwen35_fixed_causal(
                    verifier, seq, aux_ids, device=device
                )
                nxt = int(logits[0, -1].argmax(dim=-1).item())
                gen.append(nxt)
                seq.append(nxt)
            base_tokens_list.append(gen)

    e2e_chain = _run_live_vllm_proposer_e2e(
        verifier=verifier,
        drafter=drafter,
        prompt_ids_list=prompt_ids_list,
        base_tokens_list=base_tokens_list,
        aux_layer_ids=aux_ids,
        max_new_tokens=max_new_tokens,
        use_kraus_tree=False,
        device=device,
    )
    e2e_kraus = _run_live_vllm_proposer_e2e(
        verifier=verifier,
        drafter=drafter,
        prompt_ids_list=prompt_ids_list,
        base_tokens_list=base_tokens_list,
        aux_layer_ids=aux_ids,
        max_new_tokens=max_new_tokens,
        use_kraus_tree=True,
        device=device,
    )

    payload = {
        "wsl_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "vllm_version": vllm.__version__,
        "has_triton_kernels": bool(getattr(tk, "HAS_TRITON", False)),
        "triton_sp4_butterfly_scan_benchmark": triton_scan_bench,
        "vllm_v1_plugin_and_proposer_wiring": plugin_report,
        "pure_attention_tree_kv_compaction_check": pure_attn_kv_report,
        "live_wsl_e2e_linear_chain": e2e_chain,
        "live_wsl_e2e_kraus_tree_specinfer": e2e_kraus,
    }
    print("WSL_JSON:" + json.dumps(payload))  # noqa: T201


if __name__ == "__main__":
    main()
