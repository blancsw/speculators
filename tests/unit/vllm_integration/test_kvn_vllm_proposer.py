"""Unit & adversarial verification suite for ``speculators.vllm_integration``.

Covers:
1. Chunked prefill (``[16, 16, 8]`` + scheduler preemption/resume) vs monolithic prefill
   FP32 recurrent & boundary ``last_fc`` parity (``[M5]`` / ``[C6]``).
2. Axis 3 ``ZeroReplayAPCStateCache`` 16-token chained block hashing, 0-FLOP prefix
   restoration exactness (``0.0`` boundary error), byte accounting, and LRU eviction.
3. Axis 6 ``PurityLoadAwareAmdahlScheduler`` H200 (206 tok) / RTX 3070 Ti (99 tok)
   rooflines, creative ``T > 0`` high-purity preservation, and low-purity / high-batch
   load throttling.
4. Axis 5 ``specinfer_verify_tree_step`` & ``compact_tree_kv_cache``: bit-exact greedy
   tree verification + KV compaction at ``T = 0.0`` and Monte Carlo lossless sampling
   proof at ``T = 0.6`` (both deterministic top-w Kraus forks and stochastic ``q^(j)``
   proposals, including out-of-shortlist OOV verifier tokens).
5. ``vllm.general_plugins`` registration, ``KvNDSparkConverter`` checkpoint packaging,
   and ``benchmark_vllm_prod`` Prometheus/streaming metrics.
6. ``kvn_phase2_calibration_and_amdahl`` ``[C4]`` & ``[C6]`` fixes: full-vocab OOV
   rejection in ``calibrate``, cumulative prefix survival ``prod_{j<=k} c_j >= gamma``,
   true batched SWA / persistent-state Hybrid E2E parity, EOS stopping, and
   ``--use-kraus-tree`` E2E verification.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from speculators.convert.entrypoints import convert_model
from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.vllm_integration import (
    PROPOSER_IMPORT_PATH,
    HardwareRooflineProfile,
    KvNDSparkVllmProposer,
    KvNStateSlotPool,
    PurityLoadAwareAmdahlScheduler,
    ZeroReplayAPCStateCache,
    build_tree_attention_mask,
    compact_tree_kv_cache,
    configure_kvn_speculator_dict,
    specinfer_verify_tree_step,
    update_kvn_dspark_pretrained_config,
)
from speculators.vllm_integration import plugin as plugin_mod

REPO_SPECULATORS_DIR = Path(__file__).resolve().parents[3]
CALIB_SCRIPT_PATH = (
    REPO_SPECULATORS_DIR
    / "scripts"
    / "evaluate"
    / "kvn_phase2_calibration_and_amdahl.py"
)
BENCH_PROD_PATH = (
    REPO_SPECULATORS_DIR / "scripts" / "evaluate" / "benchmark_vllm_prod.py"
)


def _make_fake_vllm_hook_classes() -> tuple[type, type, dict[str, object]]:
    class _FakeRunner:
        def __init__(self, drafter_obj) -> None:
            self.drafter = drafter_obj
            self.use_aux_hidden_state_outputs = False

        def load_model(self) -> None:
            return None

        def _get_eagle3_aux_layers_from_config(self):
            return None

    class _FakeSpecConfig:
        def __init__(self, model: str | None = None, **kwargs) -> None:
            self.model = model
            for k, v in kwargs.items():
                setattr(self, k, v)

        def compute_hash(self) -> str:
            return "base_hash_const"

    fake_runner_mod = type(sys)("vllm.v1.worker.gpu_model_runner")
    fake_runner_mod.GPUModelRunner = _FakeRunner  # type: ignore[attr-defined]
    fake_spec_mod = type(sys)("vllm.config.speculative")
    fake_spec_mod.SpeculativeConfig = _FakeSpecConfig  # type: ignore[attr-defined]
    fake_arg_mod = type(sys)("vllm.engine.arg_utils")
    fake_arg_mod.EngineArgs = None  # type: ignore[attr-defined]
    mod_map = {
        "vllm.v1.worker.gpu_model_runner": fake_runner_mod,
        "vllm.config.speculative": fake_spec_mod,
        "vllm.engine.arg_utils": fake_arg_mod,
    }
    return _FakeRunner, _FakeSpecConfig, mod_map


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_tiny_verifier_and_drafter(
    tmp_path: Path,
    backbone_type: str = "kvn",
    vocab_size: int = 64,
    draft_vocab_size: int = 32,
    block_size: int = 4,
    n_modes: int = 16,
    eos_token_id: int = 63,
    sample_from_anchor: bool | None = None,
) -> tuple[Path, Path, Path, Qwen3ForCausalLM, KvNDSparkDraftModel]:
    torch.manual_seed(1234)
    v_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=32,
        use_sliding_window=True,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention"],
        eos_token_id=eos_token_id,
    )
    verifier = Qwen3ForCausalLM(v_cfg).eval()
    v_dir = tmp_path / f"verifier_{backbone_type}"
    verifier.save_pretrained(v_dir)

    data_dir = tmp_path / f"data_{backbone_type}"
    data_dir.mkdir(parents=True, exist_ok=True)
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[:draft_vocab_size] = True
    d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
    np.save(data_dir / f"t2d-{draft_vocab_size}.npy", t2d.numpy())
    np.save(data_dir / f"d2t-{draft_vocab_size}.npy", d2t.numpy())

    tl_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=32,
        use_sliding_window=True,
        layer_types=["sliding_attention", "sliding_attention"],
    )
    drafter = KvNDSparkDraftModel.from_training_args(
        verifier_config=tl_cfg,
        t2d=t2d,
        d2t=d2t,
        verifier_name_or_path=str(v_dir),
        draft_vocab_size=draft_vocab_size,
        block_size=block_size,
        target_layer_ids=[1, 2, 3],
        mask_token_id=0,
        kvn_backbone_type=backbone_type,
        kvn_n_modes=n_modes,
        kvn_reupload_layers=2,
        kvn_reupload_dim=16,
        kvn_anchor_rank=16,
        kvn_state_dtype="float32",
        markov_rank=16,
    ).eval()
    if sample_from_anchor is not None:
        drafter.config.sample_from_anchor = sample_from_anchor
    ckpt_dir = tmp_path / f"drafter_{backbone_type}"
    drafter.save_pretrained(ckpt_dir)
    return v_dir, ckpt_dir, data_dir, verifier, drafter


class TestChunkedPrefillAndZeroReplayAPC:
    def test_chunked_prefill_and_preemption_match_monolithic_scan(
        self, tmp_path: Path
    ) -> None:
        _, _, _, _, drafter = _make_tiny_verifier_and_drafter(tmp_path)
        proposer = KvNDSparkVllmProposer(
            drafter, max_num_seqs=8, device="cpu", enable_apc=False
        )
        assert proposer.slot_pool is not None

        torch.manual_seed(7)
        prompt_ids = torch.randint(0, 32, (40,), dtype=torch.long)
        aux_seq = torch.randn(40, 3 * 32)

        # Monolithic 40-token prefill
        slot_mono, _ = proposer.prefill_request("req-mono", prompt_ids, aux_seq)
        q_mono = proposer.slot_pool.q_pool[slot_mono].clone()
        p_mono = proposer.slot_pool.p_pool[slot_mono].clone()
        fc_mono = proposer.slot_pool.last_fc_pool[slot_mono].clone()

        # 3-chunk prefill [16, 16, 8] with preemption swap-out between chunks 1 and 2
        proposer.prefill_request("req-chunk", prompt_ids[:16], aux_seq[:16])
        proposer.preempt_requests(["req-chunk"])
        assert proposer.slot_pool.is_preempted("req-chunk")
        proposer.prefill_request("req-chunk", prompt_ids[16:32], aux_seq[16:32])
        slot_chunk, _ = proposer.prefill_request(
            "req-chunk", prompt_ids[32:], aux_seq[32:]
        )

        q_chunk = proposer.slot_pool.q_pool[slot_chunk]
        p_chunk = proposer.slot_pool.p_pool[slot_chunk]
        fc_chunk = proposer.slot_pool.last_fc_pool[slot_chunk]

        assert torch.max(torch.abs(fc_chunk - fc_mono)).item() < 1e-6
        assert torch.max(torch.abs(q_chunk - q_mono)).item() < 1e-5
        assert torch.max(torch.abs(p_chunk - p_mono)).item() < 1e-5
        if proposer.slot_pool.nu_pool is not None:
            nu_mono = proposer.slot_pool.nu_pool[slot_mono]
            nu_chunk = proposer.slot_pool.nu_pool[slot_chunk]
            assert torch.max(torch.abs(nu_chunk - nu_mono)).item() < 1e-5

        # Subsequent batched decode step yields identical draft tokens
        step_aux = torch.randn(1, 3 * 32)
        out = proposer.propose_batch(
            ["req-mono", "req-chunk"],
            [[5], [5]],
            [step_aux, step_aux],
        )
        assert torch.equal(out.padded_draft_ids[0], out.padded_draft_ids[1])

    def test_zero_replay_apc_16_token_block_restore_exactness(
        self, tmp_path: Path
    ) -> None:
        _, _, _, _, drafter = _make_tiny_verifier_and_drafter(tmp_path)
        proposer_apc = KvNDSparkVllmProposer(
            drafter,
            max_num_seqs=8,
            device="cpu",
            enable_apc=True,
            apc_block_size=16,
            apc_max_blocks=16,
        )
        proposer_ref = KvNDSparkVllmProposer(
            drafter, max_num_seqs=8, device="cpu", enable_apc=False
        )
        assert proposer_apc.apc_cache is not None
        assert proposer_apc.slot_pool is not None
        assert proposer_ref.slot_pool is not None

        torch.manual_seed(99)
        shared_prefix_toks = torch.randint(0, 32, (32,), dtype=torch.long)
        shared_prefix_aux = torch.randn(32, 3 * 32)

        turn1_toks = torch.cat(
            [shared_prefix_toks, torch.tensor([3, 4, 5, 6], dtype=torch.long)]
        )
        turn1_aux = torch.cat([shared_prefix_aux, torch.randn(4, 3 * 32)], dim=0)

        _, restored_t1 = proposer_apc.prefill_request("turn-1", turn1_toks, turn1_aux)
        assert restored_t1 == 0
        assert len(proposer_apc.apc_cache) == 2

        # Exact boundary check: restoring 32-token prefix alone has 0.0 error & 0 FLOPs
        turn2_suffix_toks = torch.tensor([9, 10, 11, 12, 13, 14], dtype=torch.long)
        turn2_suffix_aux = torch.randn(6, 3 * 32)
        turn2_full_toks = torch.cat([shared_prefix_toks, turn2_suffix_toks])
        turn2_full_aux = torch.cat([shared_prefix_aux, turn2_suffix_aux], dim=0)

        # Verify exact 0.0 bit-identity vs stored APC snapshot & <1e-6 vs M=32 GEMM
        boundary_slot, _ = proposer_apc.slot_pool.get_or_allocate_slot("boundary-check")
        restored_cnt, hit_hash, rem = proposer_apc.apc_cache.match_and_restore_prefix(
            turn2_full_toks, proposer_apc.slot_pool, boundary_slot
        )
        assert restored_cnt == 32
        assert len(rem) == 6
        snap_32 = proposer_apc.apc_cache._blocks[hit_hash]
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.q_pool[boundary_slot] - snap_32.q_state
                )
            ).item()
            == 0.0
        )
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.p_pool[boundary_slot] - snap_32.p_state
                )
            ).item()
            == 0.0
        )
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.last_fc_pool[boundary_slot] - snap_32.last_fc
                )
            ).item()
            == 0.0
        )
        if proposer_apc.slot_pool.nu_pool is not None and snap_32.nu_state is not None:
            assert (
                torch.max(
                    torch.abs(
                        proposer_apc.slot_pool.nu_pool[boundary_slot] - snap_32.nu_state
                    )
                ).item()
                == 0.0
            )
        ref_32_slot, _ = proposer_ref.prefill_request(
            "ref-32", shared_prefix_toks, shared_prefix_aux
        )
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.q_pool[boundary_slot]
                    - proposer_ref.slot_pool.q_pool[ref_32_slot]
                )
            ).item()
            < 1e-6
        )
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.p_pool[boundary_slot]
                    - proposer_ref.slot_pool.p_pool[ref_32_slot]
                )
            ).item()
            < 1e-6
        )

        # Full turn-2 prefill (32 tokens restored in 0 FLOPs + 6 suffix tokens computed)
        slot_t2, restored_t2 = proposer_apc.prefill_request(
            "turn-2", turn2_full_toks, turn2_full_aux
        )
        assert restored_t2 == 32
        stats = proposer_apc.apc_cache.memory_stats()
        assert stats["replay_flops_per_hit"] == 0
        assert stats["saved_prefill_tokens"] >= 32

        slot_ref_full, _ = proposer_ref.prefill_request(
            "ref-38", turn2_full_toks, turn2_full_aux
        )
        assert (
            torch.max(
                torch.abs(
                    proposer_apc.slot_pool.q_pool[slot_t2]
                    - proposer_ref.slot_pool.q_pool[slot_ref_full]
                )
            ).item()
            < 1e-5
        )

    def test_apc_byte_accounting_and_lru_eviction(self) -> None:
        pool = KvNStateSlotPool(
            max_num_seqs=4,
            num_layers=5,
            n_modes=128,
            d_model=4096,
            use_lindblad_purity=False,
            state_dtype=torch.float32,
            fc_dtype=torch.bfloat16,
        )
        # 5 layers * 2 (q, p) * 128 modes * 4B = 5,120 B; / 16 tok = 320 B/tok
        assert pool.bytes_per_seq(include_last_fc=False) == 5120

        apc = ZeroReplayAPCStateCache(block_size=16, max_cached_blocks=2)
        for idx in range(3):
            toks = list(range(idx * 100, idx * 100 + 16))
            apc.record_sequence_blocks(
                token_ids=toks,
                q_per_step=torch.zeros(5, 16, 128, dtype=torch.float32),
                p_per_step=torch.zeros(5, 16, 128, dtype=torch.float32),
                fc_per_step=torch.zeros(16, 4096, dtype=torch.bfloat16),
            )
        stats = apc.memory_stats()
        assert len(apc) == 2
        assert stats["num_evictions"] == 1
        assert stats["recurrent_bytes_per_token"] == 320.0


class TestPurityLoadAwareAmdahlScheduler:
    def test_roofline_ridge_points_and_purity_vs_load_scheduling(self) -> None:
        h200 = HardwareRooflineProfile.h200_sxm()
        rtx = HardwareRooflineProfile.rtx_3070_ti_laptop()
        assert h200.ridge_tokens == 206
        assert rtx.ridge_tokens == 99

        sched = PurityLoadAwareAmdahlScheduler(roofline=h200, max_horizon=7)
        conf = torch.full((2, 7), 0.78)
        high_entropy = torch.full((2, 7), 0.90)

        # High purity + low p_vac at T=0.6 preserves creative drafting at B=4
        dec_pure = sched.select_horizon(
            confidences=conf,
            active_batch_size=4,
            purity=torch.full((2, 7), 0.98),
            p_vac=torch.full((2, 7), 0.01),
            draft_entropy=high_entropy,
            temperature=0.6,
        )
        # Low purity + high vacuum leakage (epistemic drift) throttles horizon
        dec_mixed = sched.select_horizon(
            confidences=conf,
            active_batch_size=4,
            purity=torch.full((2, 7), 0.20),
            p_vac=torch.full((2, 7), 0.65),
            draft_entropy=high_entropy,
            temperature=0.6,
        )
        assert dec_pure.effective_alpha.mean().item() > conf.mean().item()
        assert int(dec_pure.per_req_horizons[0].item()) > int(
            dec_mixed.per_req_horizons[0].item()
        )

        # High batch size B=128 above H200 ridge throttles horizon vs B=1
        dec_b128 = sched.select_horizon(
            confidences=torch.full((2, 7), 0.55),
            active_batch_size=128,
            purity=torch.full((2, 7), 0.50),
            p_vac=torch.full((2, 7), 0.25),
            temperature=0.0,
        )
        dec_b1 = sched.select_horizon(
            confidences=torch.full((2, 7), 0.55),
            active_batch_size=1,
            purity=torch.full((2, 7), 0.50),
            p_vac=torch.full((2, 7), 0.25),
            temperature=0.0,
        )
        assert dec_b128.batch_horizon <= dec_b1.batch_horizon

        # Kraus tree budget prefers wide tree at B=1 pure vs throttled at B=128 mixed
        tree_b1 = sched.select_tree_budget(
            confidences=torch.full((2, 7), 0.72),
            active_batch_size=1,
            purity=torch.full((2, 7), 0.96),
            p_vac=torch.full((2, 7), 0.02),
        )
        tree_b128 = sched.select_tree_budget(
            confidences=torch.full((2, 7), 0.72),
            active_batch_size=128,
            purity=torch.full((2, 7), 0.25),
            p_vac=torch.full((2, 7), 0.60),
        )
        assert tree_b1.batch_tree_budget > tree_b128.batch_tree_budget


class TestSpecInferLosslessTreeVerification:
    def test_greedy_tree_walk_and_bit_exact_kv_cache_compaction(
        self, tmp_path: Path
    ) -> None:
        _, _, _, verifier, _ = _make_tiny_verifier_and_drafter(tmp_path)
        prompt = torch.tensor([[2, 5, 8, 11]], dtype=torch.long)

        # Sequential reference generation
        pkv_ref = DynamicCache()
        out0 = verifier(input_ids=prompt, past_key_values=pkv_ref, use_cache=True)
        anchor_tok = int(out0.logits[0, -1].argmax().item())
        out1 = verifier(
            input_ids=torch.tensor([[anchor_tok]]),
            past_key_values=pkv_ref,
            use_cache=True,
        )
        tok1 = int(out1.logits[0, -1].argmax().item())
        out2 = verifier(
            input_ids=torch.tensor([[tok1]]),
            past_key_values=pkv_ref,
            use_cache=True,
        )
        tok2 = int(out2.logits[0, -1].argmax().item())
        out3 = verifier(
            input_ids=torch.tensor([[tok2]]),
            past_key_values=pkv_ref,
            use_cache=True,
        )
        tok3 = int(out3.logits[0, -1].argmax().item())

        # Tree containing wrong branch at node 0, true tok1 at node 1 (root child),
        # wrong grandchild at node 2, true tok2 at node 3 (child of node 1)!
        wrong_a = (tok1 + 1) % 64
        wrong_b = (tok2 + 1) % 64
        tree_ids = torch.tensor([[wrong_a, tok1, wrong_b, tok2]], dtype=torch.long)
        tree_parents = torch.tensor([[-1, -1, 1, 1]], dtype=torch.long)

        pkv_tree = DynamicCache()
        _ = verifier(input_ids=prompt, past_key_values=pkv_tree, use_cache=True)
        prefix_len = pkv_tree.get_seq_length()
        tree_mask, depths = build_tree_attention_mask(
            tree_parents, prefix_len=prefix_len, dtype=torch.float32
        )
        verify_in = torch.cat([torch.tensor([[anchor_tok]]), tree_ids], dim=1)
        pos_ids = torch.cat([torch.tensor([[prefix_len]]), prefix_len + depths], dim=1)
        v_out = verifier(
            input_ids=verify_in,
            past_key_values=pkv_tree,
            use_cache=True,
            attention_mask=tree_mask,
            position_ids=pos_ids,
        )
        res = specinfer_verify_tree_step(
            v_out.logits, tree_ids, tree_parents, temperature=0.0
        )
        assert res.accepted_tokens[0] == [tok1, tok2]
        assert res.accepted_node_indices[0] == [1, 3]
        assert int(res.bonus_tokens[0].item()) == tok3

        # Compact pkv_tree and verify next forward pass matches sequential pkv_ref!
        compact_tree_kv_cache(pkv_tree, prefix_len, res.accepted_node_indices)
        next_tree = verifier(
            input_ids=torch.tensor([[tok3]]),
            past_key_values=pkv_tree,
            use_cache=True,
        )
        next_ref = verifier(
            input_ids=torch.tensor([[tok3]]),
            past_key_values=pkv_ref,
            use_cache=True,
        )
        assert torch.max(torch.abs(next_tree.logits - next_ref.logits)).item() < 1e-5

    @pytest.mark.parametrize("stochastic_mode", [False, True])
    def test_specinfer_monte_carlo_lossless_target_distribution_with_oov(
        self, stochastic_mode: bool
    ) -> None:
        """20,000-trial Monte Carlo proof that SpecInfer preserves p(y|x) at T=0.6."""
        vocab_size = 10
        shortlist_size = 6  # tokens 6..9 are 100% OOV!
        temperature = 0.6
        n_trials = 20000
        gen = torch.Generator().manual_seed(2026 + int(stochastic_mode))

        raw_logits = torch.tensor(
            [1.4, 0.9, 0.6, -0.2, 0.1, -0.5, 1.1, 0.3, -0.4, 0.7],
            dtype=torch.float32,
        )
        target_p = torch.softmax(raw_logits / temperature, dim=-1)
        # Replicate logits across [B=n_trials, 1 + N=4, V=10]
        v_logits = raw_logits.view(1, 1, vocab_size).expand(n_trials, 4, vocab_size)
        tree_parents = torch.full((n_trials, 3), -1, dtype=torch.long)
        d2t = torch.zeros(shortlist_size, dtype=torch.long)

        if not stochastic_mode:
            # Deterministic Kraus top-3 shortlist fork [0, 1, 2]
            tree_ids = torch.tensor([[0, 1, 2]], dtype=torch.long).expand(n_trials, 3)
            tree_q = None
        else:
            # 3 stochastic children drawn from skewed shortlist distributions q^(1..3)
            q1 = torch.softmax(torch.tensor([0.2, 1.5, -0.3, 0.8, 0.0, -1.0]), dim=-1)
            q2 = torch.softmax(torch.tensor([1.2, -0.5, 0.9, -0.2, 0.4, 0.1]), dim=-1)
            q3 = torch.softmax(torch.tensor([-0.4, 0.3, 1.1, 0.2, 0.9, -0.2]), dim=-1)
            c1 = torch.multinomial(
                q1.expand(n_trials, -1), num_samples=1, generator=gen
            )
            c2 = torch.multinomial(
                q2.expand(n_trials, -1), num_samples=1, generator=gen
            )
            c3 = torch.multinomial(
                q3.expand(n_trials, -1), num_samples=1, generator=gen
            )
            tree_ids = torch.cat([c1, c2, c3], dim=1)
            tree_q = (
                torch.stack([q1, q2, q3], dim=0)
                .unsqueeze(0)
                .expand(n_trials, 3, shortlist_size)
            )

        res = specinfer_verify_tree_step(
            verifier_logits=v_logits,
            tree_draft_ids=tree_ids,
            tree_parent_idx=tree_parents,
            temperature=temperature,
            tree_q_probs=tree_q,
            d2t=d2t,
            stochastic_proposals=stochastic_mode,
            generator=gen,
        )

        # First emitted token is accepted_tokens[b][0] if accepted else bonus_tokens[b]
        counts = torch.zeros(vocab_size, dtype=torch.float64)
        bonuses = res.bonus_tokens.tolist()
        for b_idx in range(n_trials):
            acc_b = res.accepted_tokens[b_idx]
            first_emitted = acc_b[0] if acc_b else bonuses[b_idx]
            counts[first_emitted] += 1.0

        emp_p = (counts / float(n_trials)).to(torch.float32)
        max_err = float(torch.max(torch.abs(emp_p - target_p)).item())
        assert max_err < 0.015, (
            f"Empirical {emp_p.tolist()} vs target {target_p.tolist()}"
        )
        # Verify OOV token 6 received its true ~15%+ mass!
        assert abs(float(emp_p[6].item() - target_p[6].item())) < 0.012


class TestPluginConverterAndProdBenchClient:
    def test_pyproject_entrypoint_and_plugin_config_mapping(
        self, tmp_path: Path
    ) -> None:
        pyproject_data = tomllib.loads(
            (REPO_SPECULATORS_DIR / "pyproject.toml").read_text(encoding="utf-8")
        )
        plugins = pyproject_data["project"]["entry-points"]["vllm.general_plugins"]
        assert (
            plugins["kvn_dspark"]
            == "speculators.vllm_integration.plugin:register_kvn_vllm_plugin"
        )

        vllm_spec_kvn = {"model": str(tmp_path / "kvn_ckpt")}
        configure_kvn_speculator_dict(
            {"kvn_backbone_type": "kvn", "block_size": 8, "sample_from_anchor": True},
            vllm_spec_kvn,
        )
        assert vllm_spec_kvn["method"] == "custom_class"
        assert vllm_spec_kvn["model"] == PROPOSER_IMPORT_PATH
        assert vllm_spec_kvn["num_speculative_tokens"] == 8

        vllm_spec_swa = {"model": str(tmp_path / "swa_ckpt")}
        configure_kvn_speculator_dict(
            {"kvn_backbone_type": "swa", "block_size": 8, "sample_from_anchor": True},
            vllm_spec_swa,
        )
        assert vllm_spec_swa["method"] == "dspark"
        assert vllm_spec_swa["num_speculative_tokens"] == 8

    def test_kvn_dspark_converter_exports_pruned_vllm_checkpoint(
        self, tmp_path: Path
    ) -> None:
        v_dir, ckpt_dir, _, _, _ = _make_tiny_verifier_and_drafter(tmp_path, n_modes=16)
        out_dir = tmp_path / "exported_kvn"
        convert_model(
            model=str(ckpt_dir),
            verifier=str(v_dir),
            algorithm="kvn_dspark",
            output_path=str(out_dir),
            validate_device="cpu",
            prune_modes=8,
        )
        assert (out_dir / "vllm_speculative_config.json").is_file()
        reloaded = KvNDSparkDraftModel.from_pretrained(out_dir)
        assert reloaded.config.kvn_n_modes == 8
        assert reloaded.config.kvn_state_dtype == "float32"

    def test_benchmark_vllm_prod_prometheus_and_summary_helpers(self) -> None:
        bench_mod = _load_script_module("benchmark_vllm_prod_test", BENCH_PROD_PATH)
        m_before = bench_mod.parse_prometheus_spec_decode_counters(
            'vllm:spec_decode_num_drafts_total{engine="0"} 100\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 700\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 420\n'
        )
        m_after = bench_mod.parse_prometheus_spec_decode_counters(
            'vllm:spec_decode_num_drafts_total{engine="0"} 200\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 1400\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 870\n'
        )
        tau, acc_rate = bench_mod.compute_delta_spec_metrics(m_before, m_after)
        assert tau == pytest.approx(5.5)
        assert acc_rate == pytest.approx(450.0 / 700.0, rel=1e-3)


class TestCalibrationAndBatchedE2EFixes:
    def test_no_serial_batch_loop_exists_in_eval_script(self) -> None:
        source = CALIB_SCRIPT_PATH.read_text(encoding="utf-8")
        assert "for _ in range(bs):" not in source
        tree = ast.parse(source)
        assert tree is not None

    def test_cumulative_prefix_survival_pareto_truncates_compounded_decay(self) -> None:
        calib_mod = _load_script_module("kvn_calib_test_pareto", CALIB_SCRIPT_PATH)
        # Marginal c_k = 0.75 >= 0.60, cumulative survival is [0.75, 0.5625, 0.4219]
        confs = torch.tensor([[0.75, 0.75, 0.75]], dtype=torch.float32)
        accs = torch.tensor([[True, True, True]])
        pareto = calib_mod.compute_dynamic_stopping_pareto(
            confs, accs, thresholds=[0.60]
        )
        assert pareto[0]["mean_drafted_tokens"] == pytest.approx(1.0)
        assert pareto[0]["mean_accept_length_tau"] == pytest.approx(2.0)

    def test_calibrate_uses_full_vocab_argmax_and_rejects_oov_targets(
        self, tmp_path: Path
    ) -> None:
        calib_mod = _load_script_module("kvn_calib_test_oov", CALIB_SCRIPT_PATH)
        v_dir, ckpt_dir, data_dir, verifier, _ = _make_tiny_verifier_and_drafter(
            tmp_path, vocab_size=64, draft_vocab_size=16, block_size=4
        )
        # Force verifier lm_head so OOV token 50 (>= 16) strictly dominates every input!
        with torch.no_grad():
            verifier.lm_head.weight.zero_()
            verifier.model.norm.weight.fill_(1.0)
            verifier.lm_head.weight[50, 0] = 100.0
        verifier.save_pretrained(v_dir)

        hs_dir = data_dir / "hidden_states"
        hs_dir.mkdir(parents=True, exist_ok=True)
        final_hs = torch.ones((24, 1, 32), dtype=torch.float32)
        aux_hs = torch.randn((24, 3, 32), dtype=torch.float32)
        save_file(
            {
                "hidden_states": torch.cat([aux_hs, final_hs], dim=1),
                "token_ids": torch.randint(0, 16, (24,), dtype=torch.long),
            },
            str(hs_dir / "hs_0000.safetensors"),
        )

        report = calib_mod.evaluate_checkpoint_calibration(
            checkpoint_dir=ckpt_dir,
            verifier_model=str(v_dir),
            data_dir=data_dir,
            draft_vocab_size=16,
            max_samples=4,
            device="cpu",
        )
        assert report["full_vocab_size_evaluated"] == 64
        assert report["oov_target_rate_pct"] == 100.0
        assert report["per_position"]["pos_0"]["accept_rate_pct"] == 0.0

    @pytest.mark.parametrize("backbone_type", ["kvn", "hybrid_1swa_2kvn", "swa"])
    def test_batched_e2e_greedy_parity_and_eos_stopping(
        self, tmp_path: Path, backbone_type: str
    ) -> None:
        calib_mod = _load_script_module(
            f"kvn_calib_e2e_{backbone_type}", CALIB_SCRIPT_PATH
        )
        v_dir, ckpt_dir, data_dir, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path,
            backbone_type=backbone_type,
            vocab_size=64,
            draft_vocab_size=32,
            block_size=4,
            eos_token_id=63,
        )
        prompts = [[1, 4, 7, 10, 13], [2, 6, 9, 12, 15, 18]]
        report = calib_mod.evaluate_checkpoint_e2e(
            checkpoint_dir=ckpt_dir,
            verifier_model=str(v_dir),
            prompt_token_ids_list=prompts,
            max_new_tokens=10,
            temperature=0.0,
            batch_sizes=[1, 3],
            data_dir=data_dir,
            draft_vocab_size=32,
            device="cpu",
        )
        assert report["exact_greedy_match_all_prompts"] is True
        assert "1" in report["amdahl_by_batch_size"]
        assert "3" in report["amdahl_by_batch_size"]
        assert report["amdahl_by_batch_size"]["3"]["spec_tps"] > 0.0

    def test_kraus_tree_e2e_greedy_parity(self, tmp_path: Path) -> None:
        calib_mod = _load_script_module("kvn_calib_e2e_tree", CALIB_SCRIPT_PATH)
        v_dir, ckpt_dir, data_dir, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path,
            backbone_type="kvn",
            vocab_size=64,
            draft_vocab_size=32,
            block_size=4,
        )
        report = calib_mod.evaluate_checkpoint_e2e(
            checkpoint_dir=ckpt_dir,
            verifier_model=str(v_dir),
            prompt_token_ids_list=[[3, 6, 9, 12, 15]],
            max_new_tokens=10,
            temperature=0.0,
            batch_sizes=[1, 2],
            data_dir=data_dir,
            draft_vocab_size=32,
            device="cpu",
            use_kraus_tree=True,
            tree_budget=8,
            branch_width=3,
        )
        assert report["use_kraus_tree"] is True
        assert report["exact_greedy_match_all_prompts"] is True
        assert report["amdahl_by_batch_size"]["2"]["spec_tps"] > 0.0


class TestGate1AdversarialRegressionGuards:
    def test_decode_preemption_auto_restores_in_propose_batch(
        self, tmp_path: Path
    ) -> None:
        _, _, _, _, drafter = _make_tiny_verifier_and_drafter(tmp_path)
        proposer = KvNDSparkVllmProposer(
            drafter, max_num_seqs=4, device="cpu", enable_apc=False
        )
        assert proposer.slot_pool is not None

        prompt_ids = torch.randint(0, 32, (20,), dtype=torch.long)
        aux_seq = torch.randn(20, 3 * 32)
        proposer.prefill_request("req-keep", prompt_ids, aux_seq)
        proposer.prefill_request("req-swap", prompt_ids, aux_seq)

        # Preempt during decode (no intermediate prefill_request before propose_batch!)
        proposer.preempt_requests(["req-swap"])
        assert "req-swap" in proposer.slot_pool._preempted_snapshots
        step_aux = torch.randn(1, 3 * 32)
        out = proposer.propose_batch(
            ["req-keep", "req-swap"],
            [[7], [7]],
            [step_aux, step_aux],
        )
        assert torch.equal(out.padded_draft_ids[0], out.padded_draft_ids[1])

    def test_kraus_tree_per_depth_survival_avoids_sibling_double_cumprod(
        self, tmp_path: Path
    ) -> None:
        # Depth-1 siblings [0.80, 0.20, 0.05], depth-2 [0.64, 0.16], depth-3 [0.512]
        tree_dict = {
            "tree_survival": torch.tensor(
                [[0.80, 0.20, 0.05, 0.64, 0.16, 0.512]], dtype=torch.float32
            ),
            "tree_depths": torch.tensor([[1, 1, 1, 2, 2, 3]], dtype=torch.long),
            "tree_parent_idx": torch.tensor([[-1, -1, -1, 0, 1, 3]], dtype=torch.long),
        }
        alphas = KvNDSparkVllmProposer._derive_per_depth_tree_confidences(
            tree_dict, max_horizon=8
        )
        recovered_surv = torch.cumprod(alphas, dim=-1)
        expected_surv = torch.tensor([[0.80, 0.64, 0.512]], dtype=torch.float32)
        assert torch.max(torch.abs(recovered_surv - expected_surv)).item() < 1e-6

        _, _, _, _, drafter = _make_tiny_verifier_and_drafter(tmp_path)
        proposer = KvNDSparkVllmProposer(
            drafter,
            max_num_seqs=4,
            device="cpu",
            use_kraus_tree=True,
            tree_budget=8,
            branch_width=3,
        )
        proposer.prefill_request(
            "req-tree",
            torch.randint(0, 32, (16,), dtype=torch.long),
            torch.randn(16, 3 * 32),
        )
        out = proposer.propose_batch(
            ["req-tree"], [[4]], [torch.randn(1, 3 * 32)], use_kraus_tree=True
        )
        assert out.tree_dict is not None
        assert out.tree_schedule is not None
        assert out.confidences.size(1) <= int(drafter.block_size)

    def test_hybrid_nu_list_and_engram_3gram_decode_parity(
        self, tmp_path: Path
    ) -> None:
        calib_mod = _load_script_module("kvn_calib_adv_guards", CALIB_SCRIPT_PATH)
        _, _, _, _, drafter = _make_tiny_verifier_and_drafter(
            tmp_path, backbone_type="hybrid_1swa_2kvn"
        )
        assert drafter.engram is not None
        # Amplify Engram projection so omitting Engram or prev_context_ids diverges!
        with torch.no_grad():
            drafter.engram.engram_table.weight.normal_(mean=0.0, std=1.5)
            drafter.engram.engram_proj.weight.normal_(mean=0.0, std=1.5)
            if drafter.rollout.layer_mode_logits is not None:
                drafter.rollout.layer_mode_logits.copy_(
                    torch.linspace(-1.0, 1.0, drafter.rollout.layer_mode_logits.size(0))
                    .unsqueeze(1)
                    .expand_as(drafter.rollout.layer_mode_logits)
                )

        sc = drafter.allocate_state_cache(batch_size=2, device="cpu")
        sc[:, 2] = torch.linspace(0.2, 2.0, sc[:, 2].numel()).reshape_as(sc[:, 2])
        fc_blk = torch.randn(2, 2, drafter.hidden_size)
        tok_blk = torch.tensor([[3, 7], [5, 9]], dtype=torch.long)
        vlen = torch.tensor([2, 2], dtype=torch.long)

        _, _, roll_out = calib_mod._advance_kvn_rollout_blocks(
            drafter, fc_blk, tok_blk, vlen, sc, apply_mixer=False
        )
        # Purity roll_out[3] < 1.0 proves non-zero Lindblad anc_nu reached rollout!
        assert isinstance(roll_out, tuple)
        assert len(roll_out) >= 4
        assert float(roll_out[3].mean().item()) < 0.99

        h_blk = torch.randn(2, drafter.block_size, drafter.hidden_size)
        b_logits = torch.randn(2, drafter.block_size, drafter.lm_head.out_features)
        anc_ids = torch.tensor([11, 14], dtype=torch.long)
        prev_ctx = torch.tensor([[2, 6], [4, 8]], dtype=torch.long)

        ids_eval, _ = calib_mod._sample_draft_block_with_probs(
            drafter,
            h_blk,
            b_logits,
            anc_ids,
            temperature=0.0,
            prev_context_ids=prev_ctx,
        )
        ids_core, _, _ = drafter._decode_draft_block(
            hidden_blocks=h_blk,
            base_logits=b_logits,
            anchor_token_ids=anc_ids,
            temperature=0.0,
            prev_context_ids=prev_ctx,
        )
        assert torch.equal(ids_eval, ids_core)


class TestGate2Vllm0291ProductionIntegration:
    def test_update_kvn_dspark_pretrained_config_signature_and_fields(self) -> None:
        cfg_dict = {
            "speculators_model_type": "kvn_dspark",
            "kvn_backbone_type": "kvn",
            "kvn_n_modes": 128,
            "block_size": 8,
            "sample_from_anchor": False,
            "draft_vocab_size": 32000,
            "mask_token_id": 151669,
            "aux_hidden_state_layer_ids": [9, 18, 27],
            "use_lindblad_purity": True,
        }
        pre_trained_cfg: dict[str, object] = {"model_type": "qwen3"}
        update_kvn_dspark_pretrained_config(
            config_dict=cfg_dict,
            pre_trained_config=pre_trained_cfg,
        )
        assert pre_trained_cfg["architectures"] == ["Qwen3DSparkModel"]
        assert pre_trained_cfg["dspark_bonus_anchor"] is True
        assert pre_trained_cfg["eagle_aux_hidden_state_layer_ids"] == [9, 18, 27]
        assert pre_trained_cfg["target_layer_ids"] == [8, 17, 26]
        assert pre_trained_cfg["kvn_n_modes"] == 128
        assert pre_trained_cfg["use_lindblad_purity"] is True

    def test_rewrite_kvn_and_swa_speculative_dict_and_explicit_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, kvn_dir, _, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path / "kvn_case",
            backbone_type="kvn",
            block_size=4,
            sample_from_anchor=False,
        )
        _, swa_dir, _, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path / "swa_case",
            backbone_type="swa",
            block_size=4,
            sample_from_anchor=False,
        )
        monkeypatch.delenv("KVN_DSPARK_CHECKPOINT_PATH", raising=False)

        spec_kvn: dict[str, object] = {"model": str(kvn_dir)}
        resolved = plugin_mod._maybe_rewrite_kvn_speculative_dict(spec_kvn)
        assert resolved == str(kvn_dir)
        assert spec_kvn["method"] == "custom_class"
        assert spec_kvn["model"] == PROPOSER_IMPORT_PATH
        assert spec_kvn["num_speculative_tokens"] == 3

        spec_kvn_override: dict[str, object] = {
            "model": str(kvn_dir),
            "num_speculative_tokens": 2,
        }
        plugin_mod._maybe_rewrite_kvn_speculative_dict(spec_kvn_override)
        assert spec_kvn_override["num_speculative_tokens"] == 2

        spec_swa: dict[str, object] = {"model": str(swa_dir)}
        resolved_swa = plugin_mod._maybe_rewrite_kvn_speculative_dict(spec_swa)
        assert resolved_swa is None
        assert spec_swa["method"] == "dspark"
        assert spec_swa["model"] == str(swa_dir)
        assert spec_swa["num_speculative_tokens"] == 3

    def test_proposer_ignores_target_draft_model_config_and_load_model_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        v_dir, kvn_dir, _, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path, backbone_type="kvn"
        )
        monkeypatch.setenv("KVN_DSPARK_CHECKPOINT_PATH", str(kvn_dir))

        class _DummyModelCfg:
            def __init__(self, model: str) -> None:
                self.model = model

        target_cfg = _DummyModelCfg(str(v_dir))

        class _DummySpecCfg:
            method = "custom_class"
            model = PROPOSER_IMPORT_PATH
            target_model_config = target_cfg
            # Mirrors vLLM 0.29.1 SpeculativeConfig.__post_init__ line 183!
            draft_model_config = target_cfg

        class _DummyVllmCfg:
            model_config = target_cfg
            speculative_config = _DummySpecCfg()
            scheduler_config = None
            device_config = None

        proposer = KvNDSparkVllmProposer(_DummyVllmCfg(), device="cpu")
        assert str(proposer._checkpoint_path) == str(kvn_dir)

        load_calls = 0
        orig_from_pretrained = KvNDSparkDraftModel.from_pretrained

        def _counting_from_pretrained(*args, **kwargs):
            nonlocal load_calls
            load_calls += 1
            return orig_from_pretrained(*args, **kwargs)

        monkeypatch.setattr(
            KvNDSparkDraftModel, "from_pretrained", _counting_from_pretrained
        )

        class _DummyTargetVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.applied_aux_layers: tuple[int, ...] | None = None

            def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
                self.applied_aux_layers = tuple(layers)

        target_stub = _DummyTargetVerifier()
        proposer.load_model(target_model=target_stub)
        assert load_calls == 1
        assert target_stub.applied_aux_layers == tuple(
            proposer.aux_hidden_state_layer_ids
        )

        # Prefill a request into slot_pool and verify second load_model() preserves it!
        proposer.prefill_request(
            "req-persist",
            torch.randint(0, 32, (8,), dtype=torch.long),
            torch.randn(8, 3 * 32),
        )
        assert proposer.slot_pool is not None
        slot_before = proposer.slot_pool.get_slot("req-persist")
        q_before = proposer.slot_pool.q_pool[slot_before].clone()

        target_stub_2 = _DummyTargetVerifier()
        proposer.load_model(target_model=target_stub_2)
        assert load_calls == 1
        assert target_stub_2.applied_aux_layers == tuple(
            proposer.aux_hidden_state_layer_ids
        )
        assert proposer.slot_pool.get_slot("req-persist") == slot_before
        assert torch.equal(proposer.slot_pool.q_pool[slot_before], q_before)

    def test_gpu_model_runner_aux_layers_and_compute_hash_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, kvn_dir_a, _, _, drafter_a = _make_tiny_verifier_and_drafter(
            tmp_path / "a", backbone_type="kvn"
        )
        _, kvn_dir_b, _, _, _ = _make_tiny_verifier_and_drafter(
            tmp_path / "b", backbone_type="kvn"
        )
        proposer = KvNDSparkVllmProposer(drafter_a, max_num_seqs=4, device="cpu")
        fake_runner_cls, fake_spec_cls, mod_map = _make_fake_vllm_hook_classes()
        orig_import = plugin_mod.importlib.import_module
        monkeypatch.setattr(
            plugin_mod.importlib,
            "import_module",
            lambda name, *a, **kw: mod_map.get(name) or orig_import(name, *a, **kw),
        )
        plugin_mod._patch_gpu_model_runner()
        plugin_mod._patch_speculative_config_and_engine_args()

        runner = fake_runner_cls(proposer)
        assert runner.use_aux_hidden_state_outputs is True
        assert runner._get_eagle3_aux_layers_from_config() == tuple(
            proposer.aux_hidden_state_layer_ids
        )

        sc_plain = fake_spec_cls(model="ngram")
        sc_kvn_a = fake_spec_cls(model=str(kvn_dir_a))
        sc_kvn_b = fake_spec_cls(model=str(kvn_dir_b))
        assert sc_kvn_a.model == PROPOSER_IMPORT_PATH
        assert sc_plain.compute_hash() == "base_hash_const"
        assert sc_kvn_a.compute_hash() != sc_plain.compute_hash()
        assert sc_kvn_a.compute_hash() != sc_kvn_b.compute_hash()

    def test_benchmark_vllm_prod_baseline_json_speedup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bench_mod = _load_script_module(
            "benchmark_vllm_prod_gate2_test", BENCH_PROD_PATH
        )
        baseline_file = tmp_path / "Verifier_Baseline_vllm_serve_prod.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "model_name": "Qwen/Qwen3-8B",
                    "concurrency_sweep": [
                        {"concurrency": 1, "output_throughput_tok_s": 50.0},
                        {"concurrency": 4, "output_throughput_tok_s": 160.0},
                    ],
                }
            ),
            encoding="utf-8",
        )

        tps_map = bench_mod.load_baseline_throughput_map(baseline_file)
        assert tps_map == {1: 50.0, 4: 160.0}
        assert bench_mod.load_baseline_throughput_map(tmp_path / "missing.json") == {}

        async def _fake_tier(
            base_url: str,
            model_name: str,
            concurrency: int,
            requests_per_worker: int,
            max_tokens: int,
            temperature: float,
        ):
            tps = 115.0 if concurrency == 1 else 280.0
            return bench_mod.ConcurrencyTierSummary(
                concurrency=concurrency,
                num_requests=concurrency * requests_per_worker,
                wall_time_s=1.0,
                total_output_tokens=int(tps),
                output_throughput_tok_s=tps,
                mean_ttft_ms=10.0,
                p50_ttft_ms=10.0,
                p95_ttft_ms=12.0,
                mean_tpot_ms=8.0,
                mean_itl_ms=8.0,
                p95_itl_ms=9.0,
            )

        monkeypatch.setattr(bench_mod, "run_concurrency_tier", _fake_tier)
        parser = bench_mod.build_arg_parser()
        parsed = parser.parse_args(
            [
                "--model-name",
                "Qwen/Qwen3-8B",
                "--concurrencies",
                "1,4",
                "--skip-apc-benchmark",
                "--baseline-json",
                str(baseline_file),
            ]
        )
        suite = asyncio.run(bench_mod.run_full_benchmark_suite(parsed))
        assert suite["baseline_json"] == str(baseline_file)
        assert suite["concurrency_sweep"][0]["speedup_vs_baseline"] == pytest.approx(
            2.3
        )
        assert suite["concurrency_sweep"][1]["speedup_vs_baseline"] == pytest.approx(
            1.75
        )
