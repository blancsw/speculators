"""Phase F production regression & CUDA-Graph parity suite for ``KvN-DSpark v3``.

Verifies:
1. Fused ``segmented_2d_osc_scan`` and ``fused_2d_osc_dynamics_multistep_decode``
   Triton kernels achieve ``max_abs_diff < 1e-5`` FP32 state parity against
   step-by-step recurrence across ``reset_mask`` boundaries and ``initial_state``.
2. ``KvNDSparkVllmProposer.propose_vllm_step`` eliminates rejected-draft state
   corruption across 0-accepted, partial-accepted, and all-accepted speculative steps,
   chunked prefill, vLLM prefix-cache skip + ``ZeroReplayAPCStateCache`` 0-FLOP restore,
   batched concurrent prefills, and ``RECOMPUTE`` / swap preemptions.
3. Zero-sync static-slot ``torch.cuda.CUDAGraph`` decode path matches eager reference
   bit-for-bit at ``T = 0.0`` across padded batch buckets (with zero scratch-slot
   interference) and advances Philox RNG cleanly on mixed-temperature ``T > 0`` batches.
4. Vectorized ``[B * W]`` ``draft_kraus_tree_step`` + ``propose_tree_cudagraph`` and
   ``KVN_VLLM_USE_KRAUS_TREE=1`` spine mode preserve POVM completeness ``< 1e-5`` and
   exact CUDA-Graph parity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.kvn_dspark import KvNDSparkDraftModel, KvNDSparkSpeculatorConfig
from speculators.models.kvn_dspark.triton_kernels import (
    pytorch_segmented_2d_osc_scan,
    segmented_2d_osc_scan,
)
from speculators.vllm_integration.amdahl_scheduler import (
    HardwareRooflineProfile,
    PurityLoadAwareAmdahlScheduler,
)
from speculators.vllm_integration.kvn_proposer import KvNDSparkVllmProposer
from speculators.vllm_integration.plugin import (
    PROPOSER_IMPORT_PATH,
    _patch_runner_bookkeeping_hook,
    _patch_runner_propose_hook,
    _patch_vllm_config_cls,
)


def _build_v3_2d_osc_drafter(
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    block_size: int = 5,
    n_modes: int = 16,
    hidden_size: int = 64,
    vocab_size: int = 96,
    draft_vocab_size: int = 48,
) -> KvNDSparkDraftModel:
    torch.manual_seed(2026)
    tl_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
    )
    cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        block_size=block_size,
        max_anchors=64,
        aux_hidden_state_layer_ids=[0, 1, 2],
        sample_from_anchor=False,
        draft_vocab_size=draft_vocab_size,
        kvn_backbone_type="kvn",
        kvn_num_layers=2,
        kvn_n_modes=n_modes,
        kvn_d_low=32,
        kvn_enable_sp4=False,
        use_lindblad_purity=True,
        use_engram=True,
        engram_num_buckets=64,
        engram_dim=16,
        use_kraus_head=True,
        kraus_rank=4,
        kraus_top_c=12,
        markov_head_type="rnn",
        markov_rank=16,
    )
    model = KvNDSparkDraftModel(cfg).to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.02)
        torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
        torch.nn.init.ones_(model.verifier_norm.weight)
        if model.engram is not None:
            model.engram.engram_table.weight.normal_(0.0, 0.5)
            model.engram.engram_proj.weight.normal_(0.0, 0.5)
    return model


def _compute_ground_truth_state_and_draft(
    model: KvNDSparkDraftModel,
    full_tokens: list[int],
    full_fc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance ``model`` step-by-step over ``full_tokens`` from zero state.

    ``full_tokens`` has length ``L >= 1`` (prompt + accepted + latest bonus token).
    ``full_fc`` has shape ``[L - 1, D]`` containing ``[fc_0, ..., fc_{L-2}]`` so
    token ``y_t`` receives causal verifier feature ``prev_fc_t = 0`` at ``t=0`` and
    ``full_fc[t - 1]`` for ``t >= 1``.
    """
    dev = next(model.parameters()).device
    fc_dtype = model.fc.weight.dtype
    d_model = model.hidden_size
    total_l = len(full_tokens)
    assert full_fc.shape == (max(0, total_l - 1), d_model)

    state_cache = model.allocate_state_cache(1, dev)
    zero_fc = torch.zeros((1, d_model), dtype=fc_dtype, device=dev)
    prev_fcs = (
        zero_fc
        if total_l == 1
        else torch.cat([zero_fc, full_fc.to(device=dev, dtype=fc_dtype)], dim=0)
    )

    # Feed 1 token at a time up to L - 1, then draft on the final anchor token L - 1
    for t_idx in range(total_l - 1):
        tok_1 = torch.tensor([[full_tokens[t_idx]]], dtype=torch.long, device=dev)
        fc_1 = prev_fcs[t_idx : t_idx + 1].unsqueeze(0)
        vlen_1 = torch.ones((1,), dtype=torch.long, device=dev)
        _, state_cache, _ = model.advance_and_draft_step(
            fc_1,
            tok_1,
            vlen_1,
            state_cache,
            temperature=0.0,
            return_details=True,
            include_draft_probs=False,
        )

    anc_tok = torch.tensor([[full_tokens[-1]]], dtype=torch.long, device=dev)
    anc_fc = prev_fcs[total_l - 1 : total_l].unsqueeze(0)
    vlen_1 = torch.ones((1,), dtype=torch.long, device=dev)
    p0 = full_tokens[-3] if total_l >= 3 else 0
    p1 = full_tokens[-2] if total_l >= 2 else 0
    prev2 = torch.tensor([[p0, p1]], dtype=torch.long, device=dev)
    draft_ids, state_cache, _ = model.advance_and_draft_step(
        anc_fc,
        anc_tok,
        vlen_1,
        state_cache,
        temperature=0.0,
        return_details=True,
        prev_context_ids=prev2,
        include_draft_probs=False,
    )
    q_ref = state_cache[:, 0, 0, :].float().clone()
    p_ref = state_cache[:, 1, 0, :].float().clone()
    nu_ref = state_cache[:, 2, 0, :].float().clone()
    return q_ref, p_ref, nu_ref, draft_ids[0, 1:].clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_segmented_2d_osc_scan_and_multistep_decode_fp32_parity() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(77)
    for seq_len in (1, 17, 128, 513):
        bsz, modes = 3, 64
        r_mat = torch.empty(
            (bsz, seq_len, modes), device=dev, dtype=torch.float32
        ).uniform_(0.90, 0.999)
        dtheta = (
            torch.randn((bsz, seq_len, modes), device=dev, dtype=torch.float32) * 0.25
        )
        u_q = torch.randn((bsz, seq_len, modes), device=dev, dtype=torch.float32) * 0.1
        u_p = torch.randn((bsz, seq_len, modes), device=dev, dtype=torch.float32) * 0.1
        reset_mask = torch.zeros((bsz, seq_len), device=dev, dtype=torch.bool)
        reset_mask[:, 0] = True
        if seq_len > 20:
            reset_mask[0, seq_len // 2] = True
            reset_mask[2, seq_len // 3] = True
        q0 = torch.randn((bsz, modes), device=dev, dtype=torch.float32)
        p0 = torch.randn((bsz, modes), device=dev, dtype=torch.float32)
        reset_mask[1, 0] = False  # test initial_state continuation on batch row 1

        a_cos = r_mat * torch.cos(dtheta)
        a_sin = r_mat * torch.sin(dtheta)
        a_nu = torch.empty(
            (bsz, seq_len, modes), device=dev, dtype=torch.float32
        ).uniform_(0.85, 0.99)
        u_nu = torch.rand((bsz, seq_len, modes), device=dev, dtype=torch.float32) * 0.05
        nu0 = torch.rand((bsz, modes), device=dev, dtype=torch.float32)

        q_tri, p_tri, nu_tri = segmented_2d_osc_scan(
            a_cos,
            a_sin,
            u_q,
            u_p,
            a_nu=a_nu,
            u_nu=u_nu,
            reset_mask=reset_mask,
            initial_state=(q0, p0, nu0),
        )
        q_ref, p_ref, nu_ref = pytorch_segmented_2d_osc_scan(
            a_cos,
            a_sin,
            u_q,
            u_p,
            a_nu=a_nu,
            u_nu=u_nu,
            reset_mask=reset_mask,
            initial_state=(q0, p0, nu0),
        )
        assert torch.max(torch.abs(q_tri - q_ref)).item() < 1e-5
        assert torch.max(torch.abs(p_tri - p_ref)).item() < 1e-5
        assert nu_tri is not None
        assert nu_ref is not None
        assert torch.max(torch.abs(nu_tri - nu_ref)).item() < 1e-5

    model = _build_v3_2d_osc_drafter(device=dev, dtype=torch.float32)
    layer = model.kvn_layers[0]
    x_seq = torch.randn((2, 9, model.hidden_size), device=dev, dtype=torch.float32)
    vlen = torch.tensor([9, 5], device=dev, dtype=torch.long)
    q_init = torch.randn((2, 1, 16), device=dev, dtype=torch.float32)
    p_init = torch.randn((2, 1, 16), device=dev, dtype=torch.float32)
    nu_init = torch.rand((2, 1, 16), device=dev, dtype=torch.float32)

    h_ms, q_ms, p_ms, nu_ms, _ = layer.multistep_decode(
        x_seq,
        q_init.squeeze(1),
        p_init.squeeze(1),
        valid_lens=vlen,
        nu_init=nu_init.squeeze(1),
        state_dtype=torch.float32,
    )
    assert nu_ms is not None
    for b_idx, valid_t in enumerate([9, 5]):
        st_b: tuple[torch.Tensor, ...] = (
            q_init[b_idx : b_idx + 1],
            p_init[b_idx : b_idx + 1],
            nu_init[b_idx : b_idx + 1],
        )
        h_steps = []
        for t_i in range(valid_t):
            h_t, st_b, _ = layer.recurrent_step(
                x_seq[b_idx : b_idx + 1, t_i : t_i + 1],
                st_b,
                state_dtype=torch.float32,
            )
            h_steps.append(h_t)
        h_cat = torch.cat(h_steps, dim=1)
        assert torch.max(torch.abs(h_ms[b_idx, :valid_t] - h_cat[0])).item() < 1e-5
        assert torch.max(torch.abs(q_ms[b_idx] - st_b[0][0, 0])).item() < 1e-5
        assert torch.max(torch.abs(p_ms[b_idx] - st_b[1][0, 0])).item() < 1e-5
        assert torch.max(torch.abs(nu_ms[b_idx] - st_b[2][0, 0])).item() < 1e-5


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_propose_vllm_step_rejected_drafts_chunked_apc_and_preemption(
    device: str,
) -> None:
    model = _build_v3_2d_osc_drafter(device=device, dtype=torch.float32, block_size=5)
    # Flat roofline + zero survival floor so untrained model emits full 4-token horizon
    roofline = HardwareRooflineProfile(
        name="unit-flat",
        peak_bf16_tflops=989.0,
        hbm_bandwidth_tbs=4.8,
        ridge_tokens=4096,
        base_verify_ms=10.0,
        base_draft_chain_ms=0.0,
        draft_per_req_slope_ms=0.0,
    )
    scheduler = PurityLoadAwareAmdahlScheduler(
        roofline=roofline, max_horizon=4, min_survival_floor=0.0
    )
    proposer = KvNDSparkVllmProposer(
        model,
        max_num_seqs=16,
        device=device,
        enable_apc=True,
        apc_block_size=16,
        scheduler=scheduler,
    )
    assert proposer.slot_pool is not None
    assert proposer.apc_cache is not None

    d_in = 3 * model.hidden_size
    torch.manual_seed(999)
    # True underlying sequence of 60 tokens + verifier aux hidden states
    master_toks = torch.randint(1, 45, (60,), dtype=torch.long).tolist()
    master_aux = torch.randn((60, d_in), dtype=torch.float32, device=device)
    master_fc = proposer.extract_fc_hidden(master_aux, already_projected=False)

    def _make_runner(
        req_ids: list[str],
        seq_rows: list[list[int]],
        num_no_spec: list[int],
    ) -> SimpleNamespace:
        max_col = max(len(r) for r in seq_rows) + 8
        arr = np.zeros((len(req_ids), max_col), dtype=np.int32)
        for i, r in enumerate(seq_rows):
            arr[i, : len(r)] = r
        ib = SimpleNamespace(
            req_ids=req_ids,
            num_tokens_no_spec=np.array(num_no_spec, dtype=np.int32),
            token_ids_cpu=arr,
            greedy_reqs=set(req_ids),
            random_reqs=set(),
        )
        return SimpleNamespace(input_batch=ib)

    # --- Step 0: Concurrent unchunked prefill for req-A (P=36, bonus y_36)
    #             AND chunk-0 prefill for req-B (chunk=20, no bonus yet) ---
    aux_s0 = torch.cat([master_aux[:36], master_aux[:20]], dim=0)
    runner_s0 = _make_runner(
        ["req-A", "req-B"],
        [master_toks[:37], master_toks[:20]],
        [37, 20],
    )
    sched_s0 = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"req-A": 36, "req-B": 20},
    )
    props_s0 = proposer.propose_vllm_step(
        runner_s0, sched_s0, [[master_toks[36]], []], [aux_s0]
    )
    assert len(props_s0[0]) == 4  # Step-0 immediate draft emitted!
    assert props_s0[1] == []

    q_gt0, p_gt0, nu_gt0, draft_gt0 = _compute_ground_truth_state_and_draft(
        model, master_toks[:37], master_fc[:36]
    )
    slot_a = proposer.slot_pool.get_slot("req-A")
    assert torch.max(torch.abs(proposer.slot_pool.q_pool[slot_a] - q_gt0)).item() < 1e-5
    assert torch.equal(
        torch.tensor(props_s0[0], device=draft_gt0.device), draft_gt0
    )

    # --- Step 1: req-A has ZERO drafts accepted (4 REJECTED draft aux rows!),
    #             req-B finishes chunk-1 (16 tokens + bonus y_36),
    #             req-C hits vLLM prefix cache (skips 32 tokens, schedules only 4!) ---
    rejected_garbage_aux = (
        torch.randn((4, d_in), dtype=torch.float32, device=device) * 50.0
    )
    aux_a_s1 = torch.cat([master_aux[36:37], rejected_garbage_aux], dim=0)  # n_sched=5!
    aux_b_s1 = master_aux[20:36]  # n_sched=16
    aux_c_s1 = master_aux[32:36]  # n_sched=4 (vLLM skipped 0..31!)
    aux_s1 = torch.cat([aux_a_s1, aux_b_s1, aux_c_s1], dim=0)

    runner_s1 = _make_runner(
        ["req-A", "req-B", "req-C"],
        [master_toks[:38], master_toks[:37], master_toks[:37]],
        [38, 37, 37],
    )
    sched_s1 = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"req-A": 5, "req-B": 16, "req-C": 4},
    )
    props_s1 = proposer.propose_vllm_step(
        runner_s1,
        sched_s1,
        [[master_toks[37]], [master_toks[36]], [master_toks[36]]],
        [aux_s1],
    )
    assert proposer.apc_cache.saved_prefill_tokens >= 32

    # Verify req-A (after 4 rejected garbage draft tokens!) matches ground truth at L=38
    q_gt38, p_gt38, nu_gt38, draft_gt38 = _compute_ground_truth_state_and_draft(
        model, master_toks[:38], master_fc[:37]
    )
    assert (
        torch.max(torch.abs(proposer.slot_pool.q_pool[slot_a] - q_gt38)).item() < 1e-5
    )
    assert (
        torch.max(torch.abs(proposer.slot_pool.p_pool[slot_a] - p_gt38)).item() < 1e-5
    )
    assert (
        torch.max(torch.abs(proposer.slot_pool.nu_pool[slot_a] - nu_gt38)).item() < 1e-5
    )
    assert props_s1[0] == draft_gt38.tolist()

    # Verify both chunked req-B and APC-restored req-C match req-A's step-0 L=37 state!
    for rid_bc, prop_bc in [("req-B", props_s1[1]), ("req-C", props_s1[2])]:
        slot_bc = proposer.slot_pool.get_slot(rid_bc)
        assert (
            torch.max(torch.abs(proposer.slot_pool.q_pool[slot_bc] - q_gt0)).item()
            < 1e-5
        )
        assert prop_bc == draft_gt0.tolist()

    # --- Step 2: req-A accepts 2 of 4 drafts (n_samp=3, 2 rejected trailing drafts);
    #             req-B & req-C are omitted (preempted!) ---
    aux_a_s2 = torch.cat(
        [master_aux[37:40], torch.randn((2, d_in), device=device) * 25.0], dim=0
    )
    runner_s2 = _make_runner(["req-A"], [master_toks[:41]], [41])
    sched_s2 = SimpleNamespace(
        finished_req_ids=set(), num_scheduled_tokens={"req-A": 5}
    )
    props_s2 = proposer.propose_vllm_step(
        runner_s2, sched_s2, [master_toks[38:41]], [aux_a_s2]
    )
    q_gt41, p_gt41, nu_gt41, draft_gt41 = _compute_ground_truth_state_and_draft(
        model, master_toks[:41], master_fc[:40]
    )
    assert (
        torch.max(torch.abs(proposer.slot_pool.q_pool[slot_a] - q_gt41)).item() < 1e-5
    )
    assert props_s2[0] == draft_gt41.tolist()

    # --- Step 3: req-A accepts ALL 4 drafts (n_samp=5, L=46);
    #             req-B resumes via vLLM RECOMPUTE (n_sched=37, L=38);
    #             req-C resumes via swap-decode (n_sched=1, L=38) ---
    aux_a_s3 = master_aux[40:45]
    aux_b_s3 = master_aux[:37]  # full RECOMPUTE of 0..36!
    aux_c_s3 = master_aux[36:37]  # direct swap resume of anchor 36!
    runner_s3 = _make_runner(
        ["req-A", "req-B", "req-C"],
        [master_toks[:46], master_toks[:38], master_toks[:38]],
        [46, 38, 38],
    )
    sched_s3 = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"req-A": 5, "req-B": 37, "req-C": 1},
    )
    props_s3 = proposer.propose_vllm_step(
        runner_s3,
        sched_s3,
        [master_toks[41:46], [master_toks[37]], [master_toks[37]]],
        [torch.cat([aux_a_s3, aux_b_s3, aux_c_s3], dim=0)],
    )
    q_gt46, _, _, draft_gt46 = _compute_ground_truth_state_and_draft(
        model, master_toks[:46], master_fc[:45]
    )
    assert (
        torch.max(
            torch.abs(
                proposer.slot_pool.q_pool[proposer.slot_pool.get_slot("req-A")] - q_gt46
            )
        ).item()
        < 1e-5
    )
    assert props_s3[0] == draft_gt46.tolist()
    assert props_s3[1] == draft_gt38.tolist()
    assert props_s3[2] == draft_gt38.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_cudagraph_vs_eager_buckets_stochastic_and_horizon_tensor_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = torch.device("cuda")
    model = _build_v3_2d_osc_drafter(device=dev, dtype=torch.bfloat16, block_size=5)
    sched = PurityLoadAwareAmdahlScheduler(
        roofline=HardwareRooflineProfile.h200_sxm(), max_horizon=4
    )

    # 1. Verify select_horizon_tensor matches select_horizon identically
    confs = torch.tensor(
        [[0.95, 0.85, 0.70, 0.40], [0.55, 0.30, 0.20, 0.10], [0.99, 0.98, 0.97, 0.96]],
        device=dev,
        dtype=torch.float32,
    )
    pur = torch.tensor(
        [[0.92, 0.90, 0.88, 0.85], [0.40, 0.35, 0.30, 0.25], [0.99, 0.99, 0.99, 0.99]],
        device=dev,
        dtype=torch.float32,
    )
    vac = torch.full_like(pur, 0.05)
    for temp_val in (0.0, 0.6):
        dec_ref = sched.select_horizon(
            confidences=confs,
            active_batch_size=8,
            purity=pur,
            p_vac=vac,
            temperature=temp_val,
        )
        t_vec = torch.full((3,), temp_val, device=dev, dtype=torch.float32)
        h_ten = sched.select_horizon_tensor(
            confidences=confs,
            active_batch_size=8,
            purity=pur,
            p_vac=vac,
            temperatures_tensor=t_vec,
        )
        assert torch.equal(dec_ref.per_req_horizons, h_ten)

    # 2. Verify CUDA-Graph proposer vs eager proposer across odd/padded batch sizes
    prop_cg = KvNDSparkVllmProposer(model, max_num_seqs=16, device=dev, scheduler=sched)
    monkeypatch.setenv("KVN_VLLM_DISABLE_CUDAGRAPH", "1")
    prop_eager = KvNDSparkVllmProposer(
        model, max_num_seqs=16, device=dev, scheduler=sched
    )
    monkeypatch.delenv("KVN_VLLM_DISABLE_CUDAGRAPH", raising=False)
    assert prop_cg._cg_enabled is True
    assert prop_eager._cg_enabled is False

    rng = np.random.default_rng(2026)
    d_in = 3 * model.hidden_size
    for bsz in (1, 3, 5):  # exercises padded buckets B=1, B=4, B=8!
        rids = [f"b{bsz}-r{i}" for i in range(bsz)]
        tok_mat = rng.integers(1, 40, size=(bsz, 24), dtype=np.int32)
        aux_pf = torch.randn((bsz * 16, d_in), device=dev, dtype=torch.bfloat16)
        ib_pf = SimpleNamespace(
            req_ids=rids,
            num_tokens_no_spec=np.full((bsz,), 17, dtype=np.int32),
            token_ids_cpu=tok_mat,
            greedy_reqs=set(rids),
            random_reqs=set(),
        )
        so_pf = SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens=dict.fromkeys(rids, 16),
        )
        samp_pf = [[int(tok_mat[i, 16])] for i in range(bsz)]
        out_cg_0 = prop_cg.propose_vllm_step(
            SimpleNamespace(input_batch=ib_pf), so_pf, samp_pf, [aux_pf]
        )
        out_eg_0 = prop_eager.propose_vllm_step(
            SimpleNamespace(input_batch=ib_pf), so_pf, samp_pf, [aux_pf]
        )
        assert out_cg_0 == out_eg_0

        # Decode step with variable accepted lengths 1..3
        aux_dec = torch.randn((bsz * 5, d_in), device=dev, dtype=torch.bfloat16)
        no_spec_dec = np.array([17 + (i % 3) + 1 for i in range(bsz)], dtype=np.int32)
        samp_dec = [
            [int(x) for x in tok_mat[i, 17 : int(no_spec_dec[i])]] for i in range(bsz)
        ]
        ib_dec = SimpleNamespace(
            req_ids=rids,
            num_tokens_no_spec=no_spec_dec,
            token_ids_cpu=tok_mat,
            greedy_reqs=set(rids),
            random_reqs=set(),
        )
        so_dec = SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens=dict.fromkeys(rids, 5),
        )
        out_cg_1 = prop_cg.propose_vllm_step(
            SimpleNamespace(input_batch=ib_dec), so_dec, samp_dec, [aux_dec]
        )
        out_eg_1 = prop_eager.propose_vllm_step(
            SimpleNamespace(input_batch=ib_dec), so_dec, samp_dec, [aux_dec]
        )
        assert out_cg_1 == out_eg_1

    # 3. Verify mixed-temperature stochastic CUDA-Graph replay advances RNG cleanly
    rids_mix = ["mix-greedy", "mix-stoch"]
    tok_mix = rng.integers(1, 40, size=(2, 32), dtype=np.int32)
    ib_mix0 = SimpleNamespace(
        req_ids=rids_mix,
        num_tokens_no_spec=np.array([9, 9], dtype=np.int32),
        token_ids_cpu=tok_mix,
        temperature_cpu=np.array([0.0, 2.5], dtype=np.float32),
        greedy_reqs={"mix-greedy"},
        random_reqs={"mix-stoch"},
    )
    so_mix0 = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"mix-greedy": 8, "mix-stoch": 8},
    )
    p_mix = prop_cg.propose_vllm_step(
        SimpleNamespace(input_batch=ib_mix0),
        so_mix0,
        [[int(tok_mix[0, 8])], [int(tok_mix[1, 8])]],
        [torch.randn((16, d_in), device=dev, dtype=torch.bfloat16)],
    )
    assert len(p_mix[0]) >= 1
    assert len(p_mix[1]) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_vectorized_kraus_tree_cudagraph_and_vllm_spine_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = torch.device("cuda")
    model = _build_v3_2d_osc_drafter(device=dev, dtype=torch.bfloat16, block_size=5)
    proposer = KvNDSparkVllmProposer(
        model,
        max_num_seqs=8,
        device=dev,
        use_kraus_tree=True,
        tree_budget=10,
        branch_width=3,
    )

    for bsz in (1, 4):
        fc_blk = torch.randn(
            (bsz, 2, model.hidden_size), device=dev, dtype=torch.bfloat16
        )
        tok_blk = torch.randint(1, 40, (bsz, 2), device=dev, dtype=torch.long)
        vlen = torch.tensor([2] * bsz, device=dev, dtype=torch.long)
        prev2 = torch.randint(1, 40, (bsz, 2), device=dev, dtype=torch.long)
        sc_ref = model.allocate_state_cache(bsz, dev)
        sc_cg = sc_ref.clone()

        tree_ref, upd_ref = model.draft_kraus_tree_step(
            fc_blk,
            tok_blk,
            vlen,
            sc_ref,
            tree_budget=10,
            branch_width=3,
            temperature=0.0,
            return_q_probs=False,
            prev_context_ids=prev2,
        )
        tree_cg, upd_cg = proposer.propose_tree_cudagraph(
            fc_blk,
            tok_blk,
            vlen,
            sc_cg,
            tree_budget=10,
            branch_width=3,
            temperature=0.0,
            prev_context_ids=prev2,
        )
        assert float(tree_cg["povm_completeness_max_err"].item()) < 1e-5
        assert torch.equal(tree_ref["tree_draft_ids"], tree_cg["tree_draft_ids"])
        assert torch.equal(tree_ref["tree_parent_idx"], tree_cg["tree_parent_idx"])
        assert torch.equal(tree_ref["tree_depths"], tree_cg["tree_depths"])
        assert (
            torch.max(
                torch.abs(tree_ref["tree_survival"] - tree_cg["tree_survival"])
            ).item()
            < 1e-5
        )
        assert torch.max(torch.abs(upd_ref - upd_cg)).item() < 1e-5

    # Also test KVN_VLLM_USE_KRAUS_TREE=1 in propose_vllm_step
    monkeypatch.setenv("KVN_VLLM_USE_KRAUS_TREE", "1")
    prop_tree_vllm = KvNDSparkVllmProposer(model, max_num_seqs=8, device=dev)
    assert prop_tree_vllm.use_kraus_tree is True
    rng = np.random.default_rng(777)
    tok_arr = rng.integers(1, 40, size=(2, 16), dtype=np.int32)
    ib = SimpleNamespace(
        req_ids=["kt-0", "kt-1"],
        num_tokens_no_spec=np.array([9, 9], dtype=np.int32),
        token_ids_cpu=tok_arr,
        greedy_reqs={"kt-0", "kt-1"},
        random_reqs=set(),
    )
    so = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"kt-0": 8, "kt-1": 8},
    )
    d_in = 3 * model.hidden_size
    out_spine = prop_tree_vllm.propose_vllm_step(
        SimpleNamespace(input_batch=ib),
        so,
        [[int(tok_arr[0, 8])], [int(tok_arr[1, 8])]],
        [torch.randn((16, d_in), device=dev, dtype=torch.bfloat16)],
    )
    assert len(out_spine) == 2
    assert all(len(r) >= 1 for r in out_spine)

    # Verify pre-allocated max_num_tokens aux_hidden_states slicing + FULL CUDAGraph
    # uniform_decode_query_len preservation vs high-batch Amdahl throttling.
    padded_aux = torch.randn((2048, d_in), device=dev, dtype=torch.bfloat16)
    so_padded = SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"kt-0": 5, "kt-1": 5},
        total_num_scheduled_tokens=10,
    )
    ib_dec = SimpleNamespace(
        req_ids=["kt-0", "kt-1"],
        num_tokens_no_spec=np.array([10, 10], dtype=np.int32),
        token_ids_cpu=tok_arr,
        greedy_reqs={"kt-0", "kt-1"},
        random_reqs=set(),
    )
    runner_full_cg = SimpleNamespace(
        input_batch=ib_dec,
        uniform_decode_query_len=5,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=512),
    )
    out_full_cg = prop_tree_vllm.propose_vllm_step(
        runner_full_cg,
        so_padded,
        [[int(tok_arr[0, 9])], [int(tok_arr[1, 9])]],
        [padded_aux],
    )
    assert [len(r) for r in out_full_cg] == [4, 4]

    ib_dec2 = SimpleNamespace(
        req_ids=["kt-0", "kt-1"],
        num_tokens_no_spec=np.array([11, 11], dtype=np.int32),
        token_ids_cpu=tok_arr,
        greedy_reqs={"kt-0", "kt-1"},
        random_reqs=set(),
    )
    runner_over_cg = SimpleNamespace(
        input_batch=ib_dec2,
        uniform_decode_query_len=5,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4),
    )
    out_over_cg = prop_tree_vllm.propose_vllm_step(
        runner_over_cg,
        so_padded,
        [[int(tok_arr[0, 10])], [int(tok_arr[1, 10])]],
        [padded_aux],
    )
    assert len(out_over_cg) == 2
    assert all(1 <= len(r) <= 4 for r in out_over_cg)


def _make_async_runner_fixture(
    device: str,
    req_ids: list[str],
    prompt_toks: list[list[int]],
    num_prompt: list[int],
    num_comp: list[int],
    num_no_spec: list[int],
    discard_flags: list[bool],
) -> SimpleNamespace:
    max_col = max(
        max(len(r), ns + 4)
        for r, ns in zip(prompt_toks, num_no_spec, strict=True)
    )
    arr = np.full((len(req_ids), max_col), -1, dtype=np.int32)
    for i, r in enumerate(prompt_toks):
        arr[i, : len(r)] = r
    disc_np = np.array(discard_flags, dtype=bool)
    disc_gpu = torch.tensor(discard_flags, dtype=torch.bool, device=device)
    requests_dict = {
        rid: SimpleNamespace(
            get_token_id=lambda pos, r=row: int(r[min(pos, len(r) - 1)])
        )
        for rid, row in zip(req_ids, prompt_toks, strict=True)
    }
    ib = SimpleNamespace(
        req_ids=req_ids,
        vocab_size=96,
        num_prompt_tokens=np.array(num_prompt, dtype=np.int32),
        num_computed_tokens_cpu=np.array(num_comp, dtype=np.int32),
        num_tokens_no_spec=np.array(num_no_spec, dtype=np.int32),
        token_ids_cpu=arr,
        greedy_reqs=set(req_ids),
        random_reqs=set(),
    )
    return SimpleNamespace(
        input_batch=ib,
        discard_request_mask=SimpleNamespace(np=disc_np, gpu=disc_gpu),
        requests=requests_dict,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_async_gpu_step_parity_across_chunked_apc_rejected_and_preemption(
    device: str,
) -> None:
    model = _build_v3_2d_osc_drafter(device=device, dtype=torch.float32, block_size=5)
    roofline = HardwareRooflineProfile(
        name="unit-flat",
        peak_bf16_tflops=989.0,
        hbm_bandwidth_tbs=4.8,
        ridge_tokens=4096,
        base_verify_ms=10.0,
        base_draft_chain_ms=0.0,
        draft_per_req_slope_ms=0.0,
    )
    scheduler = PurityLoadAwareAmdahlScheduler(
        roofline=roofline, max_horizon=4, min_survival_floor=0.0
    )
    proposer = KvNDSparkVllmProposer(
        model,
        max_num_seqs=16,
        device=device,
        enable_apc=True,
        apc_block_size=16,
        scheduler=scheduler,
    )
    assert proposer.slot_pool is not None
    assert proposer.apc_cache is not None

    d_in = 3 * model.hidden_size
    torch.manual_seed(1234)
    master_toks = torch.randint(1, 45, (60,), dtype=torch.long).tolist()
    master_aux = torch.randn((60, d_in), dtype=torch.float32, device=device)
    master_fc = proposer.extract_fc_hidden(master_aux, already_projected=False)

    # Step 0: req-A unchunked prefill (P=36, bonus y_36); req-B chunk-0 discarded (20)
    r_s0 = _make_async_runner_fixture(
        device,
        ["req-A", "req-B"],
        [master_toks[:36], master_toks[:36]],
        num_prompt=[36, 36],
        num_comp=[0, 0],
        num_no_spec=[36, 20],
        discard_flags=[False, True],
    )
    samp_s0 = torch.tensor(
        [[master_toks[36], -1, -1, -1, -1], [42, -1, -1, -1, -1]],
        dtype=torch.long,
        device=device,
    )
    next_s0, vcnt_s0 = proposer.prepare_async_sampled_tokens(r_s0, samp_s0)
    assert vcnt_s0.tolist() == [1, 0]
    assert next_s0.tolist() == [master_toks[36], master_toks[19]]

    out_s0 = proposer.propose_vllm_async_gpu_step(
        r_s0,
        SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens={"req-A": 36, "req-B": 20},
            total_num_scheduled_tokens=56,
        ),
        samp_s0,
        next_s0,
        vcnt_s0,
        [torch.cat([master_aux[:36], master_aux[:20]], dim=0)],
    )
    _, _, _, draft_gt37 = _compute_ground_truth_state_and_draft(
        model, master_toks[:37], master_fc[:36]
    )
    assert torch.equal(out_s0[0], draft_gt37)
    assert torch.equal(out_s0[1], torch.zeros_like(draft_gt37))

    # Step 1: req-A rejects all 4 drafts (n_sched=5, vlen=1); req-B finishes chunk-1;
    #         req-C restores 32 prefix tokens via ZeroReplayAPCStateCache!
    garbage_aux = torch.randn((4, d_in), dtype=torch.float32, device=device) * 40.0
    aux_s1 = torch.cat(
        [master_aux[36:37], garbage_aux, master_aux[20:36], master_aux[32:36]], dim=0
    )
    r_s1 = _make_async_runner_fixture(
        device,
        ["req-A", "req-B", "req-C"],
        [master_toks[:36], master_toks[:36], master_toks[:36]],
        num_prompt=[36, 36, 36],
        num_comp=[36, 20, 32],
        num_no_spec=[37, 36, 36],
        discard_flags=[False, False, False],
    )
    samp_s1 = torch.tensor(
        [
            [master_toks[37], -1, -1, -1, -1],
            [master_toks[36], -1, -1, -1, -1],
            [master_toks[36], -1, -1, -1, -1],
        ],
        dtype=torch.long,
        device=device,
    )
    next_s1, vcnt_s1 = proposer.prepare_async_sampled_tokens(r_s1, samp_s1)
    out_s1 = proposer.propose_vllm_async_gpu_step(
        r_s1,
        SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens={"req-A": 5, "req-B": 16, "req-C": 4},
            total_num_scheduled_tokens=25,
        ),
        samp_s1,
        next_s1,
        vcnt_s1,
        [aux_s1],
    )
    assert proposer.apc_cache.saved_prefill_tokens >= 32
    _, _, _, draft_gt38 = _compute_ground_truth_state_and_draft(
        model, master_toks[:38], master_fc[:37]
    )
    assert torch.equal(out_s1[0], draft_gt38)
    assert torch.equal(out_s1[1], draft_gt37)
    assert torch.equal(out_s1[2], draft_gt37)

    # Step 2: req-A accepts 2 of 4 drafts (vlen=3 -> L=41); req-B & req-C preempted
    aux_s2 = torch.cat(
        [master_aux[37:40], torch.randn((2, d_in), device=device) * 30.0], dim=0
    )
    r_s2 = _make_async_runner_fixture(
        device,
        ["req-A"],
        [master_toks[:36]],
        num_prompt=[36],
        num_comp=[37],
        num_no_spec=[38],
        discard_flags=[False],
    )
    samp_s2 = torch.tensor(
        [[master_toks[38], master_toks[39], master_toks[40], -1, -1]],
        dtype=torch.long,
        device=device,
    )
    next_s2, vcnt_s2 = proposer.prepare_async_sampled_tokens(r_s2, samp_s2)
    out_s2 = proposer.propose_vllm_async_gpu_step(
        r_s2,
        SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens={"req-A": 5},
            total_num_scheduled_tokens=5,
        ),
        samp_s2,
        next_s2,
        vcnt_s2,
        [aux_s2],
    )
    _, _, _, draft_gt41 = _compute_ground_truth_state_and_draft(
        model, master_toks[:41], master_fc[:40]
    )
    assert torch.equal(out_s2[0], draft_gt41)

    # Step 3: req-A accepts all 4 drafts (vlen=5 -> L=46); req-B RECOMPUTES (0..36);
    #         req-C swap-resumes (num_comp=36, n_sched=1)
    aux_s3 = torch.cat(
        [master_aux[40:45], master_aux[:37], master_aux[36:37]], dim=0
    )
    r_s3 = _make_async_runner_fixture(
        device,
        ["req-A", "req-B", "req-C"],
        [master_toks[:36], master_toks[:37], master_toks[:37]],
        num_prompt=[36, 36, 36],
        num_comp=[40, 0, 36],
        num_no_spec=[41, 37, 37],
        discard_flags=[False, False, False],
    )
    samp_s3 = torch.tensor(
        [
            master_toks[41:46],
            [master_toks[37], -1, -1, -1, -1],
            [master_toks[37], -1, -1, -1, -1],
        ],
        dtype=torch.long,
        device=device,
    )
    next_s3, vcnt_s3 = proposer.prepare_async_sampled_tokens(r_s3, samp_s3)
    out_s3 = proposer.propose_vllm_async_gpu_step(
        r_s3,
        SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens={"req-A": 5, "req-B": 37, "req-C": 1},
            total_num_scheduled_tokens=43,
        ),
        samp_s3,
        next_s3,
        vcnt_s3,
        [aux_s3],
    )
    q_gt46, _, _, draft_gt46 = _compute_ground_truth_state_and_draft(
        model, master_toks[:46], master_fc[:45]
    )
    slot_a = proposer.slot_pool.get_slot("req-A")
    q_diff_a = torch.max(torch.abs(proposer.slot_pool.q_pool[slot_a] - q_gt46)).item()
    assert q_diff_a < 1e-5
    assert torch.equal(out_s3[0], draft_gt46)
    assert torch.equal(out_s3[1], draft_gt38)
    assert torch.equal(out_s3[2], draft_gt38)


def test_plugin_vllm_config_and_async_runner_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_v3_2d_osc_drafter(device="cpu", dtype=torch.float32, block_size=5)
    proposer = KvNDSparkVllmProposer(model, max_num_seqs=8, device="cpu")
    d_in = 3 * model.hidden_size

    class _DummyVllmConfig:
        def __init__(self, spec_cfg: object, async_sched: bool | None) -> None:
            self.speculative_config = spec_cfg
            self.scheduler_config = SimpleNamespace(async_scheduling=async_sched)
            self.parallel_config = SimpleNamespace(
                pipeline_parallel_size=1,
                disable_nccl_for_dp_synchronization=False,
            )
            self.model_config = SimpleNamespace(runner_type="generate")
            self.__post_init__()

        def __post_init__(self) -> None:
            if (
                self.speculative_config is not None
                and getattr(self.speculative_config, "method", None) == "custom_class"
            ):
                if self.scheduler_config.async_scheduling is True:
                    raise ValueError("custom_class blocks async_scheduling")
                if self.scheduler_config.async_scheduling is None:
                    self.scheduler_config.async_scheduling = False

    _patch_vllm_config_cls(_DummyVllmConfig)
    kvn_spec = SimpleNamespace(
        method="custom_class",
        model=PROPOSER_IMPORT_PATH,
        disable_padded_drafter_batch=False,
    )
    cfg_auto = _DummyVllmConfig(kvn_spec, None)
    assert cfg_auto.scheduler_config.async_scheduling is True
    assert cfg_auto.parallel_config.disable_nccl_for_dp_synchronization is True
    cfg_engine_core = _DummyVllmConfig(kvn_spec, True)
    assert cfg_engine_core.scheduler_config.async_scheduling is True
    cfg_explicit_off = _DummyVllmConfig(kvn_spec, False)
    assert cfg_explicit_off.scheduler_config.async_scheduling is False

    monkeypatch.setenv("KVN_VLLM_DISABLE_ASYNC_SCHED", "1")
    cfg_env_off = _DummyVllmConfig(kvn_spec, None)
    assert cfg_env_off.scheduler_config.async_scheduling is False
    monkeypatch.delenv("KVN_VLLM_DISABLE_ASYNC_SCHED", raising=False)

    class _DummyRunner:
        def __init__(self, drafter_obj: object, ib_obj: object) -> None:
            self.drafter = drafter_obj
            self.input_batch = ib_obj
            self.use_async_scheduling = True
            self.speculative_config = kvn_spec
            self.valid_sampled_token_count_gpu: torch.Tensor | None = None
            self._draft_token_req_ids: list[str] | None = None
            self.copied_counts: tuple[torch.Tensor, torch.Tensor] | None = None

        def _copy_valid_sampled_token_count(
            self, next_tok: torch.Tensor, v_cnt: torch.Tensor
        ) -> None:
            self.copied_counts = (next_tok, v_cnt)
            self.valid_sampled_token_count_gpu = v_cnt
            self.input_batch.prev_sampled_token_ids = next_tok.unsqueeze(1)

        def _bookkeeping_sync(
            self, scheduler_output: object, sampler_output: object, *args: object
        ) -> str:
            del scheduler_output, sampler_output, args
            assert self.copied_counts is not None
            return "ok"

        def propose_draft_token_ids(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("Should dispatch to KvN async GPU proposer")

    _patch_runner_bookkeeping_hook(_DummyRunner)
    _patch_runner_propose_hook(_DummyRunner)
    r_fix = _make_async_runner_fixture(
        "cpu",
        ["req-A"],
        [[5, 6, 7, 8]],
        num_prompt=[4],
        num_comp=[0],
        num_no_spec=[4],
        discard_flags=[False],
    )
    dummy_runner = _DummyRunner(proposer, r_fix.input_batch)
    dummy_runner.discard_request_mask = r_fix.discard_request_mask
    dummy_runner.requests = r_fix.requests
    samp_gpu = torch.tensor([[9, -1, -1, -1, -1]], dtype=torch.long)
    assert (
        dummy_runner._bookkeeping_sync(
            SimpleNamespace(), SimpleNamespace(sampled_token_ids=samp_gpu)
        )
        == "ok"
    )
    hook_out = dummy_runner.propose_draft_token_ids(
        SimpleNamespace(
            finished_req_ids=set(),
            num_scheduled_tokens={"req-A": 4},
            total_num_scheduled_tokens=4,
        ),
        [],
        None,
        None,
        None,
        [torch.randn((4, d_in), dtype=torch.float32)],
        None,
        None,
    )
    assert isinstance(hook_out, torch.Tensor)
    assert hook_out.shape == (1, 4)
    assert dummy_runner._draft_token_req_ids == ["req-A"]
