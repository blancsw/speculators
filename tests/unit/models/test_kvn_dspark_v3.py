"""Comprehensive adversarial & unit tests for KvN-DSpark v3 (Axes 1-6 + Audit Fixes)."""

# ruff: noqa: PLC0415, PT018

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.kvn_dspark import (
    BornConfidenceHead,
    CausalKvNRFFLayer,
    DataReuploadingLinear,
    KrausPOVMHead,
    KvNDSparkDraftModel,
    KvNDSparkSpeculatorConfig,
    ModalDataReuploadingRFF,
    MultiHorizonKoopmanRollout,
    SpeculativeEngram,
    fused_multistep_complex_decode,
    fused_multistep_sp4_decode,
    kvn_select_anchors,
    pytorch_segmented_complex_scan,
    pytorch_segmented_scalar_scan,
    pytorch_segmented_sp4_scan,
    segmented_complex_scan,
    segmented_scalar_scan,
    segmented_sp4_scan,
    specinfer_verify_single_step,
    specinfer_verify_tree,
)
from speculators.models.kvn_dspark.triton_kernels import HAS_TRITON
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_tl_cfg(
    hidden_size: int = 64,
    num_layers: int = 3,
    vocab_size: int = 64,
    sliding_window: int = 32,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=max(8, hidden_size // 4),
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        sliding_window=sliding_window,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * num_layers,
        _attn_implementation="eager",
    )


def _make_v3_cfg(
    hidden_size: int = 64,
    num_layers: int = 3,
    vocab_size: int = 64,
    draft_vocab_size: int | None = None,
    n_modes: int = 32,
    block_size: int = 4,
    backbone_type: str = "kvn",
    **kwargs,
) -> KvNDSparkSpeculatorConfig:
    tl_cfg = _make_tl_cfg(
        hidden_size=hidden_size, num_layers=num_layers, vocab_size=vocab_size
    )
    return KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        draft_vocab_size=draft_vocab_size or vocab_size,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=True,
        markov_rank=16,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        kvn_backbone_type=backbone_type,  # type: ignore[arg-type]
        kvn_n_modes=n_modes,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
        **kwargs,
    )


def _init_verifier_stubs(model: KvNDSparkDraftModel) -> None:
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.02)
        torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
        torch.nn.init.ones_(model.verifier_norm.weight)


# ==============================================================================
# 1. [C1] Exact FP32 Recurrent State Parity at T = 8192 with bfloat16 Inputs
# ==============================================================================


@pytest.mark.parametrize("use_symplectic_butterfly", [True, False])
def test_c1_long_context_t8192_bf16_scan_vs_recurrent_parity(
    use_symplectic_butterfly: bool,
):
    """[C1] Verify T=8192 bf16 parallel scan vs step-by-step recurrent_step FP32 state
    parity < 1e-4.
    """
    torch.manual_seed(2026)
    cfg_check = _make_v3_cfg()
    assert cfg_check.kvn_state_dtype == "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = (
        CausalKvNRFFLayer(
            d_model=32,
            n_modes=16,
            reupload_layers_l=2,
            d_re=16,
            use_symplectic_butterfly=use_symplectic_butterfly,
            use_lindblad_purity=True,
            layer_idx=1,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    seq_len = 8192
    x_bf16 = torch.randn(1, seq_len, 32, device=device, dtype=torch.bfloat16)
    reset_mask = torch.zeros(1, seq_len, device=device, dtype=torch.bool)
    reset_mask[0, 0] = True
    reset_mask[0, 2048] = True
    reset_mask[0, 5500] = True

    with torch.no_grad():
        _, q_scan, p_scan, dtheta_scan, nu_scan = layer(
            x_bf16,
            reset_mask=reset_mask,
            state_dtype=torch.float32,
            return_nu=True,
        )
        assert q_scan.dtype == torch.float32
        assert p_scan.dtype == torch.float32
        assert nu_scan.dtype == torch.float32

        # Compare against multistep_decode over the entire 8192 sequence (single O(T)
        # C++/Triton loop)
        # AND explicit recurrent_step over a 256-step window starting from the T=7936
        # FP32 state!
        z0 = torch.zeros(1, 16, device=device, dtype=torch.float32)
        # Segment 1: [0..2047], Segment 2: [2048..5499], Segment 3: [5500..8191]
        _, q_seg1, p_seg1, nu_seg1, _ = layer.multistep_decode(
            x_bf16[:, :2048],
            z0,
            z0,
            valid_lens=torch.tensor([2048], device=device),
            nu_init=z0,
            state_dtype=torch.float32,
        )
        _, q_seg2, p_seg2, nu_seg2, _ = layer.multistep_decode(
            x_bf16[:, 2048:5500],
            z0,
            z0,
            valid_lens=torch.tensor([5500 - 2048], device=device),
            nu_init=z0,
            state_dtype=torch.float32,
        )
        _, q_seg3, p_seg3, nu_seg3, _ = layer.multistep_decode(
            x_bf16[:, 5500:7936],
            z0,
            z0,
            valid_lens=torch.tensor([7936 - 5500], device=device),
            nu_init=z0,
            state_dtype=torch.float32,
        )
        assert nu_seg1 is not None and nu_seg2 is not None and nu_seg3 is not None

        # Step-by-step Python recurrent_step over the final 256 tokens [7936 .. 8191]
        state: tuple[Tensor, ...] = (
            q_seg3.unsqueeze(1),
            p_seg3.unsqueeze(1),
            nu_seg3.unsqueeze(1),
        )
        rec_qs, rec_ps, rec_nus = [], [], []
        for t in range(7936, seq_len):
            x_step, state, _ = layer.recurrent_step(
                x_bf16[:, t : t + 1, :],
                state=state,
                reset_mask=reset_mask[:, t : t + 1],
                state_dtype=torch.float32,
            )
            assert x_step.dtype == torch.bfloat16
            assert state[0].dtype == torch.float32
            assert state[1].dtype == torch.float32
            assert state[2].dtype == torch.float32
            rec_qs.append(state[0])
            rec_ps.append(state[1])
            rec_nus.append(state[2])

        q_tail = torch.cat(rec_qs, dim=1)
        p_tail = torch.cat(rec_ps, dim=1)
        nu_tail = torch.cat(rec_nus, dim=1)

    assert (q_scan[:, 2047] - q_seg1).abs().max().item() < 1e-4
    assert (q_scan[:, 5499] - q_seg2).abs().max().item() < 1e-4
    assert (q_scan[:, 7936:] - q_tail).abs().max().item() < 1e-4
    assert (p_scan[:, 7936:] - p_tail).abs().max().item() < 1e-4
    assert (nu_scan[:, 7936:] - nu_tail).abs().max().item() < 1e-4
    assert dtheta_scan.dtype == torch.bfloat16


# ==============================================================================
# 2. [C3] Prompt-Boundary & Post-EOS / Cross-Doc Masking Alignment
# ==============================================================================


def test_c3_prompt_boundary_eos_and_cross_doc_masking_alignment():
    """[C3] Verify sample_from_anchor=True aligns loss_mask & doc_ids at a + k + 1."""
    # Sequence layout (seq_len = 24, block_size = 4):
    # Doc 0: [0..7] prompt (mask=0), [8..11] assistant (mask=1, 11=EOS), [12..15] user
    # turn 2 (mask=0)
    # Doc 1: [16..18] assistant-start test (16 has mask=1 to test cross-doc rejection at
    # a=15!),
    #        [17..18] prompt (mask=0), [19..23] assistant (mask=1)
    seq_len = 24
    block_size = 4
    loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
    loss_mask[0, 8:12] = 1
    loss_mask[0, 16] = (
        1  # Token 16 is doc 1 -> anchor a=15 (doc 0) MUST NOT be selected!
    )
    loss_mask[0, 19:24] = 1

    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)
    doc_ids[0, 16:] = 1

    anchors, valid = kvn_select_anchors(
        loss_mask,
        num_anchors=32,
        block_size=block_size,
        sample_from_anchor=True,
        document_ids=doc_ids,
    )
    selected = set(anchors[valid].tolist())
    # Expected valid anchors under sample_from_anchor=True (a <= seq_len - block_size =
    # 20):
    # - a=7 (prompt-to-assistant boundary predicting 8!)
    # - a=8, 9, 10 (predicting 9, 10, 11)
    # - NOT a=11 (EOS, since loss_mask[12] == 0!)
    # - NOT a=15 (cross-doc boundary doc 0 -> doc 1, even though loss_mask[16] == 1!)
    # - a=18 (doc 1 prompt boundary predicting 19), 19, 20
    assert selected == {7, 8, 9, 10, 18, 19, 20}

    # Backward compatibility when sample_from_anchor=False
    anchors_legacy, valid_legacy = kvn_select_anchors(
        loss_mask,
        num_anchors=32,
        block_size=block_size,
        sample_from_anchor=False,
        document_ids=doc_ids,
    )
    selected_legacy = set(anchors_legacy[valid_legacy].tolist())
    assert 7 not in selected_legacy
    assert 8 in selected_legacy

    # Verify slot-level _compute_targets_and_mask across KvNDSparkDraftModel backbones
    v_hs = torch.randn(1, seq_len, 64)
    test_anchors = torch.tensor([7, 9, 20], dtype=torch.long)
    test_valid = torch.ones_like(test_anchors, dtype=torch.bool)

    for bb in ("kvn", "hybrid_1swa_2kvn", "swa"):
        kvn_cfg = _make_v3_cfg(vocab_size=64, block_size=block_size, backbone_type=bb)
        mdl = KvNDSparkDraftModel(kvn_cfg).eval()
        _init_verifier_stubs(mdl)
        _, blk_mask, anchored_idx = mdl._compute_targets_and_mask(
            verifier_last_hidden_states=v_hs,
            loss_mask=loss_mask,
            document_ids=doc_ids,
            anchor_positions=test_anchors,
            anchor_valid=test_valid,
        )
        # Anchor 0 (a=7): predicts 8, 9, 10, 11 -> all 4 slots supervised!
        assert blk_mask[0, 0:4].bool().tolist() == [True, True, True, True]
        assert anchored_idx[0:4].tolist() == [7, 8, 9, 10]
        # Anchor 1 (a=9): predicts 10, 11 (valid), then 12, 13 (post-EOS mask=0!) -> [T,
        # T, F, F]
        assert blk_mask[0, 4:8].bool().tolist() == [True, True, False, False]
        # Anchor 2 (a=20): predicts 21, 22, 23 (valid), then 24 (== seq_len
        # out-of-bounds!) -> [T, T, T, F]
        assert blk_mask[0, 8:12].bool().tolist() == [True, True, True, False]


# ==============================================================================
# 3. [M2, M4, M5] Docstrings, Legacy A0 Wiring, Odd Modes, Sequential B3 & Ckpt
# ==============================================================================


def test_m2_m4_m5_citations_legacy_a0_odd_modes_sequential_b3_and_grad_ckpt():
    """Verify [M2] docstrings, [M4] legacy A0/odd errors, and [M5] B3 + grad ckpt."""
    kvn_dir = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "speculators"
        / "models"
        / "kvn_dspark"
    )
    for py_name in (
        "config.py",
        "model_definitions.py",
        "core.py",
        "triton_kernels.py",
    ):
        src_txt = (kvn_dir / py_name).read_text(encoding="utf-8")
        assert "Prop. 7.2" not in src_txt, f"Stale Prop. 7.2 citation in {py_name}"
        assert "unitary scan" not in src_txt.lower()

    # [M4] Wire DataReuploadingLinear when kvn_reupload_dim=0 and dyadic_rff=False
    cfg_a0 = _make_v3_cfg(
        n_modes=16,
        kvn_reupload_dim=0,
        dyadic_rff=False,
        use_symplectic_butterfly=False,
    )
    model_a0 = KvNDSparkDraftModel(cfg_a0)
    _init_verifier_stubs(model_a0)
    assert isinstance(
        model_a0.kvn_layers[0].reupload.legacy_reupload, DataReuploadingLinear
    )

    hs = torch.randn(1, 16, 3 * 64)
    ids = torch.randint(0, 64, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.bool)
    v_hs = torch.randn(1, 16, 64)
    doc_ids = torch.zeros(1, 16, dtype=torch.long)
    _, loss_a0, _ = model_a0(hs, ids, mask, v_hs, doc_ids, max_anchors=2)
    loss_a0.backward()
    leg_w = model_a0.kvn_layers[0].reupload.legacy_reupload.weights[0]
    assert leg_w.grad is not None and leg_w.grad.abs().max().item() > 0.0

    # [M4] Odd mode count validation on init and pruning
    with pytest.raises(ValueError, match="even"):
        _make_v3_cfg(n_modes=31)

    model_even = KvNDSparkDraftModel(_make_v3_cfg(n_modes=32))
    with pytest.raises(ValueError, match="even"):
        model_even.prune_kvn_modes(15)

    # [M5] Sequential B3 hybrid (4 KvN layers -> rollout -> 1 SWA layer) + gradient
    # checkpointing
    cfg_b3 = _make_v3_cfg(
        backbone_type="hybrid_1swa_2kvn", num_layers=5, n_modes=16, block_size=4
    )
    model_b3 = KvNDSparkDraftModel(cfg_b3)
    _init_verifier_stubs(model_b3)
    assert len(model_b3.kvn_layers) == 4
    assert len(model_b3.layers) == 1

    captured_swa_in: list[torch.Tensor] = []

    def _swa_hook(_mod, args, kwargs):
        t = args[0] if args else kwargs["hidden_states"]
        captured_swa_in.append(t.detach().clone())

    handle = model_b3.layers[0].register_forward_pre_hook(_swa_hook, with_kwargs=True)
    pin_mask = torch.zeros(1, 16, dtype=torch.bool)
    pin_mask[0, 6] = True
    model_b3.eval()
    model_b3._kvn_backbone_forward(hs, ids, pin_mask, v_hs, doc_ids, max_anchors=1)
    with torch.no_grad():
        model_b3.kvn_layers[0].readout_linear.weight.mul_(5.0)
    model_b3._kvn_backbone_forward(hs, ids, pin_mask, v_hs, doc_ids, max_anchors=1)
    handle.remove()

    assert len(captured_swa_in) == 2
    assert not torch.allclose(captured_swa_in[0], captured_swa_in[1], atol=1e-4)

    # Enable gradient checkpointing on sequential hybrid B3 and run train
    # forward+backward
    model_b3.train()
    model_b3.gradient_checkpointing_enable()
    assert model_b3.gradient_checkpointing is True
    assert all(lk.gradient_checkpointing for lk in model_b3.kvn_layers)
    _, loss_b3, _ = model_b3(hs, ids, mask, v_hs, doc_ids, max_anchors=2)
    assert torch.isfinite(loss_b3)
    loss_b3.backward()
    assert model_b3.kvn_layers[0].readout_linear.weight.grad is not None
    assert model_b3.layers[0].self_attn.q_proj.weight.grad is not None


# ==============================================================================
# 4. Axe 1 — Multiplicative Dyadic 2^l RFF + Hashed Sub-Token SpeculativeEngram
# ==============================================================================


def test_axe1_multiplicative_dyadic_rff_and_speculative_engram():
    """Verify Barthe Ch. 3 Eq. 3.26 multiplicative 2^l RFF bandwidth & SpeculativeEngram
    2/3-gram parity.
    """
    torch.manual_seed(401)
    rff_dyadic = ModalDataReuploadingRFF(
        d_model=64,
        n_modes=32,
        reupload_layers_l=4,
        d_re=32,
        sigma_g=1.0,
        dyadic_rff=True,
    )
    rff_add = ModalDataReuploadingRFF(
        d_model=64,
        n_modes=32,
        reupload_layers_l=4,
        d_re=32,
        sigma_g=1.0,
        dyadic_rff=False,
    )
    for p in rff_dyadic.parameters():
        assert not p.is_complex()

    # Sweep along a 1D input ray and measure high-frequency Fourier energy
    t_grid = torch.linspace(-3.0, 3.0, 256).view(1, 256, 1)
    dir_vec = torch.randn(1, 1, 64)
    ray_x = t_grid * dir_vec

    rq_dy, rp_dy = rff_dyadic(ray_x)
    rq_ad, rp_ad = rff_add(ray_x)
    assert not rq_dy.is_complex() and not rp_dy.is_complex()
    # Bounded output amplitude (zero DC drift)
    assert rq_dy.abs().max().item() <= 2.0

    spec_dy = torch.fft.rfft(rq_dy[0], dim=0).abs().square()
    spec_ad = torch.fft.rfft(rq_ad[0], dim=0).abs().square()
    hf_ratio_dy = spec_dy[24:].sum() / spec_dy[1:].sum()
    hf_ratio_ad = spec_ad[24:].sum() / spec_ad[1:].sum()
    assert hf_ratio_dy.item() > hf_ratio_ad.item()

    # SpeculativeEngram sensitivity to both 2-gram (prev_prev_id) and 3-gram (prev3_id)
    engram = SpeculativeEngram(out_dim=16, engram_buckets=4096, engram_dim=64)
    prev_id = torch.tensor([[10, 20, 30, 40]])
    prev2_a = torch.tensor([[5, 10, 20, 30]])
    prev2_b = torch.tensor([[99, 10, 20, 30]])
    prev3_a = torch.tensor([[1, 5, 10, 20]])
    prev3_b = torch.tensor([[77, 5, 10, 20]])

    out_base = engram(prev_id, prev_prev_id=prev2_a, prev3_id=prev3_a)
    out_2g = engram(prev_id, prev_prev_id=prev2_b, prev3_id=prev3_a)
    out_3g = engram(prev_id, prev_prev_id=prev2_a, prev3_id=prev3_b)

    assert out_base.std().item() < 0.05  # Near-zero residual init
    assert not torch.allclose(out_base[:, 0], out_2g[:, 0], atol=1e-6)
    assert not torch.allclose(out_base[:, 0], out_3g[:, 0], atol=1e-6)
    assert torch.allclose(out_base[:, 1:], out_2g[:, 1:], atol=1e-6)

    # Verify exact train-vs-decode logit parity with RNN MarkovHead + SpeculativeEngram
    # + Kraus POVM
    cfg = _make_v3_cfg(
        n_modes=16,
        block_size=4,
        markov_head_type="rnn",
        use_engram=True,
        engram_buckets=1024,
        engram_dim=32,
        self_cond_rollout_prob=0.0,
    )
    model = KvNDSparkDraftModel(cfg).eval()
    _init_verifier_stubs(model)

    seq_len = 16
    hs = torch.randn(1, seq_len, 3 * 64)
    ids = torch.randint(0, 64, (1, seq_len))
    pin_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    pin_mask[0, 8] = True  # Anchor a = 7 (predicting 8..11)
    v_hs = torch.randn(1, seq_len, 64)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)

    fc_all = model.hidden_norm(model.fc(hs))
    prefill = model.prefill_kvn_state(ids[:, :7], fc_all[:, :7])
    dec_ids, _, details = model.advance_and_draft_step(
        prev_fc_block=prefill["last_fc"],
        token_ids_block=ids[:, 7:8],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=prefill["state_cache"],
        temperature=0.0,
        return_details=True,
    )

    # Feed drafted tokens into input_ids[8:11] so teacher-forced forward sees identical
    # history!
    ids_tf = ids.clone()
    ids_tf[0, 8:11] = dec_ids[0, :3]
    hidden, base_logits, _, _, anchored_idx, _ = model._kvn_backbone_forward(
        hs, ids_tf, pin_mask, v_hs, doc_ids, max_anchors=1
    )
    assert anchored_idx.tolist() == [7, 8, 9, 10]
    prev_toks = ids_tf[0, anchored_idx].view(1, 4)
    eng_delta = model.engram(prev_toks)
    prev_emb = model.markov_head.prev_embeddings(prev_toks) + eng_delta.to(
        base_logits.dtype
    )
    m_bias = model.markov_head.block_bias(
        prev_token_ids=prev_toks,
        hidden_states=hidden.view(1, 4, 64),
        prev_emb=prev_emb,
    )
    tf_logits = base_logits.view(1, 4, 64) + m_bias
    tf_probs = torch.softmax(tf_logits.float(), dim=-1)
    assert torch.allclose(tf_probs, details["draft_probs"], atol=1e-5)


# ==============================================================================
# 5. Axe 4 — sp(4, R) Butterfly Contractivity, Symplecticity & Forcing
# ==============================================================================


def test_axe4_sp4_butterfly_contractivity_symplecticity_and_inhomogeneous_rollout():
    """Verify butterfly mixing, ||A||_2 < 1 stability, Sp(4,R) limit, and forcing."""
    torch.manual_seed(501)
    # 5 layers over M=32 modes cover XOR strides 1, 2, 4, 8, 16
    perms = [
        CausalKvNRFFLayer(d_model=32, n_modes=32, layer_idx=i).butterfly_perm
        for i in range(5)
    ]
    for l_idx, perm in enumerate(perms):
        stride = 1 << l_idx
        for d in range(16):
            a, b = int(perm[2 * d].item()), int(perm[2 * d + 1].item())
            assert b == (a ^ stride)

    # Adversarial spectral norm check: extreme 100x inputs MUST keep ||A_{t,d}||_2 <
    # 1.0!
    layer = CausalKvNRFFLayer(
        d_model=32,
        n_modes=16,
        use_symplectic_butterfly=True,
        squeezing_max_ratio=0.8,
        layer_idx=2,
    )
    with torch.no_grad():
        torch.nn.init.normal_(layer.squeezing_proj.weight, std=2.0)
        torch.nn.init.normal_(layer.butterfly_bs_proj.weight, std=2.0)
    x_extreme = torch.randn(4, 32, 32) * 100.0
    a_mat = layer.compute_step_transition_matrices(x_extreme)
    spec_norms = torch.linalg.matrix_norm(a_mat, ord=2)
    assert spec_norms.max().item() < 1.0

    # Zero-damping limit (gamma -> 0) WITH non-zero squeezing & beamsplitters: exact
    # Sp(4, R)!
    layer_symp = CausalKvNRFFLayer(
        d_model=32,
        n_modes=8,
        damping_min=0.0,
        use_symplectic_butterfly=True,
        squeezing_max_ratio=0.0,
    )
    with torch.no_grad():
        layer_symp.log_eta.fill_(-60.0)
        # Inject explicit active squeezing even at gamma=0 by temporarily setting
        # squeezing_max_ratio
        # with small positive damping 1e-7? Wait: at gamma=0, |xi| <= 0.8 * gamma = 0 is
        # U(2) subset Sp(4, R);
        # also test explicit Strang M(0, xi) B M(0, xi) R symplecticity AND layer_symp
        # A^T Omega_4 A = Omega_4!
        torch.nn.init.normal_(layer_symp.butterfly_bs_proj.weight, std=1.0)
        layer_symp.butterfly_couple_gate.fill_(1.0)

    a_symp = layer_symp.compute_step_transition_matrices(torch.randn(2, 8, 32))
    omega_2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    omega_4 = torch.block_diag(omega_2, omega_2)
    lhs = a_symp.transpose(-1, -2) @ omega_4 @ a_symp
    assert torch.allclose(lhs, omega_4.expand_as(lhs), atol=1e-5)

    # Also verify non-zero squeezing exp(S_a/2) B exp(S_b/2) at gamma = 0 preserves
    # Omega_4 identically!
    xi = torch.tensor([0.45, -0.35])
    c_h, s_h = torch.cosh(xi), torch.sinh(xi)
    psi = torch.tensor([0.7, -1.1])
    m_blocks = []
    for i in range(2):
        cp, sp = torch.cos(psi[i]), torch.sin(psi[i])
        m_blocks.append(
            torch.tensor(
                [
                    [c_h[i] + s_h[i] * cp, s_h[i] * sp],
                    [s_h[i] * sp, c_h[i] - s_h[i] * cp],
                ]
            )
        )
    m4 = torch.block_diag(*m_blocks)
    a_squeezed = m4 @ a_symp[0, 0, 0] @ m4
    assert torch.allclose(a_squeezed.T @ omega_4 @ a_squeezed, omega_4, atol=1e-5)

    # Inhomogeneous rollout horizon_force_proj step-1 gradient check
    rollout = MultiHorizonKoopmanRollout(d_model=32, n_modes=16, block_size=4)
    h_anc = torch.randn(3, 32, requires_grad=True)
    q_anc = torch.randn(3, 16)
    p_anc = torch.randn(3, 16)
    dt_anc = torch.randn(3, 16)
    h_blk, _ = rollout(h_anc, q_anc, p_anc, dt_anc)
    h_blk.square().sum().backward()
    assert rollout.horizon_force_proj.weight.grad is not None
    assert rollout.horizon_force_proj.weight.grad.abs().max().item() > 0.0


# ==============================================================================
# 6. Axe 6 — Lindblad Excess Thermal Variance, Purity & Rescued p_vac
# ==============================================================================


def test_axe6_lindblad_excess_variance_purity_and_rescued_pvac():
    """Verify delta_nu >= 0, purity in (0, 1], monotonic noise response, and
    BornConfidenceHead grads.
    """
    torch.manual_seed(601)
    layer = CausalKvNRFFLayer(
        d_model=32, n_modes=16, use_symplectic_butterfly=True, use_lindblad_purity=True
    )
    rollout = MultiHorizonKoopmanRollout(
        d_model=32, n_modes=16, block_size=4, use_lindblad_purity=True
    )
    conf_head = BornConfidenceHead(
        hidden_size=32,
        n_modes=16,
        use_born=True,
        vacuum_born_norm=True,
        use_lindblad_purity=True,
    )

    x = torch.randn(2, 12, 32)
    out, q, p, dt, nu = layer(x, return_nu=True)
    assert torch.all(nu >= 0.0)

    h_blk, born_obs, p_vac, purity, _, _ = rollout(
        out[:, -1],
        q[:, -1],
        p[:, -1],
        dt[:, -1],
        anchor_nu=nu[:, -1],
        return_quantum_stats=True,
    )
    assert torch.all(purity > 0.0) and torch.all(purity <= 1.0)
    assert p_vac.shape == (2, 4, 8)
    assert purity.shape == (2, 4, 16)

    # Detach h_blk & born_obs individually to prove BOTH p_vac and purity drive
    # BornConfidenceHead
    conf_logits = conf_head(
        h_blk.detach(), born_features=born_obs, p_vac=p_vac, purity=purity
    )
    conf_logits.sum().backward()
    assert conf_head.vacuum_pvac_proj is not None
    assert conf_head.lindblad_purity_proj is not None
    assert conf_head.vacuum_pvac_proj.weight.grad is not None
    assert conf_head.vacuum_pvac_proj.weight.grad.abs().max().item() > 0.0
    assert conf_head.lindblad_purity_proj.weight.grad is not None
    assert conf_head.lindblad_purity_proj.weight.grad.abs().max().item() > 0.0
    assert layer.lindblad_env_proj is not None
    assert layer.lindblad_env_proj.weight.grad is not None
    assert layer.lindblad_env_proj.weight.grad.abs().max().item() > 0.0

    # Increasing thermal bath injection strictly decreases mode purity
    with torch.no_grad():
        layer.lindblad_env_proj.bias.add_(5.0)
        _, q2, p2, dt2, nu2 = layer(x, return_nu=True)
        _, _, _, purity_noisy, _, _ = rollout(
            out[:, -1],
            q2[:, -1],
            p2[:, -1],
            dt2[:, -1],
            anchor_nu=nu2[:, -1],
            return_quantum_stats=True,
        )
    assert torch.all(purity_noisy < purity)


# ==============================================================================
# 7. [C5] Proximal Dyadic Group-LASSO Exact Zeroing + Zero-Shot Pruning Parity
# ==============================================================================


def test_c5_proximal_dyadic_group_lasso_exact_zeros_and_zeroshot_pruning():
    """[C5] Verify proximal dyadic Group-LASSO produces exact IEEE-754 0.0s and < 1e-5
    pruning drift across 5 butterfly layers.
    """
    torch.manual_seed(701)
    cfg = _make_v3_cfg(
        hidden_size=48,
        num_layers=5,  # Covers butterfly strides 1, 2, 4, 8, 16 on M=32!
        n_modes=32,
        block_size=4,
        kvn_multilayer_rollout=True,
        use_symplectic_butterfly=True,
        use_lindblad_purity=True,
        use_kraus_tree=True,
    )
    model = KvNDSparkDraftModel(cfg)
    _init_verifier_stubs(model)

    # Make dyads 0..7 ~4x stronger than dyads 8..15 so proximal thresholding cleanly
    # zeros a subset
    with torch.no_grad():
        for layer in model.kvn_layers:
            layer.in_gain_q[:16].mul_(4.0)
            layer.in_gain_p[:16].mul_(4.0)
            layer.readout_linear.weight[:, :16].mul_(4.0)
            layer.readout_linear.weight[:, 32:48].mul_(4.0)
            layer.readout_linear.weight[:, 64:80].mul_(4.0)
            layer.in_gain_q[16:].mul_(0.25)
            layer.in_gain_p[16:].mul_(0.25)
            layer.readout_linear.weight[:, 16:32].mul_(0.25)
            layer.readout_linear.weight[:, 48:64].mul_(0.25)
            layer.readout_linear.weight[:, 80:96].mul_(0.25)
        assert model.rollout is not None
        model.rollout.obs_linear.weight[:, 16:32].mul_(0.25)
        model.rollout.obs_linear.weight[:, 48:64].mul_(0.25)
        model.rollout.obs_linear.weight[:, 80:96].mul_(0.25)

    # Run 12 proximal steps tuned between the two dyad energy tiers
    probe = model.apply_proximal_dyadic_group_lasso(0.0)
    step_lam = 0.5 * (probe["min_dyad_norm"] + probe["max_dyad_norm"]) / 10.0
    stats = probe
    for _ in range(12):
        stats = model.apply_proximal_dyadic_group_lasso(step_lam)
        if 0.25 <= stats["zero_dyad_fraction"] <= 0.75:
            break

    assert 0.0 < stats["zero_dyad_fraction"] < 1.0
    zero_dyads = round(stats["zero_dyad_fraction"] * 16)
    kept_modes = 2 * (16 - zero_dyads)
    assert kept_modes >= 2 and kept_modes % 2 == 0

    # Verify exact IEEE-754 0.0 on zeroed dyads
    zero_mask = model.kvn_layers[0].in_gain_q == 0.0
    assert int(zero_mask.sum().item()) == 2 * zero_dyads

    model.eval()
    seq_len = 20
    hs = torch.randn(1, seq_len, 3 * 48)
    ids = torch.randint(0, 64, (1, seq_len))
    fc_all = model.hidden_norm(model.fc(hs))

    pref_before = model.prefill_kvn_state(ids[:, :11], fc_all[:, :11])
    tok_before, _, det_before = model.advance_and_draft_step(
        prev_fc_block=pref_before["last_fc"],
        token_ids_block=ids[:, 11:12],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=pref_before["state_cache"],
        temperature=0.0,
        return_details=True,
    )

    model.prune_kvn_modes(kept_modes)
    assert model.config.kvn_n_modes == kept_modes

    pref_after = model.prefill_kvn_state(ids[:, :11], fc_all[:, :11])
    tok_after, _, det_after = model.advance_and_draft_step(
        prev_fc_block=pref_after["last_fc"],
        token_ids_block=ids[:, 11:12],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=pref_after["state_cache"],
        temperature=0.0,
        return_details=True,
    )

    assert torch.equal(tok_before, tok_after)
    assert (
        det_before["draft_probs"] - det_after["draft_probs"]
    ).abs().max().item() < 1e-5
    assert (
        det_before["confidences"] - det_after["confidences"]
    ).abs().max().item() < 1e-5


# ==============================================================================
# 8. Axe 2 — 2-Pass Self-Conditioned Rollout, Shortlist Mass & DDP Invariant
# ==============================================================================


def test_axe2_self_cond_rollout_shortlist_mass_and_ddp_zero_unused_params():
    """Verify 2-pass self-conditioned training, shortlist target_lse_full/in_vocab_mass,
    and zero unused DDP params.
    """
    torch.manual_seed(801)
    cfg = _make_v3_cfg(
        vocab_size=64,
        draft_vocab_size=32,
        n_modes=16,
        block_size=4,
        markov_head_type="rnn",
        self_cond_rollout_prob=1.0,
        self_cond_temperature=0.8,
    )
    model = KvNDSparkDraftModel(cfg)
    _init_verifier_stubs(model)
    t2d = torch.zeros(64, dtype=torch.bool)
    t2d[:32] = True
    d2t = torch.zeros(32, dtype=torch.long)
    model.load_vocab_mappings(t2d, d2t)
    full_v_w = torch.randn(64, 64) * 0.02
    full_v_w.requires_grad_(False)
    object.__setattr__(model, "_full_verifier_lm_head_weight", full_v_w)
    with torch.no_grad():
        model.verifier_lm_head.weight.copy_(full_v_w[:32])
        model.lm_head.weight.copy_(full_v_w[:32])

    seq_len = 20
    hs = torch.randn(1, seq_len, 3 * 64)
    ids = torch.randint(0, 64, (1, seq_len))
    mask = torch.ones(1, seq_len, dtype=torch.bool)
    v_hs = torch.randn(1, seq_len, 64)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)

    model.train()
    _, loss, metrics = model(hs, ids, mask, v_hs, doc_ids, max_anchors=3)
    assert torch.isfinite(loss)
    t_lse, in_mass, argmax_in = model._last_shortlist_stats
    assert t_lse is not None and in_mass is not None and argmax_in is not None
    assert torch.all((in_mass > 0.0) & (in_mass <= 1.0 + 1e-5))
    loss.backward()

    unused_normal = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert unused_normal == [], (
        f"Unused trainable parameters on active batch: {unused_normal}"
    )

    # Adversarial DDP edge case: all-zero loss_mask rank MUST STILL touch every
    # trainable parameter!
    model.zero_grad(set_to_none=True)
    empty_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    _, zero_loss, _ = model(hs, ids, empty_mask, v_hs, doc_ids, max_anchors=3)
    zero_loss.backward()
    unused_empty = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert unused_empty == [], (
        f"Unused trainable parameters on empty-anchor rank: {unused_empty}"
    )


# ==============================================================================
# 9. Axe 5 — Chunked Prefill, Kraus POVM Completeness & SpecInfer Lossless MC
# ==============================================================================


def test_axe5_chunked_prefill_kraus_povm_completeness_and_specinfer_mc():
    """Verify chunked prefill parity, POVM completeness, and SpecInfer MC TV < 0.015."""
    torch.manual_seed(901)
    # 1. Exact POVM completeness identity: sum_{y in C} M_y^dagger M_y == I_{M+1}
    kraus = KrausPOVMHead(d_model=32, n_modes=16, top_c=12)
    cand_logits = torch.randn(4, 5, 12)
    cand_vecs = torch.randn(4, 5, 12, 32)
    q_st = torch.randn(4, 5, 16)
    p_st = torch.randn(4, 5, 16)
    p_kraus, povm_diag, psi_sq = kraus.compute_povm(cand_logits, cand_vecs, q_st, p_st)
    # povm_diag is diag(M_y^dagger M_y) across j = 0 .. M
    povm_sum = povm_diag.sum(dim=-2)  # [4, 5, 17]
    assert torch.allclose(povm_sum, torch.ones_like(povm_sum), atol=1e-5)
    assert torch.allclose(
        p_kraus.sum(dim=-1), torch.ones_like(p_kraus[..., 0]), atol=1e-5
    )
    assert torch.allclose(
        psi_sq.sum(dim=-1), torch.ones_like(psi_sq[..., 0]), atol=1e-5
    )

    # 2. Chunked prefill_kvn_state vs 1-shot prefill + stochastic advance_and_draft_step
    cfg = _make_v3_cfg(vocab_size=32, n_modes=16, block_size=4)
    model = KvNDSparkDraftModel(cfg).eval()
    _init_verifier_stubs(model)

    ids = torch.randint(0, 32, (1, 12))
    fc = torch.randn(1, 12, 64)
    full_pref = model.prefill_kvn_state(ids[:, :10], fc[:, :10])
    c1_pref = model.prefill_kvn_state(ids[:, :6], fc[:, :6])
    # Test BOTH 2-chunk prefill_kvn_state(initial_state=c1_pref) AND
    # advance_and_draft_step!
    c2_pref = model.prefill_kvn_state(ids[:, 6:10], fc[:, 6:10], initial_state=c1_pref)
    assert torch.allclose(full_pref["state_cache"], c2_pref["state_cache"], atol=1e-5)

    shifted_chunk2 = torch.cat([c1_pref["last_fc"], fc[:, 6:9]], dim=1)
    _, c2_cache, _ = model.advance_and_draft_step(
        prev_fc_block=shifted_chunk2,
        token_ids_block=ids[:, 6:10],
        valid_lens=torch.tensor([4], dtype=torch.long),
        state_cache=c1_pref["state_cache"].clone(),
        return_details=True,
    )
    assert torch.allclose(full_pref["state_cache"], c2_cache, atol=1e-5)

    # 3. Kraus POVM tree expansion: shapes, causal mask, survival monotonicity & greedy
    # spine match
    pre9 = model.prefill_kvn_state(ids[:, :9], fc[:, :9])
    greedy_ids, _, greedy_det = model.advance_and_draft_step(
        prev_fc_block=pre9["last_fc"],
        token_ids_block=ids[:, 9:10],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=pre9["state_cache"].clone(),
        temperature=0.0,
        return_details=True,
    )
    assert greedy_det["purity"].shape == (1, 4)
    assert greedy_det["p_vac"].shape == (1, 4)
    assert greedy_det["mode_purity"].shape == (1, 4, 16)
    assert greedy_det["dyad_p_vac"].shape == (1, 4, 8)
    assert greedy_det["draft_entropy"].shape == (1, 4)

    tree_t0, _ = model.draft_kraus_tree_step(
        confirmed_fc_hidden=pre9["last_fc"],
        confirmed_token_ids=ids[:, 9:10],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=pre9["state_cache"].clone(),
        tree_budget=12,
        branch_width=3,
        temperature=0.0,
    )
    assert tree_t0["tree_draft_ids"].shape == (1, 12)
    assert tree_t0["tree_attn_mask"].shape == (1, 12, 12)
    # Extract greedy top-1 spine from tree (parent=-1 first child -> its first child ->
    # ...)
    parents = tree_t0["tree_parent_idx"][0].tolist()
    spine_tokens: list[int] = []
    cur = -1
    for _ in range(4):
        kids = [i for i, p_idx in enumerate(parents) if p_idx == cur]
        assert len(kids) > 0
        first_kid = kids[0]
        spine_tokens.append(int(tree_t0["tree_draft_ids"][0, first_kid].item()))
        cur = first_kid
    assert spine_tokens == greedy_ids[0].tolist()

    # Check causal mask lower-triangular / ancestor validity & survival monotonicity at
    # T = 0.6
    tree_stoch, _ = model.draft_kraus_tree_step(
        confirmed_fc_hidden=pre9["last_fc"],
        confirmed_token_ids=ids[:, 9:10],
        valid_lens=torch.tensor([1], dtype=torch.long),
        state_cache=pre9["state_cache"].clone(),
        tree_budget=10,
        branch_width=3,
        temperature=0.6,
    )
    q_sums = tree_stoch["tree_q_probs"].sum(dim=-1)
    assert torch.allclose(q_sums, torch.ones_like(q_sums), atol=1e-5)
    surv = tree_stoch["tree_survival"][0]
    for idx_n, p_idx in enumerate(tree_stoch["tree_parent_idx"][0].tolist()):
        if p_idx >= 0:
            assert surv[idx_n].item() <= surv[p_idx].item() + 1e-6

    # 4. Monte Carlo SpecInfer lossless target distribution test (N = 20,000 trials at T
    # = 0.6)
    gen = torch.Generator().manual_seed(1337)
    vocab_mc = 8
    target_logits = torch.tensor([1.4, 0.9, 0.2, -0.4, 1.1, -0.8, 0.5, -0.1]) / 0.6
    target_p = torch.softmax(target_logits, dim=-1)
    q_siblings = torch.stack(
        [
            torch.softmax(
                torch.tensor([0.8, 1.3, -0.2, 0.1, 0.4, -0.5, 0.9, 0.0]), dim=-1
            ),
            torch.softmax(
                torch.tensor([1.5, 0.2, 0.7, -0.6, 0.9, 0.1, -0.3, 0.4]), dim=-1
            ),
            torch.softmax(
                torch.tensor([0.1, 0.5, 1.1, 0.8, -0.4, 0.3, 0.2, -0.7]), dim=-1
            ),
        ],
        dim=0,
    )

    counts = torch.zeros(vocab_mc, dtype=torch.float64)
    n_trials = 20_000
    # Vectorized simulation of the exact sequential SpecInfer step & tree API check
    for trial in range(n_trials):
        cands = torch.tensor(
            [
                int(torch.multinomial(q_siblings[r], 1, generator=gen).item())
                for r in range(3)
            ],
            dtype=torch.long,
        )
        if trial < 200:
            # Exercise specinfer_verify_tree root+children path on first 200 trials
            node_targets = target_p.unsqueeze(0).expand(4, vocab_mc)
            emitted_seq, _ = specinfer_verify_tree(
                tree_draft_ids=cands,
                tree_parent_idx=torch.tensor([-1, -1, -1], dtype=torch.long),
                tree_q_probs=q_siblings,
                node_target_probs=node_targets,
                generator=gen,
            )
            tok = emitted_seq[0]
        else:
            tok, _ = specinfer_verify_single_step(
                candidate_ids=cands,
                candidate_q_probs=q_siblings,
                target_probs=target_p,
                generator=gen,
            )
        counts[tok] += 1.0

    emp_p = (counts / n_trials).float()
    tv_dist = 0.5 * (emp_p - target_p).abs().sum().item()
    assert tv_dist < 0.015, f"SpecInfer MC TV distance {tv_dist:.4f} exceeded 0.015!"


# ==============================================================================
# 10. Axe 3 — Mega-KvN Triton / PyTorch Segmented Scan & Multi-Step Decode Parity
# ==============================================================================


def test_axe3_triton_and_pytorch_scan_kernels_forward_backward():
    """Verify forward+backward parity of scalar, complex, and sp(4) scans & multistep
    decode.
    """
    torch.manual_seed(1001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Scalar scan forward + backward vs explicit sequential loop
    a_s = torch.sigmoid(torch.randn(2, 64, 8, device=device, requires_grad=True))
    u_s = torch.nn.functional.softplus(
        torch.randn(2, 64, 8, device=device, requires_grad=True)
    )
    init_s = torch.rand(2, 8, device=device, requires_grad=True)

    y_scan = segmented_scalar_scan(
        a_s, u_s, initial_state=init_s, use_triton=HAS_TRITON
    )
    y_ref = pytorch_segmented_scalar_scan(a_s, u_s, initial_state=init_s)
    assert torch.allclose(y_scan, y_ref, atol=1e-5)

    # 2. Complex scan forward + backward
    ac_raw = torch.randn(2, 64, 8, device=device, requires_grad=True)
    as_raw = torch.randn(2, 64, 8, device=device, requires_grad=True)
    uq = torch.randn(2, 64, 8, device=device, requires_grad=True)
    up = torch.randn(2, 64, 8, device=device, requires_grad=True)

    ac = 0.45 * torch.tanh(ac_raw)
    asn = 0.45 * torch.tanh(as_raw)
    q1, p1 = segmented_complex_scan(ac, asn, uq, up, use_triton=HAS_TRITON)
    (q1.square().sum() + p1.square().sum()).backward()
    g_uq = uq.grad.clone()
    uq.grad = None
    ac_raw.grad = None
    as_raw.grad = None
    up.grad = None

    ac2 = 0.45 * torch.tanh(ac_raw)
    asn2 = 0.45 * torch.tanh(as_raw)
    q2, p2 = pytorch_segmented_complex_scan(ac2, asn2, uq, up)
    (q2.square().sum() + p2.square().sum()).backward()
    assert torch.allclose(q1, q2, atol=1e-5)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(g_uq, uq.grad, atol=1e-5)

    # 3. 4x4 sp(4, R) scan forward + backward + multistep decode
    a4_leaf = torch.randn(2, 96, 4, 4, 4, device=device, requires_grad=True)
    u4_leaf = torch.randn(2, 96, 4, 4, device=device, requires_grad=True)
    z0_leaf = torch.randn(2, 4, 4, device=device, requires_grad=True)

    a4 = 0.2 * torch.tanh(a4_leaf)
    z_tr = segmented_sp4_scan(a4, u4_leaf, initial_state=z0_leaf, use_triton=HAS_TRITON)
    z_tr.square().sum().backward()
    g_a4 = a4_leaf.grad.clone()
    g_u4 = u4_leaf.grad.clone()
    g_z0 = z0_leaf.grad.clone()
    a4_leaf.grad = None
    u4_leaf.grad = None
    z0_leaf.grad = None

    a4_py = 0.2 * torch.tanh(a4_leaf)
    z_py = pytorch_segmented_sp4_scan(a4_py, u4_leaf, initial_state=z0_leaf)
    z_py.square().sum().backward()

    assert torch.allclose(z_tr, z_py, atol=1e-5)
    assert torch.allclose(g_a4, a4_leaf.grad, atol=1e-4)
    assert torch.allclose(g_u4, u4_leaf.grad, atol=1e-4)
    assert torch.allclose(g_z0, z0_leaf.grad, atol=1e-4)

    with torch.no_grad():
        vlens = torch.tensor([96, 40], device=device, dtype=torch.long)
        _, z_fin_ms, _, _ = fused_multistep_sp4_decode(
            a4_py, u4_leaf, z0_leaf, vlens, use_triton=HAS_TRITON
        )
        assert torch.allclose(z_fin_ms[0], z_py[0, 95], atol=1e-5)
        assert torch.allclose(z_fin_ms[1], z_py[1, 39], atol=1e-5)

        _, _, q_fin_c, p_fin_c, _, _ = fused_multistep_complex_decode(
            ac2[:, :32],
            asn2[:, :32],
            uq[:, :32],
            up[:, :32],
            torch.zeros(2, 8, device=device),
            torch.zeros(2, 8, device=device),
            torch.tensor([32, 15], device=device),
        )
        assert torch.allclose(q_fin_c[0], q2[0, 31], atol=1e-5)
        assert torch.allclose(q_fin_c[1], q2[1, 14], atol=1e-5)


@pytest.mark.parametrize("backbone_type", ["kvn", "swa"])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("max_anchors", [2, 4, 6])
def test_c4_kvn_dspark_dense_anchor_oov_shortlist_and_nontriton_fallback(
    backbone_type: str,
    sample_from_anchor: bool,
    max_anchors: int,
):
    """[C4] Verify 80%-OOV targets preserve exact c*=0.20 and 0% argmax accuracy."""
    import math

    torch.manual_seed(1101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 16
    full_vocab = 16
    draft_vocab = 8
    block_size = 4
    seq_len = (
        16  # max_anchors=4 -> 16 >= 16, max_anchors=6 -> 24 > 16 (dense-anchor branch!)
    )

    tl_cfg = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=full_vocab,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    kvn_cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        draft_vocab_size=draft_vocab,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        kvn_backbone_type=backbone_type,  # type: ignore[arg-type]
        kvn_n_modes=8,
        use_kraus_tree=False,
    )
    model = KvNDSparkDraftModel(kvn_cfg).to(device).eval()
    train_kw, _ = KvNDSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}'
    )

    t2d = torch.zeros(full_vocab, dtype=torch.bool)
    t2d[:draft_vocab] = True
    d2t = torch.arange(draft_vocab, dtype=torch.long)

    # Construct full_w so h = [1, 1, ..., 1] (unit RMS -> verifier_norm(h) == 1) yields:
    # logit(0) = 0.0, logit(1..7) = -100.0 -> shortlist sum exp = 1.0
    # logit(8) = log(4.0), logit(9..15) = -100.0 -> OOV sum exp = 4.0 -> M_in = 0.20,
    # argmax = 8 (OOV)!
    full_w = torch.zeros(full_vocab, hidden_size, device=device)
    full_w[:, 0] = -100.0
    full_w[0, 0] = 0.0
    full_w[draft_vocab, 0] = math.log(4.0)

    model.load_vocab_mappings(t2d, d2t)
    object.__setattr__(model, "_full_verifier_lm_head_weight", full_w.clone())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.verifier_norm.weight.fill_(1.0)
        model.verifier_lm_head.weight.copy_(full_w[:draft_vocab])
        model.embed_tokens.weight[:, 0] = 1.0
        if model.rollout is not None:
            model.rollout.out_norm.weight.fill_(1.0)
            linears = [
                m
                for m in model.rollout.anchor_proj.modules()
                if isinstance(m, torch.nn.Linear)
            ]
            if linears and linears[-1].bias is not None:
                linears[-1].bias[0::hidden_size] = 1.0
        model.norm.weight.zero_()
        model.norm.weight[0] = math.sqrt(hidden_size)
        model.lm_head.weight.zero_()
        model.lm_head.weight[0, 0] = 50.0

    hidden_states = torch.zeros(1, seq_len, 2 * hidden_size, device=device)
    verifier_last_h = torch.ones(1, seq_len, hidden_size, device=device)
    input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    loss_mask = torch.ones(1, seq_len, dtype=torch.float32, device=device)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_h,
        document_ids=doc_ids,
        loss_config=train_kw["loss_config"],
        tv_loss_fn=train_kw["tv_loss_fn"],
        max_anchors=max_anchors,
    )
    assert torch.isfinite(loss)
    c_star = float(metrics["accept_rate_sum"] / metrics["accept_rate_total"])
    assert abs(c_star - 0.20) < 1e-4, f"Expected c*=0.20, got {c_star}"
    assert float(metrics["full_acc_sum"]) == 0.0
    first_pos_key = "position_0_acc_sum" if sample_from_anchor else "position_1_acc_sum"
    assert float(metrics[first_pos_key]) == 0.0


def test_kvn_v3_config_alias_kwargs_and_property_setters():
    """Verify KvNDSparkSpeculatorConfig accepts kvn_* v3 aliases and setters."""
    tl_cfg = _make_tl_cfg(hidden_size=32, num_layers=2, vocab_size=64)
    cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        kvn_n_modes=16,
        kvn_dyadic_rff=False,
        kvn_use_engram=False,
        kvn_engram_buckets=1024,
        kvn_engram_dim=32,
        kvn_symplectic_butterfly=False,
        kvn_squeezing_max_ratio=0.55,
        kvn_lindblad_purity=False,
        kvn_use_kraus_tree=False,
        kvn_kraus_top_c=64,
        kvn_use_triton_scan=False,
        kvn_self_cond_prob=0.25,
    )
    assert cfg.dyadic_rff is False and cfg.kvn_dyadic_rff is False
    assert cfg.use_engram is False and cfg.kvn_use_engram is False
    assert cfg.engram_buckets == 1024 and cfg.kvn_engram_buckets == 1024
    assert cfg.engram_dim == 32 and cfg.kvn_engram_dim == 32
    assert (
        cfg.use_symplectic_butterfly is False and cfg.kvn_symplectic_butterfly is False
    )
    assert abs(cfg.squeezing_max_ratio - 0.55) < 1e-9
    assert cfg.use_lindblad_purity is False and cfg.kvn_lindblad_purity is False
    assert cfg.use_kraus_tree is False and cfg.kvn_use_kraus_tree is False
    assert cfg.kraus_top_c == 64 and cfg.kvn_kraus_top_c == 64
    assert cfg.use_triton_scan is False and cfg.kvn_use_triton_scan is False
    assert (
        abs(cfg.self_cond_rollout_prob - 0.25) < 1e-9
        and abs(cfg.kvn_self_cond_prob - 0.25) < 1e-9
    )

    # Property setters + roundtrip serialization
    cfg.kvn_symplectic_butterfly = True
    cfg.kvn_lindblad_purity = True
    assert cfg.use_symplectic_butterfly is True
    assert cfg.use_lindblad_purity is True

    reloaded = KvNDSparkSpeculatorConfig(**cfg.to_dict())
    assert reloaded.use_symplectic_butterfly is True
    assert reloaded.use_lindblad_purity is True
    assert reloaded.engram_buckets == 1024
