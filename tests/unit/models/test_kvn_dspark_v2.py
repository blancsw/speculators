"""Unit tests for KvN-DSpark Phase 2 (Dequantized Damped Bosonic SSM + Vacuum Born v2)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.kvn_dspark.config import KvNDSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config.schema import TrainConfig
from speculators.train.optimizers import split_named_params_for_muon


def _make_qwen3_8b_like_tl_config(
    hidden_size: int = 4096,
    num_layers: int = 5,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    sliding_window: int = 128,
    vocab_size: int = 256,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        sliding_window=sliding_window,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * num_layers,
        _attn_implementation="eager",
    )


def _make_v2_speculator_config(
    tl_cfg: Qwen3Config,
    backbone_type: str = "kvn",
    n_modes: int = 128,
    block_size: int = 8,
    **extra_kwargs,
) -> KvNDSparkSpeculatorConfig:
    return KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        draft_vocab_size=tl_cfg.vocab_size,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=True,
        markov_rank=32,
        markov_head_type="gated",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        kvn_backbone_type=backbone_type,  # type: ignore[arg-type]
        kvn_n_modes=n_modes,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"]),
        ),
        **extra_kwargs,
    )


def test_gqa_state_memory_accounting_qwen3_8b():
    """Verify exact GQA KV dimension (8 * 128 = 1024) yields 2.62 MB/seq and 1024x ratio."""
    tl_cfg = _make_qwen3_8b_like_tl_config()
    swa_cfg = _make_v2_speculator_config(tl_cfg, backbone_type="swa", n_modes=128)
    kvn_bf16_cfg = _make_v2_speculator_config(
        tl_cfg, backbone_type="kvn", n_modes=128, kvn_state_dtype="bfloat16"
    )
    kvn_fp32_cfg = _make_v2_speculator_config(
        tl_cfg, backbone_type="kvn", n_modes=128, kvn_state_dtype="float32"
    )
    hyb_cfg = _make_v2_speculator_config(
        tl_cfg, backbone_type="hybrid_1swa_2kvn", n_modes=128, kvn_state_dtype="bfloat16"
    )

    assert swa_cfg.estimated_state_bytes_per_seq() == 2_621_440
    assert kvn_bf16_cfg.estimated_state_bytes_per_seq() == 2_560
    assert swa_cfg.estimated_state_bytes_per_seq() // kvn_bf16_cfg.estimated_state_bytes_per_seq() == 1024
    assert kvn_fp32_cfg.estimated_state_bytes_per_seq() == 5_120
    assert hyb_cfg.estimated_state_bytes_per_seq() == 526_336


def test_new_v2_cli_flags_resolution():
    """Verify TrainConfig resolves all Phase 2 KvN-DSpark CLI arguments cleanly."""
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "Qwen/Qwen3-8B",
            "--speculator-type",
            "kvn_dspark",
            "--kvn-reupload-dim",
            "512",
            "--kvn-learnable-rff",
            "--kvn-anchor-rank",
            "512",
            "--kvn-enable-interferometer",
            "--kvn-vacuum-born-norm",
            "--kvn-multilayer-rollout",
            "--kvn-intrablock-mixer",
            "--kvn-strict-causal-shift",
            "--kvn-ffn-mult",
            "1.5",
            "--kvn-state-dtype",
            "float32",
        ]
    )
    flat = cfg.flatten()
    assert flat["kvn_reupload_dim"] == 512
    assert flat["kvn_learnable_rff"] is True
    assert flat["kvn_anchor_rank"] == 512
    assert flat["kvn_enable_interferometer"] is True
    assert flat["kvn_vacuum_born_norm"] is True
    assert flat["kvn_multilayer_rollout"] is True
    assert flat["kvn_intrablock_mixer"] is True
    assert flat["kvn_strict_causal_shift"] is True
    assert flat["kvn_ffn_mult"] == pytest.approx(1.5)
    assert flat["kvn_state_dtype"] == "float32"


def test_spectral_params_routed_to_adamw_not_muon():
    """Verify spectral frequencies, dyadic angles, and multi-layer weights bypass Muon NS."""

    class _DummySpectralModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(32, 16, bias=False)
            self.omega_rff = nn.Parameter(torch.randn(16, 32))
            self.dyad_matrix = nn.Parameter(torch.randn(16, 2))
            self.layer_mode_logits = nn.Parameter(torch.randn(5, 32))
            self.vacuum_scale = nn.Parameter(torch.randn(2, 2))

    mod = _DummySpectralModule()
    muon_params, adamw_params = split_named_params_for_muon(mod)
    muon_names = {n for n, _ in muon_params}
    adamw_names = {n for n, _ in adamw_params}

    assert "in_proj.weight" in muon_names
    assert "omega_rff" in adamw_names
    assert "dyad_matrix" in adamw_names
    assert "layer_mode_logits" in adamw_names
    assert "vacuum_scale" in adamw_names


def test_modal_data_reuploading_rff_compression_and_sqrt_l_scaling():
    """Verify ModalDataReuploadingRFF compresses params, scales std as 1/sqrt(L*d_re), and keeps final sin/cos active."""
    from speculators.models.kvn_dspark.model_definitions import (
        DataReuploadingLinear,
        ModalDataReuploadingRFF,
    )

    torch.manual_seed(11)
    legacy = DataReuploadingLinear(d_model=256, reupload_layers_l=3)
    modal = ModalDataReuploadingRFF(
        d_model=256,
        n_modes=32,
        reupload_layers_l=3,
        d_re=64,
        sigma_g=1.2,
        learnable_rff=True,
    )

    legacy_params = sum(p.numel() for p in legacy.parameters())
    modal_params = sum(p.numel() for p in modal.parameters())
    assert modal_params < legacy_params * 0.25
    assert isinstance(modal.omega_rff, nn.Parameter)

    expected_std = 1.2 / (3 * 64) ** 0.5
    assert modal.omega_rff.std().item() == pytest.approx(expected_std, rel=0.15)

    x = torch.randn(2, 10, 256, requires_grad=True)
    r_q, r_p = modal(x)
    assert r_q.shape == (2, 10, 32)
    assert r_p.shape == (2, 10, 32)
    (r_q.square().sum() + r_p.square().sum()).backward()
    assert modal.reupload_x[-1].weight.grad is not None
    assert modal.reupload_x[-1].weight.grad.abs().max().item() > 0.0


def test_multipacking_document_boundary_isolation_in_layer():
    """Verify reset_mask isolates packed documents with zero cross-document state leakage."""
    from speculators.models.kvn_dspark.model_definitions import CausalKvNRFFLayer

    torch.manual_seed(21)
    layer = CausalKvNRFFLayer(d_model=64, n_modes=32, reupload_layers_l=2, d_re=32)
    layer.eval()

    x1 = torch.randn(1, 32, 64)
    x2 = x1.clone()
    x2[:, :16, :] = x2[:, :16, :] * 500.0 + 50.0

    reset_mask = torch.zeros(1, 32, dtype=torch.bool)
    reset_mask[:, 0] = True
    reset_mask[:, 16] = True

    out1, q1, p1, _ = layer(x1, reset_mask=reset_mask)
    out2, q2, p2, _ = layer(x2, reset_mask=reset_mask)

    assert torch.allclose(out1[:, 16:], out2[:, 16:], atol=1e-5)
    assert torch.allclose(q1[:, 16:], q2[:, 16:], atol=1e-5)
    assert torch.allclose(p1[:, 16:], p2[:, 16:], atol=1e-5)

    # Without document reset at t=16, mutating doc 0 MUST leak into doc 1
    no_reset = torch.zeros(1, 32, dtype=torch.bool)
    no_reset[:, 0] = True
    _, q_leak1, _, _ = layer(x1, reset_mask=no_reset)
    _, q_leak2, _, _ = layer(x2, reset_mask=no_reset)
    assert not torch.allclose(q_leak1[:, 16:], q_leak2[:, 16:], atol=1e-3)


def test_segmented_scan_matches_step_by_step_recurrent_with_resets():
    """Verify parallel Kogge-Stone segmented scan matches step-by-step O(1) recurrence across resets."""
    from speculators.models.kvn_dspark.model_definitions import CausalKvNRFFLayer

    torch.manual_seed(33)
    layer = CausalKvNRFFLayer(d_model=64, n_modes=32, reupload_layers_l=2, d_re=32)
    layer.eval()

    x = torch.randn(2, 20, 64)
    reset_mask = torch.zeros(2, 20, dtype=torch.bool)
    reset_mask[:, 0] = True
    reset_mask[0, 7] = True
    reset_mask[1, 5] = True
    reset_mask[1, 14] = True

    scan_out, scan_q, scan_p, scan_dtheta = layer(x, reset_mask=reset_mask)

    state = None
    rec_outs, rec_qs, rec_ps, rec_dthetas = [], [], [], []
    for t in range(x.size(1)):
        x_step, state, dtheta_step = layer.recurrent_step(
            x[:, t : t + 1, :],
            state=state,
            reset_mask=reset_mask[:, t : t + 1],
        )
        rec_outs.append(x_step)
        rec_qs.append(state[0])
        rec_ps.append(state[1])
        rec_dthetas.append(dtheta_step)

    assert torch.allclose(scan_out, torch.cat(rec_outs, dim=1), atol=1e-5)
    assert torch.allclose(scan_q, torch.cat(rec_qs, dim=1), atol=1e-5)
    assert torch.allclose(scan_p, torch.cat(rec_ps, dim=1), atol=1e-5)
    assert torch.allclose(scan_dtheta, torch.cat(rec_dthetas, dim=1), atol=1e-5)


def test_interferometer_restores_phase_gradient_and_vacuum_born_properties():
    """Verify beamsplitters restore non-zero phase gradients in BornConfidenceHead and simplex/decay hold."""
    from speculators.models.kvn_dspark.model_definitions import (
        BornConfidenceHead,
        MultiHorizonKoopmanRollout,
        compute_vacuum_born_observables,
    )

    torch.manual_seed(44)
    rollout = MultiHorizonKoopmanRollout(
        d_model=64,
        n_modes=32,
        block_size=8,
        anchor_rank=32,
        enable_interferometer=True,
        vacuum_born_norm=True,
    )
    conf_head = BornConfidenceHead(hidden_size=64, n_modes=32, use_born=True)

    anchor_h = torch.randn(4, 64)
    anchor_q = torch.randn(4, 32)
    anchor_p = torch.randn(4, 32)
    anchor_dtheta = torch.randn(4, 32) * 0.8 + 0.2

    hidden_blocks, born_features = rollout(anchor_h, anchor_q, anchor_p, anchor_dtheta)
    # Detach hidden_blocks so loss gradient flows EXCLUSIVELY through born_features
    conf_logits = conf_head(hidden_blocks.detach(), born_features=born_features)
    conf_logits.sum().backward()

    assert rollout.horizon_freq_scale.grad is not None
    assert rollout.horizon_phase_bias.grad is not None
    assert rollout.horizon_freq_scale.grad.abs().max().item() > 1e-5
    assert rollout.horizon_phase_bias.grad.abs().max().item() > 1e-5

    # Test resonance boundedness and monotonic horizon decay to vacuum
    q_res = torch.randn(3, 8, 32) * 1e3
    p_res = torch.randn(3, 8, 32) * 1e3
    vac_log_k = torch.zeros(16)
    born_res, p_vac_res = compute_vacuum_born_observables(q_res, p_res, vac_log_k)
    assert torch.all(born_res >= 0.0) and torch.all(born_res < 2.0 + 1e-5)
    simplex_sum = 0.5 * (born_res[..., 0::2] + born_res[..., 1::2]) + p_vac_res
    assert torch.allclose(simplex_sum, torch.ones_like(simplex_sum), atol=1e-5)

    # Monotonic decay of energy across k = 0..7 drives p_vac strictly upward
    scales = torch.exp(-0.3 * torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    q_dec = torch.ones(2, 1, 32) * scales
    p_dec = torch.ones(2, 1, 32) * scales
    _, p_vac_dec = compute_vacuum_born_observables(q_dec, p_dec, vac_log_k)
    diffs = p_vac_dec[:, 1:, :] - p_vac_dec[:, :-1, :]
    assert torch.all(diffs > 0.0)


def test_intrablock_causal_mixer_strict_horizon_causality():
    """Verify IntraBlockCausalMixer mixes horizons k=0..K-1 causally with zero future leak."""
    from speculators.models.kvn_dspark.model_definitions import IntraBlockCausalMixer

    torch.manual_seed(55)
    mixer = IntraBlockCausalMixer(d_model=64, block_size=8, n_heads=4)
    mixer.eval()

    h1 = torch.randn(3, 8, 64)
    h2 = h1.clone()
    h2[:, 4:, :] = torch.randn_like(h2[:, 4:, :]) * 200.0

    y1 = mixer(h1)
    y2 = mixer(h2)
    assert torch.allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-5)
    assert not torch.allclose(y1[:, 4:, :], y2[:, 4:, :], atol=1e-3)


@pytest.mark.parametrize("vacuum_born_norm", [True, False])
def test_exact_dyadic_lasso_pruning_zero_error_dead_modes(vacuum_born_norm: bool):
    """Verify pruning dead oscillator dyads (q=p=0, arbitrary non-zero readout weights) has zero error across all 3 heads."""
    from speculators.models.kvn_dspark.model_definitions import (
        BornConfidenceHead,
        CausalKvNRFFLayer,
        MultiHorizonKoopmanRollout,
    )

    torch.manual_seed(66)
    layer = CausalKvNRFFLayer(
        d_model=64,
        n_modes=32,
        reupload_layers_l=2,
        d_re=32,
        enable_interferometer=True,
        vacuum_born_norm=vacuum_born_norm,
    )
    rollout = MultiHorizonKoopmanRollout(
        d_model=64,
        n_modes=32,
        block_size=4,
        anchor_rank=32,
        enable_interferometer=True,
        vacuum_born_norm=vacuum_born_norm,
    )
    conf_head = BornConfidenceHead(
        hidden_size=64, n_modes=32, use_born=True, vacuum_born_norm=vacuum_born_norm
    )
    layer.eval()
    rollout.eval()
    conf_head.eval()

    # Make dyads 8..15 (modes 16..31) dead oscillators (zero input gain -> q=p=0),
    # while keeping readout weights NON-ZERO on modes 0..15 and zeroing importance ONLY
    # for ranking via tiny weights vs large weights on 0..15 — Wait: test NON-ZERO weights
    # on modes 16..31 by passing explicit keep_indices = [0..15] or making 0..15 larger!
    with torch.no_grad():
        layer.in_gain_q[16:] = 0.0
        layer.in_gain_p[16:] = 0.0
        # Ensure modes 0..15 strictly win top-k importance while modes 16..31 still have non-zero weights!
        layer.readout_linear.weight[:, :16] *= 10.0
        layer.readout_linear.weight[:, 32:48] *= 10.0
        layer.readout_linear.weight[:, 64:80] *= 10.0

    x = torch.randn(1, 12, 64)
    out_before, q_before, p_before, dtheta_before = layer(x)
    h_blk_before, born_before = rollout(
        out_before[0, [3, 7]],
        q_before[0, [3, 7]],
        p_before[0, [3, 7]],
        dtheta_before[0, [3, 7]],
    )
    conf_before = conf_head(h_blk_before, born_features=born_before)

    top_idx = layer.prune_to_top_modes(16, compensate_zero_state_bias=not vacuum_born_norm)
    assert torch.equal(top_idx.cpu(), torch.arange(16))
    rollout.prune_to_indices(top_idx, compensate_zero_state_bias=not vacuum_born_norm)
    conf_head.prune_born_modes(top_idx, compensate_zero_state_bias=not vacuum_born_norm)

    out_after, q_after, p_after, dtheta_after = layer(x)
    h_blk_after, born_after = rollout(
        out_after[0, [3, 7]],
        q_after[0, [3, 7]],
        p_after[0, [3, 7]],
        dtheta_after[0, [3, 7]],
    )
    conf_after = conf_head(h_blk_after, born_features=born_after)

    assert torch.allclose(out_before, out_after, atol=1e-5)
    assert torch.allclose(h_blk_before, h_blk_after, atol=1e-5)
    assert torch.allclose(conf_before, conf_after, atol=1e-5)


def _init_dummy_draft_weights(model) -> None:
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.02)
        torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
        torch.nn.init.ones_(model.verifier_norm.weight)


@pytest.mark.parametrize("backbone_type", ["kvn", "hybrid_1swa_2kvn"])
def test_zero_anchor_causal_leakage_in_draft_model(backbone_type: str):
    """Verify mutating hidden_states[:, a:, :] (including anchor a itself!) has ZERO effect on block logits at anchor a."""
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    torch.manual_seed(101)
    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=64,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        vocab_size=128,
    )
    cfg = _make_v2_speculator_config(
        tl_cfg, backbone_type=backbone_type, n_modes=32, block_size=4
    )
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_draft_weights(model)
    model.eval()

    seq_len = 24
    hs1 = torch.randn(1, seq_len, 3 * 64)
    hs2 = hs1.clone()
    # Mutate anchor position a=6 AND all future positions >= 6 drastically!
    hs2[:, 6:, :] = torch.randn_like(hs2[:, 6:, :]) * 5000.0

    input_ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    loss_mask[0, 6] = True
    loss_mask[0, 14] = True
    verifier_last_hs = torch.randn(1, seq_len, 64)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, logits1, _, _, _, _ = model._kvn_backbone_forward(
        hidden_states=hs1,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )
    _, logits2, _, _, _, _ = model._kvn_backbone_forward(
        hidden_states=hs2,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )

    # First block (slots 0..3) corresponds to anchor a=6
    assert torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-5)


def test_end_to_end_multipacking_document_isolation_in_draft_model():
    """Verify KvNDSparkDraftModel isolates packed documents across both fc shift and recurrent scan."""
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    torch.manual_seed(202)
    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=64,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        vocab_size=128,
    )
    cfg = _make_v2_speculator_config(tl_cfg, backbone_type="kvn", n_modes=32, block_size=4)
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_draft_weights(model)
    model.eval()

    seq_len = 32
    hs1 = torch.randn(1, seq_len, 3 * 64)
    hs2 = hs1.clone()
    hs2[:, :16, :] = hs2[:, :16, :] * 1000.0 + 250.0

    ids1 = torch.randint(0, 128, (1, seq_len))
    ids2 = ids1.clone()
    ids2[:, :16] = (ids1[:, :16] + 37) % 128

    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    loss_mask[0, 16] = True  # Anchor right at the first token of doc 1!
    loss_mask[0, 22] = True
    verifier_last_hs = torch.randn(1, seq_len, 64)
    document_ids = torch.cat(
        [torch.zeros(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.long)],
        dim=1,
    )

    _, logits1, _, _, _, _ = model._kvn_backbone_forward(
        hidden_states=hs1,
        input_ids=ids1,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )
    _, logits2, _, _, _, _ = model._kvn_backbone_forward(
        hidden_states=hs2,
        input_ids=ids2,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )
    assert torch.allclose(logits1, logits2, atol=1e-5)


def test_multilayer_rollout_and_intrablock_mixer_wiring_and_pruning():
    """Verify multi-layer modal rollout + IntraBlockCausalMixer zero-error pruning invariance."""
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    torch.manual_seed(303)
    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=64,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        vocab_size=128,
    )
    cfg = _make_v2_speculator_config(
        tl_cfg,
        backbone_type="kvn",
        n_modes=32,
        block_size=4,
        kvn_multilayer_rollout=True,
        kvn_intrablock_mixer=True,
    )
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_draft_weights(model)
    model.eval()

    assert model.intrablock_mixer is not None
    assert model.rollout is not None and model.rollout.layer_mode_logits is not None

    with torch.no_grad():
        for layer in model.kvn_layers:
            m = layer.n_modes
            layer.readout_linear.weight[:, 16:m] = 0.0
            layer.readout_linear.weight[:, m + 16 : 2 * m] = 0.0
            layer.readout_linear.weight[:, 2 * m + 16 : 3 * m] = 0.0
        m = model.rollout.n_modes
        model.rollout.obs_linear.weight[:, 16:m] = 0.0
        model.rollout.obs_linear.weight[:, m + 16 : 2 * m] = 0.0
        model.rollout.obs_linear.weight[:, 2 * m + 16 : 3 * m] = 0.0
        assert model.born_confidence_head is not None
        assert model.born_confidence_head.born_proj is not None
        model.born_confidence_head.born_proj.weight[:, 16:m] = 0.0

    seq_len = 20
    hs = torch.randn(1, seq_len, 3 * 64)
    ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    loss_mask[0, 5] = True
    loss_mask[0, 12] = True
    v_hs = torch.randn(1, seq_len, 64)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, logits_before, _, _, _, _ = model._kvn_backbone_forward(
        hs, ids, loss_mask, v_hs, doc_ids, max_anchors=2
    )
    _, loss_before, _ = model(hs, ids, loss_mask, v_hs, doc_ids, max_anchors=2)

    model.prune_kvn_modes(16)
    assert model.config.kvn_n_modes == 16

    _, logits_after, _, _, _, _ = model._kvn_backbone_forward(
        hs, ids, loss_mask, v_hs, doc_ids, max_anchors=2
    )
    _, loss_after, _ = model(hs, ids, loss_mask, v_hs, doc_ids, max_anchors=2)

    assert torch.allclose(logits_before, logits_after, atol=1e-5)
    assert torch.allclose(loss_before, loss_after, atol=1e-4)


def test_advance_and_draft_step_matches_prefill_and_fullgraph_compiles():
    """Verify static-shape vLLM advance_and_draft_step matches prefill rollout and compiles fullgraph."""
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    torch.manual_seed(404)
    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=64,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        vocab_size=128,
    )
    cfg = _make_v2_speculator_config(
        tl_cfg,
        backbone_type="kvn",
        n_modes=16,
        block_size=4,
        kvn_state_dtype="float32",
    )
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_draft_weights(model)
    model.eval()

    seq_len = 12
    hs = torch.randn(1, seq_len, 3 * 64)
    ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    loss_mask[0, 6] = True  # Anchor at position 6
    v_hs = torch.randn(1, seq_len, 64)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)

    # 1. Full sequence forward up to anchor a=6 -> extract greedy autoregressive block tokens
    hidden, base_logits, _, _, _, born_feat = model._kvn_backbone_forward(
        hs, ids, loss_mask, v_hs, doc_ids, max_anchors=1
    )
    ref_tokens, ref_conf = model._decode_draft_block_greedy(
        hidden_blocks=hidden.view(1, 4, 64),
        base_logits=base_logits.view(1, 4, 128),
        anchor_token_ids=ids[:, 6],
        born_features=born_feat,
    )

    # 2. Incremental static-shape vLLM execution in 2 chunks: [0..4] (len=5) then [5..6] (len=2, padded to S_max=5)
    fc_all = model.hidden_norm(model.fc(hs))
    fc_shifted = torch.zeros_like(fc_all)
    fc_shifted[:, 1:, :] = fc_all[:, :-1, :]

    state_cache = model.allocate_state_cache(batch_size=1, device=hs.device)
    assert state_cache.shape == (3, 2, 1, 16)

    # Chunk 1: tokens 0..4 (valid_lens = 5)
    model.advance_and_draft_step(
        prev_fc_block=fc_shifted[:, :5, :],
        token_ids_block=ids[:, :5],
        valid_lens=torch.tensor([5], dtype=torch.long),
        state_cache=state_cache,
        reset_seq_mask=torch.tensor([True]),
    )

    # Chunk 2: tokens 5..6 padded with garbage at indices 2..4 (valid_lens = 2)
    pad_fc = torch.randn(1, 3, 64) * 999.0
    pad_ids = torch.randint(0, 128, (1, 3))
    chunk2_fc = torch.cat([fc_shifted[:, 5:7, :], pad_fc], dim=1)
    chunk2_ids = torch.cat([ids[:, 5:7], pad_ids], dim=1)

    compiled_step = torch.compile(
        model.advance_and_draft_step, backend="eager", fullgraph=True
    )
    inc_tokens, inc_conf = compiled_step(
        prev_fc_block=chunk2_fc,
        token_ids_block=chunk2_ids,
        valid_lens=torch.tensor([2], dtype=torch.long),
        state_cache=state_cache,
        reset_seq_mask=torch.tensor([False]),
    )

    assert torch.equal(inc_tokens, ref_tokens)
    assert torch.allclose(inc_conf, ref_conf, atol=1e-4)


def test_lasso_weight_trainer_kwargs_override_and_eval_iso_loss():
    """Verify --kvn-lasso-weight override in get_trainer_kwargs, zero LASSO in eval, and prune_kvn_modes reset."""
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    torch.manual_seed(505)
    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        vocab_size=128,
    )
    cfg = _make_v2_speculator_config(
        tl_cfg,
        backbone_type="kvn",
        n_modes=32,
        block_size=4,
        kvn_lasso_weight=3e-4,
    )
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_draft_weights(model)

    train_kw, val_kw = KvNDSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="eager",
        kvn_lasso_weight=0.0,
    )
    assert train_kw["kvn_lasso_weight"] == pytest.approx(0.0)
    assert val_kw["kvn_lasso_weight"] == pytest.approx(0.0)

    train_kw_default, val_kw_default = KvNDSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="eager",
    )
    assert train_kw_default["kvn_lasso_weight"] is None
    assert val_kw_default["kvn_lasso_weight"] == pytest.approx(0.0)

    seq_len = 16
    hs = torch.randn(1, seq_len, 3 * 64)
    ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    loss_mask[0, 5] = True
    v_hs = torch.randn(1, seq_len, 64)
    doc_ids = torch.zeros(1, seq_len, dtype=torch.long)

    model.train()
    _, loss_with_lasso, metrics_lasso = model(
        hs, ids, loss_mask, v_hs, doc_ids, max_anchors=1
    )
    _, loss_override_zero, metrics_zero = model(
        hs, ids, loss_mask, v_hs, doc_ids, max_anchors=1, kvn_lasso_weight=0.0
    )
    assert loss_with_lasso.item() > loss_override_zero.item() + 1e-6
    assert "lasso_loss_sum" in metrics_lasso
    assert metrics_zero["lasso_loss_sum"].item() == pytest.approx(0.0)

    model.eval()
    _, loss_eval, _ = model(hs, ids, loss_mask, v_hs, doc_ids, max_anchors=1)
    assert torch.allclose(loss_eval, loss_override_zero, atol=1e-6)

    model.prune_kvn_modes(16)
    assert model.config.kvn_n_modes == 16
    assert model.config.kvn_lasso_weight == pytest.approx(0.0)


def test_config_roundtrip_and_slim_checkpoint_key_ownership(tmp_path):
    """Verify KvNDSparkSpeculatorConfig registry round-trip and verifier weight omission on save."""
    from safetensors.torch import load_file
    from speculators import SpeculatorModel, SpeculatorModelConfig
    from speculators.models.kvn_dspark.core import KvNDSparkDraftModel

    tl_cfg = _make_qwen3_8b_like_tl_config(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        vocab_size=64,
    )
    cfg_reduced = _make_v2_speculator_config(
        tl_cfg, backbone_type="kvn", n_modes=16, block_size=4
    )
    cfg_reduced.draft_vocab_size = 32

    for data in (cfg_reduced.to_dict(), cfg_reduced.to_diff_dict()):
        restored = SpeculatorModelConfig.from_dict(data)
        assert type(restored) is KvNDSparkSpeculatorConfig
        assert (
            SpeculatorModel.registered_model_class_from_config(restored)
            is KvNDSparkDraftModel
        )

    model_red = KvNDSparkDraftModel(cfg_reduced)
    _init_dummy_draft_weights(model_red)
    t2d = torch.zeros(64, dtype=torch.bool)
    t2d[:32] = True
    d2t = torch.zeros(32, dtype=torch.long)
    model_red.load_vocab_mappings(t2d, d2t)
    red_dir = tmp_path / "reduced"
    model_red.save_pretrained(red_dir)
    red_keys = set(load_file(red_dir / "model.safetensors"))

    omitted = {"embed_tokens.weight", "verifier_lm_head.weight", "verifier_norm.weight"}
    assert omitted.isdisjoint(red_keys)
    assert "lm_head.weight" in red_keys
    assert {"t2d", "d2t"}.issubset(red_keys)

