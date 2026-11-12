"""Unit tests for KvN-DSpark (Zero-KV-Cache Koopman-von Neumann RFF Speculator)."""

# ruff: noqa: PT018

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.kvn_dspark import (
    CausalKvNRFFLayer,
    KvNDSparkDraftModel,
    KvNDSparkSpeculatorConfig,
    MultiHorizonKoopmanRollout,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config.resolution import _ALGORITHM_GROUP_USERS
from speculators.train.config.schema import TrainConfig
from speculators.train.optimizers import split_named_params_for_muon


def _make_qwen3_tl_config(
    hidden_size: int = 64, num_layers: int = 3, vocab_size: int = 128
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=32,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * num_layers,
        _attn_implementation="eager",
    )


def _make_kvn_config(
    backbone_type: str = "kvn",
    n_modes: int = 32,
    block_size: int = 4,
    hidden_size: int = 64,
    num_layers: int = 3,
    vocab_size: int = 128,
) -> KvNDSparkSpeculatorConfig:
    tl_cfg = _make_qwen3_tl_config(
        hidden_size=hidden_size, num_layers=num_layers, vocab_size=vocab_size
    )
    return KvNDSparkSpeculatorConfig(
        transformer_layer_config=tl_cfg,
        draft_vocab_size=vocab_size,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=True,
        markov_rank=16,
        markov_head_type="gated",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        kvn_backbone_type=backbone_type,  # type: ignore[arg-type]
        kvn_n_modes=n_modes,
        kvn_reupload_layers=2,
        kvn_sigma_g=1.0,
        kvn_lasso_weight=1e-4,
        kvn_use_born_confidence=True,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )


def _init_dummy_verifier_weights(model: KvNDSparkDraftModel) -> None:
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.02)
        torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
        torch.nn.init.ones_(model.verifier_norm.weight)


def test_kvn_dspark_registration_and_train_config():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["kvn_dspark"] is KvNDSparkDraftModel

    cfg = TrainConfig(speculator_type="kvn_dspark")
    assert cfg.kvn_dspark.kvn_backbone_type == "kvn"
    assert cfg.kvn_dspark.kvn_n_modes == 256
    assert cfg.draft.num_layers == 5
    assert "kvn_dspark" in _ALGORITHM_GROUP_USERS["dflash"]
    assert "kvn_dspark" in _ALGORITHM_GROUP_USERS["dspark"]
    assert "kvn_dspark" in _ALGORITHM_GROUP_USERS["kvn_dspark"]

    flat = cfg.flatten()
    assert flat["kvn_n_modes"] == 256
    assert flat["kvn_backbone_type"] == "kvn"


def test_causal_kvn_rff_layer_and_lasso_pruning():
    torch.manual_seed(42)
    layer = CausalKvNRFFLayer(d_model=64, n_modes=64, reupload_layers_l=2)
    x = torch.randn(1, 24, 64)
    out, q_seq, p_seq, dtheta_seq = layer(x)

    assert out.shape == (1, 24, 64)
    assert q_seq.shape == (1, 24, 64)
    assert p_seq.shape == (1, 24, 64)
    assert dtheta_seq.shape == (1, 24, 64)

    lasso = layer.lasso_penalty()
    assert lasso.item() > 0.0

    top_idx = layer.prune_to_top_modes(16)
    assert top_idx.shape == (16,)
    assert layer.n_modes == 16
    out2, q2, p2, _ = layer(x)
    assert out2.shape == (1, 24, 64)
    assert q2.shape == (1, 24, 16)
    assert p2.shape == (1, 24, 16)


def test_multi_horizon_koopman_rollout_and_muon_routing():
    rollout = MultiHorizonKoopmanRollout(d_model=64, n_modes=32, block_size=4)
    h = torch.randn(5, 64)
    q = torch.randn(5, 32)
    p = torch.randn(5, 32)
    dtheta = torch.randn(5, 32)

    hidden_blocks, born_features = rollout(h, q, p, dtheta)
    assert hidden_blocks.shape == (5, 4, 64)
    assert born_features.shape == (5, 4, 32)

    _, adamw_params = split_named_params_for_muon(rollout)
    adamw_names = {n for n, _ in adamw_params}
    assert "horizon_freq_scale" in adamw_names
    assert "horizon_log_decay" in adamw_names
    assert "horizon_phase_bias" in adamw_names
    assert "anchor_proj.weight" in adamw_names
    assert "obs_linear.weight" in adamw_names


@pytest.mark.parametrize("backbone_type", ["kvn", "hybrid_1swa_2kvn", "swa"])
def test_kvn_dspark_draft_model_forward_backward_and_pruning(backbone_type: str):
    torch.manual_seed(7)
    cfg = _make_kvn_config(backbone_type=backbone_type, n_modes=32, block_size=4)
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_verifier_weights(model)

    seq_len = 32
    hidden_states = torch.randn(1, seq_len, 3 * 64)
    input_ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.ones(1, seq_len, dtype=torch.bool)
    verifier_last_hs = torch.randn(1, seq_len, 64)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=4,
    )
    assert torch.isfinite(loss)
    loss.backward()

    grad_count = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    assert grad_count > 0
    assert "accept_len_sum" in metrics
    assert "confidence_loss_sum" in metrics

    if backbone_type == "kvn":
        swa_cfg = _make_kvn_config(backbone_type="swa", n_modes=32)
        assert (
            cfg.estimated_state_bytes_per_seq()
            < swa_cfg.estimated_state_bytes_per_seq() * 0.1
        )
        model.prune_kvn_modes(16)
        assert model.config.kvn_n_modes == 16
        _, loss_after, _ = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hs,
            document_ids=document_ids,
            max_anchors=4,
        )
        assert torch.isfinite(loss_after)


def test_strict_causality_no_future_leakage():
    """Verify modifying tokens t > t0 has zero effect on outputs/states at t <= t0."""
    torch.manual_seed(123)
    layer = CausalKvNRFFLayer(d_model=64, n_modes=32, reupload_layers_l=2)
    layer.eval()

    x1 = torch.randn(1, 20, 64)
    x2 = x1.clone()
    t0 = 10
    # Mutate all future positions t > t0 drastically
    x2[:, t0 + 1 :, :] = torch.randn_like(x2[:, t0 + 1 :, :]) * 100.0

    out1, q1, p1, dtheta1 = layer(x1)
    out2, q2, p2, dtheta2 = layer(x2)

    assert torch.allclose(out1[:, : t0 + 1], out2[:, : t0 + 1], atol=1e-5)
    assert torch.allclose(q1[:, : t0 + 1], q2[:, : t0 + 1], atol=1e-5)
    assert torch.allclose(p1[:, : t0 + 1], p2[:, : t0 + 1], atol=1e-5)
    assert torch.allclose(dtheta1[:, : t0 + 1], dtheta2[:, : t0 + 1], atol=1e-5)


def test_fft_equals_recurrent_step_o1_state():
    """Verify parallel O(T log T) FFT dynamics match O(1) recurrent inference."""
    torch.manual_seed(99)
    layer = CausalKvNRFFLayer(d_model=64, n_modes=32, reupload_layers_l=2)
    layer.eval()

    x = torch.randn(2, 16, 64)
    fft_out, fft_q, fft_p, fft_dtheta = layer(x)

    state = None
    rec_outs, rec_qs, rec_ps, rec_dthetas = [], [], [], []
    for t in range(x.size(1)):
        x_step, state, dtheta_step = layer.recurrent_step(x[:, t : t + 1, :], state)
        rec_outs.append(x_step)
        rec_qs.append(state[0])
        rec_ps.append(state[1])
        rec_dthetas.append(dtheta_step)

    rec_out = torch.cat(rec_outs, dim=1)
    rec_q = torch.cat(rec_qs, dim=1)
    rec_p = torch.cat(rec_ps, dim=1)
    rec_dtheta = torch.cat(rec_dthetas, dim=1)

    assert torch.allclose(fft_out, rec_out, atol=1e-5)
    assert torch.allclose(fft_q, rec_q, atol=1e-5)
    assert torch.allclose(fft_p, rec_p, atol=1e-5)
    assert torch.allclose(fft_dtheta, rec_dtheta, atol=1e-5)


def test_exact_pruning_invariance_when_inactive_modes_zero():
    """Verify Barthe LASSO pruning theorem: zero-weight inactive modes prune with zero
    error.
    """
    torch.manual_seed(314)
    cfg = _make_kvn_config(backbone_type="kvn", n_modes=32, block_size=4, num_layers=2)
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_verifier_weights(model)
    model.eval()

    # Zero out readout/head columns for modes 16..31 across all layers, rollout, Born,
    # and Kraus heads
    with torch.no_grad():
        for layer in model.kvn_layers:
            m = layer.n_modes
            layer.readout_linear.weight[:, 16:m] = 0.0
            layer.readout_linear.weight[:, m + 16 : 2 * m] = 0.0
            layer.readout_linear.weight[:, 2 * m + 16 : 3 * m] = 0.0
            if layer.butterfly_couple_gate is not None:
                layer.butterfly_couple_gate[16:m] = 0.0
        assert model.rollout is not None
        m = model.rollout.n_modes
        model.rollout.obs_linear.weight[:, 16:m] = 0.0
        model.rollout.obs_linear.weight[:, m + 16 : 2 * m] = 0.0
        model.rollout.obs_linear.weight[:, 2 * m + 16 : 3 * m] = 0.0
        assert model.born_confidence_head is not None
        assert model.born_confidence_head.born_proj is not None
        model.born_confidence_head.born_proj.weight[:, 16:m] = 0.0
        if model.born_confidence_head.vacuum_pvac_proj is not None:
            model.born_confidence_head.vacuum_pvac_proj.weight[:, 8 : m // 2] = 0.0
        if model.born_confidence_head.lindblad_purity_proj is not None:
            model.born_confidence_head.lindblad_purity_proj.weight[:, 16:m] = 0.0
        if model.kraus_head is not None:
            model.kraus_head.kraus_mode_gate[16:m] = 0.0
            model.kraus_head.kraus_backaction_proj.weight[:, 16:m] = 0.0
            model.kraus_head.kraus_backaction_proj.weight[:, m + 16 : 2 * m] = 0.0

    seq_len = 24
    hidden_states = torch.randn(1, seq_len, 3 * 64)
    input_ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    # Fix exactly 2 deterministic anchors at positions 4 and 10 (predicting 5 and 11)
    loss_mask[0, 5] = True
    loss_mask[0, 11] = True
    verifier_last_hs = torch.randn(1, seq_len, 64)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, logits_before, _, _, _, born_before = model._kvn_backbone_forward(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )
    _, loss_before, _ = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )

    model.prune_kvn_modes(16)
    assert model.config.kvn_n_modes == 16

    _, logits_after, _, _, _, born_after = model._kvn_backbone_forward(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )
    _, loss_after, _ = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=2,
    )

    assert torch.allclose(logits_before, logits_after, atol=1e-5)
    assert torch.allclose(loss_before, loss_after, atol=1e-4)


def test_save_and_reload_pruned_checkpoint(tmp_path):
    """Verify pruned checkpoint saves to safetensors and reloads via from_pretrained."""
    torch.manual_seed(55)
    cfg = _make_kvn_config(backbone_type="kvn", n_modes=32, block_size=4, num_layers=2)
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_verifier_weights(model)
    model.prune_kvn_modes(16)

    ckpt_dir = tmp_path / "pruned_m16"
    model.save_pretrained(ckpt_dir)

    reloaded = KvNDSparkDraftModel.from_pretrained(ckpt_dir)
    _init_dummy_verifier_weights(reloaded)
    assert reloaded.config.kvn_n_modes == 16
    assert reloaded.kvn_layers[0].n_modes == 16
    assert reloaded.rollout is not None and reloaded.rollout.n_modes == 16


def test_torch_dynamo_eager_compile():
    """Verify torch._dynamo traces KvNDSparkDraftModel.forward cleanly for H200."""
    torch.manual_seed(77)
    cfg = _make_kvn_config(backbone_type="kvn", n_modes=16, block_size=4, num_layers=2)
    model = KvNDSparkDraftModel(cfg)
    _init_dummy_verifier_weights(model)

    raw_fn = getattr(
        KvNDSparkDraftModel.forward, "__wrapped__", KvNDSparkDraftModel.forward
    )
    compiled_fn = torch.compile(
        raw_fn.__get__(model, KvNDSparkDraftModel), backend="eager"
    )
    seq_len = 24
    hidden_states = torch.randn(1, seq_len, 3 * 64)
    input_ids = torch.randint(0, 128, (1, seq_len))
    loss_mask = torch.ones(1, seq_len, dtype=torch.bool)
    verifier_last_hs = torch.randn(1, seq_len, 64)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, loss, metrics = compiled_fn(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hs,
        document_ids=document_ids,
        max_anchors=3,
    )
    assert torch.isfinite(loss)
    loss.backward()
