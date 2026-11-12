"""Comprehensive unit tests for KvN-DSpark v3 training, OOV-exact calibration, DEAO,
optimizer routing, proximal Group-LASSO, and zero-leak multi-turn data pipeline.
"""

# ruff: noqa: PLC0415, PT018

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset as HFDataset
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.kvn_dspark.config import KvNDSparkSpeculatorConfig
from speculators.models.kvn_dspark.core import KvNDSparkDraftModel
from speculators.models.kvn_dspark.losses import (
    compute_kvn_metrics,
    deao_loss_decay,
    kvn_tv_loss,
    resolve_kvn_loss_config,
)
from speculators.models.kvn_dspark.optimizers import (
    BlockwiseMuon,
    SinkhornAdamW,
    build_kvn_optimizers,
    infer_muon_block_partition,
    sinkhorn_normalize_2d,
)
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train import cli as train_cli
from speculators.train.config import TrainConfig
from speculators.train.data import ArrowDataset
from speculators.train.optimizers import (
    build_optimizers,
    split_named_params_for_muon,
)
from speculators.train.trainer import Trainer, TrainerConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(name: str, rel_path: str):
    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_qwen3_config(vocab_size: int = 128, hidden_size: int = 32) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
    )


# ---------------------------------------------------------------------------
# 1. [C2] & [M6] Decontaminated Multi-Turn Zero-Leak Split + <think> Filter + Lock Fix
# ---------------------------------------------------------------------------


def test_decontaminated_mix_unique_conversations_think_filter_and_zero_leak_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    decontam = _load_script_module(
        "build_decontaminated_mix_v3", "scripts/build_decontaminated_mix.py"
    )

    # 5 source multi-turn conversations:
    # - conv 0: 2 turns, turn 2 truncated via finish_reason="length" -> keeps turn 1
    # only
    # - conv 1: 1 turn with unclosed <think> -> completely dropped (0 rows, 0 accepted
    # convs)
    # - conv 2, 3, 4: 2 clean turns each -> 2 sibling rows each sharing conversation_id
    source_convs = [
        [
            {"role": "user", "content": "Conv0 turn1 prompt"},
            {"role": "assistant", "content": "orig0_1"},
            {"role": "user", "content": "Conv0 turn2 prompt"},
            {"role": "assistant", "content": "orig0_2"},
        ],
        [
            {"role": "user", "content": "Conv1 unclosed think prompt"},
            {"role": "assistant", "content": "orig1_1"},
        ],
        [
            {"role": "user", "content": "Conv2 turn1 prompt"},
            {"role": "assistant", "content": "orig2_1"},
            {"role": "user", "content": "Conv2 turn2 prompt"},
            {"role": "assistant", "content": "orig2_2"},
        ],
        [
            {"role": "user", "content": "Conv3 turn1 prompt"},
            {"role": "assistant", "content": "orig3_1"},
            {"role": "user", "content": "Conv3 turn2 prompt"},
            {"role": "assistant", "content": "orig3_2"},
        ],
        [
            {"role": "user", "content": "Conv4 turn1 prompt"},
            {"role": "assistant", "content": "orig4_1"},
            {"role": "user", "content": "Conv4 turn2 prompt"},
            {"role": "assistant", "content": "orig4_2"},
        ],
    ]

    def fake_iter_clean(domain: str, index, seed: int = 42):
        for c in source_convs:
            yield c, 0

    async def fake_post_chat(
        session, endpoint, model, messages, temperature, max_tokens, timeout_s=300.0
    ):
        last_user = messages[-1]["content"]
        if "Conv0 turn2" in last_user:
            return {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "<think>cut off mid reasoning"},
                        "token_ids": [3, 4],
                    }
                ],
            }
        if "Conv1 unclosed think" in last_user:
            return {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<think>never closed thought block"},
                        "token_ids": [5, 6],
                    }
                ],
            }
        return {
            "prompt_token_ids": [10, 11],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": f"<think>closed</think> Clean answer for {last_user}"
                    },
                    "token_ids": [12, 13, 14],
                }
            ],
        }

    monkeypatch.setattr(
        decontam,
        "load_benchmark_index",
        lambda *a, **k: decontam.BenchmarkContaminationIndex(),
    )
    monkeypatch.setattr(decontam, "_iter_clean_domain_conversations", fake_iter_clean)
    monkeypatch.setattr(decontam, "_post_chat_completion_async", fake_post_chat)

    train_out = tmp_path / "train_mix.jsonl"
    val_out = tmp_path / "val_mix.jsonl"
    # Request 4 unique conversations (conv0, conv2, conv3, conv4 -> 1 + 2 + 2 + 2 = 7
    # rows!)
    decontam.main(
        [
            "--ultrachat-samples",
            "4",
            "--openmath-samples",
            "0",
            "--magicoder-samples",
            "0",
            "--vllm-endpoint",
            "http://localhost:8000/v1",
            "--output-jsonl",
            str(train_out),
            "--val-output-jsonl",
            str(val_out),
            "--val-samples",
            "2",
            "--seed",
            "7",
        ]
    )

    train_rows = [
        json.loads(line)
        for line in train_out.read_text(encoding="utf-8").splitlines()
        if line
    ]
    val_rows = [
        json.loads(line)
        for line in val_out.read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = json.loads(
        train_out.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )

    train_cids = {r["conversation_id"] for r in train_rows}
    val_cids = {r["conversation_id"] for r in val_rows}

    # 1) Exact 4 unique conversations accepted (7 total turn-rows > 4!)
    assert len(train_cids) + len(val_cids) == 4
    assert len(train_rows) + len(val_rows) == 7
    assert manifest["truncated_generations_rejected"] == 2
    assert manifest["unique_conversations_val"] == 2
    assert manifest["unique_conversations_train"] == 2

    # 2) ZERO sibling prefix leakage across train and val!
    assert train_cids.isdisjoint(val_cids)


def test_prepare_arrow_dataset_zero_sibling_leak_and_train_only_token_freq(
    tmp_path: Path,
):
    decontam = _load_script_module(
        "build_decontaminated_mix_v3_arrow", "scripts/build_decontaminated_mix.py"
    )

    # 3 multi-turn conversations (2 turns each = 6 rows).
    # Plant token ID 999/998 ONLY in one conversation and verify whichever conversation
    # lands in the held-out validation split never leaks into token_freq.pt!
    raw = HFDataset.from_dict(
        {
            "conversation_id": ["c_a", "c_a", "c_b", "c_b", "c_c", "c_c"],
            "input_ids": [
                [1, 2, 10],
                [1, 2, 10, 3, 11],
                [1, 4, 12],
                [1, 4, 12, 5, 13],
                [1, 6, 999],
                [1, 6, 999, 7, 998],
            ],
            "loss_mask": [
                [0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 1],
                [0, 0, 0, 0, 1],
            ],
        }
    )

    arrow_dir = tmp_path / "arrow_ds"
    freq_path = tmp_path / "token_freq.pt"
    decontam.prepare_conversation_disjoint_arrow_dataset(
        raw,
        arrow_dir,
        seq_length=32,
        train_ratio=0.67,
        seed=42,
        token_freq_path=freq_path,
    )

    train_ds = ArrowDataset(arrow_dir, train_ratio=0.67, split="train")
    val_ds = ArrowDataset(arrow_dir, train_ratio=0.67, split="val")

    train_cids = set(train_ds.data.with_format(None)["conversation_id"])
    val_cids = set(val_ds.data.with_format(None)["conversation_id"])
    assert len(train_ds) == 4
    assert len(val_ds) == 2
    assert train_cids.isdisjoint(val_cids)

    # Verify token_freq.pt was computed strictly on train_ds (no val-exclusive tokens!)
    freq_dict = torch.load(freq_path, weights_only=True)
    val_supervised_tokens: set[int] = set()
    train_supervised_tokens: set[int] = set()
    for i in range(len(train_ds)):
        rec = train_ds[i]
        train_supervised_tokens.update(rec.input_ids[rec.loss_mask].tolist())
    for i in range(len(val_ds)):
        rec = val_ds[i]
        val_supervised_tokens.update(rec.input_ids[rec.loss_mask].tolist())

    val_exclusive = val_supervised_tokens - train_supervised_tokens
    assert val_exclusive, (
        "Expected held-out validation conversation to have unique tokens"
    )
    for tok in val_exclusive:
        assert freq_dict.get(tok, 0) == 0, (
            f"Val-only token {tok} leaked into token_freq.pt!"
        )

    # Adversarial check: scattered conversation_ids AND straddling train_ratio (0.5 on
    # 3x2=6 rows -> s=3!)
    scattered = HFDataset.from_dict(
        {
            "conversation_id": ["cid_0", "cid_1", "cid_0", "cid_1", "cid_2", "cid_2"],
            "input_ids": [[1, 2]] * 6,
            "loss_mask": [[False, True]] * 6,
        }
    )
    scattered_dir = tmp_path / "scattered_arrow"
    decontam.prepare_conversation_disjoint_arrow_dataset(
        scattered,
        scattered_dir,
        seq_length=16,
        train_ratio=0.5,
        seed=7,
    )

    sc_train = ArrowDataset(scattered_dir, train_ratio=0.5, split="train")
    sc_val = ArrowDataset(scattered_dir, train_ratio=0.5, split="val")
    sc_train_cids = set(sc_train.data.with_format(None)["conversation_id"])
    sc_val_cids = set(sc_val.data.with_format(None)["conversation_id"])
    assert sc_train_cids.isdisjoint(sc_val_cids)


# ---------------------------------------------------------------------------
# 2. [C4] Full-Vocab OOV Mass (M_in), c_star <= 0.20, & Non-Persistent Full Head
# ---------------------------------------------------------------------------


def test_oov_80_percent_shortlist_c_star_exactness_and_checkpoint_cleanliness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vocab_full = 64
    vocab_draft = 8
    hidden = 16
    qwen_cfg = _tiny_qwen3_config(vocab_size=vocab_full, hidden_size=hidden)

    spec_cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=qwen_cfg,
        draft_vocab_size=vocab_draft,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        markov_rank=8,
        kvn_n_modes=8,
        enable_confidence_head=True,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="dummy/verifier",
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    model = KvNDSparkDraftModel(spec_cfg)

    # Shortlist covers tokens 0..7; OOV tokens are 8..63.
    t2d = torch.zeros(vocab_full, dtype=torch.bool)
    t2d[:vocab_draft] = True
    d2t = torch.zeros(vocab_draft, dtype=torch.long)
    model.load_vocab_mappings(t2d, d2t)

    full_lm_w = torch.zeros(vocab_full, hidden, dtype=torch.float32)
    full_lm_w[:, 0] = -60.0
    full_lm_w[0, 0] = float(torch.log(torch.tensor(0.20)).item())
    full_lm_w[8, 0] = float(torch.log(torch.tensor(0.80)).item())

    fake_verifier_layers = {
        "embed_tokens.weight": torch.randn(vocab_full, hidden),
        "lm_head.weight": full_lm_w,
        "model.norm.weight": torch.ones(hidden),
    }
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        lambda *_a, **_k: fake_verifier_layers,
    )
    model.load_verifier_weights()

    # 1) _full_verifier_lm_head_weight MUST NOT pollute state_dict or saved checkpoints!
    assert "_full_verifier_lm_head_weight" not in model.state_dict()
    assert not any("full_verifier" in k for k in model.state_dict())
    ckpt_dir = tmp_path / "ckpt_clean"
    model.save_pretrained(ckpt_dir)
    from safetensors import safe_open

    with safe_open(ckpt_dir / "model.safetensors", framework="pt") as sf:
        assert not any("full_verifier" in k for k in sf.keys())  # noqa: SIM118

    # 2) verifier_lm_head stays a standard nn.Linear (no Tensor monkey-patching!),
    # while model.compute_full_verifier_target_stats returns exact M_in == 0.20!
    h = torch.zeros(1, 4, hidden)
    h[:, :, 0] = 1.0
    targets = model.verifier_lm_head(h)
    assert not hasattr(targets, "_target_lse_full")
    assert not hasattr(targets, "_in_vocab_mass")
    stats = model.compute_full_verifier_target_stats(h, shortlist_logits=targets)
    assert stats is not None
    t_lse, in_vocab_mass, argmax_in_vocab = stats
    torch.testing.assert_close(
        in_vocab_mass,
        torch.full((1, 4), 0.20),
        atol=1e-5,
        rtol=1e-4,
    )
    assert not bool(argmax_in_vocab.any())

    # 3) Even if drafter puts 100% probability on shortlist argmax (token 0),
    # true acceptance rate alpha_k == 0.20 (NOT 1.00!), c_star <= 0.20, and greedy acc
    # == 0!
    draft_logits = torch.full((1, 4, vocab_draft), -50.0, requires_grad=True)
    with torch.no_grad():
        draft_logits[:, :, 0] = 20.0
    conf_logits = torch.zeros((1, 4), requires_grad=True)
    loss_mask = torch.ones((1, 4), dtype=torch.bool)
    loss_cfg = resolve_kvn_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")

    loss, metrics = compute_kvn_metrics(
        draft_logits,
        targets,
        conf_logits,
        loss_mask,
        block_size=4,
        loss_config=loss_cfg,
        tv_loss_fn=kvn_tv_loss,
        per_position_loss_weight="deao",
        target_lse_full=t_lse,
        in_vocab_mass=in_vocab_mass,
        target_argmax_in_vocab=argmax_in_vocab,
        loss_implementation="eager",
    )
    accept_mean = (metrics["accept_rate_sum"] / metrics["accept_rate_total"]).item()
    assert accept_mean <= 0.20 + 1e-5
    assert abs(accept_mean - 0.20) < 1e-4
    assert metrics["full_acc_sum"].item() == 0.0

    loss.backward()
    assert conf_logits.grad is not None
    assert bool((conf_logits.grad > 0).all())


# ---------------------------------------------------------------------------
# 3. Axe 2 — Direct Expected-Acceptance Optimization (DEAO) Survival Weights
# ---------------------------------------------------------------------------


def test_deao_survival_weights_cold_start_and_exact_jacobian():
    pos_idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.float32)
    mask = torch.ones((1, 4), dtype=torch.bool)

    # 1) Cold start (alpha = 0 everywhere): w_1* == 1.0 >= 1.0 (never vanishes!)
    w_cold = deao_loss_decay(
        pos_idx=pos_idx,
        loss_mask=mask,
        block_size=4,
        accept_rate=torch.zeros((1, 4)),
        sample_from_anchor=True,
    )
    torch.testing.assert_close(w_cold, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    # 2) Bottleneck alpha = [0.15, 0.75, 0.75, 0.75]: verify against autograd Jacobian
    # of E[tau]!
    alpha = torch.tensor([[0.15, 0.75, 0.75, 0.75]], requires_grad=True)
    expected_tau = 1.0 + torch.cumprod(alpha, dim=-1).sum()
    (exact_grad,) = torch.autograd.grad(expected_tau, alpha)

    w_deao = deao_loss_decay(
        pos_idx=pos_idx,
        loss_mask=mask,
        block_size=4,
        accept_rate=alpha,
        sample_from_anchor=True,
    )
    assert w_deao[0, 0].item() > 2.7  # >= 1.0 and amplified by deterministic suffix!
    torch.testing.assert_close(w_deao, exact_grad, atol=1e-6, rtol=1e-6)

    # 3) Masked trailing slots contribute zero phantom suffix leverage
    partial_mask = torch.tensor([[True, True, False, False]])
    w_masked = deao_loss_decay(
        pos_idx=pos_idx,
        loss_mask=partial_mask,
        block_size=4,
        accept_rate=torch.tensor([[0.4, 0.8, 0.99, 0.99]]),
        sample_from_anchor=True,
    )
    torch.testing.assert_close(
        w_masked, torch.tensor([[1.8, 0.4, 0.0, 0.0]]), atol=1e-6, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# 4. [M3] & [C5] & Axe 1 Optimizer Routing, Decay Split, BlockwiseMuon & Proximal
# Group-LASSO
# ---------------------------------------------------------------------------


class _SyntheticKvNV3Module(nn.Module):
    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * d, d, bias=False)  # [D, 3*D] -> BlockwiseMuon (1, 3)
        self.qkv = nn.Linear(d, 3 * d, bias=False)  # [3*D, D] -> BlockwiseMuon (0, 3)
        self.mlp_up = nn.Linear(d, 2 * d, bias=False)  # -> Muon
        self.anchor_proj = nn.Linear(d, d, bias=False)  # -> AdamW decay
        self.dt_proj = nn.Linear(
            d, d, bias=True
        )  # weight -> AdamW decay, bias -> no_decay
        self.kraus_proj = nn.Linear(d, d, bias=False)  # -> AdamW decay
        self.squeezing_proj = nn.Parameter(torch.randn(d, d))  # -> AdamW no_decay
        self.butterfly_angles = nn.Parameter(torch.randn(d, d))  # -> AdamW no_decay
        self.lindblad_gamma = nn.Parameter(torch.randn(d, d))  # -> AdamW no_decay
        self.omega_rff = nn.Parameter(torch.randn(2, d, d))  # -> AdamW no_decay
        self.phase_rff = nn.Parameter(torch.zeros(d))  # -> AdamW no_decay
        self.log_eta = nn.Parameter(torch.zeros(d))  # -> AdamW no_decay
        self.kvn_omega = nn.Parameter(torch.ones(d))  # -> AdamW no_decay
        self.dyad_theta = nn.Parameter(torch.zeros(d // 2))  # -> AdamW no_decay
        self.vacuum_log_kappa = nn.Parameter(torch.zeros(1))  # -> AdamW no_decay
        self.horizon_step = nn.Parameter(torch.ones(4, d))  # -> AdamW no_decay
        self.layer_mode_logits = nn.Parameter(torch.zeros(3, d))  # -> AdamW no_decay
        self.engram_table = nn.Parameter(
            torch.randn(32, d)
        )  # -> SinkhornAdamW engram group (wd=0)
        self.norm = nn.LayerNorm(d)


def test_muon_adamw_routing_decay_groups_blockwise_and_sinkhorn():
    mod = _SyntheticKvNV3Module(d=16)
    muon_named, adamw_named = split_named_params_for_muon(mod)
    muon_names = {n for n, _ in muon_named}
    adamw_names = {n for n, _ in adamw_named}

    assert muon_names == {"fc.weight", "qkv.weight", "mlp_up.weight"}
    for forbidden in (
        "anchor_proj.weight",
        "dt_proj.weight",
        "kraus_proj.weight",
        "squeezing_proj",
        "butterfly_angles",
        "lindblad_gamma",
        "engram_table",
        "omega_rff",
    ):
        assert forbidden in adamw_names
        assert forbidden not in muon_names

    assert infer_muon_block_partition("fc.weight", mod.fc.weight.shape) == (1, 3)
    assert infer_muon_block_partition("qkv.weight", mod.qkv.weight.shape) == (0, 3)

    cfg = SimpleNamespace(
        optimizer="muon",
        optimizer_implementation="eager",
        lr=1e-3,
        weight_decay=0.05,
        muon_lr=2e-2,
        muon_momentum=0.95,
        muon_weight_decay=0.1,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    )
    opts = build_kvn_optimizers(mod, cfg)
    assert len(opts) == 2
    muon_opt, adamw_opt = opts
    assert isinstance(muon_opt, BlockwiseMuon)
    assert isinstance(adamw_opt, SinkhornAdamW)
    assert len(muon_opt._block_partitions) == 2

    # Non-KvN modules routed through upstream build_optimizers keep standard Muon +
    # AdamW!
    upstream_opts = build_optimizers(mod, cfg)
    assert type(upstream_opts[0]) is torch.optim.Muon
    assert type(upstream_opts[1]) is torch.optim.AdamW

    # Verify decay vs no_decay vs engram groups inside AdamW
    wd_by_param_id = {
        id(p): grp["weight_decay"]
        for grp in adamw_opt.param_groups
        for p in grp["params"]
    }
    assert wd_by_param_id[id(mod.anchor_proj.weight)] == 0.05
    assert wd_by_param_id[id(mod.dt_proj.weight)] == 0.05
    assert wd_by_param_id[id(mod.kraus_proj.weight)] == 0.05
    for no_wd_param in (
        mod.dt_proj.bias,
        mod.squeezing_proj,
        mod.butterfly_angles,
        mod.lindblad_gamma,
        mod.omega_rff,
        mod.phase_rff,
        mod.log_eta,
        mod.kvn_omega,
        mod.dyad_theta,
        mod.vacuum_log_kappa,
        mod.horizon_step,
        mod.layer_mode_logits,
        mod.engram_table,
        mod.norm.weight,
    ):
        assert wd_by_param_id[id(no_wd_param)] == 0.0

    # Verify Sinkhorn equilibration preserves exact zero on unvisited rows of
    # engram_table!
    mod.engram_table.data.fill_(1.0)
    for p in mod.parameters():
        p.grad = torch.randn_like(p) * 0.01
    sparse_grad = torch.zeros_like(mod.engram_table)
    sparse_grad[2] = torch.linspace(0.1, 5.0, 16)
    sparse_grad[7] = torch.linspace(10.0, 0.2, 16)
    mod.engram_table.grad = sparse_grad.clone()

    eq = sinkhorn_normalize_2d(sparse_grad, num_iters=5)
    assert torch.all(eq[0] == 0.0)
    active_row_rms = eq[[2, 7]].square().mean(dim=-1).sqrt()
    torch.testing.assert_close(
        active_row_rms[0], active_row_rms[1], rtol=1e-2, atol=1e-3
    )

    unvisited_before = mod.engram_table.data[0].clone()
    muon_opt.step()
    adamw_opt.step()
    # Unvisited row 0 has 0 grad AND weight_decay=0.0 => ZERO drift!
    torch.testing.assert_close(mod.engram_table.data[0], unvisited_before)


def test_trainer_proximal_dyadic_group_lasso_zeros_weak_dyads_under_muon_adamw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    qwen_cfg = _tiny_qwen3_config(vocab_size=32, hidden_size=16)
    kvn_cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=qwen_cfg,
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        markov_rank=8,
        kvn_n_modes=8,
        kvn_reupload_dim=16,
        kvn_anchor_rank=8,
        kvn_lasso_weight=2.5,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    model = KvNDSparkDraftModel(kvn_cfg)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    nn.init.normal_(model.verifier_lm_head.weight, std=0.02)

    # Dyad 0 (modes 0, 1) stays weak; dyads 1..3 (modes 2..7) carry strong initial
    # amplitude.
    for layer in model.kvn_layers:
        m = layer.n_modes
        nn.init.normal_(layer.readout_linear.weight, std=1.2)
        with torch.no_grad():
            weak_cols = [0, 1, m, m + 1, 2 * m, 2 * m + 1]
            layer.readout_linear.weight[:, weak_cols] *= 1e-4
            layer.in_gain_q[:2].fill_(1e-6)
            layer.in_gain_p[:2].fill_(1e-6)
            if layer.butterfly_couple_gate is not None:
                layer.butterfly_couple_gate[:2].fill_(1e-6)
            if layer.lindblad_gain is not None:
                layer.lindblad_gain[:2].fill_(1e-6)

    forward_lasso_seen: list[float | None] = []
    orig_total_lasso = model.total_lasso_penalty

    def spy_total_lasso(lasso_weight: float | None = None) -> Tensor:
        forward_lasso_seen.append(lasso_weight)
        return orig_total_lasso(lasso_weight=lasso_weight)

    monkeypatch.setattr(model, "total_lasso_penalty", spy_total_lasso)

    prox_calls: list[float] = []
    orig_prox = model.apply_proximal_dyadic_group_lasso

    def wrapped_prox(step_lambda: float):
        prox_calls.append(step_lambda)
        return orig_prox(step_lambda)

    monkeypatch.setattr(model, "apply_proximal_dyadic_group_lasso", wrapped_prox)

    seq_len = 16
    fake_batches = [
        {
            "hidden_states": torch.randn(1, seq_len, 3 * 16),
            "input_ids": torch.randint(0, 32, (1, seq_len)),
            "verifier_last_hidden_states": torch.randn(1, seq_len, 16),
            "loss_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "position_ids": torch.arange(seq_len).unsqueeze(0),
            "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "error_records": 0,
        }
        for _ in range(12)
    ]

    train_kw, val_kw = KvNDSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="eager",
        per_position_loss_weight="deao",
        max_anchors=2,
        kvn_lasso_weight=2.5,
    )

    t_cfg = TrainerConfig(
        lr=1e-2,
        num_epochs=1,
        save_path=str(tmp_path / "prox_ckpt"),
        train_call_kwargs=train_kw,
        val_call_kwargs=val_kw,
        optimizer="muon",
        optimizer_implementation="eager",
        muon_lr=2e-2,
        scheduler_type="none",
        hidden_states_dtype=torch.float32,
        kvn_lasso_weight=2.5,
    )
    loader = DataLoader(fake_batches, batch_size=None)
    trainer = Trainer(model, t_cfg, loader, val_loader=None)
    trainer.train_epoch(0)

    # 1) Proximal operator ran on every step with step_lambda == lr * kvn_lasso_weight
    assert len(prox_calls) == 12
    assert all(abs(lam - 1e-2 * 2.5) < 1e-9 for lam in prox_calls)

    # 2) Non-proximal backward LASSO penalty was suppressed (0.0) on every step!
    assert all(w == 0.0 for w in forward_lasso_seen)

    # 3) Weak dyad 0 was driven to EXACT IEEE-754 0.0 across layers & heads
    # simultaneously!
    col_norms = model.kvn_layers[0].readout_linear.weight.detach().norm(dim=0)
    m = model.kvn_layers[0].n_modes
    weak_norms = col_norms[[0, 1, m, m + 1, 2 * m, 2 * m + 1]]
    strong_norms = col_norms[[2, 3, 4, 5, 6, 7]]
    assert bool((weak_norms == 0.0).all()), (
        f"Expected weak dyad exact 0.0, got {weak_norms}"
    )
    assert bool((strong_norms > 0.1).all()), (
        f"Expected strong dyads active, got {strong_norms}"
    )
    if model.rollout is not None:
        rollout_weak = model.rollout.obs_linear.weight.detach()[
            :, [0, 1, m, m + 1, 2 * m, 2 * m + 1]
        ]
        assert bool((rollout_weak == 0.0).all())


# ---------------------------------------------------------------------------
# 5. KvN-DSpark v3 CLI & Schema Resolution
# ---------------------------------------------------------------------------


def test_kvn_v3_cli_flags_and_build_draft_model_aliases(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy/qwen3",
            "--speculator-type",
            "kvn_dspark",
            "--loss-fn",
            '{"ce": 0.1, "tv": 0.9}',
            "--loss-decay",
            "deao",
            "--no-kvn-dyadic-rff",
            "--kvn-use-engram",
            "--kvn-engram-buckets",
            "32768",
            "--kvn-engram-dim",
            "64",
            "--kvn-symplectic-butterfly",
            "--kvn-squeezing-max-ratio",
            "0.75",
            "--kvn-lindblad-purity",
            "--kvn-use-kraus-tree",
            "--kvn-kraus-top-c",
            "128",
            "--kvn-self-cond-prob",
            "0.25",
            "--no-kvn-use-triton-scan",
        ]
    )
    assert cfg.kvn_dspark.kvn_state_dtype == "float32"
    assert cfg.loss.loss_decay == "deao"
    assert cfg.dflash.per_position_loss_weight == "deao"
    assert cfg.kvn_dspark.kvn_dyadic_rff is False
    assert cfg.kvn_dspark.kvn_use_engram is True
    assert cfg.kvn_dspark.kvn_engram_buckets == 32768
    assert cfg.kvn_dspark.kvn_engram_dim == 64
    assert cfg.kvn_dspark.kvn_symplectic_butterfly is True
    assert cfg.kvn_dspark.kvn_squeezing_max_ratio == 0.75
    assert cfg.kvn_dspark.kvn_lindblad_purity is True
    assert cfg.kvn_dspark.kvn_use_kraus_tree is True
    assert cfg.kvn_dspark.kvn_kraus_top_c == 128
    assert cfg.kvn_dspark.kvn_self_cond_prob == 0.25
    assert cfg.kvn_dspark.kvn_use_triton_scan is False

    # Round-trip flatten -> from_flat
    assert TrainConfig.from_flat(cfg.flatten()).model_dump() == cfg.model_dump()

    # Non-KvN speculators reject --loss-decay=deao cleanly
    with pytest.raises((ValueError, SystemExit)):
        TrainConfig.resolve(
            [
                "--verifier-name-or-path",
                "dummy/qwen3",
                "--speculator-type",
                "dspark",
                "--loss-decay",
                "deao",
            ]
        )

    # Verify unmodified train_cli.build_draft_model constructs KvNDSparkDraftModel with
    # all v3 flags
    qwen_cfg = _tiny_qwen3_config()
    monkeypatch.setattr(
        train_cli, "create_transformer_layer_config", lambda **_k: qwen_cfg
    )
    monkeypatch.setattr(train_cli, "resolve_mask_token_id", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        "speculators.models.dflash.core.resolve_target_layer_ids",
        lambda *_a, **_k: [0, 1],
    )
    monkeypatch.setattr(
        "speculators.config.VerifierConfig.from_pretrained",
        lambda *_a, **_k: VerifierConfig(
            name_or_path=None, architectures=["Qwen3ForCausalLM"]
        ),
    )

    args = argparse.Namespace(**cfg.flatten())
    built = train_cli.build_draft_model(args, KvNDSparkDraftModel, None, None, 128)
    assert isinstance(built, KvNDSparkDraftModel)
    assert built.config.sample_from_anchor is True
    assert built.config.dyadic_rff is False
    assert built.config.use_engram is True
    assert built.config.engram_buckets == 32768
    assert built.config.engram_dim == 64
    assert built.config.use_symplectic_butterfly is True
    assert built.config.squeezing_max_ratio == 0.75
    assert built.config.use_lindblad_purity is True
    assert built.config.use_kraus_tree is True
    assert built.config.kraus_top_c == 128
    assert built.config.self_cond_rollout_prob == 0.25
    assert built.config.use_triton_scan is False
