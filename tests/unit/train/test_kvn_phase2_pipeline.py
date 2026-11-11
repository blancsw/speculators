"""TDD suite for KvN-DSpark Phase 2 data decontamination, calibration/Amdahl evaluation, and H200 CLI recipes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import pytest
import torch

from speculators.train.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(module_name: str, rel_path: str):
    script_path = REPO_ROOT / rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Missing expected Phase 2 script: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Phase B: 13-gram & SHA-256 Decontamination + Stratified Mix Builder
# --------------------------------------------------------------------------- #


def test_decontamination_index_catches_sha256_and_13gram_leaks():
    mod = _load_script_module(
        "build_decontaminated_mix", "scripts/build_decontaminated_mix.py"
    )
    idx = mod.BenchmarkContaminationIndex(ngram_size=13)
    idx.add_benchmark_text(
        "gsm8k",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
    )
    idx.add_benchmark_text(
        "humaneval",
        "def has_close_elements(numbers: List[float], threshold: float) -> bool: Check if in given list of numbers, are any two numbers closer to each other than given threshold.",
    )

    # 1. Exact prompt match (even with casing / extra spaces / punctuation differences) -> rejected
    exact_leak = [
        {
            "role": "user",
            "content": "  NATALIA sold clips to 48 of her friends in April,   and then she sold half as many clips in May! ",
        },
        {"role": "assistant", "content": "48 + 24 = 72."},
    ]
    leaked_exact, reason_exact = idx.is_contaminated(exact_leak)
    assert leaked_exact is True
    assert "gsm8k" in reason_exact

    # 2. Embedded 13-gram overlap inside assistant completion -> rejected
    ngram_leak = [
        {"role": "user", "content": "Give me a practice Python problem about floats."},
        {
            "role": "assistant",
            "content": "Sure! Try solving: Check if in given list of numbers, are any two numbers closer to each other than given threshold in O(n log n).",
        },
    ]
    leaked_ngram, reason_ngram = idx.is_contaminated(ngram_leak)
    assert leaked_ngram is True
    assert "13gram" in reason_ngram and "humaneval" in reason_ngram

    # 3. Clean domain conversation (< 13 shared consecutive words) -> accepted
    clean_conv = [
        {
            "role": "user",
            "content": "Derive the eigenvalues of a damped harmonic oscillator in state-space form.",
        },
        {
            "role": "assistant",
            "content": "For z_{t+1} = exp((-eta + i*omega)*dt) * z_t, the discrete poles have modulus exp(-eta*dt) < 1.",
        },
    ]
    leaked_clean, reason_clean = idx.is_contaminated(clean_conv)
    assert leaked_clean is False
    assert reason_clean == ""


def test_domain_normalizers_and_stratified_temperature_assignment():
    mod = _load_script_module(
        "build_decontaminated_mix", "scripts/build_decontaminated_mix.py"
    )

    uc = mod.normalize_domain_row(
        "ultrachat",
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
    )
    om = mod.normalize_domain_row(
        "openmath",
        {
            "problem": "Solve 2x + 3 = 11.",
            "generated_solution": "Subtract 3 to get 2x = 8, so x = 4.",
        },
    )
    mc = mod.normalize_domain_row(
        "magicoder",
        {
            "problem": "Implement binary search.",
            "solution": "def bsearch(a, x): ...",
        },
    )
    assert uc is not None and len(uc) == 2 and uc[0]["role"] == "user"
    assert om is not None and om[1]["role"] == "assistant" and "x = 4" in om[1]["content"]
    assert mc is not None and mc[0]["content"] == "Implement binary search."

    temps_uc = [mod.assign_sampling_temperature("ultrachat", i) for i in range(100)]
    temps_om = [mod.assign_sampling_temperature("openmath", i) for i in range(100)]
    temps_mc = [mod.assign_sampling_temperature("magicoder", i) for i in range(100)]
    assert temps_uc.count(0.0) == 50 and temps_uc.count(0.7) == 50
    assert temps_om.count(0.0) == 50 and temps_om.count(0.6) == 50
    assert temps_mc.count(0.0) == 60 and temps_mc.count(0.6) == 40

    parsed = mod.parse_args(["--output-jsonl", "dummy.jsonl"])
    assert (
        parsed.ultrachat_samples + parsed.openmath_samples + parsed.magicoder_samples
        == 100_000
    )
    assert parsed.val_samples == 5_000


def test_livecodebench_fallback_and_token_exact_on_policy_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json
    import datasets
    import huggingface_hub

    mod = _load_script_module(
        "build_decontaminated_mix", "scripts/build_decontaminated_mix.py"
    )

    # 1. Verify livecodebench script-less fallback when load_dataset rejects legacy scripts
    fake_lcb = tmp_path / "test_lcb.jsonl"
    fake_lcb.write_text(
        json.dumps(
            {
                "question_content": "Given a directed acyclic graph compute the longest topological path modulo prime p."
            }
        )
        + "\n",
        encoding="utf-8",
    )

    orig_load_dataset = datasets.load_dataset

    def _fake_load_dataset(path, *args, **kwargs):
        if path == "livecodebench/code_generation_lite":
            raise RuntimeError(
                "Dataset scripts are no longer supported, but found code_generation_lite.py"
            )
        return orig_load_dataset(path, *args, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, repo_type=None: ["test.jsonl", "README.md"],
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, repo_type=None: str(fake_lcb),
    )

    idx = mod.load_benchmark_index(["livecodebench"], ngram_size=13)
    leaked, reason = idx.is_contaminated(
        [
            {
                "role": "user",
                "content": "Given a directed acyclic graph compute the longest topological path modulo prime p.",
            }
        ]
    )
    assert leaked is True and "livecodebench" in reason

    # 2. Verify token-exact boundary rows (input_ids + loss_mask), exact target_count replacement
    #    after post-generation 13-gram rejection, and --resume idempotency
    fake_stream = [
        # Multi-turn conversation (2 assistant turns -> 2 boundary rows; 2nd turn leaks -> rejected!)
        {
            "messages": [
                {"role": "user", "content": "Explain damped harmonic oscillators."},
                {"role": "assistant", "content": "orig 1"},
                {"role": "user", "content": "Give a graph problem."},
                {"role": "assistant", "content": "orig 2"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Derive Bochner theorem for Gaussian kernels."},
                {"role": "assistant", "content": "orig 3"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Show how symplectic integrators conserve energy."},
                {"role": "assistant", "content": "orig 4"},
            ]
        },
    ]

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: datasets.Dataset.from_list(fake_stream),
    )

    call_counter = {"n": 0}

    async def _fake_post_chat_completion(
        session, endpoint, model, messages, temperature, max_tokens, timeout_s=300.0
    ):
        call_counter["n"] += 1
        last_user = messages[-1]["content"]
        if "graph problem" in last_user:
            # Post-generation 13-gram leak -> must be rejected and replaced!
            reply = "Try this: Given a directed acyclic graph compute the longest topological path modulo prime p."
            comp_ids = [21, 22, 23]
        else:
            reply = f"Clean on-policy response #{call_counter['n']}"
            comp_ids = [30 + call_counter["n"], 40 + call_counter["n"]]
        prompt_ids = [10, 11, 12]
        return {
            "prompt_token_ids": prompt_ids,
            "choices": [
                {
                    "message": {"role": "assistant", "content": reply},
                    "token_ids": comp_ids,
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(mod, "_post_chat_completion_async", _fake_post_chat_completion)

    out_jsonl = tmp_path / "mix.jsonl"
    mod.main(
        [
            "--verifier-model",
            "Qwen/Qwen3-8B",
            "--ultrachat-samples",
            "3",
            "--openmath-samples",
            "0",
            "--magicoder-samples",
            "0",
            "--exclude-benchmarks",
            "livecodebench",
            "--vllm-endpoint",
            "http://localhost:8000/v1",
            "--output-jsonl",
            str(out_jsonl),
            "--val-samples",
            "0",
            "--resume",
        ]
    )

    written = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Exact target_count = 3 clean boundary rows despite 1 post-gen 13-gram leak!
    assert len(written) == 3
    for row in written:
        assert "input_ids" in row and "loss_mask" in row
        assert len(row["input_ids"]) == len(row["loss_mask"]) == 5
        assert row["loss_mask"] == [0, 0, 0, 1, 1]

    # Re-running with --resume when partial cache already has 3 clean rows makes 0 new LLM calls
    prev_calls = call_counter["n"]
    out_jsonl_resume = tmp_path / "mix.jsonl"
    mod.main(
        [
            "--verifier-model",
            "Qwen/Qwen3-8B",
            "--ultrachat-samples",
            "3",
            "--openmath-samples",
            "0",
            "--magicoder-samples",
            "0",
            "--exclude-benchmarks",
            "livecodebench",
            "--vllm-endpoint",
            "http://localhost:8000/v1",
            "--output-jsonl",
            str(out_jsonl_resume),
            "--val-samples",
            "0",
            "--resume",
        ]
    )
    assert call_counter["n"] == prev_calls


# --------------------------------------------------------------------------- #
# Phase D: Calibration (ECE, AUROC, Dynamic Stopping Pareto) & Amdahl Metrics
# --------------------------------------------------------------------------- #


def test_ece_auroc_dynamic_stopping_and_amdahl_metrics():
    eval_mod = _load_script_module(
        "kvn_phase2_calibration_and_amdahl",
        "scripts/evaluate/kvn_phase2_calibration_and_amdahl.py",
    )

    # 1. Perfect calibration -> ECE == 0.0; overconfident wrong predictions -> high ECE
    probs_perfect = torch.tensor([0.0, 0.0, 1.0, 1.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert eval_mod.compute_ece(probs_perfect, labels, num_bins=10) == pytest.approx(
        0.0, abs=1e-6
    )

    probs_overconf = torch.tensor([0.9, 0.9, 0.9, 0.9])
    labels_half = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert eval_mod.compute_ece(probs_overconf, labels_half, num_bins=10) == pytest.approx(
        0.4, abs=1e-6
    )

    # 2. Rejection AUROC: perfect separation -> 1.0, random inverted -> 0.0
    reject_scores = torch.tensor([0.95, 0.85, 0.20, 0.10])
    reject_labels = torch.tensor([1, 1, 0, 0])
    assert eval_mod.compute_binary_auroc(reject_scores, reject_labels) == pytest.approx(
        1.0, abs=1e-6
    )

    # 3. Dynamic block stopping Pareto curve
    # Block 0: accepts first 3 tokens (k=0,1,2), rejects k=3..7; confidences drop below 0.5 at k=3
    # Block 1: accepts first 1 token (k=0), rejects k=1..7; confidences drop below 0.5 at k=1
    confidences = torch.tensor(
        [
            [0.95, 0.88, 0.76, 0.30, 0.20, 0.15, 0.10, 0.05],
            [0.91, 0.25, 0.18, 0.12, 0.09, 0.06, 0.04, 0.02],
        ]
    )
    accepted_flags = torch.tensor(
        [
            [True, True, True, False, True, False, False, False],
            [True, False, True, False, False, False, False, False],
        ]
    )
    pareto = eval_mod.compute_dynamic_stopping_pareto(
        confidences, accepted_flags, thresholds=[0.0, 0.5]
    )
    p_full = pareto[0]  # gamma=0.0 -> drafts all 8 slots, tau = 1 + (3 + 1)/2 = 3.0
    p_dyn = pareto[1]   # gamma=0.5 -> drafts 3 and 1 slots (mean 2.0), preserves exact tau = 3.0 while saving 75% draft/verify slots!
    assert p_full["mean_drafted_tokens"] == pytest.approx(8.0)
    assert p_full["mean_accept_length_tau"] == pytest.approx(3.0)
    assert p_dyn["mean_drafted_tokens"] == pytest.approx(2.0)
    assert p_dyn["mean_accept_length_tau"] == pytest.approx(3.0)
    assert p_dyn["verifier_slot_flops_saved_pct"] == pytest.approx(75.0)

    # 4. End-to-End Amdahl speedup metrics
    amdahl = eval_mod.compute_amdahl_e2e_metrics(
        baseline_tps=120.0,
        spec_tps=300.0,
        draft_latency_ms=1.8,
        verify_latency_ms=10.2,
        mean_accept_len=3.6,
    )
    assert amdahl["e2e_speedup"] == pytest.approx(2.5)
    assert amdahl["draft_overhead_fraction"] == pytest.approx(1.8 / 12.0)
    assert amdahl["amdahl_theoretical_ceiling"] == pytest.approx(3.6 / (1.0 + 1.8 / 10.2))


def test_evaluate_checkpoint_calibration_end_to_end_cpu(tmp_path: Path):
    import numpy as np
    from safetensors.torch import save_file
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.kvn_dspark import (
        KvNDSparkDraftModel,
        KvNDSparkSpeculatorConfig,
    )
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    eval_mod = _load_script_module(
        "kvn_phase2_calibration_and_amdahl",
        "scripts/evaluate/kvn_phase2_calibration_and_amdahl.py",
    )

    vocab_size = 64
    draft_vocab_size = 32
    hidden_size = 32
    layer_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        use_sliding_window=True,
        layer_types=["sliding_attention", "sliding_attention"],
        _attn_implementation="eager",
    )
    verifier_dir = tmp_path / "verifier"
    Qwen3ForCausalLM(layer_cfg).save_pretrained(verifier_dir)

    cfg = KvNDSparkSpeculatorConfig(
        transformer_layer_config=layer_cfg,
        draft_vocab_size=draft_vocab_size,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=16,
        markov_head_type="gated",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        kvn_backbone_type="kvn",
        kvn_n_modes=16,
        kvn_reupload_layers=2,
        kvn_reupload_dim=16,
        kvn_anchor_rank=16,
        kvn_use_born_confidence=True,
        speculators_config=SpeculatorsConfig(
            algorithm="kvn_dspark",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(verifier_dir), architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )

    d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[:draft_vocab_size] = True
    model = KvNDSparkDraftModel(config=cfg)
    model.load_vocab_mappings(t2d, d2t)
    model.load_verifier_weights()
    ckpt_dir = tmp_path / "ckpt"
    model.save_pretrained(ckpt_dir)

    data_dir = tmp_path / "data"
    hs_dir = data_dir / "hidden_states"
    hs_dir.mkdir(parents=True)
    np.save(data_dir / f"t2d-{draft_vocab_size}.npy", t2d.numpy())
    np.save(data_dir / f"d2t-{draft_vocab_size}.npy", d2t.numpy())

    seq_len = 24
    save_file(
        {
            "hidden_states": torch.randn(seq_len, 3, hidden_size, dtype=torch.bfloat16),
            "token_ids": torch.randint(0, draft_vocab_size, (seq_len,), dtype=torch.long),
        },
        str(hs_dir / "hs_0.safetensors"),
    )

    report = eval_mod.evaluate_checkpoint_calibration(
        checkpoint_dir=ckpt_dir,
        verifier_model=str(verifier_dir),
        data_dir=data_dir,
        draft_vocab_size=draft_vocab_size,
        max_samples=4,
        device="cpu",
    )
    assert report["num_evaluated_blocks"] > 0
    assert 0.0 <= report["overall_ece_15bin"] <= 1.0
    assert 0.0 <= report["overall_rejection_auroc"] <= 1.0
    assert len(report["dynamic_stopping_pareto"]) == 6


def test_online_arrow_calibration_fallback_and_exact_e2e_spec_decoding(tmp_path: Path):
    """Verify online-mode calibration (no hidden_states/ dir) and lossless E2E speculative decoding for kvn & swa."""
    from datasets import Dataset
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.kvn_dspark import (
        KvNDSparkDraftModel,
        KvNDSparkSpeculatorConfig,
    )
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    eval_mod = _load_script_module(
        "kvn_phase2_calibration_and_amdahl",
        "scripts/evaluate/kvn_phase2_calibration_and_amdahl.py",
    )

    torch.manual_seed(777)
    vocab_size = 48
    draft_vocab_size = 32
    hidden_size = 32
    layer_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        use_sliding_window=True,
        layer_types=["sliding_attention", "sliding_attention"],
        _attn_implementation="eager",
    )
    verifier = Qwen3ForCausalLM(layer_cfg).eval()
    verifier_dir = tmp_path / "verifier"
    verifier.save_pretrained(verifier_dir)

    # Save an online-mode Arrow dataset WITHOUT any data_dir/hidden_states/ directory
    online_data_dir = tmp_path / "online_arrow_data"
    rows = [
        {
            "input_ids": torch.randint(1, draft_vocab_size, (24,)).tolist(),
            "loss_mask": [False] * 4 + [True] * 20,
            "seq_len": 24,
        }
        for _ in range(10)
    ]
    Dataset.from_list(rows).with_format("torch").save_to_disk(str(online_data_dir))
    assert not (online_data_dir / "hidden_states").exists()

    d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[:draft_vocab_size] = True

    for backbone in ("kvn", "swa"):
        cfg = KvNDSparkSpeculatorConfig(
            transformer_layer_config=layer_cfg,
            draft_vocab_size=draft_vocab_size,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            markov_rank=16,
            markov_head_type="gated",
            enable_confidence_head=True,
            confidence_head_with_markov=True,
            kvn_backbone_type=backbone,  # type: ignore[arg-type]
            kvn_n_modes=16,
            kvn_reupload_layers=2,
            kvn_reupload_dim=16,
            kvn_anchor_rank=16,
            kvn_use_born_confidence=(backbone == "kvn"),
            speculators_config=SpeculatorsConfig(
                algorithm="kvn_dspark",
                proposal_methods=[GreedyTokenProposalConfig()],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=str(verifier_dir), architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
        model = KvNDSparkDraftModel(config=cfg)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        ckpt_dir = tmp_path / f"ckpt_{backbone}"
        model.save_pretrained(ckpt_dir)

        if backbone == "kvn":
            cal_report = eval_mod.evaluate_checkpoint_calibration(
                checkpoint_dir=ckpt_dir,
                verifier_model=str(verifier_dir),
                data_dir=online_data_dir,
                draft_vocab_size=draft_vocab_size,
                max_samples=2,
                train_data_ratio=0.8,
                device="cpu",
            )
            assert cal_report["num_evaluated_blocks"] > 0
            assert 0.0 <= cal_report["overall_ece_15bin"] <= 1.0

        prompt_ids = [[2, 5, 7, 11, 13], [3, 6, 9, 12, 15]]
        e2e_greedy = eval_mod.evaluate_checkpoint_e2e(
            checkpoint_dir=ckpt_dir,
            verifier_model=str(verifier_dir),
            prompt_token_ids_list=prompt_ids,
            max_new_tokens=10,
            temperature=0.0,
            batch_sizes=[1, 2],
            device="cpu",
        )
        assert e2e_greedy["exact_greedy_match_all_prompts"] is True
        assert e2e_greedy["mean_accept_len_tau"] >= 1.0
        assert "1" in e2e_greedy["amdahl_by_batch_size"]
        assert "2" in e2e_greedy["amdahl_by_batch_size"]
        assert e2e_greedy["amdahl_by_batch_size"]["1"]["e2e_speedup"] > 0.0

        if backbone == "kvn":
            e2e_stoch = eval_mod.evaluate_checkpoint_e2e(
                checkpoint_dir=ckpt_dir,
                verifier_model=str(verifier_dir),
                prompt_token_ids_list=prompt_ids[:1],
                max_new_tokens=8,
                temperature=0.6,
                batch_sizes=[1],
                device="cpu",
            )
            assert e2e_stoch["mean_accept_len_tau"] >= 1.0


# --------------------------------------------------------------------------- #
# Phase C: H200 Suite & Recipe CLI Resolution Guard (Zero ConfigError/Warnings)
# --------------------------------------------------------------------------- #


def _find_posix_bash() -> str | None:
    """Prefer Git Bash on Windows over the System32 WSL stub; use PATH bash on POSIX."""
    git_bash_candidates = (
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files (x86)/Git/bin/bash.exe"),
    )
    for cand in git_bash_candidates:
        if cand.exists():
            return str(cand)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


@pytest.mark.skipif(_find_posix_bash() is None, reason="requires POSIX bash")
def test_h200_shell_scripts_syntax_and_all_11_configs_resolve_cleanly():
    bash_bin = _find_posix_bash()
    assert bash_bin is not None

    shell_scripts = [
        REPO_ROOT / "examples" / "train" / "kvn_dspark_2xh200_online.sh",
        REPO_ROOT / "examples" / "train" / "kvn_dspark_phase2_h200_suite.sh",
        REPO_ROOT / "examples" / "evaluate" / "kvn_dspark_phase2_e2e_eval.sh",
    ]
    for sh_path in shell_scripts:
        assert sh_path.exists(), f"Missing {sh_path}"
        raw_bytes = sh_path.read_bytes()
        assert b"\r\n" not in raw_bytes, f"{sh_path.name} must use LF line endings"
        subprocess.run(
            [bash_bin, "-n"],
            input=sh_path.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=True,
        )

    # 1. Verify standalone 2xH200 online v2 recipe resolves with zero warnings
    from tests.unit.train.config.test_resolution import (
        _config_block,
        _train_invocation_args,
    )

    online_text = (
        REPO_ROOT / "examples" / "train" / "kvn_dspark_2xh200_online.sh"
    ).read_text(encoding="utf-8")
    assert "tutorial_regen" not in online_text
    assert "scripts/build_decontaminated_mix.py" in online_text
    online_lines = online_text.splitlines()
    online_probe = (
        "\n".join(_config_block(online_lines))
        + '\n__emit() { printf "%s\\n" "$@"; }\n__emit '
        + _train_invocation_args(online_lines)
        + "\n"
    )
    out_online = subprocess.run(
        [bash_bin, "-s"],
        input=online_probe,
        capture_output=True,
        text=True,
        check=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        online_flat = TrainConfig.resolve(out_online.stdout.splitlines()).flatten()
    assert online_flat["speculator_type"] == "kvn_dspark"
    assert online_flat["train_data_ratio"] == pytest.approx(0.95)
    assert online_flat["kvn_n_modes"] == 512
    assert online_flat["kvn_reupload_dim"] == 512
    assert online_flat["kvn_anchor_rank"] == 512
    assert online_flat["kvn_enable_interferometer"] is True
    assert online_flat["kvn_vacuum_born_norm"] is True
    assert online_flat["kvn_multilayer_rollout"] is True
    assert online_flat["kvn_intrablock_mixer"] is True
    assert online_flat["kvn_strict_causal_shift"] is True

    # 2. Verify all 9 orthogonal ablation models + post-prune recalibration in Phase 2 H200 suite
    suite_text = (
        REPO_ROOT / "examples" / "train" / "kvn_dspark_phase2_h200_suite.sh"
    ).read_text(encoding="utf-8")
    models = [
        "B1_DSpark_SWA128",
        "B2_DSpark_SWA512",
        "B3_Hybrid_1SWA_4KvN",
        "A0_KvN_v1_CausalFix",
        "A1_KvN_ModalRFF_Comp",
        "A2_KvN_Multilayer",
        "A3_KvN_IntraBlock",
        "A4_KvN_v2_Full128",
        "A5_KvN_v2_LASSO_512to128",
        "A5_KvN_v2_Pruned128_Recal",
    ]
    for model_id in models:
        probe = suite_text.replace(
            'if [[ "${1:-}" == "--dry-run-lib" ]]; then\n    return 0 2>/dev/null || exit 0\nfi',
            f'emit_train_argv "{model_id}" 42\nexit 0',
        )
        out = subprocess.run(
            [bash_bin, "-s"],
            input=probe,
            capture_output=True,
            text=True,
            check=True,
        )
        argv = [line for line in out.stdout.splitlines() if line]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = TrainConfig.resolve(argv)
        flat = cfg.flatten()
        assert flat["speculator_type"] == "kvn_dspark"
        assert flat["train_data_ratio"] == pytest.approx(0.95)
        if model_id == "A5_KvN_v2_Pruned128_Recal":
            assert flat["from_pretrained"] is not None
            assert flat["epochs"] == 1
            assert flat["kvn_lasso_weight"] == pytest.approx(0.0)
        else:
            assert flat["num_layers"] == 5
            assert flat["draft_vocab_size"] == 32768
            assert flat["epochs"] == 6
            if model_id == "A0_KvN_v1_CausalFix":
                assert flat["kvn_reupload_dim"] == 0

    # 3. Pre-flight ISO guards on vLLM --max-model-len (> SEQ_LENGTH) and Phase D eval script
    assert "VLLM_MAX_MODEL_LEN" in online_text
    assert '--max-model-len "$SEQ_LENGTH"' not in online_text
    assert "VLLM_MAX_MODEL_LEN" in suite_text
    assert '--max-model-len "$SEQ_LENGTH"' not in suite_text

    eval_text = (
        REPO_ROOT / "examples" / "evaluate" / "kvn_dspark_phase2_e2e_eval.sh"
    ).read_text(encoding="utf-8")
    assert "benchmark-e2e" in eval_text
    assert "calibrate" in eval_text
    assert 'if [[ -d "${DATA_DIR}/hidden_states" ]]; then' not in eval_text
    assert 'vllm serve "$model_path"' not in eval_text
