# Master Execution Runbook & Scientific Protocol — KvN-DSpark Phase 3 (`v3` Full Stack, 100k Zero-Leak On-Policy & Production `vLLM V1`)

> **AUTONOMOUS H200 CLUSTER HANDOFF (`Carbon` / Linux NVIDIA H200 SXM Node)**
>
> This document is the **definitive ICML/NeurIPS execution protocol and production serving runbook** for **`KvN-DSpark v3`** on branch `kvn-rff-rnn-mtp` (`Qwen/Qwen3-8B` primary 11-model orthogonal suite + `Qwen/Qwen3-32B` scale-transfer).
>
> - **100% IMPLEMENTED, AUDIT-HARDENED & LOCALLY GPU-VERIFIED (`[x]` — DO NOT RE-IMPLEMENT):**
>   1. **`KvN-DSpark v3` Architecture, Losses, Optimizers & Fused `Mega-KvN` Triton Kernels** (`src/speculators/models/kvn_dspark/{config.py,model_definitions.py,core.py,losses.py,optimizers.py,triton_kernels.py}`): FP32 recurrent state scan (`[C1]`), `document_ids` multi-pack boundary resets (`[C3]`), Multiplicative $2^\ell$ Dyadic RFF + Zero-KV `SpeculativeEngram` (**Axe 1**), `DEAO` survival loss + 2-pass Self-Conditioned Markov/Engram rollout (**Axe 2**), Exact $\mathfrak{sp}(4,\mathbb{R})$ Symplectic Squeeze-Rotation Butterfly Scan (**Axe 3**), Open-System Lindblad Decoherence & Wigner Purity $\mathcal{P}_{t,m}$ (**Axe 4**), Diagonal Kraus POVM Measurement Back-Action & SpecInfer Multi-Branch Tree `draft_kraus_tree_step` (**Axe 5**), and Cross-Layer Joint Proximal Dyadic Group-LASSO `apply_proximal_dyadic_group_lasso` + analytical 3-head pruning compensation (**Axe 6**, `[C5]`).
>   2. **Zero-Leak Data Pipeline & Self-Contained OOV-Exact Losses** (`scripts/build_decontaminated_mix.py`, `src/speculators/models/kvn_dspark/losses.py`, `src/speculators/train/trainer.py`): `conversation_id`-grouped train/val splitting (`0.0%` shared UltraChat sibling prefixes, `[C2]`), `<think>` truncation filter, **train-only** `token_freq.pt` (`t2d`/`d2t`), and exact full-vocab OOV-aware $c^*_{b,k} \le M_{\text{in}}$ (`[C4]`).
>   3. **Native `vLLM V1` Production Integration & Evaluation Suite** (`src/speculators/vllm_integration/`, `scripts/evaluate/{kvn_phase2_calibration_and_amdahl.py,benchmark_vllm_prod.py}`, `examples/train/kvn_dspark_phase3_h200_suite.sh`, `examples/evaluate/kvn_dspark_phase3_e2e_eval.sh`): `vllm.general_plugins` auto-registration (`register_kvn_vllm_plugin`, `[C6]`), `KvNDSparkVllmProposer`, `KvNStateSlotPool` (chunked prefill + preemption swap), `ZeroReplayAPCStateCache` (0-FLOP prefix restore), `PurityLoadAwareAmdahlScheduler`, and lossless `specinfer_verify_tree_step`.
>
> - **ORDERED EXECUTION MISSION ON THE H200 CLUSTER (`[ ]` — EXECUTE IN SEQUENCE):**
>   1. **[ ] Step 0 (Pre-Flight & Gate):** Verify `/dev/shm >= 32 GB` (`--shm-size=64g`), install `hs_connectors` + `speculators` + `vllm`, and pass the Linux unit/Triton/vLLM gate.
>   2. **[ ] Phase B (Zero-Leak 100k Corpus):** Generate and preprocess `qwen3_8b_mix100k_seq8192` (`95,000` train + `5,000` val conversations grouped strictly by `conversation_id`, `0` 13-gram benchmark overlap, train-only `token_freq.pt`).
>   3. **[ ] Phase C (Strict 6-Epoch Iso-Compute Ablation Matrix + `Qwen3-32B` Transfer):** Execute `examples/train/kvn_dspark_phase3_h200_suite.sh` across the 11 orthogonal configurations (`3` seeds `42, 1337, 2026` on headline models `B1`, `V2_Fixed`, `V3_A5`, `V3_A6`, including `5 + 1 = 6` iso-compute proximal Group-LASSO pruning `512 -> 128` and the FFN-matched reviewer-shield control `V3_FFNMatched_4p5`), plus `Qwen/Qwen3-32B` scale transfer.
>   4. **[ ] Phase D (Lossless Chain vs Kraus-Tree, Full-Vocab Calibration & Production `vllm serve`):** Execute `examples/evaluate/kvn_dspark_phase3_e2e_eval.sh` (`D.1a` linear chain, `D.1b` Kraus POVM SpecInfer tree, `D.2` full-vocab 15-bin ECE/AUROC/Pareto, `D.3` controlled PyTorch batched roofline, `D.4` live `vllm serve` + `benchmark_vllm_prod.py` across concurrencies `1..128`), verify Section 8 Go/No-Go gates, and compile `kvn_rff_rnn_mtp_report_phase3.md` (never commit `.safetensors` weights).

---

## 1. Scientific Architecture & Audit-Hardened Guarantees (`KvN-DSpark v3`)

### 1.1 Resolution of All `[C1]`–`[C6]` & `[M1]`–`[M6]` Methodological Invariants

| Invariant ID | Failure Mode Eliminated | Exact Phase 3 Mathematical & Code Guarantee |
| :--- | :--- | :--- |
| **`[C1]` FP32 Phase-Space Recurrence** | BF16 rounding drift in $e^{(-\gamma + i\omega)\Delta t}$ over $8\text{k}$–$32\text{k}$ tokens | `--kvn-state-dtype float32` maintains $(q_{t,m}, p_{t,m}, \nu_{t,m}) \in \mathbb{R}^3$ strictly in `torch.float32` across both PyTorch Kogge-Stone and `Mega-KvN` Triton scans while keeping projections in `bfloat16`. |
| **`[C2]` Hermetic `conversation_id` Split & Train-Only Shortlist** | Multi-turn sibling prefix overlap in UltraChat & val-split leakage into `token_freq.pt` | `scripts/build_decontaminated_mix.py` groups all sub-conversations by canonical `conversation_id` **prior** to shuffling/splitting (`0.0%` prefix overlap), rejects truncated `<think>` generations, and restricts `token_freq.pt` (`t2d`/`d2t`) strictly to the `95%` training split. |
| **`[C3]` Multi-Pack Document Reset & Chunked Boundary Shift** | Cross-document state bleed under sequence packing & chunked-prefill `fc_{t-1}` discontinuity | `_prepare_causal_kvn_inputs` zeroes $\tilde{h}^{\text{fc}}_t = h^{\text{fc}}_{t-1}$ and sets `reset_mask=True` whenever `document_ids[t] != document_ids[t-1]`. `KvNStateSlotPool` persists `last_fc_pool` (`8.19 KB/seq` on `Qwen3-8B`) so chunked prefill $32+32$ matches $64$ to `< 1e-6`. |
| **`[C4]` Full-Vocab OOV-Exact Loss & Confidence Targets** | Shortlist-only denominator inflating `pos0_acc` and corrupting `BornConfidenceHead` BCE | Self-contained `src/speculators/models/kvn_dspark/losses.py` + `KvNDSparkDraftModel.load_verifier_weights()` preserve full-vocab verifier probability mass ($p_{\text{OOV}} = 1 - \sum_{v \in \mathcal{V}_{\text{draft}}} p_v$) in TV/CE and enforce $c^*_{b,k} \le M_{\text{in}}$ with OOV-aware greedy accuracy. |
| **`[C5]` Iso-Compute Cross-Layer Proximal Group-LASSO** | Smooth L1 failing to produce exact structural zeros; non-zero LASSO on unpruned baselines; `6+1=7` epoch unfairness | Every unpruned model sets `--kvn-lasso-weight 0.0`. `V3_A6` runs **5 epochs** of exact post-step joint cross-layer proximal soft-thresholding (`apply_proximal_dyadic_group_lasso`) + zero-shot `512 -> 128` pruning + **1 epoch** warm recalibration = **6 total epochs (100% iso-compute)**. |
| **`[C6]` Native `vLLM V1` Plugin, Zero-Replay APC & SpecInfer** | Standalone toy loop without PagedAttention/APC/preemption integration | `[project.entry-points."vllm.general_plugins"]` registers `kvn_dspark`, routing checkpoints to `KvNDSparkVllmProposer` with `KvNStateSlotPool` CPU preemption swap, 16-token `ZeroReplayAPCStateCache`, `PurityLoadAwareAmdahlScheduler`, and `specinfer_verify_tree_step`. |

### 1.2 Summary of the Six Orthogonal `KvN-DSpark v3` Axes

1. **Axe 1 — Multiplicative $2^\ell$ Dyadic Multi-Scale RFF + Zero-KV `SpeculativeEngram` (`V3_A1`):**
   Initializes re-uploading spectral frequencies on a geometric dyadic lattice $\Omega^{(\ell)} = 2^\ell \Omega_{\text{base}}$ ($\ell \in \{0, \dots, L_{\text{re}}-1\}$) spanning sharp token transitions and long-horizon discourse envelopes, paired with an $O(1)$ hash-indexed 3-gram `SpeculativeEngram` table (`--kvn-engram-buckets 65536 --kvn-engram-dim 128`, `0 B` recurrent state) injected directly into Markov rollout embeddings.
2. **Axe 2 — `DEAO` Survival-Weighted Loss + 2-Pass Self-Conditioned Rollout (`V3_A2`):**
   Replaces static geometric horizon weights with detached prefix-survival weights $w_{b,k}^{\text{DEAO}} \propto \prod_{j < k} \text{sg}(\hat{\alpha}_{b,j})$ (`--per-position-loss-weight deao`) concentrating gradient capacity on reachable speculative horizons, and eliminates teacher-forcing exposure bias via 2-pass scheduled self-conditioning (`--kvn-self-cond-prob 0.5`, feeding Pass-1 argmax $\hat{y}_{k-1}$ into Pass-2 Markov/Engram transitions with detached inputs).
3. **Axe 3 — $\mathfrak{sp}(4,\mathbb{R})$ Symplectic Squeeze-Rotation Butterfly Scan + Fused `Mega-KvN` Triton (`V3_A3`):**
   Upgrades independent 2D damped oscillators to coupled 4D dyadic phase-space blocks $(q_{2g}, p_{2g}, q_{2g+1}, p_{2g+1}) \in \mathbb{R}^4$ evolved by exact symplectic squeeze-rotation-beam-splitter operators $S_{t,g} \in \text{Sp}(4,\mathbb{R})$ ($\det S_{t,g} = +1$) composed with isotropic contraction $e^{-\gamma_{t,g}\Delta t_{t,g}} < 1$, fused into a single register-resident Kogge-Stone Triton kernel (`--kvn-symplectic-butterfly --kvn-use-triton-scan`).
4. **Axe 4 — Open-System Lindblad Decoherence & Wigner Purity Calibration (`V3_A4`):**
   Tracks per-mode thermal/decoherence variance $\nu_{t,m} = a_{t,m}^2 \nu_{t-1,m} + \delta\nu_{t,m} \ge 0$ alongside coherent amplitudes $(q_{t,m}, p_{t,m})$ to compute exact Gaussian Wigner purity $\mathcal{P}_{t,m} = (1 + 2\nu_{t,m})^{-1} \in (0, 1]$ (`--kvn-lindblad-purity`), separating epistemic context ambiguity ($\mathcal{P} \ll 1$) from peak sharpness in `BornConfidenceHead` and `PurityLoadAwareAmdahlScheduler`.
5. **Axe 5 — Kraus POVM Measurement Back-Action & SpecInfer Multi-Branch Tree (`V3_A5`):**
   Replaces single-chain rollout collapse at bimodal syntactic forks with top-$C$ diagonal Kraus operators satisfying exact POVM completeness $\sum_{y \in \mathcal{C}_k} M_y^\dagger M_y = I_{M+1}$ and Lüders post-measurement state collapse $(q^{(y)}, p^{(y)}) \propto M_y (q, p)$ (`--kvn-use-kraus-tree`), verified in a single target pass via `build_tree_attention_mask` + `specinfer_verify_tree_step` + `compact_tree_kv_cache`.
6. **Axe 6 — Scale-Invariant Proximal Dyadic Group-LASSO Pruning $512 \to 128$ (`V3_A6`):**
   Trains an over-parameterized spectral lottery ($M_{\text{train}}=512$, $256$ dyads) for 5 epochs under exact post-step joint cross-layer proximal soft-thresholding:
   $$w_g \leftarrow \max\!\left(0,\; 1 - \frac{\eta_t \lambda_{\text{prox}}}{\|w_g\|_{\text{RMS}}}\right) w_g, \qquad \lambda_{\text{prox}} = \kappa \frac{\|w_g^{(0)}\|_{\text{RMS}}}{\sum_{t=1}^{T_{\text{LASSO}}} \eta_t}, \quad \kappa \in [1.05, 1.15]$$
   driving redundant dyads to **exact `0.0000`** across all $L$ layers simultaneously, followed by zero-shot structural pruning `prune_kvn_modes(128)` (`64` dyads) with analytical 3-head bias compensation and 1 warm recalibration epoch (`--kvn-lasso-weight 0.0`, `5 + 1 = 6` strict iso-compute epochs).

---

## 2. Exact Machine-Verified Parameter & Serving Memory Accounting

> [!IMPORTANT]
> **Strict Anti-Double-Subtraction & Reviewer-Shield Accounting Policy:**
> 1. **Parameter Columns Disaggregated Explicitly:** Across all models on `Qwen/Qwen3-8B` (`d=4096, V=151,936, V_draft=32,768`), the frozen shortlist `lm_head` (`32,768 × 4,096 = 134,217,728` params) and frozen `embed_tokens` (`151,936 × 4,096 = 622,329,856` params) are identical and never double-subtracted. We report **(i) Trainable Parameters** (`requires_grad=True`), **(ii) Trainable + Shortlist `lm_head`** (co-located `vLLM` footprint when sharing target token embeddings), and **(iii) Standalone Checkpoint Total** (including `embed_tokens` + `lm_head`).
> 2. **Explicit FFN-Matched Control (`V3_FFNMatched_4p5`):** In `Qwen3-8B`, each `DSpark` SWA layer uses a 3-matrix SwiGLU MLP (`intermediate_size = 12,288`, $3 \times 4096 \times 12288 = \mathbf{150,994,944}$ params/layer), whereas standard `KvN` uses a 2-matrix channel MLP (`ffn_mult = 1.5`, $2 \times 4096 \times 6144 = 50,331,648$ weight elements/layer; `50,341,888` w/ biases). `V3_FFNMatched_4p5` (`--kvn-ffn-mult 4.5`, hidden width $18,432$) sets $2 \times 4096 \times 18432 = \mathbf{150,994,944}$ FFN weight elements/layer (`151,017,472` w/ biases) — **matching `B1_DSpark_SWA128` FFN capacity** while retaining $100\%$ of `KvN`'s `15.87 KB/seq` Zero-KV state compression!
> 3. **Complete Per-Sequence & APC Byte Accounting:** Every `KvN` serving slot accounts for **all three** state tensors: FP32 phase space $(q, p)$ (`5,120 B`), FP32 Lindblad variance $\nu$ (`2,560 B` when active), and the BF16 chunk-boundary `last_fc` vector $h^{\text{fc}}_{t-1}$ (`8,192 B` on `Qwen3-8B`, `10,240 B` on `Qwen3-32B`), plus exact `16`-token `ZeroReplayAPCStateCache` overhead in bytes/token.

### Table 2A — `Qwen/Qwen3-8B` (`d=4096, L=5, V_draft=32,768, GQA n_kv=8, d_head=128`) Exact Meta-Device Audit

| Run ID | Seeds | Backbone ($M_{\text{train}} \to M_{\text{infer}}$) | Trainable Params ($\Delta$ vs `B1`) | Trainable + Shortlist `lm_head` | Standalone Total (w/ `embed`) | Dynamical State / Seq (`(q,p)` + $\nu$) | Chunk `fc_{t-1}` / Seq | **Total Slot / Seq** (vs `B1`) | **APC `16`-Tok** Dyn / Full (`B/tok`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B1_DSpark_SWA128`** | `42,1337,2026` | `swa` ($W=128$) | `1,076,551,041` *(0.0%)* | `1,210,768,769` | `1,833,098,625` | `2,621,440 B` *(2.62 MB)* | `0 B` | **`2,621,440 B`** *(1.0×)* | `163,840` / `163,840` |
| **`B2_DSpark_SWA512`** | `42` | `swa` ($W=512$) | `1,076,551,041` *(0.0%)* | `1,210,768,769` | `1,833,098,625` | `10,485,760 B` *(10.49 MB)* | `0 B` | **`10,485,760 B`** *(0.25×)* | `655,360` / `655,360` |
| **`B3_Hybrid_1SWA_4KvN`** | `42` | `hybrid` ($128$) | `658,514,241` *(-38.83%)* | `792,731,969` | `1,415,061,825` | `530,432 B` *(524 KB + 6.1 KB)* | `8,192 B` | **`538,624 B`** *(4.87×)* | `33,152` / `33,664` |
| **`V2_Fixed`** | `42,1337,2026` | `kvn` ($128$) | `499,311,233` *(-53.62%)* | `633,528,961` | `1,255,858,817` | `5,120 B` *(5.12 KB FP32)* | `8,192 B` | **`13,312 B`** *(196.9×)* | **`320.0`** / **`832.0`** |
| **`V3_A1_DyadicRFF_Engram`** | `42` | `kvn` ($128$) | `510,486,401` *(-52.58%)* | `644,704,129` | `1,267,033,985` | `5,120 B` *(5.12 KB FP32)* | `8,192 B` | **`13,312 B`** *(196.9×)* | **`320.0`** / **`832.0`** |
| **`V3_A2_DEAO_SelfCond`** | `42` | `kvn` ($128$) | `510,486,401` *(-52.58%)* | `644,704,129` | `1,267,033,985` | `5,120 B` *(5.12 KB FP32)* | `8,192 B` | **`13,312 B`** *(196.9×)* | **`320.0`** / **`832.0`** |
| **`V3_A3_SymplecticButterfly`** | `42` | `kvn` ($128$) | `520,975,361` *(-51.61%)* | `655,193,089` | `1,277,522,945` | `5,120 B` *(5.12 KB FP32)* | `8,192 B` | **`13,312 B`** *(196.9×)* | **`320.0`** / **`832.0`** |
| **`V3_A4_LindbladPurity`** | `42` | `kvn` ($128$) | `523,663,617` *(-51.36%)* | `657,881,345` | `1,280,211,201` | `7,680 B` *(5.12 KB + 2.56 KB)* | `8,192 B` | **`15,872 B`** *(165.2×)* | **`480.0`** / **`992.0`** |
| **`V3_A5_KrausTree`** | `42,1337,2026` | `kvn` ($128$) | **`525,240,836`** *(-51.21%)* | **`659,458,564`** | `1,281,788,420` | **`7,680 B`** *(341.3× vs `B1`)* | `8,192 B` | **`15,872 B`** *(165.2×)* | **`480.0`** / **`992.0`** |
| **`V3_FFNMatched_4p5`** | `42` | `kvn` ($128$, `4.5×` FFN) | `1,028,618,756` *(-4.45%)* | `1,162,836,484` | `1,785,166,340` | `7,680 B` *(341.3× vs `B1`)* | `8,192 B` | **`15,872 B`** *(165.2×)* | **`480.0`** / **`992.0`** |
| **`V3_A6_ProxLASSO_512to128`** *(pre-prune)* | `42,1337,2026` | `kvn` ($512$) | `633,130,628` *(-41.19%)* | `767,348,356` | `1,389,678,212` | `30,720 B` *(20.48 + 10.24 KB)* | `8,192 B` | `38,912 B` *(67.4×)* | `1,920.0` / `2,432.0` |
| **`V3_A6_Pruned128_Recal`** *(deployed)* | `42,1337,2026` | `kvn` ($512 \to 128$) | **`525,240,836`** *(-51.21%)* | **`659,458,564`** | `1,281,788,420` | **`7,680 B`** *(341.3× vs `B1`)* | `8,192 B` | **`15,872 B`** *(165.2×)* | **`480.0`** / **`992.0`** |

*Exact Batch-128 Serving Memory Impact (`Qwen3-8B`):* At concurrency $B = 128$, `B2_DSpark_SWA512` consumes **`1.342 GB`** and `B1_DSpark_SWA128` consumes **`335.5 MB`** of PagedAttention drafter KV-cache, whereas `V3_A5` / `V3_A6_Pruned128_Recal` consumes **`2.03 MB` total** (`0.98 MB` FP32 `(q,p,\nu)` + `1.05 MB` `last_fc`) with **`0 B` KV-cache** and O(1) decode bandwidth independent of window length.

### Table 2B — `Qwen/Qwen3-32B` Scale-Transfer (`d=5120, intermediate=25600, L=5, target_layer_ids=[2, 32, 61]`)

| Run ID | Backbone ($M$) | Trainable Params ($\Delta$ vs `B1`) | Trainable + Shortlist `lm_head` | Standalone Total | Dynamical `(q,p,\nu)` / Seq | Chunk `fc_{t-1}` / Seq | **Total Slot / Seq** | **APC `16`-Tok** Dyn / Full (`B/tok`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B1_DSpark_SWA128`** | `swa` ($W=128$) | `2,579,435,905` *(0.0%)* | `2,747,208,065` | `3,525,120,385` | `2,621,440 B` | `0 B` | **`2,621,440 B`** *(1.0×)* | `163,840` / `163,840` |
| **`V3_A5_KrausTree`** | `kvn` ($128$) | **`764,506,116`** *(**-70.36%**)* | **`932,278,276`** | `1,710,190,596` | **`7,680 B`** | `10,240 B` | **`17,920 B`** *(146.3×)* | **`480.0`** / **`1,120.0`** |
| **`V3_A6_ProxLASSO_512to128`** *(pre-prune)* | `kvn` ($512$) | `898,741,380` *(-65.16%)* | `1,066,513,540` | `1,844,425,860` | `30,720 B` | `10,240 B` | `40,960 B` *(64.0×)* | `1,920.0` / `2,560.0` |
| **`V3_A6_Pruned128_Recal`** *(deployed)* | `kvn` ($512 \to 128$) | **`764,506,116`** *(**-70.36%**)* | **`932,278,276`** | `1,710,190,596` | **`7,680 B`** | `10,240 B` | **`17,920 B`** *(146.3×)* | **`480.0`** / **`1,120.0`** |

---

## 3. Step 0 — H200 Pre-Flight Container Check & Automated Regression Gate

> [!CAUTION]
> **Docker `/dev/shm` Trap (`--shm-size=64g` Mandatory):** Each `seq_len = 8192` BF16 hidden-state bundle (`4` captured layers $\times 4096 \times 2\text{ B}$) occupies **`268.4 MB`** in shared memory before `VllmHiddenStatesSource` deletes the transient handle. Default Docker containers allocate `64 MB` to `/dev/shm` and will crash immediately on step 1. Always launch the H200 container with `--shm-size=64g --ipc=host` (enforced automatically by `examples/train/kvn_dspark_phase3_h200_suite.sh`).

- [ ] **Step 0.1: Environment Setup, `/dev/shm` Verification & Full Unit + Triton + `vLLM V1` Gate**
  ```bash
  # 1. Verify >= 32 GB free on /dev/shm BEFORE starting any job
  df -h /dev/shm
  SHM_MB=$(df -Pm /dev/shm | awk 'NR==2 {print $4}')
  test "$SHM_MB" -ge 32768 || { echo "FATAL: Relaunch container with --shm-size=64g"; exit 1; }

  # 2. Clone/update branch & install hs_connectors + speculators (with vLLM general_plugins entry point!)
  cd /workspace
  git clone -b kvn-rff-rnn-mtp git@github.com:blancsw/speculators.git || (cd speculators && git pull origin kvn-rff-rnn-mtp)
  cd speculators

  uv venv .venv --python 3.11
  source .venv/bin/activate
  uv pip install -e hs_connectors -e ".[dev]"
  uv pip install "vllm>=0.11.0" "triton>=3.3.0" -r scripts/evaluate/requirements.txt

  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

  # 3. Mandatory Pre-Flight Gate (must report 0 failures across models, fused Triton OOV losses & vLLM proposer)
  pytest tests/unit/models/test_kvn_dspark.py \
         tests/unit/models/test_kvn_dspark_v2.py \
         tests/unit/models/test_kvn_dspark_v3.py \
         tests/unit/models/test_fused_losses.py \
         tests/unit/train/test_kvn_phase2_pipeline.py \
         tests/unit/train/test_kvn_v3_train_pipeline.py \
         tests/unit/vllm_integration/ -q

  # 4. Dry-run CLI verification of all 12 Phase 3 model emitters
  bash -n examples/train/kvn_dspark_phase3_h200_suite.sh
  bash -n examples/evaluate/kvn_dspark_phase3_e2e_eval.sh
  ```

---

## 4. Phase B — Zero-Leak `conversation_id`-Disjoint 100k On-Policy Corpus (`qwen3_8b_mix100k_seq8192`)

| Domain | HF Source (Train Split Only) | Unique Source Convs | Target Volume | On-Policy `Qwen3-8B` Temp Mix | Role in `KvN-DSpark v3` |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **General Multi-Turn Dialogue** | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | `207,865` | **`45,000`** (45%) | 50% $T=0.0$ / 50% $T=0.7$ | Broad lexical coverage, multi-turn APC prefix structure |
| **Math & CoT Reasoning** | `nvidia/OpenMathInstruct-2` (`train`) | `14,000,000` | **`32,000`** (32%) | 50% $T=0.0$ / 50% $T=0.6$ | Deep horizons $k=2..7$, LaTeX, symbolic recurrence |
| **Code & Algorithms** | `ise-uiuc/Magicoder-OSS-Instruct-75K` (`train`) | `75,197` | **`23,000`** (23%) | 60% $T=0.0$ / 40% $T=0.6$ | Bimodal syntactic forks (`KrausTree`), 3-gram `Engram` |
| **Total Decontaminated Mix** | Grouped by `conversation_id` (`95%` Train / `5%` Val) | — | **`100,000`** (`95k` + `5k`) | Stratified | **~170M–195M on-policy tokens packed at `seq_len=8192`** |

- [ ] **Step B.1: High-Throughput Data-Parallel On-Policy Generation across All Node H200 GPUs**
  *(Note: `examples/train/kvn_dspark_phase3_h200_suite.sh` runs Phase B automatically when `SKIP_DATA_PREP=0`. Running Step B.1 explicitly first lets you dedicate **all** H200 GPUs to `vLLM` data-parallel generation before partitioning GPUs for Phase C.)*
  ```bash
  export VERIFIER_MODEL="Qwen/Qwen3-8B"
  export ARTIFACT_ROOT="/workspace/kvn_phase3_artifacts"
  export DATA_DIR="${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192"
  export RAW_JSONL="${ARTIFACT_ROOT}/raw_onpolicy_100k_conv_disjoint.jsonl"
  mkdir -p "$DATA_DIR" /dev/shm/kvn_prep_hs

  # On 8xH200: ALL_GPUS="0,1,2,3,4,5,6,7" DP_GEN=8 | On 2xH200: ALL_GPUS="0,1" DP_GEN=2
  ALL_GPUS="${ALL_GPUS:-0,1,2,3,4,5,6,7}"
  DP_GEN="${DP_GEN:-8}"

  CUDA_VISIBLE_DEVICES="$ALL_GPUS" python scripts/launch_vllm.py "$VERIFIER_MODEL" \
      --target-layer-ids 2 18 33 \
      --hidden-states-path "/dev/shm/kvn_prep_hs" \
      -- --data-parallel-size "$DP_GEN" --port 8000 --max-model-len 12288 --gpu-memory-utilization 0.90 &
  VLLM_PREP_PID=$!
  until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done

  # Generates 100k unique conversations, filters 13-gram/SHA-256 overlap with 6 benchmarks,
  # drops truncated <think> completions, and groups strictly by conversation_id!
  python scripts/build_decontaminated_mix.py \
      --verifier-model "$VERIFIER_MODEL" \
      --ultrachat-samples 45000 \
      --openmath-samples 32000 \
      --magicoder-samples 23000 \
      --exclude-benchmarks mt_bench gsm8k math500 humaneval mbpp livecodebench \
      --vllm-endpoint "http://localhost:8000/v1" \
      --concurrency "$((64 * DP_GEN))" \
      --output-jsonl "$RAW_JSONL" \
      --prepare-arrow-dir "$DATA_DIR" \
      --seq-length 8192 \
      --train-data-ratio 0.95 \
      --val-samples 0 \
      --resume \
      --seed 42

  kill "$VLLM_PREP_PID" && wait "$VLLM_PREP_PID" 2>/dev/null || true
  rm -rf /dev/shm/kvn_prep_hs
  ```
  *Phase B Gate Check:* Inspect `${RAW_JSONL}.manifest.json` and confirm `unique_conversations_train == 100000`, `pretokenized_boundary_rows == true`, and `contaminated_candidates_rejected >= 0`. `--prepare-arrow-dir "$DATA_DIR"` aligns the `95%` `ArrowDataset` boundary strictly on whole `conversation_id` groups and computes `${DATA_DIR}/token_freq.pt` (`t2d-32768.npy`/`d2t-32768.npy`) exclusively on the 95% train split.

---

## 5. Phase C — Strict 6-Epoch Iso-Compute Orthogonal Ablation Suite & `Qwen3-32B` Transfer

All `Qwen/Qwen3-8B` runs share identical optimization and sequence geometry:
- **Optimizer:** `Muon` (`--muon-lr 2e-3`, 5-step Newton-Schulz orthogonalization on 2D weights) + `AdamW` (`--lr 3e-4`, cosine decay, routing 1D norms, biases, `omega_rff`, `dyad_`, `lindblad_`, `kraus_`, `engram`, and `vacuum_` parameters cleanly out of Newton-Schulz).
- **Loss & Sequence Geometry:** `--loss-fn '{"ce": 0.1, "tv": 0.9}'` (`--loss-implementation fused` with full-vocab OOV mass preserved), `--total-seq-len 8192`, `--max-anchors 512`, `--block-size 8`, `--num-layers 5`, `--draft-vocab-size 32768`, `--target-layer-ids 2 18 33`, `--kvn-state-dtype float32`.
- **Strict 6-Epoch Iso-Compute Schedule:** Every baseline and `v3` model trains for **`EPOCHS=6`**. `V3_A6_ProxLASSO_512to128` trains for **`LASSO_EPOCHS=5`** + zero-shot structural pruning `512 -> 128` + **`RECAL_EPOCHS=1`** (`--lr 5e-5 --muon-lr 2e-4 --kvn-lasso-weight 0.0`) = **`6` total epochs**.
- **Multi-GPU Proximal Threshold Scaling Note:** Measured cross-layer `_compute_joint_dyad_rms_norms()` on 5-layer `Qwen3-8B` (`d=4096, M=512`) `V3_A6_ProxLASSO_512to128` is $\|w_g^{(0)}\|_{\text{RMS}} \approx 0.01423$. On `8×H200` (`NPROC=6` data-parallel training GPUs, `~18.5k`–`19.7k` steps over 5 epochs, cosine $\sum_t \eta_t \approx 2.8$–$2.96$), the script default `PROX_LASSO_WEIGHT=5.2e-3` achieves $\sum_t \eta_t \lambda_{\text{prox}} \approx 0.0146\text{–}0.0154 \approx 1.05 \|w_g^{(0)}\|_{\text{RMS}}$; on `2×H200` (`NPROC=1`), pass `PROX_LASSO_WEIGHT=9e-4`; on `Qwen3-32B` (`50k` mix, `NPROC=4`, `~13.4k` steps, $\sum_t \eta_t \approx 2.0$), pass `PROX_LASSO_WEIGHT=7.5e-3`.

### Table 5A — Orthogonal 11-Model Experimental Matrix (`examples/train/kvn_dspark_phase3_h200_suite.sh`)

| Run ID | Seeds | Epochs | Active `v3` Axes (Cumulative Orthogonal Ladder) | Isolated Scientific Hypothesis |
| :--- | :---: | :---: | :--- | :--- |
| **`B1_DSpark_SWA128`** | `42,1337,2026` | `6` | Official SWA-128 baseline (`--markov-head-type rnn`, `lasso=0`) | Reference SOTA attention drafter (`2.62 MB/seq`) |
| **`B2_DSpark_SWA512`** | `42` | `6` | SWA-512 wide-window baseline (`lasso=0`) | Quantifies `10.49 MB/seq` KV-cache roofline penalty at `8k` |
| **`B3_Hybrid_1SWA_4KvN`** | `42` | `6` | 4 `KvN v3` layers + 1 top `SWA-128` layer (`lasso=0`) | Hybrid memory/quality Pareto anchor (`538.6 KB/seq`) |
| **`V2_Fixed`** | `42,1337,2026` | `6` | `KvN v2` w/ `[C1]`–`[C5]` fixes (`FP32` state, OOV loss, `lasso=0`) | Clean bug-free Phase 2 Zero-KV baseline (`13.31 KB/seq`) |
| **`V3_A1_DyadicRFF_Engram`** | `42` | `6` | `V2_Fixed` + **Dyadic $2^\ell$ RFF** + **3-gram `Engram`** + `rnn` Markov | Multi-scale spectral features + O(1) lexical retrieval (**Axe 1**) |
| **`V3_A2_DEAO_SelfCond`** | `42` | `6` | `V3_A1` + **`DEAO` survival loss** + **Self-Cond (`p=0.5`)** | Survival-aligned gradients + exposure-bias removal (**Axe 2**) |
| **`V3_A3_SymplecticButterfly`** | `42` | `6` | `V3_A2` + **$\mathfrak{sp}(4,\mathbb{R})$ Butterfly Scan** (`Triton`) | Coupled 4D squeezing/beam-splitting phase space (**Axe 3**) |
| **`V3_A4_LindbladPurity`** | `42` | `6` | `V3_A3` + **Lindblad Decoherence $\nu_{t,m}$ & Purity $\mathcal{P}_{t,m}$** | Mixed-state epistemic uncertainty calibration (**Axe 4**) |
| **`V3_A5_KrausTree`** | `42,1337,2026` | `6` | `V3_A4` + **Kraus POVM Back-Action & SpecInfer Tree** | Multi-branch syntactic fork resolution (**Axe 5**, full `v3` @ $M=128$) |
| **`V3_FFNMatched_4p5`** | `42` | `6` | `V3_A5` + **`--kvn-ffn-mult 4.5`** (`150,994,944` FFN params/layer) | **Reviewer-shield control:** proves parameter savings aren't an FFN artifact |
| **`V3_A6_ProxLASSO_512to128`** | `42,1337,2026` | **`5 + 1 = 6`** | `V3_A5` @ $M=512 \xrightarrow{\text{ProxLASSO}} 128$ + 1-ep warm recal | **Spectral lottery gain** over direct $M=128$ (`V3_A5`) under strict iso-compute (**Axe 6**) |

- [ ] **Step C.1: Execute Headline 3-Seed Models (`B1`, `V2_Fixed`, `V3_A5`, `V3_A6`)**
  ```bash
  # Recommended 8xH200 topology (GPUs 0,1 = vLLM DP=2 producer; GPUs 2..7 = 6-GPU DDP training):
  SKIP_DATA_PREP=1 \
  VLLM_GPUS="0,1" VLLM_DP=2 TRAIN_GPUS="2,3,4,5,6,7" NPROC=6 PROX_LASSO_WEIGHT="5.2e-3" \
  MODELS="B1_DSpark_SWA128 V2_Fixed V3_A5_KrausTree V3_A6_ProxLASSO_512to128" \
  SEEDS="42 1337 2026" \
  bash examples/train/kvn_dspark_phase3_h200_suite.sh

  # Alternative 2xH200 topology (GPU 0 = vLLM DP=1; GPU 1 = training):
  # SKIP_DATA_PREP=1 VLLM_GPUS="0" VLLM_DP=1 TRAIN_GPUS="1" NPROC=1 PROX_LASSO_WEIGHT="9e-4" \
  # MODELS="B1_DSpark_SWA128 V2_Fixed V3_A5_KrausTree V3_A6_ProxLASSO_512to128" SEEDS="42 1337 2026" \
  # bash examples/train/kvn_dspark_phase3_h200_suite.sh
  ```

- [ ] **Step C.2: Execute Orthogonal Single-Axis Progression & `V3_FFNMatched_4p5` Control (`seed=42`)**
  ```bash
  SKIP_DATA_PREP=1 \
  VLLM_GPUS="0,1" VLLM_DP=2 TRAIN_GPUS="2,3,4,5,6,7" NPROC=6 \
  MODELS="B2_DSpark_SWA512 B3_Hybrid_1SWA_4KvN V3_A1_DyadicRFF_Engram V3_A2_DEAO_SelfCond V3_A3_SymplecticButterfly V3_A4_LindbladPurity V3_FFNMatched_4p5" \
  SEEDS="42" \
  bash examples/train/kvn_dspark_phase3_h200_suite.sh
  ```

- [ ] **Step C.3: `Qwen/Qwen3-32B` Scale-Transfer (`-70.36%` Trainable Params vs `DSpark` SWA-128)**
  *(On `8×H200` 141 GB: `Qwen3-32B` BF16 weights occupy ~65 GB, fitting on 1 H200 per DP replica (`VLLM_GPUS="0,1,2,3" VLLM_DP=4`), leaving GPUs `4,5,6,7` (`NPROC=4`) for DDP drafter training.)*
  ```bash
  VERIFIER_MODEL="Qwen/Qwen3-32B" \
  TARGET_LAYER_IDS="2 32 61" \
  ARTIFACT_ROOT="/workspace/kvn_phase3_qwen3_32b" \
  DATA_DIR="/workspace/kvn_phase3_qwen3_32b/data/qwen3_32b_mix50k_seq8192" \
  ULTRACHAT_SAMPLES=22500 OPENMATH_SAMPLES=16000 MAGICODER_SAMPLES=11500 \
  VLLM_GPUS="0,1,2,3" VLLM_DP=4 TRAIN_GPUS="4,5,6,7" NPROC=4 PROX_LASSO_WEIGHT="7.5e-3" \
  MODELS="B1_DSpark_SWA128 V3_A5_KrausTree V3_A6_ProxLASSO_512to128" \
  SEEDS="42" \
  bash examples/train/kvn_dspark_phase3_h200_suite.sh
  ```

---

## 6. Phase D — Lossless E2E, Kraus-Tree SpecInfer, Full-Vocab Calibration & Production `vllm serve`

`examples/evaluate/kvn_dspark_phase3_e2e_eval.sh` executes four complementary evaluation stages across **all 25 checkpoints** (including all 3 seeds `42, 1337, 2026` and all 3 pruning stages: unpruned `M=512` `ep4`, zero-shot pruned `M=128`, and 1-epoch warm-recalibrated `M=128` `ep0`):

1. **Phase D.1a & D.1b — Lossless Linear Chain vs Kraus POVM Speculative Tree (`mt_bench`, `gsm8k`, `math500`, `humaneval`, `mbpp` at $T=0.0$ & $T=0.6$):**
   Verifies 100% bit-exact greedy token parity at $T=0.0$ and lossless multi-round residual rejection sampling (`specinfer_verify_tree_step`) at $T=0.6$, isolating the exact $\Delta\tau$ gain of Kraus POVM tree branching over linear chains on code and math forks.
2. **Phase D.2 — Full-Vocab (`151,936`) OOV-Exact Calibration & Survival Pareto (`calibrate`):**
   Evaluates 15-bin `ECE`, binary rejection `AUROC`, and cumulative prefix-survival dynamic stopping ($\prod_{j \le k} c_j \ge \gamma_{\text{conf}}$) on the held-out `5%` validation split with exact OOV ground truth.
3. **Phase D.3 — Controlled PyTorch Batched Roofline Sweep ($B \in \{1, 4, 8, 16, 32, 64, 128\}$):**
   Measures true incremental KV-cache `SWA` vs O(1) `KvN` latency and VRAM scaling under identical PyTorch verifier kernels.
4. **Phase D.4 — Production `vllm serve` + `KvNDSparkVllmProposer` + `ZeroReplayAPCStateCache` Benchmark (`scripts/evaluate/benchmark_vllm_prod.py`):**
   Launches real `vllm serve` processes (`vllm.general_plugins` auto-wires `KvNDSparkVllmProposer` for `kvn` checkpoints and native `dspark` for `B1_DSpark_SWA128`), sweeps asynchronous OpenAI client concurrencies `1, 4, 8, 16, 32, 64, 128`, measures end-to-end Amdahl wall-clock speedup $\mathcal{S}_{\text{e2e}}(B)$, TTFT/TPOT/ITL p50/p99, and multi-turn `ZeroReplayAPCStateCache` prefix-hit speedup!

- [ ] **Step D.1: Run the Complete Automated Phase 3 Evaluation Suite**
  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  ARTIFACT_ROOT="/workspace/kvn_phase3_artifacts" \
  LAST_EPOCH=5 LAST_LASSO_EPOCH=4 RUN_VLLM_PROD=1 \
  bash examples/evaluate/kvn_dspark_phase3_e2e_eval.sh
  ```

- [ ] **Step D.2 (Optional External Cross-Check): Standardized NVIDIA `aiperf` Profile against `vllm serve`**
  *(If `aiperf` is installed on the cluster node, run an additional standardized external load-generator cross-check on `V3_A6_Pruned128_Recal_s42/0` at concurrency 32:)*
  ```bash
  vllm serve Qwen/Qwen3-8B --port 8015 \
      --enable-prefix-caching \
      --speculative-config '{"model": "/workspace/kvn_phase3_artifacts/checkpoints/V3_A6_Pruned128_Recal_s42/0"}' &
  VLLM_AIPERF_PID=$!
  until curl -sf http://localhost:8015/health > /dev/null; do sleep 2; done

  if command -v aiperf >/dev/null 2>&1; then
      aiperf profile \
          --model Qwen/Qwen3-8B \
          --url http://localhost:8015 \
          --endpoint-type chat \
          --concurrency 32 \
          --request-count 128 \
          --output-tokens-mean 256 \
          --artifact-dir /workspace/kvn_phase3_artifacts/eval_results_phase3/aiperf_v3_a6_c32
  fi
  kill "$VLLM_AIPERF_PID" && wait "$VLLM_AIPERF_PID" 2>/dev/null || true
  ```

---

## 7. Pre-H200 Live Local GPU Validation (`RTX 3070 Ti` 8 GB + WSL2 `Triton 3.7.1` & `vLLM 0.29.1`)

Prior to H200 cluster launch, the entire `KvN-DSpark v3` training, joint proximal Group-LASSO pruning, Kraus POVM tree SpecInfer verification, `Mega-KvN` Triton `sp4_butterfly` kernel, and `vLLM V1` `KvNDSparkVllmProposer` + `ZeroReplayAPCStateCache` stack was executed end-to-end on a local **NVIDIA GeForce RTX 3070 Ti Laptop GPU (`8,191.5 MiB` VRAM; Windows `PyTorch 2.14.1+cu132` + WSL2 Ubuntu `PyTorch 2.13.0+cu130`, `Triton 3.7.1`, `vLLM 0.29.1rc1`)** against **`Qwen/Qwen3.5-0.8B`** (`d=1024, L=3, V=248,320, V_draft=4,096, target_layer_ids=[2, 12, 21]`, strict `6`-epoch iso-compute over `32` train / `8` val on-policy conversations, `sibling_conversation_overlap=0`, `train_only` `token_freq.pt`, and `8` held-out technical evaluation prompts). Full machine-readable artifacts are recorded in [`rtx3070ti_kvn_v3_validation_results.json`](kvn_rff_mtp/rtx3070ti_kvn_v3_validation_results.json).

### 7.1 Strict 6-Epoch Iso-Compute Local GPU Ablation (`Qwen/Qwen3.5-0.8B`, Held-Out E2E & Full-Vocab Calibration)

| Model ID (6 Iso-Compute Epochs) | Trainable / Active Params | Drafter Slot State / Seq (Dyn / Total) | Uniform Val Loss / Val EAL | Linear E2E $\tau$ ($T=0.0$ / $T=0.6$) | Kraus Tree $\tau$ ($N=9, W=2$, $T=0.0$ / $T=0.6$) | Greedy Parity ($T=0.0$) | Full-Vocab `ECE` / `AUROC` | Batched Peak VRAM (`B=1 / 8 / 32`, MiB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B1_SWA`** (`SWA-128`, RNN Markov) | `85.31M` / `89.50M` | `768.0 KB` / `768.0 KB` (`1.0×`) | `0.7083` / `1.2292` | `1.1018` / `1.1220` | — (linear only) | **`100%` (`True`)** | `0.0763` / `0.7623` | `776.1 / 797.0 / 861.3` |
| **`M_v2_Fixed`** (`KvN v2`, `[C1]`–`[C5]` fixed) | `59.45M` / `63.64M` | `3.00 KB` / `5.00 KB` (**`153.6×` less**) | `0.7030` / `1.2734` | `1.2078` / `1.2026` | — (linear only) | **`100%` (`True`)** | `0.0873` / `0.7469` | `727.5 / 751.6 / 836.1` |
| **`M_v3_Full`** (`KvN v3`, `M=128`, 6 ep) | `64.44M` / `68.63M` (`-24.5%` vs `B1`) | `4.50 KB` / `6.50 KB` (**`118.2×` less**) | **`0.6958`** / **`1.2760`** | `1.1481` / `1.1847` | **`1.2065` (`+0.058`)** / **`1.2603` (`+0.076`)** | **`100%` (`True`)** | `0.0824` / **`0.7863`** | `743.4 / 811.4 / 1045.1` |
| **`M_v3_ProxLASSO_Pruned`** (`512->128`, `5+1=6` ep) | `64.44M` / `68.63M` (`-24.5%` vs `B1`) | `4.50 KB` / `6.50 KB` (**`118.2×` less**) | `0.7000` / `1.2578` | `1.1481` / **`1.2065`** | **`1.2222` (`+0.074`)** / `1.2267` (`+0.020`) | **`100%` (`True`)** | **`0.0572`** / `0.7387` | `743.4 / 811.4 / 1045.1` |

*Key Local Empirical Findings (even at tiny 32-conversation micro-scale):*
- **Exact Lossless Speculative Parity (`100%`):** Every model (`B1_SWA`, `M_v2_Fixed`, `M_v3_Full`, `M_v3_ProxLASSO_Pruned`, and zero-shot pruned `M=128`) achieved **100% bit-exact greedy token parity (`exact_greedy_parity = True`)** with standalone `Qwen/Qwen3.5-0.8B` across both linear-chain and Kraus POVM tree (`specinfer_verify_tree_step`) decoding.
- **v3 Representation & Kraus POVM Tree Gains:** `M_v3_Full` achieved the **lowest uniform validation loss (`0.6958` vs `0.7083` `B1_SWA` and `0.7030` `M_v2_Fixed`)**, **highest teacher-forced greedy EAL (`1.2760`)**, and **highest rejection `AUROC` (`0.7863`)**. Enabling Kraus POVM speculative tree verification (`N=9, W=2`) strictly improved E2E acceptance $\tau$ over linear chains on every v3 checkpoint (`+0.0583` at $T=0.0$ and `+0.0756` at $T=0.6$ on `M_v3_Full`; `+0.0741` at $T=0.0$ on `M_v3_ProxLASSO_Pruned`, reaching `1.2222` at $T=0.0$ and `1.2603` at $T=0.6$).
- **Exact Zero-Drift Proximal Group-LASSO (`512 -> 128`):** Joint cross-layer proximal dyadic Group-LASSO (`kvn_lasso_weight=0.656`, 5 epochs) drove **`97.27%` (`249/256`) of dyads to exact FP32 zero (`<= 1e-8`)**. Zero-shot `prune_kvn_modes(128)` (`4×` state reduction) incurred **`0.000000` layer-0 output drift** and **`0.0000` E2E greedy $\tau$ degradation** (`1.1481` unpruned `M=512` $\to$ `1.1481` zero-shot `M=128`), and 1-epoch warm recalibration (`5+1=6` iso-compute epochs) achieved the **lowest 15-bin `ECE` (`0.0572`)** and highest $T=0.0$ Kraus-tree $\tau$ (`1.2222`). *(Note: Windows lacks `triton`, so Windows drafter step latency reflects pure PyTorch fallback (`34–68 ms`), whereas WSL2 Linux executes compiled `Triton 3.7.1` kernels (`30.4 ms` propose + up to `5.25×` scan speedup below).)*

### 7.2 Live WSL2 Linux CUDA `Mega-KvN` (`Triton 3.7.1`) & `vLLM V1` (`0.29.1`) Verification

| Subsystem / Kernel Under Test (WSL2 Ubuntu CUDA, RTX 3070 Ti) | Configuration | Measured Result | Numerical / Exactness Check |
| :--- | :---: | :---: | :---: |
| **`Mega-KvN` Triton `sp4_butterfly` Parallel Scan (`T=512`)** | `B=2, M=128, FP32 state` | **`5.20 ms` Triton vs `8.73 ms` PyTorch (`1.68×` speedup)** | `max_q/p/nu_fp32_abs_diff = 0.0` |
| **`Mega-KvN` Triton `sp4_butterfly` Parallel Scan (`T=2048`)** | `B=2, M=128, FP32 state` | **`6.03 ms` Triton vs `30.35 ms` PyTorch (`5.04×` speedup)** | `max_q/p/nu_fp32_abs_diff = 0.0` |
| **`Mega-KvN` Triton `sp4_butterfly` Parallel Scan (`T=8192`)** | `B=1, M=128, FP32 state` | **`11.30 ms` Triton vs `59.28 ms` PyTorch (`5.25×` speedup)** | `max_q/p/nu_fp32_abs_diff = 0.0` |
| **`vllm.general_plugins` + `KvNDSparkVllmProposer` Wiring** | `vLLM 0.29.1rc1` | `plugin_registered=True`, `method="custom_class"` | Routes `kvn` $\to$ `KvNDSparkVllmProposer` |
| **`ZeroReplayAPCStateCache` + Preemption + Chunked `32+32` Prefill** | `block_size=16`, `80`-tok prompt | **`0` replay FLOPs**, `64` prefix toks restored (`288 B/tok` dyn / `416 B/tok` full) | Preempt err `0.0`; APC err `4.2e-3` (BF16) |
| **Pure-Attention `Qwen3ForCausalLM` 4D Tree Mask + `compact_tree_kv_cache`** | `N=5` tree $\to$ `3` accepted | Exact branch-isolated verification & in-place KV compaction | Next-step logit max abs err **`6.0e-7`** |
| **Live WSL2 E2E `KvNDSparkVllmProposer` (`M_v3_Full` on `Qwen3.5-0.8B`)** | Linear Chain vs Kraus Tree (`N=9, W=2`) | Chain $\tau = \mathbf{1.1899}$ (`30.4 ms`) $\to$ **Kraus Tree $\tau = \mathbf{1.2877}$ (`+0.0978`)** | **`100%` exact greedy parity (`True` / `True`)** |

---

## 8. Quantitative ICML/NeurIPS Go / No-Go Acceptance Gates (`Qwen/Qwen3-8B` & `Qwen/Qwen3-32B`)

| Scientific / Systems Metric | Phase 2 Baseline (`V2_Fixed`) | Minimum Production Go-Live Gate | ICML / NeurIPS Spotlight Target (`KvN-DSpark v3`) |
| :--- | :---: | :---: | :---: |
| **Causal / Conversation / OOV Leakage (`[C2]`–`[C4]`)** | Fixed in `V2_Fixed` | **`0.00%` (`atol < 1e-6`)** | **`0.00%` formally unit-tested & manifest-verified** |
| **Drafter State / Seq on `Qwen3-8B` (Dyn / Full Slot)** | `5.12 KB` / `13.31 KB` | **`<= 16 KB/seq` (`>= 160×` vs SWA-128)** | **`7.68 KB` dyn (`341×`) / `15.87 KB` full (`165×` vs SWA-128, `661×` vs SWA-512)** |
| **Zero-Replay APC Prefix Cache Overhead (`block=16`)** | N/A (replay required) | **`<= 1.0 KB/tok`, `0` replay FLOPs** | **`480 B/tok` dyn / `992 B/tok` full (`165×` vs SWA-128, `0` FLOPs)** |
| **Trainable Drafter Params (`Qwen3-8B` / `Qwen3-32B`)** | `499.3M` / `732.1M` | **`<= 550M` (`8B`) / `<= 800M` (`32B`)** | **`525.2M` (`-51.21%` on `8B`) / `764.5M` (`-70.36%` on `32B`)** |
| **Linear-Chain Acceptance $\tau$ ($T=0.0$ & $T=0.6$, 5-bench mean)** | `~92–95%` of `B1` | **`>= 98%` of `B1_DSpark_SWA128`** | **`>= 100%` of `B1_DSpark_SWA128` ($\tau \ge 3.8$–$4.4$)** |
| **Kraus POVM Tree (`V3_A5`/`V3_A6`) Gain $\Delta\tau$ vs Linear Chain** | N/A (chain only) | **`+0.15` tokens/step on Code/Math** | **`+0.25` to `+0.55` tokens/step (100% lossless SpecInfer)** |
| **Full-Vocab OOV-Aware 15-Bin `ECE` & Rejection `AUROC`** | Born v2 (`no purity`) | **`ECE < 0.035`, `AUROC > 0.82`** | **`ECE < 0.020`, `AUROC > 0.87` (`V3_A4` Lindblad > `V3_A3`)** |
| **Iso-Compute (`5+1=6` ep) Proximal LASSO `512 -> 128` (`V3_A6`)** | Smooth L1 (`0%` exact zeros) | **`> 25%` exact zeroed dyads, `< 0.03` ZS drop** | **`40–75%` exact zeroed dyads & `V3_A6_Recal > V3_A5` at identical 6 epochs** |
| **Production `vllm serve` E2E Speedup $\mathcal{S}_{\text{e2e}}(B)$** | Unmeasured in `vllm serve` | **`>= 2.2×` ($B=1$), `>= 1.35×` ($B=64$)** | **`2.5×–3.0×` ($B=1..8$) & `>= 1.25×` faster than `DSpark` SWA at $B \ge 64$** |
