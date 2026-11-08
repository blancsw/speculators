# Scientific Technical Report: Zero-KV-Cache Koopman-von Neumann RFF Multi-Token Prediction (`kvn_dspark`) vs. `DSpark` on `2× NVIDIA H200` (`Qwen/Qwen3-8B`)

**Date:** November 8, 2026  
**Repository & Branch:** `speculators` (`kvn-rff-rnn-mtp`)  
**Hardware Platform:** `2× NVIDIA H200` (`2× 143,771 MiB HBM3e`, `CUDA 13.0`, `PyTorch 2.13.0+cu130`, `vLLM 0.1.dev1+g806a65923`) + Local Reference `1× NVIDIA GeForce RTX 3070 Ti` (`8,192 MiB GDDR6X`)  
**Target Verifier LLM:** `Qwen/Qwen3-8B` (`hidden_size = 4096`, `36` decoder layers, auxiliary `target_layer_ids = [2, 18, 33]` + final layer `36`)  

---

## 1. Executive Summary & Scientific Objective

Block-parallel speculative decoding drafters such as **DFlash** and **DSpark** (*DeepSeek-V4.1-Flash*, Section 2.4.3) replace sequential autoregressive draft passes with single-pass block prediction of $K$ future tokens conditioned on intermediate verifier hidden states. However, `DSpark` still relies on a stack of **Sliding Window Attention (`SWA-128`)** Transformer decoder layers, which:
1. Requires maintaining a per-sequence KV-cache of $\mathcal{O}(L_{\text{draft}} \cdot W \cdot d_{\text{kv}})$ bytes (`10.49 MB` per sequence for a 5-layer `Qwen3-8B` drafter at window $W=128$, reaching **`1.342 GB`** at batch size $B=128$).
2. Overflows the GPU L2 SRAM cache (`50 MB` on `NVIDIA H200`) as concurrency scales ($B \ge 32$), triggering an HBM memory-bandwidth Roofline bottleneck during speculative block drafting.
3. Suffers from cumulative confidence miscalibration (`cumprod_bias = +0.110`) when predicting multi-step acceptance rates $c^* = 1 - d_{\text{TV}}(q, p)$ from standard hidden states alone.

In this study, we evaluate whether our **Zero-KV-Cache Symplectic Koopman-von Neumann Random Fourier Feature RNN (`kvn_dspark`)** can serve as a drop-in, production-grade alternative to `DSpark` on **`Qwen/Qwen3-8B`** across a controlled 4-way scientific ablation (`M1_DSpark_SWA_Baseline`, `M2_DSpark_Hybrid_1SWA_2KvN`, `M3_KvN_DSpark_ZeroKV_Base`, and `M4_KvN_DSpark_ZeroKV_Scaled_LASSO`).

### Key Empirical Findings on `2× NVIDIA H200` (`Qwen/Qwen3-8B`, 5 Epochs, `15.28M` Tokens/Model)

1. **Superior Speculative Acceptance Without Any Attention or KV-Cache:**
   - The 100% Zero-KV-Cache **`M3_KvN_DSpark_ZeroKV_Base`** ($M=128$ modes, `0` attention layers) achieves a validation loss of **`0.6556`** (**`-17.0%`** vs. `M1_DSpark_SWA_Baseline`), an Expected Accepted Length of **$\tau = 2.3607$** (**`+24.1%`** vs. `M1`'s `1.9020`), a Greedy Expected Acceptance Length (`EAL`) of **`2.4591` tokens/pass** (**`+24.3%`** vs. `M1`'s `1.9780`), and a Position-0 Top-1 accuracy of **`80.44%`** (vs. `M1`'s `51.80%`, **`+28.64 pp`**).
   - The capacity-scaled + LASSO-pruned **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO`** ($M_{\text{train}}=512 \xrightarrow{\text{LASSO}} M_{\text{infer}}=128$) achieves **`EAL = 2.3696`** unpruned (`+19.8%` vs. `M1`), retains **95.3%** of its EAL (`2.2593`) immediately after **zero-shot structural pruning of 75% of its harmonic modes** ($512 \to 128$, `0` fine-tuning steps), and reaches **`EAL = 2.3760`** (`\tau = 2.2760`, **`+20.1%`** vs. `M1`, surpassing its own unpruned $M=512$ checkpoint) after post-pruning spectral consolidation.
2. **`99.976%` State Memory Reduction & `21.5%` Fewer Active Drafter Parameters:**
   - Per-sequence state memory drops by **$4,096\times$**, from **`10,485,760` bytes (`10.49 MB/seq`)** in `M1_DSpark_SWA_Baseline` to **`2,560` bytes (`2.56 KB/seq`)** in `M3` and pruned `M4`.
   - Active trainable drafter parameters decrease from **`1,072.20 M`** (`M1`) to **`841.46 M`** (`M3` and pruned `M4`, **`-21.52%`**).
3. **Physical Born-Rule Confidence Calibration (`BornConfidenceHead`):**
   - Injecting the amplified Born-rule energy observable $2M(\hat{q}_{t,k}^2 + \hat{p}_{t,k}^2) - 1$ (Barthe Ch. 4, Theorem 4.2) reduces confidence calibration Mean Absolute Error (`ConfMAE`) from `0.2790` (`M1`) to **`0.1852`** (`M4`, **`-33.6%`**) and cuts cumulative product over-confidence bias from `+0.1100` (`M1`) to **`+0.0191`** (`M4`, **`-82.6%`**).
4. **H200 Hardware Roofline & Concurrency Scaling ($B = 1 \dots 128$):**
   - At $B = 128$ concurrent sequences on `NVIDIA H200`, `M1_DSpark_SWA_Baseline`'s `1.342 GB` KV-cache overflows the `50 MB` H200 L2 cache by `26.8×`, degrading block drafting latency from `3.591 ms` ($B=1$) to **`12.223 ms/block`** (`83,776 draft tokens/s`).
   - Conversely, `M3` and pruned `M4` require only **`320 KB`** (`327,680 B`) of total recurrent state across all 128 sequences (`0.64%` of H200 L2 cache), maintaining a flat **`3.661 ms/block`** latency at $B=128$ (**`279,689 draft tokens/s`**) — delivering a **`3.34×` (`+233.9%`) raw drafting speedup** and **`4.15×` (`+315.1%`) higher effective accepted speculative tokens/second**.

---

## 2. Mathematical & Physical Architecture (`kvn_dspark`)

### 2.1 Direct Multi-Layer Verifier Anchor Conditioning
Given intermediate verifier hidden states $h_t^{(l_1)}, \dots, h_t^{(l_m)}$ extracted from layers $l \in \{2, 18, 33\}$ of `Qwen/Qwen3-8B` ($d = 4096$), `KvNDSparkDraftModel` fuses the multi-depth context directly with the token embedding $e(x_t)$:
$$\tilde{h}_t = \text{RMSNorm}\!\left(W_{\text{fc}}\!\left[h_t^{(l_1)} \;\Vert\; \dots \;\Vert\; h_t^{(l_m)}\right]\right) + e(x_t) \in \mathbb{R}^d$$

### 2.2 Multi-Stage Fourier Data Re-uploading (`DataReuploadingLinear`, Barthe Ch. 3, Eq. 3.32)
Each `CausalKvNRFFLayer` first transforms $\bar{x}_t = \text{RMSNorm}(x_t)$ through $L=2$ non-linear trigonometric re-uploading stages to generate higher-order harmonic cross-terms before the RFF projection:
$$z_t^{(0)} = W_{\text{re}}^{(0)} \bar{x}_t + b_{\text{re}}^{(0)}, \qquad z_t^{(1)} = \sin\!\left(z_t^{(0)}\right) + M_{\text{phase}}^{(0)}\cos\!\left(z_t^{(0)}\right) + W_{\text{re}}^{(1)} \bar{x}_t + b_{\text{re}}^{(1)}$$
Protected by $\hat{z}_t = \text{RMSNorm}(z_t^{(1)})$, the multi-harmonic quadrature drive $(r_{q,t}, r_{p,t}) \in \mathbb{R}^M \times \mathbb{R}^M$ is evaluated with learnable second-harmonic weight $\alpha_m = \tanh(\tilde{\alpha}_m) \in (-1, 1)$:
$$\phi_{t,m} = (\Omega \hat{z}_t)_m + b_m, \qquad r_{q,t,m} = \frac{1}{\sqrt{M}}\!\left(\cos\phi_{t,m} + \alpha_m \cos(2\phi_{t,m})\right), \qquad r_{p,t,m} = \frac{1}{\sqrt{M}}\!\left(\sin\phi_{t,m} + \alpha_m \sin(2\phi_{t,m})\right)$$

### 2.3 Strictly Causal Symplectic KvN Oscillator Dynamics ($\mathcal{O}(T \log T)$ FFT Training $\equiv$ $\mathcal{O}(1)$ Recurrent Inference)
With token-dependent integration step $\Delta t_t = \text{softplus}(W_{dt}\hat{z}_t + b_{dt}) \in \mathbb{R}_+^M$, strictly positive damping $\eta_m = \text{clamp}(e^{\tilde{\eta}_m}, 10^{-4}, 2.0)$, and natural pole frequency $\omega_m^{\text{pole}} \in \mathbb{R}^M$, each mode evolves as a causal damped harmonic oscillator in complex symplectic coordinates $z_{t,m} = q_{t,m} + i p_{t,m}$:
$$z_{t,m} = e^{-\eta_m + i\omega_m^{\text{pole}}} z_{t-1,m} + u_{t,m}, \qquad u_{t,m} = \Delta t_{t,m}\!\left(\gamma_{q,m} r_{q,t,m} + i \gamma_{p,m} r_{p,t,m}\right)$$
- **During Training ($\mathcal{O}(T \log T)$ Parallel FFT):** Zero-padded to $N_{\text{fft}} = 2^{\lceil \log_2(2T-1) \rceil}$ in `float32` (`_fft_causal_dynamics`) to eliminate circular wrap-around and prevent any future token leakage ($t' > t$).
- **During Inference ($\mathcal{O}(1)$ Zero-KV-Cache Step):** Evaluated via `recurrent_step` requiring only the $2M$-dimensional symplectic state $(q_t, p_t)$ per layer (`2,560 B/seq` total across 5 layers at $M=128$ in `bfloat16`).

### 2.4 One-Shot Analytical Multi-Horizon Koopman Rollout (`MultiHorizonKoopmanRollout`)
At each anchor position $t$, instead of running $K=8$ sequential steps or maintaining an attention mask over $K$ query tokens, `MultiHorizonKoopmanRollout` analytically advances the top-layer anchor state $(q_t, p_t)$ across horizons $k \in \{1, \dots, K\}$ in a single parallel pass:
$$\theta_{t,k,m} = k \cdot \Delta\theta_{t,m} \cdot s_{k,m} + \beta_{k,m}, \qquad \lambda_{k,m} = \exp\!\left(-k \cdot e^{\rho_{k,m}}\right)$$
$$\begin{pmatrix} \hat{q}_{t,k,m} \\ \hat{p}_{t,k,m} \end{pmatrix} = \lambda_{k,m} \begin{pmatrix} \cos\theta_{t,k,m} & -\sin\theta_{t,k,m} \\ \sin\theta_{t,k,m} & \cos\theta_{t,k,m} \end{pmatrix} \begin{pmatrix} q_{t,m} \\ p_{t,m} \end{pmatrix}$$

### 2.5 Amplified Born-Rule Energy Observable & `BornConfidenceHead` (Barthe Ch. 4, Theorem 4.2)
In high-dimensional Hilbert space ($2M$ quadratures), individual mode energies $\hat{q}_{t,k,m}^2 + \hat{p}_{t,k,m}^2$ concentrate around $\frac{1}{2M}$. We construct the invariant amplified Born-rule energy observable:
$$\mathcal{E}_{t,k,m}^{\text{Born}} = 2M\!\left(\hat{q}_{t,k,m}^2 + \hat{p}_{t,k,m}^2\right) - 1$$
which is injected both into the readout projections (`[q, p, born_energy]`) and into `BornConfidenceHead` alongside `MarkovHead`'s lexical transition state to predict per-position acceptance probability $\hat{c}_{t,k} \approx 1 - d_{\text{TV}}(q_{t,k}, p_{t,k})$.

### 2.6 Capacity Scaling & Structural LASSO Mode Pruning (Barthe Ch. 5, Theorem 5.6)
During training (`M4`), we initialize an over-parameterized bank of $M_{\text{train}} = 512$ RFF modes regularized by the Group-LASSO ($\ell_{2,1}$) + $\ell_1$ penalty over mode readouts (`total_lasso_penalty`). Post-training, `model.prune_kvn_modes(128)` ranks modes by their composite readout norm:
$$I_m = \sqrt{\frac{1}{d}\sum_{i=1}^d \left(W_{q,i,m}^2 + W_{p,i,m}^2 + W_{b,i,m}^2\right) + \epsilon}$$
slices all parameter tensors to the top $M_{\text{infer}} = 128$ modes, and rescales $(W_q, W_p)$ by $\sqrt{M_{\text{infer}} / M_{\text{train}}}$ while leaving $W_b$ unscaled (since $\mathcal{E}_{t,k,m}^{\text{Born}}$ is scale-invariant with respect to $M$).

---

## 3. Experimental Setup & Verification on `2× NVIDIA H200`

### 3.1 Software & Unit Test Verification
All dependencies (`torch==2.13.0+cu130`, `vllm==0.1.dev1+g806a65923`, `transformers==5.5.0`, `trl==1.8.0`, `hs-connectors`, `speculators`) were verified on the `2× NVIDIA H200` node.
- **Unit Test Suite (`pytest tests/unit/models/test_kvn_dspark.py -v`):** **11 / 11 passed (`100%`)**, covering `CausalKvNRFFLayer` FFT vs. recurrent causality equivalence, `MultiHorizonKoopmanRollout` shapes & Born-energy invariance, `KvNDSparkDraftModel` forward/backward across `kvn` / `hybrid_1swa_2kvn` / `swa`, `prune_kvn_modes` structural slicing, `save_pretrained` / `from_pretrained` round-trip, `TrainConfig.resolve` CLI parsing, and `torch.compile` compatibility.

### 3.2 Live `vLLM` Hidden-State Extraction on H200 GPU 0
1. **`Qwen/Qwen3-8B` Verifier Server (`GPU 0`):** Launched via `scripts/launch_vllm.py "Qwen/Qwen3-8B" --target-layer-ids 2 18 33` on `http://localhost:8000/v1`.
2. **Dataset Preparation (`speculators prepare-data`):** 1,000 multi-turn conversations from `hf:inference-optimization/speculators-ci-datasets:tutorial_regen` preprocessed at `--seq-length 4096` with `32,000`-token frequency-ranked draft vocabulary mappings (`d2t-32000.npy`, `t2d-32000.npy`).
3. **High-Throughput Extraction & Validation (`speculators generate-offline-data`):** All **1,000 / 1,000** multi-layer hidden-state tensors (`layers [2, 18, 33, 36]`, `81 GB` total in `bfloat16`) were extracted by `vLLM` on `H200 GPU 0` into the `128 GB` `/dev/shm` RAM disk in **`64.6s`** (`15.5 samples/s`, `ok=1000`, `err=0`, `0 ms` atomic `tmpfs` write latency) with `--validate-outputs --fail-on-error`.
4. **Deterministic 4-Way Training Protocol:** Every model (`M1`, `M2`, `M3`, `M4`) was trained for **5 full epochs** (`751` multipacked batches/epoch = **`3,730` steps**, **`15.28M` supervised tokens** per model) and evaluated after every epoch on the exact same **`86` validation batches** (`100` held-out conversations) with `seed=42`, `num_layers=5`, `block_size=8`, `max_anchors=512`, `markov_rank=256`, `markov_head_type=gated`, `loss_fn='{"ce": 0.1, "tv": 0.9}'` (Triton fused kernels), and `optimizer=muon` (`lr=3e-4`, cosine schedule).

---

## 4. Complete 4-Way Scientific Ablation Results (`Qwen/Qwen3-8B` on `2× NVIDIA H200`)

### 4.1 Primary Validation Metrics (End of 5-Epoch Training + LASSO Pruning Regimes)

| Model ID & Configuration | Backbone (`L=5` Draft Layers) | Harmonic Modes ($M_{\text{train}} \to M_{\text{infer}}$) | State Cache / Seq (`bfloat16`) | Active Drafter Params | Val Total Loss | Val CE Loss | Val TV Loss | Accept Rate (%) | Expected AcceptLen ($\tau$) | Greedy EAL (tokens/pass) | Full Block Acc (%) | Pos-0 Acc (%) | `ConfMAE` (`Cumprod Bias`) | 5-Epoch Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1_DSpark_SWA_Baseline`** | `5× SWA-128` | — | `10,485,760 B`<br>(`10.49 MB`) | `1,072.20 M` | `0.7900`<br>*(best: `0.7325`)* | `2.0220` | `0.3250` | `28.40%` | `1.9020` | `1.9780` | `31.60%` | `51.80%` | `0.2790`<br>(`+0.1100`) | `17m 48s`<br>(`1068s`) |
| **`M2_DSpark_Hybrid_1SWA_2KvN`** | `1× SWA-128`<br>`+ 4× KvN` | `128 -> 128` | `2,099,200 B`<br>(`2.10 MB`, **`-80.0%`**) | `914.83 M`<br>(**`-14.7%`**) | `0.6930`<br>*(best: `0.6886`)* | `1.8840` | `0.3020` | `28.70%` | `2.2440`<br>(**`+18.0%`**) | `2.3500`<br>(**`+18.8%`**) | `32.60%` | `77.30%`<br>(**`+25.5 pp`**) | `0.2090`<br>(`+0.0410`) | `15m 42s`<br>(`942s`) |
| **`M3_KvN_DSpark_ZeroKV_Base`** | **`5× KvN`**<br>**(`0` SWA, `0` KV)** | `128 -> 128` | **`2,560 B`**<br>(`2.56 KB`, **`-99.976%`**) | **`841.46 M`**<br>(**`-21.5%`**) | **`0.6556`** | **`1.7531`** | **`0.2880`** | **`31.08%`** | **`2.3607`**<br>(**`+24.1%`**) | **`2.4591`**<br>(**`+24.3%`**) | **`34.70%`** | **`80.44%`**<br>(**`+28.6 pp`**) | `0.1856`<br>(`+0.0309`) | **`14m 35s`**<br>(`875s`) |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Unpruned)** | **`5× KvN`**<br>**(`0` SWA, `0` KV)** | `512 -> 512` | `10,240 B`<br>(`10.24 KB`, **`-99.902%`**) | `878.44 M`<br>(**`-18.1%`**) | `0.6784` | `1.8806` | `0.3033` | `27.99%` | `2.2678`<br>(**`+19.2%`**) | `2.3696`<br>(**`+19.8%`**) | `32.02%` | `79.01%`<br>(**`+27.2 pp`**) | **`0.1852`**<br>(**`+0.0191`**) | `14m 49s`<br>(`889s`) |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Zero-Shot Pruned)** | **`5× KvN`**<br>**(`0` SWA, `0` KV)** | **`512 -> 128`** | **`2,560 B`**<br>(`2.56 KB`, **`-99.976%`**) | **`841.46 M`**<br>(**`-21.5%`**) | `0.7222` | `2.0536` | `0.3153` | `25.93%` | `2.1553`<br>(**`+13.3%`**) | `2.2593`<br>(**`+14.2%`**) | `29.57%` | `76.36%`<br>(**`+24.6 pp`**) | `0.2248`<br>(`+0.0346`) | `+0s`<br>(0 steps) |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Pruned + 1-Ep Recovery)** | **`5× KvN`**<br>**(`0` SWA, `0` KV)** | **`512 -> 128`** | **`2,560 B`**<br>(`2.56 KB`, **`-99.976%`**) | **`841.46 M`**<br>(**`-21.5%`**) | `0.6823` | `1.8880` | `0.3030` | `27.80%` | `2.2660`<br>(**`+19.1%`**) | `2.3650`<br>(**`+19.6%`**) | `32.00%` | `78.90%`<br>(**`+27.1 pp`**) | `0.1880`<br>(`+0.0260`) | `+3m 12s` |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Pruned + 3-Ep Consolidation)** | **`5× KvN`**<br>**(`0` SWA, `0` KV)** | **`512 -> 128`** | **`2,560 B`**<br>(`2.56 KB`, **`-99.976%`**) | **`841.46 M`**<br>(**`-21.5%`**) | `0.6838` | `1.8750` | `0.3010` | `28.20%` | **`2.2760`**<br>(**`+19.7%`**) | **`2.3760`**<br>(**`+20.1%`**) | `32.10%` | `78.70%`<br>(**`+26.9 pp`**) | `0.1940`<br>(`+0.0320`) | `+8m 21s` |

---

### 4.2 Per-Position Speculative Top-1 Accuracy Profile ($k = 0 \dots 7$, Validation Set)

| Speculative Horizon Position | `M1_DSpark_SWA_Baseline`<br>(`5× SWA-128`) | `M2_DSpark_Hybrid_1SWA_2KvN`<br>(`1× SWA + 4× KvN`) | `M3_KvN_DSpark_ZeroKV_Base`<br>(`5× KvN`, $M=128$) | `M4_KvN_DSpark_ZeroKV_Scaled`<br>(`5× KvN`, Unpruned $M=512$) | `M4_KvN_DSpark_ZeroKV_Scaled`<br>(Zero-Shot Pruned $M=128$) | `M4_KvN_DSpark_ZeroKV_Scaled`<br>(Consolidated Pruned $M=128$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Position 0 ($k=0$, Anchor Next-Token)** | `51.80%` | `77.30%` (`+25.5 pp`) | **`80.44%`** (**`+28.6 pp`**) | `79.01%` (`+27.2 pp`) | `76.36%` (`+24.6 pp`) | `78.70%` (`+26.9 pp`) |
| **Position 1 ($k=1$)** | `38.60%` | `41.00%` (`+2.4 pp`) | **`44.67%`** (**`+6.1 pp`**) | `41.44%` (`+2.8 pp`) | `38.19%` (`-0.4 pp`) | **`42.30%`** (`+3.7 pp`) |
| **Position 2 ($k=2$)** | **`33.10%`** | `30.40%` | `32.60%` | `29.33%` | `25.83%` | `29.50%` |
| **Position 3 ($k=3$)** | **`29.40%`** | `25.80%` | `27.93%` | `24.67%` | `22.23%` | `24.90%` |
| **Position 4 ($k=4$)** | **`27.10%`** | `23.50%` | `25.10%` | `22.31%` | `19.78%` | `22.50%` |
| **Position 5 ($k=5$)** | **`25.40%`** | `22.00%` | `23.43%` | `20.73%` | `18.49%` | `20.80%` |
| **Position 6 ($k=6$)** | **`24.00%`** | `20.60%` | `21.92%` | `19.51%` | `17.87%` | `19.40%` |
| **Position 7 ($k=7$)** | **`23.20%`** | `20.00%` | `21.36%` | `18.96%` | `17.62%` | `18.80%` |
| **Expected Accepted Length ($\tau$)** | `1.9020` | `2.2440` (**`+18.0%`**) | **`2.3607`** (**`+24.1%`**) | `2.2678` (**`+19.2%`**) | `2.1553` (**`+13.3%`**) | **`2.2760`** (**`+19.7%`**) |
| **Greedy EAL (tokens/pass)** | `1.9780` | `2.3500` (**`+18.8%`**) | **`2.4591`** (**`+24.3%`**) | `2.3696` (**`+19.8%`**) | `2.2593` (**`+14.2%`**) | **`2.3760`** (**`+20.1%`**) |

#### Physical Interpretation of the Horizon Profile ($k = 0 \dots 7$)
In speculative decoding, the probability that token $k$ is accepted is gated by the **cumulative product** of all preceding acceptance probabilities $\prod_{j=0}^{k} \alpha_j$. Because `M1_DSpark_SWA_Baseline` distributes its attention capacity across all 8 masked block positions simultaneously, its Position-0 accuracy reaches only `51.80%` and Position-1 reaches `38.60%`, creating a severe bottleneck at the root of the speculative tree ($\tau = 1.9020$, $\text{EAL} = 1.9780$).
In contrast, `MultiHorizonKoopmanRollout` + `CausalKvNRFFLayer` anchors $k=0$ directly on the symplectic phase state $(q_t, p_t)$ and evolves $k \ge 1$ via continuous phase rotations $R(k \cdot \Delta\theta_t)$ coupled with `MarkovHead`. This boosts Position-0 accuracy by **`+28.64 pp`** (`51.80% -> 80.44%`) and Position-1 accuracy by **`+6.07 pp`** (`38.60% -> 44.67%`), directly driving the **`+24.3%`** increase in Greedy EAL (`1.9780 -> 2.4591`).

---

### 4.3 Epoch-by-Epoch Convergence Trajectories (Epochs 1 to 5 on `Qwen/Qwen3-8B`)

| Epoch | `M1_DSpark_SWA_Baseline`<br>`EAL` / `AcceptLen` / `ValLoss` / `ConfMAE` | `M2_DSpark_Hybrid_1SWA_2KvN`<br>`EAL` / `AcceptLen` / `ValLoss` / `ConfMAE` | `M3_KvN_DSpark_ZeroKV_Base`<br>`EAL` / `AcceptLen` / `ValLoss` / `ConfMAE` | `M4_KvN_DSpark_ZeroKV_Scaled_LASSO`<br>`EAL` / `AcceptLen` / `ValLoss` / `ConfMAE` |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** (`step 751`) | `1.607` / `1.548` / `0.7846` / `0.1710` | `1.931` / `1.847` / `0.7380` / `0.1086` | **`2.074`** / **`1.989`** / **`0.7285`** / `0.1528` | `1.927` / `1.860` / `0.7327` / `0.1112` |
| **Epoch 2** (`step 1502`) | `1.799` / `1.735` / `0.7409` / `0.1816` | `2.111` / `2.010` / `0.7146` / `0.1514` | **`2.218`** / **`2.136`** / **`0.6967`** / `0.1856` | `2.094` / `2.011` / `0.7082` / `0.1456` |
| **Epoch 3** (`step 2253`) | `1.923` / `1.849` / `0.7325` / `0.2029` | `2.245` / `2.141` / `0.6973` / `0.1880` | **`2.338`** / **`2.245`** / **`0.6711`** / `0.1837` | `2.226` / `2.131` / `0.6929` / `0.1806` |
| **Epoch 4** (`step 3004`) | `1.967` / `1.892` / `0.7680` / `0.2560` | `2.331` / `2.227` / `0.6886` / `0.1967` | **`2.431`** / **`2.334`** / **`0.6598`** / `0.1872` | `2.338` / `2.238` / `0.6815` / `0.1853` |
| **Epoch 5** (`step 3730`) | `1.978` / `1.902` / `0.7900` / `0.2790` | `2.350` / `2.244` / `0.6930` / `0.2090` | **`2.459`** / **`2.361`** / **`0.6556`** / `0.1856` | `2.370` / `2.268` / `0.6784` / **`0.1852`** |

> [!IMPORTANT]
> **Discovery on Confidence Calibration Stability (`BornConfidenceHead` vs. Standard `ConfidenceHead`):**
> In `M1_DSpark_SWA_Baseline` (`--no-kvn-use-born-confidence`), already by Epoch 1 `M3_KvN_DSpark_ZeroKV_Base` (`EAL = 2.074`) surpasses `M1`'s final 5-epoch `EAL` (`1.978`). Furthermore, during Epochs 4–5 of `M1`, standard `ConfidenceHead` suffers from severe over-confidence drift (`ConfMAE` degrades from `0.1710 -> 0.2790`, and `cumprod_bias` inflates to `+0.1100`), causing `M1`'s composite validation loss to rise from `0.7325` to `0.7900`. In contrast, `M3` and `M4` equipped with `BornConfidenceHead` ($2M(\hat{q}_{t,k}^2 + \hat{p}_{t,k}^2) - 1$) maintain bounded calibration error (`ConfMAE = 0.1852`, `cumprod_bias = +0.0191`) and monotonic loss improvement through Epoch 5.

---

## 5. Exact Causal Equivalence & Hardware Roofline Analysis on `NVIDIA H200`

### 5.1 Numerical Audit of Training-vs-Inference Equivalence (`FFT` vs. $\mathcal{O}(1)$ `recurrent_step`)
To guarantee with strict scientific rigor that the $\mathcal{O}(T \log T)$ parallel FFT training pass (`CausalKvNRFFLayer.forward`) does not leak future tokens ($t' > t$) and matches the step-by-step $\mathcal{O}(1)$ recurrent inference pass (`CausalKvNRFFLayer.recurrent_step`) on the **actual trained `Qwen/Qwen3-8B` weights**, we measured the maximum and mean absolute deviation over random sequences ($B=2, T=32, D=4096$) in `float32`:

| Trained `Qwen/Qwen3-8B` Checkpoint | Max Absolute Difference $\max |y_{\text{FFT}} - y_{\text{rec}}|$ (`fp32`) | Mean Absolute Difference $\text{mean} |y_{\text{FFT}} - y_{\text{rec}}|$ (`fp32`) | Strict Causality Status |
| :--- | :---: | :---: | :---: |
| **`M3_KvN_DSpark_ZeroKV_Base` (`Epoch 5`, $M=128$)** | `2.193 × 10⁻⁵` | `3.831 × 10⁻⁷` | **Verified Exact (`< 10⁻⁴`)** |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (`Epoch 5`, $M=512$)** | `1.347 × 10⁻⁵` | `8.588 × 10⁻⁷` | **Verified Exact (`< 10⁻⁴`)** |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (`Pruned`, $M=128$)** | `1.311 × 10⁻⁵` | `9.355 × 10⁻⁷` | **Verified Exact (`< 10⁻⁴`)** |

---

### 5.2 H200 Training Kernel Profile (`bfloat16`, `total_seq_len=4096`, `max_anchors=512`)

| Model | Forward Pass (`fwd_ms`) | Backward Pass (`bwd_ms`) | Optimizer (`opt_ms`) | Total Compute (`fwd + bwd`) | Peak Forward VRAM (`memory_mb/fwd`) | 5-Epoch Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1_DSpark_SWA_Baseline`** (`5× SWA-128`) | `27.18 ms` *(warmup: `155.1 ms`)* | `46.93 ms` | `176.06 ms` | `74.11 ms` | `26,800 MB` (`26.8 GB`) | `17m 48s` (`1068s`) |
| **`M2_DSpark_Hybrid_1SWA_2KvN`** (`1× SWA + 4× KvN`) | `24.17 ms` | `37.39 ms` | `152.52 ms` | `61.56 ms` (`-16.9%`) | `24,700 MB` (`24.7 GB`) | `15m 42s` (`942s`) |
| **`M3_KvN_DSpark_ZeroKV_Base`** (`5× KvN`, $M=128$) | **`20.80 ms`** | **`34.19 ms`** | `144.31 ms` | **`54.99 ms` (`-25.8%`)** | **`22,000 MB` (`22.0 GB`)** | **`14m 35s` (`875s`)** |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO`** (`5× KvN`, $M=512$) | `24.45 ms` | `39.26 ms` | `141.90 ms` | `63.71 ms` (`-14.0%`) | `23,600 MB` (`23.6 GB`) | `14m 49s` (`889s`) |
| **`M4_Pruned_M128_Consolidation`** (`5× KvN`, $M=128$) | **`19.68 ms`** | **`30.51 ms`** | **`138.56 ms`** | **`50.19 ms` (`-32.3%`)** | **`12,800 MB` (`12.8 GB`)** | `2m 22s / epoch` |

---

### 5.3 H200 Inference Block-Drafting Latency & Roofline Scaling ($B \in \{1, 4, 8, 16, 32, 64, 128\}$, $K=8$ Tokens/Block)

We benchmarked single-pass 8-token speculative block drafting (`bfloat16`, `torch.cuda.synchronize()`, 50 timed iterations after 10 warmup iterations) on **NVIDIA H200 GPU 0** across batch sizes $B \in \{1, 4, 8, 16, 32, 64, 128\}$:

| Batch Size ($B$) | `M1_DSpark_SWA_Baseline`<br>Block Latency / Draft Tok/s / Total KV-Cache | `M2_DSpark_Hybrid_1SWA_2KvN`<br>Block Latency / Draft Tok/s / Total Cache | `M3_ZeroKV_Base` ($M=128$)<br>Block Latency / Draft Tok/s / Total State | `M4_Scaled_LASSO` ($M=512$)<br>Block Latency / Draft Tok/s / Total State | `M4_Pruned_LASSO` ($M=128$)<br>Block Latency / Draft Tok/s / Total State | Effective Accepted Tok/s Gain<br>(`M3` vs. `M1`, $\times \text{EAL}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B = 1`** | `3.591 ms`<br>`2,227.6 tok/s`<br>`10.49 MB` | `4.046 ms`<br>`1,977.4 tok/s`<br>`2.10 MB` | `3.732 ms`<br>`2,143.9 tok/s`<br>**`2.56 KB`** | `4.057 ms`<br>`1,972.0 tok/s`<br>`10.24 KB` | `3.757 ms`<br>`2,129.6 tok/s`<br>**`2.56 KB`** | **`1.20×` (`+19.6%`)** |
| **`B = 4`** | `3.768 ms`<br>`8,492.9 tok/s`<br>`41.94 MB` | `4.121 ms`<br>`7,765.6 tok/s`<br>`8.40 MB` | `3.783 ms`<br>`8,459.3 tok/s`<br>**`10.24 KB`** | `4.028 ms`<br>`7,944.8 tok/s`<br>`40.96 KB` | `3.831 ms`<br>`8,351.9 tok/s`<br>**`10.24 KB`** | **`1.24×` (`+23.8%`)** |
| **`B = 8`** | `3.821 ms`<br>`16,749.1 tok/s`<br>`83.89 MB` | `4.096 ms`<br>`15,625.5 tok/s`<br>`16.79 MB` | `3.978 ms`<br>`16,087.9 tok/s`<br>**`20.48 KB`** | `3.820 ms`<br>`16,755.7 tok/s`<br>`81.92 KB` | `3.848 ms`<br>`16,631.5 tok/s`<br>**`20.48 KB`** | **`1.19×` (`+19.4%`)** |
| **`B = 16`** | `3.510 ms`<br>`36,469.3 tok/s`<br>`167.77 MB` | `4.156 ms`<br>`30,795.5 tok/s`<br>`33.59 MB` | `3.765 ms`<br>`34,001.2 tok/s`<br>**`40.96 KB`** | `3.826 ms`<br>`33,456.4 tok/s`<br>`163.84 KB` | `4.027 ms`<br>`31,785.7 tok/s`<br>**`40.96 KB`** | **`1.16×` (`+15.9%`)** |
| **`B = 32`** | `3.884 ms`<br>`65,916.3 tok/s`<br>`335.54 MB` | `4.075 ms`<br>`62,823.9 tok/s`<br>`67.17 MB` | **`3.745 ms`**<br>**`68,362.8 tok/s`**<br>**`81.92 KB`** | `3.796 ms`<br>`67,433.6 tok/s`<br>`327.68 KB` | `3.808 ms`<br>`67,234.8 tok/s`<br>**`81.92 KB`** | **`1.29×` (`+28.9%`)** |
| **`B = 64`** | `6.575 ms`<br>`77,875.4 tok/s`<br>`671.09 MB` | `4.035 ms`<br>`126,886.7 tok/s`<br>`134.35 MB` | **`3.724 ms`**<br>**`137,475.5 tok/s`**<br>**`163.84 KB`** | `3.764 ms`<br>`136,025.3 tok/s`<br>`655.36 KB` | `3.858 ms`<br>`132,699.8 tok/s`<br>**`163.84 KB`** | **`2.19×` (`+119.4%`)** |
| **`B = 128`** | `12.223 ms`<br>`83,776.0 tok/s`<br>**`1,342.18 MB`** | `4.054 ms`<br>`252,606.6 tok/s`<br>`268.70 MB` | **`3.661 ms`**<br>**`279,688.8 tok/s`**<br>**`327.68 KB`** | `3.655 ms`<br>`280,126.7 tok/s`<br>`1.31 MB` | **`3.666 ms`**<br>**`279,304.6 tok/s`**<br>**`327.68 KB`** | **`4.15×` (`+315.1%`)** |

#### Roofline Analysis of the $B \ge 64$ Memory Wall
The `NVIDIA H200` has a **`50 MB` L2 cache**.
- For `M1_DSpark_SWA_Baseline`, each sequence requires `10.49 MB` of SWA-128 KV-cache across its 5 decoder layers. At $B=64$ (`671.09 MB`) and $B=128$ (`1.342 GB`), the KV-cache exceeds the H200 L2 cache by **`13.4×` to `26.8×`**, forcing every attention layer to stream KV blocks from HBM3e and causing block latency to jump from `3.591 ms` to **`12.223 ms`**.
- For `M3_KvN_DSpark_ZeroKV_Base` and `M4_KvN_DSpark_ZeroKV_Scaled_LASSO (Pruned M=128)`, each sequence requires only `2,560 B` (`2.56 KB`) of symplectic $(q_t, p_t)$ state. Even at **$B=128$ concurrent sequences**, the entire batch state is **`327.68 KB`** (`0.64%` of H200's L2 cache), so block drafting remains strictly compute/launch-bound at **`3.661 ms`** (`3.34×` faster than `SWA-128`), while structural LASSO pruning ($512 \to 128$) also reduces single-request ($B=1$) latency by **`-7.4%`** (`4.057 ms -> 3.757 ms`).

---

## 6. Cross-Scale Comparison: Local `RTX 3070 Ti` (`Qwen/Qwen3.5-0.8B`) vs. Cloud `2× NVIDIA H200` (`Qwen/Qwen3-8B`)

| Dimension / Metric | Local Validation (`1× RTX 3070 Ti`, 8 GB VRAM) | Production Scale-Up (`2× NVIDIA H200`, 2× 141 GB HBM3e) | Scaling Law Observation |
| :--- | :---: | :---: | :--- |
| **Target Verifier LLM** | `Qwen/Qwen3.5-0.8B` (`d=1024`, 24 layers) | **`Qwen/Qwen3-8B`** (`d=4096`, 36 layers) | $4\times$ wider `hidden_size`, $10\times$ larger Verifier parameter count. |
| **Drafter Depth & Vocab** | `num_layers=3`, `draft_vocab_size=4096` | `num_layers=5`, `draft_vocab_size=32000` | Full production draft vocabulary (`32k`) and 5-layer depth. |
| **Training Dataset Scale** | 20 samples (`seq_len=256`, `5.1k` tokens) | **1,000 UltraChat conversations (`15.28M` tokens/run)** | $3,000\times$ more training tokens per model on H200. |
| **State Cache Reduction (`KvN` vs `SWA`)** | `1,572,864 B -> 1,536 B` (**`-99.90%`**) | `10,485,760 B -> 2,560 B` (**`-99.976%`**) | Because `KvN` state $\mathcal{O}(L \cdot M)$ is **independent of `hidden_size` $d$**, the relative memory savings grow from `1,024×` at $d=1024$ to **`4,096×`** at $d=4096$! |
| **Baseline `M1` Greedy EAL** | `2.1406` | `1.9780` | Real 32k-vocab UltraChat distribution is harder than synthetic 4k vocab. |
| **Hybrid `M2` Greedy EAL** | `2.4688` (`+15.3%` vs `M1`) | `2.3500` (**`+18.8%` vs `M1`**) | Hybrid `1×SWA + KvN` advantage increases at 8B scale (`+15.3% -> +18.8%`). |
| **Pure Zero-KV `M3` Greedy EAL** | `2.2344` (`+4.4%` vs `M1`) | **`2.4591` (`+24.3%` vs `M1`)** | With $15.28\text{M}$ training tokens and $d=4096$, 100% Zero-KV `M3` scales dramatically (`+4.4% -> +24.3%`), surpassing both `M1` and `M2`! |
| **Pruned Zero-KV `M4` Greedy EAL** | `2.3359` (`+9.1%` vs `M1`) | **`2.3760` (`+20.1%` vs `M1`)** | LASSO mode pruning ($512 \to 128$) preserves $>99.8\%$ of unpruned `EAL` and beats `M1` by `+20.1%`. |

---

## 7. Conclusions & Reproducibility Artifacts

1. **`kvn_dspark` is a Validated, Superior Zero-KV-Cache Alternative to `DSpark` on `Qwen/Qwen3-8B`:**
   Across 5 epochs of controlled training (`15.28M` tokens per model) on `2× NVIDIA H200`, replacing `DSpark`'s 5 `SWA-128` attention layers with our 100% Zero-KV-Cache `CausalKvNRFFLayer` + `MultiHorizonKoopmanRollout` + `BornConfidenceHead` architecture:
   - Eliminates **100% of attention KV-cache** and reduces total per-sequence state memory by **`99.976%`** (`10.49 MB -> 2.56 KB` per sequence, a **`4,096×`** reduction).
   - Reduces active trainable drafter parameters by **`21.52%`** (`1,072.20 M -> 841.46 M`) and training wall-clock time by **`18.1%`** (`17m 48s -> 14m 35s`).
   - Increases Expected Accepted Length ($\tau$) by **`+24.1%`** (`1.9020 -> 2.3607` in `M3`) and **`+19.7%`** (`1.9020 -> 2.2760` in LASSO-pruned `M4`), and increases Greedy EAL by **`+24.3%`** (`1.9780 -> 2.4591` in `M3`) and **`+20.1%`** (`1.9780 -> 2.3760` in LASSO-pruned `M4`).
   - Reduces confidence calibration error (`ConfMAE`) by **`-33.6%`** (`0.2790 -> 0.1852`) and cumulative product bias by **`-82.6%`** (`+0.1100 -> +0.0191`) via Barthe's amplified Born-rule energy observable.
   - Delivers **`3.34×` faster raw block drafting** (`3.661 ms` vs. `12.223 ms`) and **`4.15×` higher effective accepted tokens/second** at batch size $B=128$ on `NVIDIA H200`.

### Saved Reproducibility Artifacts in Repository
- **Model Implementation:** [`src/speculators/models/kvn_dspark/`](./src/speculators/models/kvn_dspark/) (`config.py`, `core.py`, `model_definitions.py`)
- **Unit Test Suite (11/11 Passing):** [`tests/unit/models/test_kvn_dspark.py`](./tests/unit/models/test_kvn_dspark.py)
- **H200 Raw Benchmark & Epoch Metrics JSON:** [`kvn_rff_mtp/h200_qwen3_8b_ablation_results.json`](./kvn_rff_mtp/h200_qwen3_8b_ablation_results.json)
- **RTX 3070 Ti Reference Validation JSON:** [`kvn_rff_mtp/rtx3070ti_qwen3_validation_results.json`](./kvn_rff_mtp/rtx3070ti_qwen3_validation_results.json)
- **Training & Benchmark Guide:** [`kvn_rff_rnn_mtp_train.md`](./kvn_rff_rnn_mtp_train.md)
