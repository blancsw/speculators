# KvN-RFF-RNN MTP (`kvn_dspark`) — Guide Complet d'Entraînement & Benchmark (`RTX 3070 Ti` & `2× NVIDIA H200`)

Ce document décrit comment entraîner, élaguer (LASSO) et évaluer le modèle de décodage spéculatif **Zero-KV-Cache Koopman-von Neumann Multi-Token Prediction (`kvn_dspark`)** intégré dans `speculators` (branche `kvn-rff-rnn-mtp`).

---

## 1. Architecture & Principe (`--speculator-type kvn_dspark`)

`kvn_dspark` remplace les couches d'attention `SWA-128` de `DSpark` (*DeepSeek-V4.1-Flash*, Section 2.4.3) par un backbone symplectique **100 % sans KV-cache** :

1. **Conditionnement d'ancre direct :** Les états cachés intermédiaires du *Verifier* gelé sont fusionnés via $\tilde{h}_t = \text{RMSNorm}(W_{\text{fc}}[h_t^{(l_1)} \Vert \dots \Vert h_t^{(l_m)}]) + e(x_t)$.
2. **Couches causales `CausalKvNRFFLayer` :** Encodage par *Data Re-uploading* de Fourier (Barthe Ch. 3, Éq. 3.32) et oscillateurs harmoniques amortis de Koopman-von Neumann ($\mathcal{O}(T \log T)$ par FFT à l'entraînement, état récurrent $\mathcal{O}(1)$ $(q_t, p_t) \in \mathbb{R}^{2M}$ en inférence).
3. **Rollout Multi-Horizon en 1 passe (`MultiHorizonKoopmanRollout`) :** Projette analytiquement l'état d'ancre $(q_t, p_t)$ vers les $K$ positions spéculatives futures par rotations de phase $R(k \cdot \Delta\theta_t)$ sans exécuter $K$ passes séquentielles.
4. **Décodage Semi-Autorégressif (`MarkovHead`) & Calibration Physique (`BornConfidenceHead`) :** Combine le biais lexical `MarkovHead` avec l'observable d'énergie de Born amplifiée $2M(\hat{q}_{t,k}^2 + \hat{p}_{t,k}^2) - 1$ (Barthe Ch. 4, Théorème 4.2) pour prédire le taux d'acceptation analytique $c^* = 1 - d_{\text{TV}}(q, p)$.
5. **Scaling Capacitif + Pruning Structurel LASSO (Barthe Ch. 5, Théorème 5.6) :** Entraînement avec une large banque de modes RFF ($M_{\text{train}} = 512$) sous régularisation $\ell_1$, suivi d'un élagage structurel (`model.prune_kvn_modes(128)`) avant l'inférence pour respecter la bande passante HBM GPU (*Roofline Model*).

---

## 2. Installation sur votre Serveur `2× NVIDIA H200`

```bash
git clone -b kvn-rff-rnn-mtp git@github.com:blancsw/speculators.git
cd speculators

# Installation de hs_connectors et speculators en mode éditable
pip install -e hs_connectors -e .
```

---

## 3. Choix du LLM de Base (*Verifier*) & `target_layer_ids`

| Matériel Cible | LLM Verifier Recommandé (`--verifier-name-or-path`) | `--target-layer-ids` | `hidden_size` | Usage Recommandé |
| :--- | :--- | :---: | :---: | :--- |
| **1× RTX 3070 Ti (8 Go VRAM)** | `Qwen/Qwen3.5-0.8B` ou `Qwen/Qwen3-0.6B` | `2 12 21` (0.8B) / `2 14 25` (0.6B) | `1024` | Validation locale E2E de la pipeline `speculators.train` + ablation rapide. |
| **2× NVIDIA H200 (2×141 Go)** | **`Qwen/Qwen3-8B`** | `2 18 33` | `4096` | **Benchmark standard `speculators`** (comparaison directe avec DFlash / DSpark / EAGLE-3 sur UltraChat, ShareGPT, HumanEval, GSM8K). |
| **2× NVIDIA H200 (2×141 Go)** | **`Qwen/Qwen3-32B`** | `2 32 61` | `5120` | **Scale-Up Haute Capacité** (`GPU 0` : vLLM en `bfloat16` avec $>70\text{ Go}$ de KV-cache ; `GPU 1` : entraînement `kvn_dspark`). |

---

## 4. Lancement de l'Entraînement sur `2× NVIDIA H200`

### 4.1 Entraînement Automatisé en 1 Commande (vLLM Online + Train + Pruning LASSO)

Le script `examples/train/kvn_dspark_2xh200_online.sh` orchestre automatiquement :
1. **Étape 0 :** Régénération des réponses cibles avec `vLLM` (si le fichier JSONL n'existe pas encore),
2. **Étape 1 :** Démarrage du serveur `vLLM` sur **`GPU 0`** avec extraction des `target_layer_ids`,
3. **Étape 2 :** Préparation du dataset Arrow et calcul de `token_freq.pt` (`speculators prepare-data`),
4. **Étape 3 :** Entraînement de `kvn_dspark` sur **`GPU 1`** (`torchrun -m speculators.train` avec optimiseur `Muon` + `AdamW` et $M_{\text{train}} = 512$),
5. **Étape 4 :** Élagage structurel LASSO $M_{\text{train}} = 512 \to M_{\text{infer}} = 128$ et sauvegarde du checkpoint final élagué.

```bash
# Sur Qwen/Qwen3-8B (Standard Benchmark)
MODEL="Qwen/Qwen3-8B" \
TARGET_LAYER_IDS="2 18 33" \
KVN_BACKBONE_TYPE="kvn" \
KVN_N_MODES_TRAIN=512 \
KVN_N_MODES_INFER=128 \
bash examples/train/kvn_dspark_2xh200_online.sh

# Sur Qwen/Qwen3-32B (Haute Capacité 2xH200)
MODEL="Qwen/Qwen3-32B" \
TARGET_LAYER_IDS="2 32 61" \
KVN_BACKBONE_TYPE="kvn" \
KVN_N_MODES_TRAIN=512 \
KVN_N_MODES_INFER=128 \
bash examples/train/kvn_dspark_2xh200_online.sh
```

### 4.2 Commande `torchrun` Directe (Protocole d'Ablation 4-Voies sur `2×H200`)

Une fois le serveur `vLLM` actif sur `GPU 0` (`http://localhost:8000/v1`) et `prepare-data` exécuté dans `./output/qwen3_8b_data`, vous pouvez lancer les **4 régimes de l'ablation scientifique** sur `GPU 1` :

#### 1) `M1_DSpark_SWA_Baseline` (3 couches SWA-128)
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 -m speculators.train \
  --verifier-name-or-path "Qwen/Qwen3-8B" \
  --data-path "./output/qwen3_8b_data" \
  --vllm-endpoint "http://localhost:8000/v1" \
  --save-path "./output/checkpoints/M1_DSpark_SWA_Baseline" \
  --speculator-type kvn_dspark \
  --kvn-backbone-type swa \
  --no-kvn-use-born-confidence \
  --draft-vocab-size 32000 \
  --num-layers 5 --sliding-window 128 --block-size 8 --max-anchors 1024 \
  --target-layer-ids 2 18 33 \
  --markov-rank 256 --markov-head-type gated \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --optimizer muon --lr 3e-4 --scheduler-type cosine --epochs 5 \
  --on-missing generate
```

#### 2) `M2_DSpark_Hybrid_1SWA_2KvN` (1 couche SWA-128 + couches KvN-RFF)
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 -m speculators.train \
  --verifier-name-or-path "Qwen/Qwen3-8B" \
  --data-path "./output/qwen3_8b_data" \
  --vllm-endpoint "http://localhost:8000/v1" \
  --save-path "./output/checkpoints/M2_DSpark_Hybrid_1SWA_2KvN" \
  --speculator-type kvn_dspark \
  --kvn-backbone-type hybrid_1swa_2kvn \
  --kvn-n-modes 128 --kvn-reupload-layers 2 --kvn-lasso-weight 1e-5 --kvn-use-born-confidence \
  --draft-vocab-size 32000 \
  --num-layers 5 --sliding-window 128 --block-size 8 --max-anchors 1024 \
  --target-layer-ids 2 18 33 \
  --markov-rank 256 --markov-head-type gated \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --optimizer muon --lr 3e-4 --scheduler-type cosine --epochs 5 \
  --on-missing generate
```

#### 3) `M3_KvN_DSpark_ZeroKV_Base` (100 % KvN-RFF, 0 KV-Cache, $M=128$)
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 -m speculators.train \
  --verifier-name-or-path "Qwen/Qwen3-8B" \
  --data-path "./output/qwen3_8b_data" \
  --vllm-endpoint "http://localhost:8000/v1" \
  --save-path "./output/checkpoints/M3_KvN_DSpark_ZeroKV_Base" \
  --speculator-type kvn_dspark \
  --kvn-backbone-type kvn \
  --kvn-n-modes 128 --kvn-reupload-layers 2 --kvn-lasso-weight 1e-5 --kvn-use-born-confidence \
  --draft-vocab-size 32000 \
  --num-layers 5 --block-size 8 --max-anchors 1024 \
  --target-layer-ids 2 18 33 \
  --markov-rank 256 --markov-head-type gated \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --optimizer muon --lr 3e-4 --scheduler-type cosine --epochs 5 \
  --on-missing generate
```

#### 4) `M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (100 % KvN-RFF, 0 KV-Cache, $M_{\text{train}}=512 \xrightarrow{\text{LASSO}} M_{\text{infer}}=128$)
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 -m speculators.train \
  --verifier-name-or-path "Qwen/Qwen3-8B" \
  --data-path "./output/qwen3_8b_data" \
  --vllm-endpoint "http://localhost:8000/v1" \
  --save-path "./output/checkpoints/M4_KvN_DSpark_ZeroKV_Scaled_LASSO" \
  --speculator-type kvn_dspark \
  --kvn-backbone-type kvn \
  --kvn-n-modes 512 --kvn-reupload-layers 2 --kvn-lasso-weight 2e-4 --kvn-use-born-confidence \
  --draft-vocab-size 32000 \
  --num-layers 5 --block-size 8 --max-anchors 1024 \
  --target-layer-ids 2 18 33 \
  --markov-rank 256 --markov-head-type gated \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --optimizer muon --lr 3e-4 --scheduler-type cosine --epochs 5 \
  --on-missing generate

# Élagage structurel LASSO post-entraînement (512 -> 128 modes actifs) :
python -c "
from speculators.models.kvn_dspark import KvNDSparkDraftModel
model = KvNDSparkDraftModel.from_pretrained('./output/checkpoints/M4_KvN_DSpark_ZeroKV_Scaled_LASSO/4')
model.prune_kvn_modes(128)
model.save_pretrained('./output/checkpoints/M4_KvN_DSpark_ZeroKV_Scaled_LASSO/pruned_m128')
"
```

---

## 5. Référence des Arguments CLI `kvn_dspark`

| Flag CLI | Défaut | Description |
| :--- | :---: | :--- |
| `--speculator-type kvn_dspark` | — | Sélectionne l'architecture `KvNDSparkDraftModel`. |
| `--kvn-backbone-type` | `kvn` | `kvn` (100 % sans KV-cache), `hybrid_1swa_2kvn` (1 couche SWA + couches KvN), ou `swa` (baseline DSpark SWA). |
| `--kvn-n-modes` | `256` | Nombre de modes harmoniques RFF $M$ par couche `CausalKvNRFFLayer`. |
| `--kvn-reupload-layers` | `2` | Nombre d'étages de *Data Re-uploading* $L$ de Barthe (Ch. 3, Éq. 3.32). |
| `--kvn-sigma-g` | `1.0` | Bande passante spectrale $\sigma_g$ d'initialisation des fréquences RFF. |
| `--kvn-lasso-weight` | `1e-4` | Poids de la pénalité LASSO $\ell_1$ (Barthe Ch. 5, Théorème 5.6) sur les colonnes $(q_m, p_m, q_m^2+p_m^2)$ du readout. |
| `--kvn-use-born-confidence` / `--no-kvn-use-born-confidence` | `True` | Injecte l'observable d'énergie de Born amplifiée $2M(\hat{q}_{t,k}^2+\hat{p}_{t,k}^2)-1$ dans `BornConfidenceHead`. |

---

## 6. Validation Locale sur `RTX 3070 Ti` (8 Go VRAM) & Résultats de Référence

Pour exécuter la validation complète de bout en bout sur la machine locale (`Qwen/Qwen3.5-0.8B`) ainsi que les tests unitaires :

```bash
# 1. Tests unitaires du modèle, du rollout de Koopman, du pruning LASSO et de la config CLI
pytest tests/unit/models/test_kvn_dspark.py -v

# 2. Validation E2E de speculators.train sur RTX 3070 Ti (extraction Qwen/Qwen3.5-0.8B + ablation M1..M4)
python examples/train/kvn_dspark_local_rtx3070ti_validate.py
```

### Tableau des Résultats Obtenus sur `RTX 3070 Ti` (`Qwen/Qwen3.5-0.8B`, 5 époques, Causalité Stricte Auditée)

| Modèle (`speculators.train`) | Backbone | Modes RFF ($M_{\text{train}} \to M_{\text{infer}}$) | Cache d'État / Séquence | Params Actifs Drafter | Accept Rate (%) | AcceptLen ($\tau$) | Greedy EAL (tokens/passe) | Erreur Calibration Confiance (`ConfMAE`) | Temps Train (5 époques) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1_DSpark_SWA_Baseline`** | `3× SWA-128` | — | `1,572,864 B` (1.57 Mo) | `84.96 M` | `12.81%` | `1.5613` | `2.1406` | `0.0938` | `18.70s` |
| **`M2_DSpark_Hybrid_1SWA_2KvN`** | `1× SWA + 2× KvN` | `128 -> 128` | `525,312 B` (`-66.6%`) | `77.06 M` | `12.25%` | **`1.5805`** | **`2.4688` (`+15.3%`)** | `0.0742` | `19.70s` |
| **`M3_KvN_DSpark_ZeroKV_Base`** | **`3× KvN (0 SWA)`** | `128 -> 128` | **`1,536 B` (`-99.90%`)** | **`68.68 M` (`-19.2%`)** | `8.71%` | `1.3325` | `2.2344` (`+4.4%`) | **`0.0627`** | `24.01s` |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO`** | **`3× KvN (0 SWA)`** | **`512 -> 128`** | **`1,536 B` (`-99.90%`)** | **`68.68 M` (`-19.2%`)** | **`13.10%`** | **`1.5588`** | **`2.3359` (`+9.1%`)** | `0.0794` | `21.13s` |

---

## 7. Résultats Réels sur `2× NVIDIA H200` (`Qwen/Qwen3-8B`, 5 Époques, `15.28M` Tokens/Modèle)

L'ablation scientifique complète à 4 voies a été exécutée sur `2× NVIDIA H200 (2× 143,771 MiB HBM3e)` avec le Verifier réel **`Qwen/Qwen3-8B`** (`target_layer_ids = [2, 18, 33]` + couche finale `36`, `hidden_size = 4096`, `num_layers = 5`, `draft_vocab_size = 32000`, `block_size = 8`, `max_anchors = 512`, `total_seq_len = 4096`, 1 000 conversations UltraChat régénérées = 900 train / 100 val, 3 730 pas d'optimisation `Muon + AdamW` par modèle).

### 7.1 Tableau Comparatif Principal (`Qwen/Qwen3-8B`, Validation Époque 5)

| Modèle (`speculators.train`) | Backbone (`num_layers=5`) | Modes RFF ($M_{\text{train}} \to M_{\text{infer}}$) | Cache d'État / Séquence | Params Actifs Drafter | Val Loss (`0.1 CE + 0.9 TV`) | Accept Rate (%) | AcceptLen ($\tau$) | Greedy EAL (tokens/passe) | Pos-0 Acc (%) | `ConfMAE` (`Bias`) | Temps Train (5 époques) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1_DSpark_SWA_Baseline`** | `5× SWA-128` | — | `10,485,760 B` (`10.49 Mo`) | `1,072.20 M` | `0.7900` (`best: 0.7325`) | `28.40%` | `1.9020` | `1.9780` | `51.80%` | `0.2790` (`+0.110`) | `17m 48s` (`1068s`) |
| **`M2_DSpark_Hybrid_1SWA_2KvN`** | `1× SWA + 4× KvN` | `128 -> 128` | `2,099,200 B` (`-80.0%`) | `914.83 M` (`-14.7%`) | `0.6930` (`best: 0.6886`) | `28.70%` | `2.2440` (`+18.0%`) | `2.3500` (`+18.8%`) | `77.30%` | `0.2090` (`+0.041`) | `15m 42s` (`942s`) |
| **`M3_KvN_DSpark_ZeroKV_Base`** | **`5× KvN (0 SWA)`** | `128 -> 128` | **`2,560 B` (`-99.976%`)** | **`841.46 M` (`-21.5%`)** | **`0.6556`** | **`31.08%`** | **`2.3607` (`+24.1%`)** | **`2.4591` (`+24.3%`)** | **`80.44%`** | `0.1856` (`+0.031`) | **`14m 35s` (`875s`)** |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Unpruned)** | **`5× KvN (0 SWA)`** | `512 -> 512` | `10,240 B` (`-99.902%`) | `878.44 M` (`-18.1%`) | `0.6784` | `28.00%` | `2.2678` (`+19.2%`) | `2.3696` (`+19.8%`) | `79.01%` | **`0.1852` (`+0.019`)** | `14m 49s` (`889s`) |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Zero-Shot Pruned)** | **`5× KvN (0 SWA)`** | **`512 -> 128`** | **`2,560 B` (`-99.976%`)** | **`841.46 M` (`-21.5%`)** | `0.7222` | `25.93%` | `2.1553` (`+13.3%`) | `2.2593` (`+14.2%`) | `76.36%` | `0.2248` (`+0.035`) | `+0s` (0 step) |
| **`M4_KvN_DSpark_ZeroKV_Scaled_LASSO` (Pruned + Consolidation)** | **`5× KvN (0 SWA)`** | **`512 -> 128`** | **`2,560 B` (`-99.976%`)** | **`841.46 M` (`-21.5%`)** | `0.6838` (`ce=1.875, tv=0.301`) | `28.20%` | **`2.2760` (`+19.7%`)** | **`2.3760` (`+20.1%`)** | `78.70%` | `0.1940` (`+0.032`) | `+3m 12s` (1-3 ep) |

### 7.2 Benchmark Roofline & Scaling de Batch en Inférence sur `NVIDIA H200` ($K=8$ tokens spéculatifs/passe)

| Taille de Batch ($B$) | `M1_DSpark_SWA_Baseline` (`5× SWA-128`)<br>Latence / Débit / KV-Cache Total | `M2_DSpark_Hybrid_1SWA_2KvN`<br>Latence / Débit / Cache Total | `M3` & `M4 (Pruned M=128)` (`5× KvN`, **0 KV-Cache**)<br>Latence / Débit / État Récurrent Total | Gain Débit Brut (`KvN` vs `SWA`) | Gain Débit Tokens Acceptés (`× EAL`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`B = 1`** | `3.591 ms` / `2,228 tok/s` / `10.49 Mo` | `4.046 ms` / `1,977 tok/s` / `2.10 Mo` | `3.732 ms` / `2,144 tok/s` / **`2.56 Ko`** | `0.96×` | **`+19.6%` (`1.20×`)** |
| **`B = 16`** | `3.510 ms` / `36,469 tok/s` / `167.77 Mo` | `4.156 ms` / `30,796 tok/s` / `33.59 Mo` | `3.765 ms` / `34,001 tok/s` / **`40.96 Ko`** | `0.93×` | **`+16.0%` (`1.16×`)** |
| **`B = 32`** | `3.884 ms` / `65,916 tok/s` / `335.54 Mo` | `4.075 ms` / `62,824 tok/s` / `67.17 Mo` | **`3.745 ms`** / **`68,363 tok/s`** / **`81.92 Ko`** | **`1.04×` (`+3.7%`)** | **`+28.9%` (`1.29×`)** |
| **`B = 64`** | `6.575 ms` / `77,875 tok/s` / `671.09 Mo` | `4.035 ms` / `126,887 tok/s` / `134.35 Mo` | **`3.724 ms`** / **`137,476 tok/s`** / **`163.84 Ko`** | **`1.77×` (`+76.5%`)** | **`+119.4%` (`2.19×`)** |
| **`B = 128`** | `12.223 ms` / `83,776 tok/s` / **`1.342 Go`** | `4.054 ms` / `252,607 tok/s` / `268.70 Mo` | **`3.661 ms`** / **`279,689 tok/s`** / **`327.68 Ko`** | **`3.34×` (`+233.9%`)** | **`+315.1%` (`4.15×`)** |

> **Rapport scientifique complet :** Voir [`kvn_rff_rnn_mtp_report.md`](./kvn_rff_rnn_mtp_report.md) et les métriques JSON brutes dans [`kvn_rff_mtp/h200_qwen3_8b_ablation_results.json`](./kvn_rff_mtp/h200_qwen3_8b_ablation_results.json).

