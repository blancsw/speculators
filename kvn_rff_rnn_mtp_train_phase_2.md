# Runbook d'Exécution & Plan d'Entraînement SOTA — KvN-DSpark Phase 2 (100k On-Policy)

> **HANDOFF AGENT AUTONOME (`Antigravity Gemini / Carbon` — Cluster Linux NVIDIA H200)**
>
> Ce document est le **plan d'exécution opérationnel de relais** sur la branche `kvn-rff-rnn-mtp` du dépôt `speculators`.
> - **CE QUI EST DÉJÀ 100 % TERMINÉ ET COMMITÉ (`[x]` — NE PAS RÉ-IMPLÉMENTER) :**
>   1. L'architecture complète **`KvN-DSpark v2`** dans `src/speculators/models/kvn_dspark/{config.py,model_definitions.py,core.py}` et `src/speculators/train/{config/schema.py,optimizers.py}`.
>   2. Le générateur de corpus multi-domaines **100k décontaminé** (`scripts/build_decontaminated_mix.py`), l'orchestrateur d'ablations H200 (`examples/train/kvn_dspark_phase2_h200_suite.sh`), la recette 2×H200 (`examples/train/kvn_dspark_2xh200_online.sh`), et l'évaluateur ECE / AUROC / Pareto / Loi d'Amdahl (`scripts/evaluate/kvn_phase2_calibration_and_amdahl.py`, `examples/evaluate/kvn_dspark_phase2_e2e_eval.sh`).
>   3. La suite de **31 tests unitaires dédiés** (`tests/unit/models/test_kvn_dspark.py`, `tests/unit/models/test_kvn_dspark_v2.py`, `tests/unit/train/test_kvn_phase2_pipeline.py`) et la validation GPU locale sur `Qwen/Qwen3.5-0.8B` (`kvn_rff_mtp/rtx3070ti_qwen3_validation_results.json`).
> - **MISSION DE L'AGENT `CARBON` SUR LE CLUSTER H200 (`[ ]` — À EXÉCUTER DANS L'ORDRE) :**
>   1. **Étape 0 :** Cloner/mettre à jour `kvn-rff-rnn-mtp`, installer `hs_connectors` + `speculators` et passer le gate `pytest` Linux (30 secondes).
>   2. **Phase B :** Générer et préparer le corpus multi-domaines on-policy décontaminé de **100 000 conversations** (`95 000` train + `5 000` validation held-out, `--train-data-ratio 0.95`, `seq_len = 8192`).
>   3. **Phase C :** Exécuter la matrice d'ablations orthogonales à **9 modèles** (dont **3 seeds `42, 1337, 2026`** sur `B1`, `A4` et `A5` + élagage dyadique zéro-shot `512 -> 128` + micro-recalibration 1 époque).
>   4. **Phase D :** Exécuter la suite d'évaluation de bout en bout sous `vLLM` + `AIPerf` (`examples/evaluate/kvn_dspark_phase2_e2e_eval.sh`), vérifier les métriques Go/No-Go (Section 5), et générer le rapport scientifique final `kvn_rff_rnn_mtp_report_phase2.md` (sans commiter les poids `.safetensors` dans Git).

---

## Vue d'Ensemble Scientifique & Système (`KvN-DSpark v2`)

**Objectif :** Valider à l'échelle industrielle (**`Qwen/Qwen3-8B`**, puis transfert **`Qwen/Qwen3-32B`**) le drafter spéculatif parallèle par blocs **Zero-Drafter-KV-Cache** (`2.56 KB/seq` en BF16, soit **`1 024×`** de réduction réelle face au KV-cache GQA `2.62 MB/seq` de `DSpark` SWA-128 sur `Qwen3-8B`, et **`-53 %`** de paramètres actifs à `~395M` contre `841.5M` en v1 et `1 072M` pour `DSpark`).

**Briques `KvN-DSpark v2` actives par défaut :**
1. **Alignement causal strict** (`_prepare_causal_kvn_inputs`, `--kvn-strict-causal-shift`) : $\tilde{h}^{\text{fc}}_t = h^{\text{fc}}_{t-1}$ masqué aux frontières `document_ids` — élimine 100 % de la fuite d'ancre de Phase 1 et supprime tout *rollback* sur le jeton bonus en `vLLM`.
2. **Scan parallèle segmenté Kogge-Stone $O(T \log T)$** (`parallel_segmented_real_scan`) : exact aux frontières de multipacking `reset_mask` et unifié avec `recurrent_step` ($\Delta t_{t,m}$ dans l'exponentielle amortie).
3. **Encodeur spectral compressé** (`ModalDataReuploadingRFF`, `--kvn-reupload-dim 512`, `--kvn-learnable-rff`) + projection d'ancre factorisée (`--kvn-anchor-rank 512`) : échelle $\sigma_g / \sqrt{L \cdot d_{\text{re}}}$ (Corollaire 3.2 d'Alice Barthe) et fréquences $\Omega$ routées vers `AdamW` (jamais orthogonalisées par `Muon`).
4. **Rollout de Koopman multi-couches + `IntraBlockCausalMixer` $K\times K$ Zéro-KV** (`--kvn-multilayer-rollout`, `--kvn-intrablock-mixer`) : attention causale éphémère $8\times 8$ intra-bloc (`0 octet` de KV-cache persistant) brisant le plafond d'acceptation $\tau=2.36$ des SSM purs identifié par *H-Spec*.
5. **Interféromètre symplectique dyadique $2\times 2$ + Mode du Vide $|M+1\rangle$ + Élagage Group-LASSO dyadique** (`DyadicSymplecticInterferometer`, `compute_vacuum_born_observables`, `prune_kvn_modes(128)`) : restaure le gradient de phase de Koopman dans `BornConfidenceHead` et permet l'élagage exact par paires dyadiques $M_{\text{train}}=512 \to M_{\text{infer}}=128$ avec compensation analytique de biais sur les 3 têtes.

---

## 0. Règles Méthodologiques Strictes & Notes Système H200 (À Respecter Impérativement)

1. **Séparation Hermétique Train / Validation / Benchmarks :** Ne **jamais** utiliser `test_sft` ni `tutorial_regen` pour l'entraînement. `scripts/build_decontaminated_mix.py` filtre tout prompt ou complétion entrant en collision par **SHA-256 normalisé ou 13-grammes** avec **MT-Bench**, **GSM8K (`test`)**, **MATH-500**, **HumanEval**, **MBPP** et **LiveCodeBench**.
2. **Sécurité `/dev/shm` à 100k exemples :** En mode online (`--on-missing generate`), `VllmHiddenStatesSource` supprime automatiquement (`self.transfer.delete(receipt.handle)`) chaque fichier `.safetensors` transitoire dans `/dev/shm` sitôt lu dans le batch GPU. Même sur **100 000 conversations**, l'occupation pic de `/dev/shm` reste **`< 4 Go`** (aucun risque de saturation RAM). En revanche, placez toujours `ARTIFACT_ROOT` (`raw_onpolicy_100k.jsonl` ~350 Mo, dataset Arrow ~1.5 Go, et checkpoints ~800 Mo chacun) sur un **disque NVMe local à gros volume** (ex. `/workspace` ou `/scratch`, prévoir `~150 Go` libres pour l'ensemble des 15 runs + checkpoints).
3. **Conflit `--from-pretrained` vs `DECODER_SHAPING_FLAGS` :** Dans `speculators.train`, passer `--from-pretrained` interdit de passer simultanément `--num-layers` ou `--sliding-window`. Le script `examples/train/kvn_dspark_phase2_h200_suite.sh` gère déjà cette séparation automatiquement (`base_args` vs `scratch_common`).
4. **Deux Métriques de Vitesse Obligatoires dans le Rapport :** Toujours rapporter : (a) la latence isolée du proposer $T_{\text{draft}}(B)$ et l'empreinte d'état GQA exacte (`2.56 KB/seq` vs `2.62 MB/seq` pour SWA-128 sur `Qwen3-8B`, ratio exact **`1 024×`**), et (b) le **speedup réel de bout en bout sous `vllm serve`** $\mathcal{S}_{\text{e2e}}(B) = \text{TPS}_{\text{spec}}(B) / \text{TPS}_{\text{autoregressive}}(B)$ incluant le vérifieur `Qwen3-8B` (Loi d'Amdahl).

---

## 1. Statut Phase A (Déjà Complétée) & Gate d'Arrivée Linux pour `Carbon`

- [x] **Tâche A.1 :** Schéma `KvNDSparkSpeculatorConfig` + `KvNDSparkArgs`, comptabilité mémoire GQA exacte (`_kv_hidden_dim`) et routage spectral `AdamW` (`omega_rff`, `dyad_`, `layer_mode_`, `vacuum_`) — **IMPLÉMENTÉ & VÉRIFIÉ**.
- [x] **Tâche A.2 :** Briques v2 dans `model_definitions.py` (`ModalDataReuploadingRFF`, `parallel_segmented_real_scan`, `DyadicSymplecticInterferometer`, `compute_vacuum_born_observables`, `IntraBlockCausalMixer`, Group-LASSO dyadique & compensation exacte des 3 têtes) — **IMPLÉMENTÉ & VÉRIFIÉ**.
- [x] **Tâche A.3 :** Alignement causal strict `_prepare_causal_kvn_inputs` + step d'inférence statique compatible CUDA-Graph `advance_and_draft_step` dans `core.py` + validation GPU locale (`M4` élagué `512 -> 128` + 1 ép. recal. : `EAL = 2.1328 > M3 = 2.0391`, `ECE_15bin = 0.0424`) — **IMPLÉMENTÉ & VÉRIFIÉ**.

### Étape 0 (À exécuter par `Carbon` dès connexion au nœud H200)

- [ ] **Étape 0.1 : Installation de l'environnement Linux H200 et Gate `pytest` (31 tests)**
  ```bash
  git clone -b kvn-rff-rnn-mtp git@github.com:blancsw/speculators.git
  cd speculators
  git pull origin kvn-rff-rnn-mtp

  uv venv .venv --python 3.11
  source .venv/bin/activate
  # IMPORTANT : installer à la fois le connecteur local hs_connectors ET speculators
  uv pip install -e hs_connectors -e ".[train,dev]"
  uv pip install -r scripts/evaluate/requirements.txt

  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

  # Gate de non-régression obligatoire avant lancement GPU (doit afficher : 31 passed)
  pytest tests/unit/models/test_kvn_dspark.py \
         tests/unit/models/test_kvn_dspark_v2.py \
         tests/unit/train/test_kvn_phase2_pipeline.py -v
  ```

---

## 2. Phase B — Corpus Multi-Domaines **100 000 Conversations** On-Policy Décontaminé

Le passage de `68k` à **`100 000` conversations** (`95 000` entraînement + `5 000` validation held-out via `--train-data-ratio 0.95`) aligne notre protocole sur les standards SOTA industriels (**EAGLE-3 / DFlash / H-Spec**) et alimente pleinement le régime sur-paramétré `A5` ($M_{\text{train}}=512$ modes + fréquences RFF apprenables) sans aucun risque d'épuisement des splits d'entraînement sources :

### Composition Stratifiée du Corpus `qwen3_8b_mix100k_seq8192`

| Domaine | Source Hugging Face (Split Train uniquement) | Dispo Source | Volume Retenu | Température Génération `Qwen3-8B` | Rôle Scientifique |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dialogue Général & Instructions** | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | `207 865` | **`45 000`** (45 %) | 50 % $T=0.0$ / 50 % $T=0.7$ | Couverture lexicale large, multi-tours |
| **Mathématiques & Raisonnement CoT** | `nvidia/OpenMathInstruct-2` (`train`) | `14 000 000` | **`32 000`** (32 %) | 50 % $T=0.0$ / 50 % $T=0.6$ | Horizons longs structurés ($k=2\dots 7$), LaTeX, CoT |
| **Code & Algorithmique** | `ise-uiuc/Magicoder-OSS-Instruct-75K` (`train`) | `75 197` | **`23 000`** (23 %) | 60 % $T=0.0$ / 40 % $T=0.6$ | Syntaxe stricte, indentation, identifiants multi-tokens |
| **Total Assemblé & Décontaminé** | Split `95 %` Train / `5 %` Validation Held-Out | — | **`100 000`** (`95k` + `5k`) | Stratifié | **~160M–190M tokens on-policy packés à `seq_len=8192`** |

### Exécution de la Phase B (Automatique ou Étape par Étape)

> **Astuce :** `examples/train/kvn_dspark_phase2_h200_suite.sh` exécute automatiquement la Phase B si `SKIP_DATA_PREP` vaut `0` (défaut). Pour paralléliser plus vite la régénération des `100k` conversations sur **tous** les GPUs du nœud avant l'entraînement (fortement recommandé sur 2×H200 ou 8×H200 !), exécutez l'Étape B.1 dédiée ci-dessous avec `--data-parallel-size` maximal, puis lancez la Phase C avec `SKIP_DATA_PREP=1`.

- [ ] **Étape B.1 (Recommandée — Génération On-Policy Haute Vitesse en Data-Parallel sur tous les H200 disponibles) :**
  ```bash
  export VERIFIER_MODEL="Qwen/Qwen3-8B"
  export ARTIFACT_ROOT="/workspace/kvn_phase2_artifacts"
  export DATA_DIR="$ARTIFACT_ROOT/data/qwen3_8b_mix100k_seq8192"
  export RAW_JSONL="$ARTIFACT_ROOT/raw_onpolicy_100k.jsonl"
  mkdir -p "$DATA_DIR"

  # Sur 2xH200 : ALL_GPUS="0,1" et DP_GEN=2  |  Sur 8xH200 : ALL_GPUS="0,1,2,3,4,5,6,7" et DP_GEN=8
  ALL_GPUS="${ALL_GPUS:-0,1}"
  DP_GEN="${DP_GEN:-2}"

  # 1. Serveur vLLM dédié à la régénération rapide des 100k conversations + rendu prepare-data
  CUDA_VISIBLE_DEVICES="$ALL_GPUS" python scripts/launch_vllm.py "$VERIFIER_MODEL" \
      --target-layer-ids 2 18 33 \
      --hidden-states-path "/dev/shm/kvn_prep_hs" \
      -- --data-parallel-size "$DP_GEN" --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90 &
  VLLM_PREP_PID=$!
  until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done

  # 2. Construction du mix 100k (45k UltraChat + 32k OpenMath + 23k Magicoder) + décontamination 13-grammes/SHA-256
  python scripts/build_decontaminated_mix.py \
      --verifier-model "$VERIFIER_MODEL" \
      --ultrachat-samples 45000 \
      --openmath-samples 32000 \
      --magicoder-samples 23000 \
      --exclude-benchmarks mt_bench gsm8k math500 humaneval mbpp livecodebench \
      --vllm-endpoint "http://localhost:8000/v1" \
      --concurrency $((64 * DP_GEN)) \
      --output-jsonl "$RAW_JSONL" \
      --val-samples 0 \
      --seed 42

  # 3. Tokenisation exacte via l'endpoint render vLLM, masques assistant et token_freq.pt
  speculators prepare-data \
      --model "$VERIFIER_MODEL" \
      --data "$RAW_JSONL" \
      --render-endpoint "http://localhost:8000" \
      --output "$DATA_DIR" \
      --seq-length 8192 \
      --no-skip-token-freq

  # Arrêt du serveur de data-prep avant le partitionnement GPU de la Phase C
  kill "$VLLM_PREP_PID" && wait "$VLLM_PREP_PID" 2>/dev/null || true
  rm -rf /dev/shm/kvn_prep_hs
  ```
  *Vérification attendue fin Phase B :* présence de `$RAW_JSONL.manifest.json` (`total_written_train == 100000` ou proche après filtrage post-génération), des shards `.arrow` et de `$DATA_DIR/token_freq.pt`.

---

## 3. Phase C — Matrice d'Ablations Orthogonales Contrôlées (9 Modèles × Multi-Seeds)

Sur le corpus de **`100 000` conversations** (`95k` train / `5k` val, `--train-data-ratio 0.95`), **`EPOCHS=6`** couvre ~1.0 milliard de jetons on-policy (davantage de jetons et +47 % de diversité unique que 8 époques sur 68k), avec hyperparamètres strictement constants :
- **Optimiseur :** `Muon` (`--muon-lr 2e-3`, Newton-Schulz 5 pas) + `AdamW` (`--lr 3e-4`, cosine decay)
- **Perte :** `{"ce": 0.1, "tv": 0.9}` + BCE de confiance (`--confidence-head-alpha 1.0`)
- **Architecture commune :** `--total-seq-len 8192`, `--max-anchors 512`, `--block-size 8`, `--num-layers 5`, `--draft-vocab-size 32768`, `--target-layer-ids 2 18 33`.

### Tableau de la Matrice Expérimentale Orthogonale (9 Modèles)

| ID Run | Seeds | Backbone | Causal Shift & Scan Reset | RFF ($d_{\text{re}}$, $\Omega$) & `anchor_rank` | Rollout Multi-Couches | Mixer Intra-Bloc $K\times K$ | Confiance (Born / Vide / Interf.) | Modes $M_{\text{train}} \to M_{\text{infer}}$ | Params Actifs | État / Séq | Question Scientifique Isolée |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B1_DSpark_SWA128`** | `42,1337,2026` | `swa` ($W=128$) | Natif SWA | — | Natif Attn | Natif 5× | MLP `DSpark` | — | `1 072 M` | **`2.62 MB`** | Baseline SOTA officielle `DSpark` |
| **`B2_DSpark_SWA512`** | `42` | `swa` ($W=512$) | Natif SWA | — | Natif Attn | Natif 5× | MLP `DSpark` | — | `1 072 M` | **`10.49 MB`** | Coût mémoire SWA sur contexte 8k |
| **`B3_Hybrid_1SWA_4KvN`** | `42` | `hybrid` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide** | $128 \to 128$ | `462 M` | **`526 KB`** | Compromis hybride (`5×` moins de KV) |
| **`A0_KvN_v1_CausalFix`** | `42` | `kvn` | **Oui (v2)** | Dense ($4096$, figé), Dense | Non (couche 4) | Non | Born v1 (sans interf.) | $128 \to 128$ | `841 M` | **`2.56 KB`** | **Vraie base Phase 1 sans fuite causale** |
| **`A1_KvN_ModalRFF_Comp`** | `42` | `kvn` | **Oui (v2)** | **$512$, Appren., $r_a=512$** | Non (couche 4) | Non | Born v1 | $128 \to 128$ | **`361 M`** | **`2.56 KB`** | Gain spectral Barthe + `-57%` params |
| **`A2_KvN_Multilayer`** | `42` | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | **Oui (5 couches)** | Non | Born v1 | $128 \to 128$ | `362 M` | **`2.56 KB`** | Gain des échelles fréquentielles basses |
| **`A3_KvN_IntraBlock`** | `42` | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | **Oui ($8\times 8$)** | MLP `DSpark` *(no Born)* | $128 \to 128$ | `395 M` | **`2.56 KB`** | **Briser le plafond *H-Spec* sur $k=2..7$** |
| **`A4_KvN_v2_Full128`** | `42,1337,2026` | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide (v2)** | $128 \to 128$ | **`395 M`** | **`2.56 KB`** | **Gain pur de la calibration Born v2** *(vs `A3`)* |
| **`A5_KvN_v2_LASSO_512to128`** | `42,1337,2026` | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide (v2)** | **$512 \to 128$** | **`395 M`** *(post-prune)* | **`2.56 KB`** | **Gain de loterie spectrale Group-LASSO dyadique** |

### Exécution de la Phase C par `Carbon`

- [ ] **Étape C.1 : Lancer les 3 configurations phares multi-seeds (`B1`, `A4`, `A5` × seeds `42, 1337, 2026`) + élagage dyadique `512 -> 128` + recalibration 1 époque**
  ```bash
  # Configuration 2xH200 (défaut) : GPU 0 = vLLM producteur, GPU 1 = torchrun
  SKIP_DATA_PREP=1 \
  VLLM_GPUS="0" VLLM_DP=1 TRAIN_GPUS="1" NPROC=1 \
  MODELS="B1_DSpark_SWA128 A4_KvN_v2_Full128 A5_KvN_v2_LASSO_512to128" \
  SEEDS="42 1337 2026" \
  bash examples/train/kvn_dspark_phase2_h200_suite.sh

  # --- OU sur 8xH200 (GPUs 0-1 = vLLM DP=2, GPUs 2-7 = torchrun 6 GPUs) ---
  # SKIP_DATA_PREP=1 VLLM_GPUS="0,1" VLLM_DP=2 TRAIN_GPUS="2,3,4,5,6,7" NPROC=6 \
  # MODELS="B1_DSpark_SWA128 A4_KvN_v2_Full128 A5_KvN_v2_LASSO_512to128" SEEDS="42 1337 2026" \
  # bash examples/train/kvn_dspark_phase2_h200_suite.sh
  ```

- [ ] **Étape C.2 : Lancer les 6 ablations orthogonales à facteur unique sur la graine canonique `42`**
  ```bash
  SKIP_DATA_PREP=1 \
  VLLM_GPUS="0" VLLM_DP=1 TRAIN_GPUS="1" NPROC=1 \
  MODELS="B2_DSpark_SWA512 B3_Hybrid_1SWA_4KvN A0_KvN_v1_CausalFix A1_KvN_ModalRFF_Comp A2_KvN_Multilayer A3_KvN_IntraBlock" \
  SEEDS="42" \
  bash examples/train/kvn_dspark_phase2_h200_suite.sh
  ```

- [ ] **Étape C.3 (Extension Papier si budget GPU disponible) : Transfert d'échelle `Qwen/Qwen3-32B`**
  ```bash
  MODEL="Qwen/Qwen3-32B" TARGET_LAYER_IDS="4 32 60" \
  OUTPUT_DIR="/workspace/kvn_phase2_artifacts/qwen3_32b_kvn_v2" \
  MAX_SAMPLES=50000 EPOCHS=5 \
  bash examples/train/kvn_dspark_2xh200_online.sh
  ```

---

## 4. Phase D — Évaluation de Bout en Bout (`vLLM` + `AIPerf` + Calibration + Loi d'Amdahl) & Livrables

> **Note `LAST_EPOCH` :** Avec `EPOCHS=6` (époques indexées `0..5`), le dernier checkpoint d'entraînement est le dossier **`5`** (`LAST_EPOCH=5`, valeur par défaut de `examples/evaluate/kvn_dspark_phase2_e2e_eval.sh`). Pour le checkpoint recalibré 1 époque (`A5_KvN_v2_Pruned128_Recal_s42`), le dossier est toujours `0`.

- [ ] **Étape D.1 : Exécuter la campagne complète `examples/evaluate/kvn_dspark_phase2_e2e_eval.sh`**
  Ce script exécute séquentiellement :
  1. **Qualité d'acceptation spéculative ($\tau$)** via `scripts/evaluate/evaluate.py acceptance` sur **5 benchmarks hors-distribution** (`spec_al_mtbench`, `spec_al_gsm8k`, `spec_al_math500`, `spec_al_humaneval`, `spec_al_mbpp`) en régimes **Greedy ($T=0.0$)** et **Stochastique ($T=0.6$)**.
  2. **Calibration physique de confiance** via `scripts/evaluate/kvn_phase2_calibration_and_amdahl.py calibrate` comparant `A0` (Born v1 aveugle à la phase), `A3` (MLP `DSpark`), `A4` et `A5` (**Born v2 avec interféromètre dyadique + mode du vide $|M+1\rangle$**) sur l'**ECE à 15 bacs**, l'**AUROC de rejet**, et la **courbe de Pareto d'arrêt dynamique** ($\gamma_{\text{conf}} \in \{0.0, 0.2, 0.35, 0.5, 0.65, 0.8\}$).
  3. **Speedup réel de bout en bout (Loi d'Amdahl)** via `scripts/evaluate/evaluate.py speedup` balayant la concurrence **$B \in \{1, 4, 8, 16, 32, 64, 128, 256\}$** face au vérifieur `Qwen/Qwen3-8B` sans spéculation et face à `B1_DSpark_SWA128`.
  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  ARTIFACT_ROOT="/workspace/kvn_phase2_artifacts" \
  LAST_EPOCH=5 \
  bash examples/evaluate/kvn_dspark_phase2_e2e_eval.sh
  ```

- [ ] **Étape D.2 : Synthèse Amdahl et Rédaction du Rapport Final `kvn_rff_rnn_mtp_report_phase2.md`**
  Compiler tous les résultats (`mean ± std` sur les 3 seeds pour `B1`, `A4`, `A5`, tableau d'ablation orthogonale complet `A0 -> A5`, courbes ECE/Pareto et speedups $\mathcal{S}_{\text{e2e}}(B)$) dans `speculators/kvn_rff_rnn_mtp_report_phase2.md`, puis commiter uniquement le rapport et les fichiers JSON/CSV de métriques légers (ne jamais commiter les `.safetensors` dans Git).

---

## 5. Critères de Succès Quantitatifs (Go / No-Go Publication ICML/NeurIPS & Production)

| Métrique Cible | Phase 1 (Auditée) | Seuil Minimum Go-Prod | Objectif SOTA Papier (ICML/NeurIPS) |
| :--- | :---: | :---: | :---: |
| **Fuite causale à l'ancre / inter-documents** | Présente (`+28.6 pts` pos 0) | **`0.00` (`atol < 1e-6`)** *(Acquis)* | **`0.00` prouvé par test unitaire formel** *(Acquis)* |
| **Empreinte d'état drafter / séquence (`Qwen3-8B`)** | `2.56 KB` | **`2.56 KB` (`1 024×` vs SWA-128)** *(Acquis)* | **`2.56 KB` (`0 B` KV-cache, `4 096×` vs SWA-512)** *(Acquis)* |
| **Paramètres actifs du drafter** | `841.5 M` | `< 450 M` *(Acquis)* | **`~395 M` (`-53 %` vs v1, `-63 %` vs `DSpark`)** *(Acquis)* |
| **Corpus d'entraînement on-policy décontaminé** | `1k` (`tutorial_regen`, fuité) | `68k` multi-domaines | **`100 000` (`95k` train + `5k` val, 0 fuite 13-grammes)** |
| **Longueur moyenne d'acceptation $\tau$ ($K=8$, Greedy, moy. 5 benchs)** | Biaisée (`tutorial_regen`) | $\ge 97\%$ du $\tau$ de `B1_DSpark_SWA128` | **$\ge 100\%$ du $\tau$ de `B1_DSpark_SWA128` ($\tau \ge 3.6$–$4.2$)** |
| **Calibration de confiance (ECE 15 bacs sur rejet)** | Non mesurée (MAE seule) | `< 0.045` | **`< 0.025` (`A4` Born-Vide bat `A3` MLP significativement)** |
| **Élagage dyadique Group-LASSO $512 \to 128$** | `+0.17` non isolé | Perte `< 0.05` zéro-shot | **Perte `< 0.02` zéro-shot et Gain net `A5 > A4` après 1 ep recal.** |
| **Speedup de bout en bout `vLLM` ($\mathcal{S}_{\text{e2e}}$, Loi d'Amdahl incluse)** | Non mesuré (`4.15×` isolé) | **`\ge 2.1×` à $B=1$ ; `\ge 1.7×` à $B=64$** | **`2.4×–2.8×` à $B=1..8$ ; `> 1.30×` vs `DSpark` à $B \ge 64$** |
