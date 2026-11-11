# Plan d'Implémentation, d'Entraînement SOTA et d'Industrialisation — KvN-DSpark Phase 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement and execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** Implémenter l'architecture rectifiée **`KvN-DSpark v2`** spécifiée dans [kvn_rff_dspark_phase2.md](../kvn_rff_dspark_phase2.md), valider ses 7 invariants mathématiques par TDD local, entraîner une matrice d'ablations orthogonales rigoureuse sur **65 000 conversations multi-domaines on-policy** (sans aucune fuite de données de test), et mesurer les performances de bout en bout sous `vLLM` (incluant la passe de vérification cible selon la Loi d'Amdahl) pour une soumission ICML / NeurIPS / ICLR et une mise en production industrielle.

**Architecture :** Drafter spéculatif parallèle par blocs **Zero-Drafter-KV-Cache** (`2.56 KB/seq` en BF16, soit **`1 024×`** de réduction réelle face au KV-cache GQA `2.62 MB/seq` de `DSpark` SWA-128 sur `Qwen3-8B`, et **`-53 %`** de paramètres actifs à `~395M`), combinant :
1. Alignement causal strict $\tilde{h}^{\text{fc}}_t = h^{\text{fc}}_{t-1}$ masqué par `document_ids` (zéro fuite d'ancre, zéro rollback sur token bonus en `vLLM`).
2. Scan associatif parallèle segmenté Kogge-Stone $O(T \log T)$ sur GPU en `complex64`.
3. Encodeur `ModalDataReuploadingRFF` ($d_{\text{re}}=512$, échelle $\sigma_g / \sqrt{L \cdot d_{\text{re}}}$, fréquences $\Omega$ apprenables sous `AdamW`).
4. Banque de Koopman multi-couches + `IntraBlockCausalMixer` $K\times K$ Zéro-KV (brisant le plafond $\tau=2.36$ des SSM purs identifié par *H-Spec*).
5. Interféromètre symplectique par dyades $2\times 2$ + observable de Born normalisée par mode du vide $|M+1\rangle$ + élagage dyadique Group-LASSO zéro-shot exact ($512 \to 128$ modes).

**Tech Stack :** Python 3.11+, PyTorch 2.6+ (`torch.optim.Muon` + `AdamW`, `FSDP2`, `torch.compile`, `CUDAGraph`), `vLLM` V1 (`extract_hidden_states` & `SpecDecodeBaseProposer`), `speculators`, `AIPerf`, NVIDIA RTX 3070 Ti (8 Go, gate TDD local) & Cluster 2× à 8× NVIDIA H200 SXM (141 Go HBM3e).

---

## 0. Règles Méthodologiques Strictes (Garde-Fous Publication Rang A*)

1. **Séparation Hermétique Train / Validation / Benchmarks :** Ne **plus jamais** utiliser le split `test_sft` ni le sous-ensemble `tutorial_regen` (1 000 conversations) pour l'entraînement. Tout prompt présent dans **MT-Bench**, **GSM8K (test)**, **MATH-500**, **HumanEval**, **MBPP** ou **LiveCodeBench** est filtré par empreinte SHA-256 et chevauchement 13-grammes avant génération des états cachés.
2. **Parité d'Information Causale absolue :** Toutes les comparaisons principales imposent `--kvn-strict-causal-shift` (`True` par défaut). L'ancien comportement Phase 1 (`--no-kvn-strict-causal-shift`) n'est autorisé que dans une unique ligne d'audit dédiée pour quantifier rétrospectivement l'artefact de Phase 1.
3. **Comptabilité Mémoire & Débit Honnêtes :**
   - Toujours rapporter l'empreinte KV-cache avec la vraie dimension GQA (`num_key_value_heads * head_dim = 1024` sur `Qwen3-8B` $\implies$ `2.62 MB/seq` pour SWA-128 et `10.49 MB/seq` pour SWA-512).
   - Toujours rapporter **deux métriques de vitesse distinctes** : (a) la latence isolée du proposer $T_{\text{draft}}(B)$ (profil Roofline mémoire), et (b) le **speedup effectif de bout en bout sous `vllm serve`** $\mathcal{S}_{\text{e2e}}(B) = \frac{\text{TPS}_{\text{spec}}(B)}{\text{TPS}_{\text{autoregressive}}(B)}$ incluant la vérification `Qwen3-8B`.
4. **Rigueur Statistique :** Les configurations phares (`B1_DSpark_SWA128`, `A4_KvN_v2_Dense128`, `A5_KvN_v2_LASSO_512to128`) sont entraînées et évaluées sur **3 graines aléatoires indépendantes** (`--seed 42`, `1337`, `2026`) avec rapport `moyenne ± écart-type`.

---

## 1. Phase A — Implémentation & Gate TDD Local (PC Windows RTX 3070 Ti 8 Go)

L'objectif de cette phase est d'implémenter et vérifier mathématiquement toutes les optimisations de [kvn_rff_dspark_phase2.md](../kvn_rff_dspark_phase2.md) localement en moins de 15 minutes avant d'allouer la moindre heure GPU H200.

### Tâche 1 : Schéma de Configuration, Comptabilité GQA Exacte & Routage Optimiseur `Muon`/`AdamW`

**Fichiers cibles :**
- Modifier : `src/speculators/models/kvn_dspark/config.py`
- Modifier : `src/speculators/train/config/schema.py`
- Modifier : `src/speculators/train/optimizers.py`
- Créer : `tests/unit/models/test_kvn_dspark_v2.py`

- [ ] **Étape 1.1 : Écrire les tests unitaires en échec pour la comptabilité GQA et le routage `AdamW`**
  Dans `tests/unit/models/test_kvn_dspark_v2.py`, écrire :
  - `test_gqa_state_memory_accounting_qwen3_8b()` : instancie un `Qwen3Config(hidden_size=4096, num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8, head_dim=128, sliding_window=128)` ; vérifie `swa -> 2_621_440` octets, `kvn (M=128, bfloat16) -> 2_560` octets (`ratio == 1024.0`), `hybrid_1swa_2kvn -> 526_336` octets.
  - `test_new_v2_cli_flags_resolution()` : vérifie que `TrainConfig.resolve([...])` accepte `--kvn-reupload-dim 512`, `--kvn-anchor-rank 512`, `--kvn-enable-interferometer`, `--kvn-vacuum-born-norm`, `--kvn-multilayer-rollout`, `--kvn-intrablock-mixer`, `--kvn-strict-causal-shift`.
  - `test_spectral_params_routed_to_adamw_not_muon()` : vérifie que `omega_rff`, `dyad_theta`, `dyad_phase`, `layer_mode_logits` et `vacuum_log_kappa` tombent tous dans `adamw_params` via `split_named_params_for_muon`.

- [ ] **Étape 1.2 : Exécuter `pytest` pour confirmer l'échec initial (Red)**
  ```powershell
  cd speculators
  pytest tests/unit/models/test_kvn_dspark_v2.py -v
  ```

- [ ] **Étape 1.3 : Implémenter les mises à jour dans `config.py`, `schema.py` et `optimizers.py`**
  - Ajouter les champs `kvn_reupload_dim`, `kvn_learnable_rff`, `kvn_enable_interferometer`, `kvn_vacuum_born_norm`, `kvn_anchor_rank`, `kvn_multilayer_rollout`, `kvn_intrablock_mixer`, `kvn_strict_causal_shift`, `kvn_ffn_mult`, `kvn_state_dtype` dans `KvNDSparkSpeculatorConfig` et `KvNDSparkArgs`.
  - Corriger `estimated_state_bytes_per_seq` avec `_kv_hidden_dim()`.
  - Ajouter `"omega_rff"`, `"dyad_"`, `"layer_mode_"`, `"vacuum_"` à `_ADAMW_NAME_HINTS` dans `src/speculators/train/optimizers.py`.

- [ ] **Étape 1.4 : Vérifier le passage au vert (Green)**
  ```powershell
  pytest tests/unit/models/test_kvn_dspark_v2.py -k "gqa or cli_flags or routed_to_adamw" -v
  ```

---

### Tâche 2 : Briques Mathématiques v2 dans `model_definitions.py`

**Fichiers cibles :**
- Modifier : `src/speculators/models/kvn_dspark/model_definitions.py`
- Modifier : `src/speculators/models/kvn_dspark/__init__.py`
- Modifier : `tests/unit/models/test_kvn_dspark_v2.py`

- [ ] **Étape 2.1 : Écrire les tests d'invariants mathématiques dans `test_kvn_dspark_v2.py`**
  1. `test_multipacking_document_boundary_isolation()` : séquence `T=32` avec `document_ids = [0]*16 + [1]*16` ; mutile `x[:, :16]` par $\times 1000$ et vérifie `torch.allclose(out1[:, 16:], out2[:, 16:], atol=1e-6)`.
  2. `test_segmented_scan_matches_step_by_step_recurrent_v2()` : compare `CausalKvNRFFLayer.forward(..., reset_mask)` au déroulement séquentiel `recurrent_step_v2` sur `T=24` avec frontières de documents multiples (`atol=1e-5`).
  3. `test_interferometer_phase_gradient_and_vacuum_born_monotonicity()` : vérifie que `horizon_freq_scale.grad` est non nul lorsque seule la perte de `BornConfidenceHead` est rétropropagée, que `(born_obs / (2*M)).sum(-1, keepdim=True) + p_vac == 1.0`, et que `p_vac` croît strictement avec l'horizon $k \in \{0, \dots, 7\}$.
  4. `test_exact_dyadic_lasso_pruning_zero_error_all_three_heads()` : annule les dyades $g \ge M/4$ dans `readout_linear`, `obs_linear` et `born_proj`, appelle `prune_kvn_modes(M // 2)`, et vérifie l'égalité exacte (`atol=1e-6`) des logits de jetons **et** des logits de confiance (en testant les deux modes `kvn_vacuum_born_norm=True` et `False`).

- [ ] **Étape 2.2 : Exécuter `pytest` pour vérifier l'échec initial**
  ```powershell
  pytest tests/unit/models/test_kvn_dspark_v2.py -v
  ```

- [ ] **Étape 2.3 : Implémenter dans `model_definitions.py`**
  - `ModalDataReuploadingRFF` ($\sin/\cos$ actif à tous les étages $l = 0 \dots L-1$, initialisation $\sigma_g / \sqrt{L \cdot d_{\text{re}}}$, `omega_rff` apprenable).
  - `parallel_segmented_complex_scan` (scan associatif parallèle Kogge-Stone en `complex64` prenant en compte `reset_mask` et $\Delta t_{t,m}$ unifié).
  - `DyadicSymplecticInterferometer` ($2\times 2$ séparateurs de faisceaux par dyade en $O(M)$) + `compute_vacuum_born_observables` (normalisation projective par mode du vide $|M+1\rangle$ de Barthe Éq. 4.34).
  - Factorisation rang-bas optionnelle de `anchor_proj` (`nn.Sequential(nn.Linear(d, r_a, bias=False), nn.Linear(r_a, K*d, bias=False))`), agrégation multi-couches `layer_mode_logits` (`[L_draft, M]`), et `IntraBlockCausalMixer` ($K\times K$ causal Zéro-KV).
  - Group-LASSO dyadique + compensation analytique du biais résiduel sur `readout_linear.bias`, `obs_linear.bias` **et** `born_confidence_head.base_linear.bias`.

- [ ] **Étape 2.4 : Vérifier les tests v1 et v2**
  ```powershell
  pytest tests/unit/models/test_kvn_dspark.py tests/unit/models/test_kvn_dspark_v2.py -v
  ```

---

### Tâche 3 : Alignement Causal Strict & Inférence `vLLM` CUDA-Graph dans `core.py`

**Fichiers cibles :**
- Modifier : `src/speculators/models/kvn_dspark/core.py`
- Modifier : `tests/unit/models/test_kvn_dspark_v2.py`
- Modifier : `examples/train/kvn_dspark_local_rtx3070ti_validate.py`

- [ ] **Étape 3.1 : Écrire les tests d'absence de fuite d'ancre et de capture `torch.compile` / `CUDAGraph`**
  - `test_zero_anchor_causal_leakage_vs_swa()` : fixe deux ancres $a_1 = 6, a_2 = 14$ ; remplace `hidden_states[:, 6:, :]` par du bruit $\times 10^4$ ; vérifie que les logits des $K$ positions générées depuis l'ancre $a_1 = 6$ sont **inchangés à `atol=1e-6`** (tant pour `kvn` que pour `hybrid_1swa_2kvn`).
  - `test_advance_and_draft_step_matches_prefill_and_compiles()` : vérifie que l'avancement incrémental par blocs variables `valid_lens` via `advance_and_draft_step` produit exactement les mêmes états d'ancres et propositions que le forward séquence complète, et compile avec `torch.compile(..., fullgraph=True)`.

- [ ] **Étape 3.2 : Implémenter `_prepare_causal_kvn_inputs` et `advance_and_draft_step` dans `core.py`**
  - Remplacer `x = fc_output + tok_emb` par `x, reset_mask = self._prepare_causal_kvn_inputs(fc_output, tok_emb, document_ids)`.
  - Propager `reset_mask` et agréger les états $(q^{(l)}, p^{(l)}, \Delta\theta^{(l)}, \Delta t^{(l)})$ de toutes les couches vers `self.rollout` puis `self.intrablock_mixer`.
  - Implémenter `advance_and_draft_step` à forme statique `[B, block_size + 1, D]`.

- [ ] **Étape 3.3 : Exécuter la validation complète locale sur RTX 3070 Ti (`Qwen/Qwen3.5-0.8B`)**
  ```powershell
  pytest tests/unit/models/test_kvn_dspark.py tests/unit/models/test_kvn_dspark_v2.py -v
  python examples/train/kvn_dspark_local_rtx3070ti_validate.py
  ```
  **Critère Gate Local :** 100 % des tests unitaires verts, aucune fuite causale (`pos0_acc` alignée honnêtement avec `M1`), zéro `NaN`, rechargement `safetensors` et élagage dyadique $512 \to 128$ sans perte à poids nuls.

---

## 2. Phase B — Préparation du Corpus Multi-Domaines 65k On-Policy (Cluster H200)

Pour éliminer définitivement le sur-apprentissage observé sur `tutorial_regen` (1k conversations) et entraîner le drafter sur la véritable distribution stationnaire du vérifieur `Qwen/Qwen3-8B` (raisonnement CoT, code et dialogue), nous construisons un corpus **strictement décontaminé de 65 000 conversations d'entraînement + 3 000 de validation**.

### Composition Stratifiée du Corpus `kvn_v2_mix_68k`

| Domaine | Source Hugging Face (Split Train uniquement) | Volume | Température Génération `Qwen3-8B` | Rôle Scientifique |
| :--- | :--- | :---: | :---: | :--- |
| **Dialogue Général & Instructions** | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | `30 000` | 50 % $T=0.0$ / 50 % $T=0.7$ | Couverture lexicale large, suivi d'instructions multi-tours |
| **Mathématiques & Raisonnement CoT** | `nvidia/OpenMathInstruct-2` (`train`) | `20 000` | 50 % $T=0.0$ / 50 % $T=0.6$ | Horizons longs structurés ($k=2\dots 7$), équations LaTeX, logique |
| **Code & Algorithmique** | `ise-uiuc/Magicoder-OSS-Instruct-75K` (`train`) | `15 000` | 60 % $T=0.0$ / 40 % $T=0.6$ | Indentation syntaxique stricte, identifiants multi-tokens |
| **Validation Held-Out (Stratifiée)** | Sous-échantillon disjoint des 3 sources ci-dessus | `3 000` | $T=0.0$ & $T=0.6$ | Sélection de checkpoint & arrêt précoce (jamais utilisé en test final) |

> **Décontamination stricte :** Le script de construction calcule la table des 13-grammes normalisés de **MT-Bench**, **GSM8K (`test`)**, **MATH-500**, **HumanEval**, **MBPP** et **LiveCodeBench**, et rejette tout prompt ou complétion présentant une collision avant sauvegarde.

### Commandes d'Environnement et de Préparation sur Nœud H200

- [ ] **Étape B.1 : Initialisation de l'environnement Linux H200**
  ```bash
  cd /workspace/KvN-RFF-RNN/speculators
  uv venv .venv --python 3.11
  source .venv/bin/activate
  uv pip install -e ".[train,dev]"
  uv pip install -r scripts/evaluate/requirements.txt

  export HF_HOME=/workspace/hf_cache
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ```

- [x] **Étape B.2 : Démarrage du serveur `vLLM` producteur d'états cachés et construction du dataset décontaminé (`draft_vocab_size = 32768`, `total_seq_len = 8192`)**
  Pour `Qwen/Qwen3-8B` (36 couches), les couches auxiliaires optimales sont **`2, 18, 33`** (+ couche finale 35 automatique pour la distillation TV/CE). Le vocabulaire réduit fréquentiel (`t2d-32768.npy` / `d2t-32768.npy`) est généré automatiquement par `speculators.train` à partir de `token_freq.pt` :
  ```bash
  export VERIFIER_MODEL="Qwen/Qwen3-8B"
  export ARTIFACT_ROOT="/workspace/kvn_phase2_artifacts"
  export DATA_DIR="$ARTIFACT_ROOT/data/qwen3_8b_mix68k_seq8192"
  export HS_CACHE_DIR="/dev/shm/kvn_phase2_hs_cache"
  mkdir -p "$DATA_DIR" "$HS_CACHE_DIR"

  # 1. Démarrage du serveur vLLM avec extraction des couches [2, 18, 33] en RAM (/dev/shm)
  CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py "$VERIFIER_MODEL" \
      --target-layer-ids 2 18 33 \
      --hidden-states-path "$HS_CACHE_DIR" \
      -- --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.88 &
  until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done

  # 2. Assemblage multi-domaines + décontamination SHA-256 & 13-grammes + régénération on-policy
  python scripts/build_decontaminated_mix.py \
      --verifier-model "$VERIFIER_MODEL" \
      --ultrachat-samples 31500 \
      --openmath-samples 21000 \
      --magicoder-samples 15500 \
      --exclude-benchmarks mt_bench gsm8k math500 humaneval mbpp livecodebench \
      --vllm-endpoint "http://localhost:8000/v1" \
      --concurrency 64 \
      --output-jsonl "$ARTIFACT_ROOT/raw_onpolicy_68k.jsonl" \
      --val-samples 0

  # 3. Rendu exact par chat-template vLLM, masques assistant et distribution token_freq.pt
  speculators prepare-data \
      --model "$VERIFIER_MODEL" \
      --data "$ARTIFACT_ROOT/raw_onpolicy_68k.jsonl" \
      --render-endpoint "http://localhost:8000" \
      --output "$DATA_DIR" \
      --seq-length 8192 \
      --no-skip-token-freq
  ```

---

## 3. Phase C — Matrice d'Ablations Orthogonales Contrôlées (Protocole Papier ICML/NeurIPS)

Chaque critique de l'audit externe (fuite d'ancre, compression RFF, lecture multi-couches, couplage intra-bloc $K\times K$, interféromètre + normalisation par mode du vide, et élagage dyadique Group-LASSO) possède sa propre ligne d'ablation **à hyperparamètres d'optimisation constants** :
- **Optimiseur :** `Muon` (`--muon-lr 2e-3`, Newton-Schulz 5 pas) + `AdamW` (`--lr 3e-4`, cosine decay, warmup 5 %)
- **Perte :** `{"ce": 0.1, "tv": 0.9}` (alignement direct sur le taux d'acceptation théorique $1 - \text{TV}(p, q)$) + perte BCE de confiance (`--confidence-head-alpha 1.0`)
- **Séquence & Ancres :** `--total-seq-len 8192`, `--max-anchors 512`, `--block-size 8`, `--num-layers 5`, `--draft-vocab-size 32768`, **`8 époques`** sur les 65k conversations (`--train-data-ratio 0.956`).

### Tableau de la Matrice Expérimentale Orthogonale (9 Modèles)

| ID Run | Backbone | Causal Shift & Scan Reset | RFF ($d_{\text{re}}$, $\Omega$) & `anchor_rank` | Rollout Multi-Couches | Mixer Intra-Bloc $K\times K$ | Confiance (Born / Vide / Interf.) | Modes $M_{\text{train}} \to M_{\text{infer}}$ | Params Actifs | État / Séq | Question Scientifique Isolée |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B1_DSpark_SWA128`** | `swa` ($W=128$) | Natif SWA | — | Natif Attn | Natif 5× | MLP `DSpark` | — | `1 072 M` | **`2.62 MB`** | Baseline SOTA officielle `DSpark` |
| **`B2_DSpark_SWA512`** | `swa` ($W=512$) | Natif SWA | — | Natif Attn | Natif 5× | MLP `DSpark` | — | `1 072 M` | **`10.49 MB`** | Coût mémoire SWA sur contexte 8k |
| **`B3_Hybrid_1SWA_4KvN`** | `hybrid` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide** | $128 \to 128$ | `462 M` | **`526 KB`** | Compromis hybride (`5×` moins de KV) |
| **`A0_KvN_v1_CausalFix`** | `kvn` | **Oui (v2)** | Dense ($4096$, figé), Dense | Non (couche 4) | Non | Born v1 (sans interf.) | $128 \to 128$ | `841 M` | **`2.56 KB`** | **Vraie base Phase 1 sans fuite causale** |
| **`A1_KvN_ModalRFF_Comp`** | `kvn` | **Oui (v2)** | **$512$, Appren., $r_a=512$** | Non (couche 4) | Non | Born v1 | $128 \to 128$ | **`361 M`** | **`2.56 KB`** | Gain spectral Barthe Thm 2.1 + `-57%` params |
| **`A2_KvN_Multilayer`** | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | **Oui (5 couches)** | Non | Born v1 | $128 \to 128$ | `362 M` | **`2.56 KB`** | Gain des échelles fréquentielles basses |
| **`A3_KvN_IntraBlock`** | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | **Oui ($8\times 8$)** | MLP `DSpark` *(no Born)* | $128 \to 128$ | `395 M` | **`2.56 KB`** | **Briser le plafond *H-Spec* sur $k=2..7$** |
| **`A4_KvN_v2_Full128`** | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide (v2)** | $128 \to 128$ | **`395 M`** | **`2.56 KB`** | **Gain pur de la calibration Born v2** *(vs `A3`)* |
| **`A5_KvN_v2_LASSO_512to128`** | `kvn` | **Oui (v2)** | $512$, Appren., $r_a=512$ | Oui | Oui | **Interf. + Vide (v2)** | **$512 \to 128$** | **`395 M`** *(post-prune)* | **`2.56 KB`** | **Gain de loterie spectrale Group-LASSO dyadique** |

### Script d'Orchestration d'Entraînement ([`kvn_dspark_phase2_h200_suite.sh`](examples/train/kvn_dspark_phase2_h200_suite.sh))

- [x] **Étape C.1 : Lancer la suite orchestrée (ou un modèle/seed individuel) sur 2×H200 ou 8×H200**
  Le script [`examples/train/kvn_dspark_phase2_h200_suite.sh`](examples/train/kvn_dspark_phase2_h200_suite.sh) encapsule l'intégralité des Phases B et C, sépare proprement `base_args` et `scratch_common` (afin que `--from-pretrained` lors de la recalibration post-élagage n'entre jamais en conflit avec `DECODER_SHAPING_FLAGS` `--num-layers` / `--sliding-window`), et est vérifié automatiquement par `pytest` :
  ```bash
  # Exécution complète (Phases B + C : 3 seeds sur B1/A4/A5 + seed 42 sur B2/B3/A0/A1/A2/A3)
  bash examples/train/kvn_dspark_phase2_h200_suite.sh

  # Ou exécution ciblée sur 8xH200 (GPUs 0-1 pour vLLM DP=2, GPUs 2-7 pour torchrun FSDP2) :
  VLLM_GPUS="0,1" VLLM_DP=2 TRAIN_GPUS="2,3,4,5,6,7" NPROC=6 \
      MODELS="A4_KvN_v2_Full128 A5_KvN_v2_LASSO_512to128" SEEDS="42 1337 2026" \
      bash examples/train/kvn_dspark_phase2_h200_suite.sh
  ```

- [ ] **Étape C.3 : Ablation Horizon Étendu ($K = 16$) et Échelle Modèle Vérifieur (`Qwen3.5-0.8B` & `Qwen3-32B`)**
  - Entraîner `B1_SWA128_K16` et `A5_KvN_v2_K16` avec `--block-size 16` pour démontrer que le coût de l'`IntraBlockCausalMixer` ($16 \times 16$) reste à `0 octet` de KV-cache alors que la latence et les lectures KV de SWA augmentent.
  - Exécuter un run de transfert d'échelle sur **`Qwen/Qwen3-32B`** (`hidden_size = 5120`, 64 couches, `--target-layer-ids 4 32 60`, TP=2) pour établir la scalabilité industrielle sur grands modèles dans le papier.

---

## 4. Phase D — Évaluation SOTA de Bout en Bout (`vLLM` + `AIPerf` + Loi d'Amdahl)

Contrairement à la Phase 1 qui mesurait uniquement le débit isolé du drafter sur un benchmark synthétique, la Phase 2 évalue le **système complet `Verifier + Drafter`** sous `vLLM` avec vérification spéculative exacte (rejection sampling sans perte de qualité).

### 4.1. Évaluation de la Qualité d'Acceptation ($\tau$) sur 5 Benchmarks Standards

- [ ] **Étape D.1 : Lancer la campagne `evaluate.py acceptance` (`AIPerf`) sur les 5 jeux hors-distribution**
  Pour chaque checkpoint final (`B1`, `B2`, `B3`, `A0`..`A5`) et chaque régime de décodage (**Greedy $T=0.0$** et **Stochastique $T=0.6$**) :
  ```bash
  SCRIPT_DIR="scripts/evaluate"
  OUT_EVAL="/workspace/kvn_phase2_artifacts/eval_results"
  mkdir -p "$OUT_EVAL"

  for BENCH in spec_al_mt_bench spec_al_gsm8k spec_al_math500 spec_al_humaneval spec_al_livecodebench; do
      for MODEL_DIR in \
          "$SAVE_ROOT/B1_DSpark_SWA128_s42/7" \
          "$SAVE_ROOT/A0_KvN_v1_CausalFix_s42/7" \
          "$SAVE_ROOT/A3_KvN_IntraBlock_s42/7" \
          "$SAVE_ROOT/A4_KvN_v2_Full128_s42/7" \
          "$SAVE_ROOT/A5_KvN_v2_Pruned128_ZeroShot_s42" \
          "$SAVE_ROOT/A5_KvN_v2_Pruned128_Recal_s42/0"; do

          TAG="$(basename "$(dirname "$MODEL_DIR")")_$(basename "$MODEL_DIR")"
          vllm serve "$MODEL_DIR" --port 8108 --dtype bfloat16 \
              --per-request-spec-decode-metrics summary &
          VLLM_PID=$!
          until curl -sf http://localhost:8108/health > /dev/null; do sleep 2; done

          python "$SCRIPT_DIR/evaluate.py" acceptance \
              --url "http://localhost:8108" \
              --model "$MODEL_DIR" \
              --dataset "$BENCH" \
              --output-artifact-dir "$OUT_EVAL/${TAG}_${BENCH}"

          kill "$VLLM_PID" && wait "$VLLM_PID" 2>/dev/null || true
      done
  done
  ```

### 4.2. Évaluation de la Calibration de Confiance (ECE, AUROC & Arrêt Dynamique de Bloc)

- [ ] **Étape D.2 : Générer la courbe de Pareto d'arrêt dynamique ($\gamma_{\text{conf}} \in \{0.0, 0.2, 0.35, 0.5, 0.65, 0.8\}$)**
  Comparer directement `A3` (MLP standard `DSpark`), `A0` (Born v1 aveugle à la phase) et `A4` (**Born v2 avec interféromètre dyadique + mode du vide $|M+1\rangle$**) sur :
  1. **ECE** (*Expected Calibration Error* à 15 bacs sur $\mathbb{P}(\text{accept}_k = 1)$),
  2. **AUROC** de prédiction de rejet par horizon $k \in \{0, \dots, 7\}$,
  3. **Nombre moyen de FLOPs de vérification économisés** par troncature dynamique du bloc dès que $\hat{c}_{t,k} < \gamma_{\text{conf}}$.

### 4.3. Benchmark Système de Bout en Bout (Loi d'Amdahl, Latence TTFT/TPOT & Capacité Max Batch)

- [ ] **Étape D.3 : Mesurer le débit réel (tokens/s) et le speedup $\mathcal{S}_{\text{e2e}}(B, L_{\text{ctx}})$ face à `Qwen3-8B` sans spéculation et face à `DSpark` / `EAGLE-3`**
  Balayer systématiquement :
  - **Concurrence $B \in \{1, 4, 8, 16, 32, 64, 128, 256\}$** (mettant en évidence : (i) le gain de compression paramétrique `-53 %` à $B \in [1, 16]$, et (ii) le franchissement des `50 MB` du cache L2 du H200 par `DSpark` SWA dès $B = 20$ !).
  - **Longueurs de contexte $L_{\text{ctx}} \in \{2\,048,\; 8\,192,\; 32\,768\}$** (démontrant l'invariance stricte en $O(1)$ de `advance_and_draft_step` et l'augmentation du nombre maximal de requêtes simultanées logeables en HBM3e).

---

## 5. Critères de Succès Quantitatifs (Go / No-Go Publication & Production)

| Métrique Cible | Phase 1 (Auditée) | Seuil Minimum Go-Prod | Objectif SOTA Papier (ICML/NeurIPS) |
| :--- | :---: | :---: | :---: |
| **Fuite causale à l'ancre / inter-documents** | Présente (`+28.6 pts` pos 0) | **`0.00` (`atol < 1e-6`)** | **`0.00` prouvé par test unitaire formel** |
| **Empreinte d'état drafter / séquence (`Qwen3-8B`)** | `2.56 KB` | **`2.56 KB` (`1 024×` vs SWA-128)** | **`2.56 KB` (`0 B` KV-cache, `4 096×` vs SWA-512)** |
| **Paramètres actifs du drafter** | `841.5 M` | `< 450 M` | **`~395 M` (`-53 %` vs v1, `-63 %` vs `DSpark`)** |
| **Longueur moyenne d'acceptation $\tau$ ($K=8$, Greedy, moy. 5 benchs)** | Biaisée (`tutorial_regen`) | $\ge 97\%$ du $\tau$ de `B1_DSpark_SWA128` | **$\ge 100\%$ du $\tau$ de `B1_DSpark_SWA128` ($\tau \ge 3.6$–$4.2$)** |
| **Calibration de confiance (ECE sur rejet spéculatif)** | Non mesurée (MAE seule) | `< 0.045` | **`< 0.025` (`A4` Born-Vide bat `A3` MLP significativement)** |
| **Perte de $\tau$ lors de l'élagage dyadique $512 \to 128$** | `+0.17` non isolé | `< 0.05` zéro-shot | **`< 0.02` zéro-shot et Gain net `A5 > A4` après 1 ep recal.** |
| **Speedup de bout en bout `vLLM` ($\mathcal{S}_{\text{e2e}}$, Loi d'Amdahl incluse)** | Non mesuré (`4.15×` isolé) | **`\ge 2.1×` à $B=1$ ; `\ge 1.7×` à $B=64$** | **`2.4×–2.8×` à $B=1..8$ ; `> 1.30×` vs `DSpark` à $B \ge 64$** |
