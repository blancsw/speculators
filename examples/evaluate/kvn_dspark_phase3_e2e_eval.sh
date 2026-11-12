#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Phase 3 — Master End-to-End Evaluation, SpecInfer Kraus-Tree,
# Full-Vocab OOV Calibration & Production vLLM V1 Benchmark Suite
# Implements Phase D.1, D.2, D.3 & D.4 of speculators/kvn_rff_rnn_mtp_train_phase_3.md
#
# Key Scientific & Systems Guarantees:
#   1. Complete multi-seed coverage (_s{42,1337,2026}) across B1, V2_Fixed, V3_A5,
#      and ALL THREE stages of V3_A6 (unpruned M=512 ep4, zero-shot pruned M=128,
#      and 1-ep warm-recalibrated M=128 ep0) — fixing [M6].
#   2. Linear chain vs Kraus POVM Speculative Tree (--use-kraus-tree, SpecInfer)
#      lossless evaluation at T=0.0 (exact greedy match) and T=0.6 (lossless MC).
#   3. Full-vocab (151,936) OOV-exact 15-bin ECE, Rejection AUROC & cumulative
#      prefix survival Pareto curve (prod_{j<=k} c_j >= gamma) — fixing [C4]/[M5].
#   4. True production `vllm serve` E2E Amdahl & Zero-Replay APC benchmark via
#      `scripts/evaluate/benchmark_vllm_prod.py` across B in {1,4,8,16,32,64,128}!
# ==============================================================================

set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/kvn_phase3_artifacts}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192}"
SAVE_ROOT="${SAVE_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
OUT_EVAL="${OUT_EVAL:-${ARTIFACT_ROOT}/eval_results_phase3}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32768}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LAST_EPOCH="${LAST_EPOCH:-5}"             # 6-epoch runs (indexed 0..5)
LAST_LASSO_EPOCH="${LAST_LASSO_EPOCH:-4}" # 5-epoch LASSO runs (indexed 0..4)
RUN_VLLM_PROD="${RUN_VLLM_PROD:-1}"
VLLM_EVAL_PORT="${VLLM_EVAL_PORT:-8010}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_TOOL="${REPO_ROOT}/scripts/evaluate/kvn_phase2_calibration_and_amdahl.py"
VLLM_PROD_TOOL="${REPO_ROOT}/scripts/evaluate/benchmark_vllm_prod.py"
mkdir -p "${OUT_EVAL}/acceptance" "${OUT_EVAL}/kraus_tree" "${OUT_EVAL}/calibration" \
         "${OUT_EVAL}/controlled_speedup" "${OUT_EVAL}/vllm_prod"

# Complete multi-seed (_s{42,1337,2026}) + orthogonal single-factor checkpoint roster
CHECKPOINTS=(
    # Headline 3-seed baselines & v2 fixed anchor
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/B1_DSpark_SWA128_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/B1_DSpark_SWA128_s2026/${LAST_EPOCH}"
    "${SAVE_ROOT}/V2_Fixed_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V2_Fixed_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/V2_Fixed_s2026/${LAST_EPOCH}"
    # Orthogonal v3 progression + FFN-matched reviewer control (seed 42)
    "${SAVE_ROOT}/B2_DSpark_SWA512_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/B3_Hybrid_1SWA_4KvN_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A1_DyadicRFF_Engram_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A2_DEAO_SelfCond_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A3_SymplecticButterfly_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A4_LindbladPurity_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_FFNMatched_4p5_s42/${LAST_EPOCH}"
    # Headline v3 Full (3 seeds)
    "${SAVE_ROOT}/V3_A5_KrausTree_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A5_KrausTree_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A5_KrausTree_s2026/${LAST_EPOCH}"
    # Headline v3 Proximal Group-LASSO 512->128 across ALL 3 seeds & ALL 3 stages!
    "${SAVE_ROOT}/V3_A6_ProxLASSO_512to128_s42/${LAST_LASSO_EPOCH}"
    "${SAVE_ROOT}/V3_A6_ProxLASSO_512to128_s1337/${LAST_LASSO_EPOCH}"
    "${SAVE_ROOT}/V3_A6_ProxLASSO_512to128_s2026/${LAST_LASSO_EPOCH}"
    "${SAVE_ROOT}/V3_A6_Pruned128_ZeroShot_s42"
    "${SAVE_ROOT}/V3_A6_Pruned128_ZeroShot_s1337"
    "${SAVE_ROOT}/V3_A6_Pruned128_ZeroShot_s2026"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s42/0"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s1337/0"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s2026/0"
)

KRAUS_TREE_CHECKPOINTS=(
    "${SAVE_ROOT}/V3_A5_KrausTree_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A5_KrausTree_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A5_KrausTree_s2026/${LAST_EPOCH}"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s42/0"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s1337/0"
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s2026/0"
)

resolve_tag() {
    local ckpt="$1"
    local leaf
    leaf="$(basename "$ckpt")"
    if [[ "$leaf" =~ ^[0-9]+$ ]]; then
        echo "$(basename "$(dirname "$ckpt")")_ep${leaf}"
    else
        echo "$leaf"
    fi
}

# ==============================================================================
# Phase D.1a: Linear-Chain Lossless Speculative Acceptance (T=0.0 & T=0.6)
# ==============================================================================
for CKPT in "${CHECKPOINTS[@]}"; do
    [[ -d "$CKPT" ]] || { echo "Skipping missing checkpoint: $CKPT"; continue; }
    TAG="$(resolve_tag "$CKPT")"

    for BENCH in mt_bench gsm8k math500 humaneval mbpp; do
        for TEMP in 0.0 0.6; do
            python "$EVAL_TOOL" benchmark-e2e \
                --checkpoint "$CKPT" \
                --verifier-model "$VERIFIER_MODEL" \
                --data-dir "$DATA_DIR" \
                --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
                --benchmark "$BENCH" \
                --num-prompts "$NUM_PROMPTS" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --temperature "$TEMP" \
                --batch-sizes 1 \
                --output-json "${OUT_EVAL}/acceptance/${TAG}_${BENCH}_T${TEMP}.json"
        done
    done
done

# ==============================================================================
# Phase D.1b: Kraus POVM Speculative Tree + SpecInfer Verification (T=0.0 & T=0.6)
# ==============================================================================
for CKPT in "${KRAUS_TREE_CHECKPOINTS[@]}"; do
    [[ -d "$CKPT" ]] || continue
    TAG="$(resolve_tag "$CKPT")"

    for BENCH in mt_bench gsm8k math500 humaneval mbpp; do
        for TEMP in 0.0 0.6; do
            python "$EVAL_TOOL" benchmark-e2e \
                --checkpoint "$CKPT" \
                --verifier-model "$VERIFIER_MODEL" \
                --data-dir "$DATA_DIR" \
                --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
                --benchmark "$BENCH" \
                --num-prompts "$NUM_PROMPTS" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --temperature "$TEMP" \
                --use-kraus-tree \
                --batch-sizes 1 \
                --output-json "${OUT_EVAL}/kraus_tree/${TAG}_kraus_${BENCH}_T${TEMP}.json"
        done
    done
done

# ==============================================================================
# Phase D.2: Full-Vocab (151,936) OOV-Exact Calibration & Lindblad Purity Pareto
# ==============================================================================
if [[ -d "$DATA_DIR" ]]; then
    for CKPT in "${CHECKPOINTS[@]}"; do
        [[ -d "$CKPT" ]] || continue
        TAG="$(resolve_tag "$CKPT")"
        python "$EVAL_TOOL" calibrate \
            --checkpoint "$CKPT" \
            --verifier-model "$VERIFIER_MODEL" \
            --data-dir "$DATA_DIR" \
            --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
            --train-data-ratio "$TRAIN_DATA_RATIO" \
            --max-samples 64 \
            --output-json "${OUT_EVAL}/calibration/${TAG}_calibration.json"
    done
fi

# ==============================================================================
# Phase D.3: Controlled PyTorch Batched Roofline Sweep (B in 1..128, True KV Cache)
# ==============================================================================
for CKPT in \
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/B2_DSpark_SWA512_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/B3_Hybrid_1SWA_4KvN_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/V2_Fixed_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/V3_A5_KrausTree_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s42/0"; do
    [[ -d "$CKPT" ]] || continue
    TAG="$(resolve_tag "$CKPT")"
    python "$EVAL_TOOL" benchmark-e2e \
        --checkpoint "$CKPT" \
        --verifier-model "$VERIFIER_MODEL" \
        --data-dir "$DATA_DIR" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --benchmark gsm8k \
        --num-prompts 32 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --batch-sizes 1 4 8 16 32 64 128 \
        --output-json "${OUT_EVAL}/controlled_speedup/${TAG}_pytorch_batches.json"
done

# ==============================================================================
# Phase D.4: Production `vllm serve` + KvNDSparkVllmProposer + ZeroReplayAPC Benchmark
# Uses `vllm.general_plugins` (`register_kvn_vllm_plugin`) to auto-route `kvn`
# checkpoints to `KvNDSparkVllmProposer` and `swa` checkpoints to `dspark`.
# ==============================================================================
if [[ "$RUN_VLLM_PROD" == "1" && -f "$VLLM_PROD_TOOL" ]]; then
    # 1. Non-speculative vLLM V1 baseline throughput across concurrency B in {1..128}
    python "$VLLM_PROD_TOOL" \
        --launch-server \
        --model-name "$VERIFIER_MODEL" \
        --port "$VLLM_EVAL_PORT" \
        --concurrencies "1,4,8,16,32,64,128" \
        --requests-per-worker 2 \
        --warmup-requests 5 \
        --max-tokens "$MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --skip-apc-benchmark \
        --output-json "${OUT_EVAL}/vllm_prod/Verifier_Baseline_vllm_serve_prod.json"

    # 2. Speculative vLLM V1 serving + Zero-Replay APC prefix-hit benchmark
    for CKPT in \
        "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}" \
        "${SAVE_ROOT}/V3_A5_KrausTree_s42/${LAST_EPOCH}" \
        "${SAVE_ROOT}/V3_A6_Pruned128_Recal_s42/0"; do
        [[ -d "$CKPT" ]] || continue
        TAG="$(resolve_tag "$CKPT")"
        python "$VLLM_PROD_TOOL" \
            --launch-server \
            --model-name "$VERIFIER_MODEL" \
            --port "$VLLM_EVAL_PORT" \
            --speculative-config "{\"model\": \"${CKPT}\"}" \
            --baseline-json "${OUT_EVAL}/vllm_prod/Verifier_Baseline_vllm_serve_prod.json" \
            --concurrencies "1,4,8,16,32,64,128" \
            --requests-per-worker 2 \
            --warmup-requests 5 \
            --max-tokens "$MAX_NEW_TOKENS" \
            --temperature 0.0 \
            --output-json "${OUT_EVAL}/vllm_prod/${TAG}_vllm_serve_prod.json"
    done
fi

echo "=== Phase 3 Evaluation Suite Complete: Results written to ${OUT_EVAL} ==="
