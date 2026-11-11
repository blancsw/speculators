#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Phase 2 — End-to-End Speculative Acceptance, Calibration & Amdahl Speedup Suite
# Implements Phase D.1, D.2 & D.3 of speculators/kvn_rff_rnn_mtp_train_phase_2.md
#
# Runs natively in PyTorch on all KvN-DSpark (`kvn`), Hybrid (`hybrid_1swa_2kvn`), and
# DSpark (`swa`) checkpoints under a strictly identical verifier & rejection-sampling harness,
# and automatically supports online-trained Arrow datasets (no `hidden_states/` dir needed).
# ==============================================================================

set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/kvn_phase2_artifacts}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192}"
SAVE_ROOT="${SAVE_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
OUT_EVAL="${OUT_EVAL:-${ARTIFACT_ROOT}/eval_results}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32768}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LAST_EPOCH="${LAST_EPOCH:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_TOOL="${REPO_ROOT}/scripts/evaluate/kvn_phase2_calibration_and_amdahl.py"
mkdir -p "${OUT_EVAL}/acceptance" "${OUT_EVAL}/calibration" "${OUT_EVAL}/speedup"

# Full orthogonal ablation matrix (seed 42) + multi-seed headline runs (seeds 1337, 2026)
CHECKPOINTS=(
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/B1_DSpark_SWA128_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/B1_DSpark_SWA128_s2026/${LAST_EPOCH}"
    "${SAVE_ROOT}/B2_DSpark_SWA512_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/B3_Hybrid_1SWA_4KvN_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A0_KvN_v1_CausalFix_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A1_KvN_ModalRFF_Comp_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A2_KvN_Multilayer_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A3_KvN_IntraBlock_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s1337/${LAST_EPOCH}"
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s2026/${LAST_EPOCH}"
    "${SAVE_ROOT}/A5_KvN_v2_LASSO_512to128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_ZeroShot_s42"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s42/0"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s1337/0"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s2026/0"
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
# Phase D.1: Multi-Benchmark Lossless Speculative Acceptance (Greedy T=0 & T=0.6)
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
# Phase D.2: Confidence Calibration (15-bin ECE, Rejection AUROC, Pareto Curve)
# Works seamlessly on BOTH online Arrow datasets (split="val") and offline hidden_states!
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
# Phase D.3: End-to-End Batched Throughput & Amdahl Speedup (B in 1..128)
# ==============================================================================
for CKPT in \
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/B2_DSpark_SWA512_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s42/0"; do
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
        --output-json "${OUT_EVAL}/speedup/${TAG}_amdahl_batches.json"
done

echo "=== Phase D Evaluation Complete: Results written to ${OUT_EVAL} ==="
