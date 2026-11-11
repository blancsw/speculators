#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Phase 2 — End-to-End vLLM Acceptance, Calibration & Amdahl Speedup Suite
# Implements Phase D.1, D.2 & D.3 of speculators/kvn_rff_rnn_mtp_train_phase_2.md
#
# Prerequisites:
#   pip install -r scripts/evaluate/requirements.txt
# ==============================================================================

set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/kvn_phase2_artifacts}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192}"
SAVE_ROOT="${SAVE_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
OUT_EVAL="${OUT_EVAL:-${ARTIFACT_ROOT}/eval_results}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32768}"
VLLM_PORT="${VLLM_PORT:-8108}"
SERVER_URL="http://localhost:${VLLM_PORT}"
LAST_EPOCH="${LAST_EPOCH:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_SCRIPT_DIR="${REPO_ROOT}/scripts/evaluate"
mkdir -p "$OUT_EVAL"

VLLM_PID=""
stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap stop_vllm EXIT

launch_vllm_for_model() {
    local model_path="$1"
    stop_vllm
    vllm serve "$model_path" --port "$VLLM_PORT" --dtype bfloat16 \
        --per-request-spec-decode-metrics summary &
    VLLM_PID=$!
    until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
        sleep 2
    done
}

CHECKPOINTS=(
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A0_KvN_v1_CausalFix_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A3_KvN_IntraBlock_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s42/${LAST_EPOCH}"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_ZeroShot_s42"
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s42/0"
)

# ==============================================================================
# Phase D.1: Multi-Benchmark Speculative Acceptance Length (Greedy T=0 & T=0.6)
# ==============================================================================
for CKPT in "${CHECKPOINTS[@]}"; do
    [[ -d "$CKPT" ]] || { echo "Skipping missing checkpoint: $CKPT"; continue; }
    TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
    launch_vllm_for_model "$CKPT"

    for BENCH in spec_al_mtbench spec_al_gsm8k spec_al_math500 spec_al_humaneval spec_al_mbpp; do
        for TEMP in 0.0 0.6; do
            python "${EVAL_SCRIPT_DIR}/evaluate.py" acceptance \
                --url "$SERVER_URL" \
                --model "$CKPT" \
                --dataset "$BENCH" \
                --output-artifact-dir "${OUT_EVAL}/acceptance/${TAG}_${BENCH}_T${TEMP}" \
                -- --extra-inputs "{\"temperature\":${TEMP},\"max_completion_tokens\":2048}"
        done
    done
    stop_vllm
done

# ==============================================================================
# Phase D.2: Confidence Calibration (15-bin ECE, Rejection AUROC, Pareto Curve)
# ==============================================================================
if [[ -d "${DATA_DIR}/hidden_states" ]]; then
    for CKPT in \
        "${SAVE_ROOT}/A0_KvN_v1_CausalFix_s42/${LAST_EPOCH}" \
        "${SAVE_ROOT}/A3_KvN_IntraBlock_s42/${LAST_EPOCH}" \
        "${SAVE_ROOT}/A4_KvN_v2_Full128_s42/${LAST_EPOCH}" \
        "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s42/0"; do
        [[ -d "$CKPT" ]] || continue
        TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
        python "${EVAL_SCRIPT_DIR}/kvn_phase2_calibration_and_amdahl.py" calibrate \
            --checkpoint "$CKPT" \
            --verifier-model "$VERIFIER_MODEL" \
            --data-dir "$DATA_DIR" \
            --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
            --output-json "${OUT_EVAL}/calibration/${TAG}_calibration.json"
    done
fi

# ==============================================================================
# Phase D.3: End-to-End vLLM Throughput & Amdahl Speedup vs Autoregressive Base
# ==============================================================================
echo "=== Benchmarking Autoregressive Verifier Baseline (${VERIFIER_MODEL}) ==="
launch_vllm_for_model "$VERIFIER_MODEL"
python "${EVAL_SCRIPT_DIR}/evaluate.py" speedup \
    --url "$SERVER_URL" \
    --model "$VERIFIER_MODEL" \
    --dataset qualitative \
    --output-artifact-dir "${OUT_EVAL}/speedup/baseline_autoregressive" \
    -- --concurrency "1,4,8,16,32,64,128,256"
stop_vllm

for CKPT in \
    "${SAVE_ROOT}/B1_DSpark_SWA128_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/A4_KvN_v2_Full128_s42/${LAST_EPOCH}" \
    "${SAVE_ROOT}/A5_KvN_v2_Pruned128_Recal_s42/0"; do
    [[ -d "$CKPT" ]] || continue
    TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
    launch_vllm_for_model "$CKPT"
    python "${EVAL_SCRIPT_DIR}/evaluate.py" speedup \
        --url "$SERVER_URL" \
        --model "$CKPT" \
        --dataset qualitative \
        --output-artifact-dir "${OUT_EVAL}/speedup/${TAG}" \
        -- --concurrency "1,4,8,16,32,64,128,256"
    stop_vllm
done

echo "=== Phase D Evaluation Complete: Results written to ${OUT_EVAL} ==="
