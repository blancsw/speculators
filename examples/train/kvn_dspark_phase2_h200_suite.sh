#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Phase 2 — Full Multi-Domain Data Prep & 9-Model Orthogonal Ablation Suite
# Implements Phase B & Phase C of speculators/kvn_rff_rnn_mtp_train_phase_2.md
#
# Usage:
#   # Full cluster execution (2xH200 or 8xH200):
#   bash examples/train/kvn_dspark_phase2_h200_suite.sh
#
#   # Single model / seed execution:
#   MODELS="A5_KvN_v2_LASSO_512to128" SEEDS="42" SKIP_DATA_PREP=1 \
#       bash examples/train/kvn_dspark_phase2_h200_suite.sh
#
#   # Dry-run CLI verification (used by pytest test_kvn_phase2_pipeline.py):
#   source examples/train/kvn_dspark_phase2_h200_suite.sh --dry-run-lib
#   emit_train_argv A5_KvN_v2_LASSO_512to128 42
# ==============================================================================

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/kvn_phase2_artifacts}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192}"
SAVE_ROOT="${SAVE_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
HS_CACHE_DIR="${HS_CACHE_DIR:-/dev/shm/kvn_phase2_hs_cache}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Stratified 100k Multi-Domain On-Policy Mix (95k train + 5k held-out val via TRAIN_DATA_RATIO=0.95)
ULTRACHAT_SAMPLES="${ULTRACHAT_SAMPLES:-45000}"
OPENMATH_SAMPLES="${OPENMATH_SAMPLES:-32000}"
MAGICODER_SAMPLES="${MAGICODER_SAMPLES:-23000}"

# Hardware Topology:
# - Default 2xH200: VLLM_GPUS="0", TRAIN_GPUS="1", NPROC=1, VLLM_DP=1
# - For 8xH200:     VLLM_GPUS="0,1", TRAIN_GPUS="2,3,4,5,6,7", NPROC=6, VLLM_DP=2
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_DP="${VLLM_DP:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
NPROC="${NPROC:-1}"

DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32768}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
# Keep vLLM context strictly > SEQ_LENGTH so /render and /chat/completions never hit HTTP 400 (max_tokens >= 1)
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((SEQ_LENGTH + 4096))}"
EPOCHS="${EPOCHS:-6}"
RECAL_EPOCHS="${RECAL_EPOCHS:-1}"
ISO_COMPUTE_PRUNE="${ISO_COMPUTE_PRUNE:-0}"
LASSO_EPOCHS="${LASSO_EPOCHS:-$((ISO_COMPUTE_PRUNE == 1 ? EPOCHS - RECAL_EPOCHS : EPOCHS))}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
MIN_SHM_MB="${MIN_SHM_MB:-32768}"
SKIP_SHM_CHECK="${SKIP_SHM_CHECK:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"

# Build exact speculators.train CLI argv for any of the 9 orthogonal ablation models
# or the post-pruning recalibration job (which omits DECODER_SHAPING_FLAGS --num-layers
# and --sliding-window as required when --from-pretrained is passed).
emit_train_argv() {
    local model_id="$1"
    local seed="${2:-42}"
    local target_layers=()
    read -r -a target_layers <<< "$TARGET_LAYER_IDS"

    # Base arguments shared across both from-scratch and --from-pretrained runs
    local base_args=(
        --verifier-name-or-path "$VERIFIER_MODEL"
        --speculator-type kvn_dspark
        --data-path "$DATA_DIR"
        --hidden-states-path "$HS_CACHE_DIR"
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1"
        --draft-vocab-size "$DRAFT_VOCAB_SIZE"
        --total-seq-len "$SEQ_LENGTH"
        --train-data-ratio "$TRAIN_DATA_RATIO"
        --optimizer muon
        --scheduler-type cosine
        --loss-implementation fused
        --loss-fn '{"ce": 0.1, "tv": 0.9}'
        --confidence-head-alpha 1.0
        --block-size 8
        --max-anchors 512
        --target-layer-ids "${target_layers[@]}"
        --on-missing generate
        --num-workers 8
        --logger tensorboard
        --seed "$seed"
    )

    # Scratch-init arguments (decoder shaping + architecture hyper-parameters)
    local scratch_common=(
        --epochs "$EPOCHS"
        --lr 3e-4
        --muon-lr 2e-3
        --num-layers 5
        --markov-rank 256
        --markov-head-type gated
        --enable-confidence-head
        --confidence-head-with-markov
    )

    local specific_args=()
    case "$model_id" in
        B1_DSpark_SWA128)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type swa
                --no-kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        B2_DSpark_SWA512)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 512
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type swa
                --no-kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        B3_Hybrid_1SWA_4KvN)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type hybrid_1swa_2kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --kvn-enable-interferometer
                --kvn-vacuum-born-norm
                --kvn-multilayer-rollout
                --kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A0_KvN_v1_CausalFix)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 0
                --no-kvn-learnable-rff
                --kvn-anchor-rank 0
                --no-kvn-enable-interferometer
                --no-kvn-vacuum-born-norm
                --no-kvn-multilayer-rollout
                --no-kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A1_KvN_ModalRFF_Comp)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --no-kvn-enable-interferometer
                --no-kvn-vacuum-born-norm
                --no-kvn-multilayer-rollout
                --no-kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A2_KvN_Multilayer)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --no-kvn-enable-interferometer
                --no-kvn-vacuum-born-norm
                --kvn-multilayer-rollout
                --no-kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A3_KvN_IntraBlock)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --no-kvn-enable-interferometer
                --no-kvn-vacuum-born-norm
                --kvn-multilayer-rollout
                --kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --no-kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A4_KvN_v2_Full128)
            specific_args=(
                "${scratch_common[@]}"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 128
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --kvn-enable-interferometer
                --kvn-vacuum-born-norm
                --kvn-multilayer-rollout
                --kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 0.0
            )
            ;;
        A5_KvN_v2_LASSO_512to128)
            specific_args=(
                "${scratch_common[@]}"
                --epochs "$LASSO_EPOCHS"
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-backbone-type kvn
                --kvn-n-modes 512
                --kvn-reupload-layers 2
                --kvn-reupload-dim 512
                --kvn-learnable-rff
                --kvn-anchor-rank 512
                --kvn-enable-interferometer
                --kvn-vacuum-born-norm
                --kvn-multilayer-rollout
                --kvn-intrablock-mixer
                --kvn-strict-causal-shift
                --kvn-use-born-confidence
                --kvn-lasso-weight 3e-4
            )
            ;;
        A5_KvN_v2_Pruned128_Recal)
            specific_args=(
                --from-pretrained "${SAVE_ROOT}/A5_KvN_v2_Pruned128_ZeroShot_s${seed}"
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --epochs "$RECAL_EPOCHS"
                --lr 5e-5
                --muon-lr 2e-4
                --kvn-lasso-weight 0.0
            )
            ;;
        *)
            echo "Unknown ablation model ID: $model_id" >&2
            return 1
            ;;
    esac

    printf "%s\n" "${base_args[@]}" "${specific_args[@]}"
}

# Allow unit tests to source this file purely as a shell library without executing GPU jobs
if [[ "${1:-}" == "--dry-run-lib" ]]; then
    return 0 2>/dev/null || exit 0
fi

set -euo pipefail

if [[ "$SKIP_SHM_CHECK" != "1" && -d "/dev/shm" ]]; then
    SHM_AVAIL_MB="$(df -Pm /dev/shm | awk 'NR==2 {print $4}')"
    if [[ -n "$SHM_AVAIL_MB" && "$SHM_AVAIL_MB" -lt "$MIN_SHM_MB" ]]; then
        echo "ERROR: /dev/shm has only ${SHM_AVAIL_MB} MB free (< MIN_SHM_MB=${MIN_SHM_MB} MB)." >&2
        echo "Relaunch container with '--shm-size=64g' (or point HS_CACHE_DIR to fast NVMe)." >&2
        exit 1
    fi
fi

mkdir -p "$DATA_DIR" "$SAVE_ROOT" "$HS_CACHE_DIR"

VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ==============================================================================
# Phase B: Launch vLLM Hidden-State Producer + Build Decontaminated 100k Dataset
# ==============================================================================
echo "=== Phase B.1: Launching vLLM server on GPUs ${VLLM_GPUS} (${VERIFIER_MODEL}) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$VERIFIER_MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HS_CACHE_DIR" \
    -- --data-parallel-size "$VLLM_DP" --port "$VLLM_PORT" --max-model-len "$VLLM_MAX_MODEL_LEN" --gpu-memory-utilization 0.88 &
VLLM_PID=$!

until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready on port ${VLLM_PORT}."

if [[ "$SKIP_DATA_PREP" != "1" ]]; then
    RAW_JSONL="${ARTIFACT_ROOT}/raw_onpolicy_100k.jsonl"
    if [[ ! -f "$RAW_JSONL" ]]; then
        echo "=== Phase B.2a: Building 13-gram decontaminated 100k multi-domain on-policy JSONL ==="
        python scripts/build_decontaminated_mix.py \
            --verifier-model "$VERIFIER_MODEL" \
            --ultrachat-samples "$ULTRACHAT_SAMPLES" \
            --openmath-samples "$OPENMATH_SAMPLES" \
            --magicoder-samples "$MAGICODER_SAMPLES" \
            --exclude-benchmarks mt_bench gsm8k math500 humaneval mbpp livecodebench \
            --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
            --concurrency "$((64 * VLLM_DP))" \
            --output-jsonl "$RAW_JSONL" \
            --val-samples 0 \
            --resume \
            --seed 42
    fi

    echo "=== Phase B.2b: Running speculators prepare-data (seq_len=${SEQ_LENGTH}) ==="
    speculators prepare-data \
        --model "$VERIFIER_MODEL" \
        --data "$RAW_JSONL" \
        --render-endpoint "http://localhost:${VLLM_PORT}" \
        --output "$DATA_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --no-skip-token-freq
fi

# ==============================================================================
# Phase C: Run Orthogonal Ablation Matrix + Multi-Seed Statistical Protocol
# ==============================================================================
run_model_seed() {
    local model_id="$1"
    local seed="$2"
    echo "=== Training ${model_id} (seed=${seed}) on GPUs ${TRAIN_GPUS} ==="
    mapfile -t train_argv < <(emit_train_argv "$model_id" "$seed")
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
        --standalone --nproc_per_node="$NPROC" \
        -m speculators.train "${train_argv[@]}"

    if [[ "$model_id" == "A5_KvN_v2_LASSO_512to128" ]]; then
        local last_ep=$((LASSO_EPOCHS - 1))
        local ckpt_dir="${SAVE_ROOT}/${model_id}_s${seed}/${last_ep}"
        local pruned_dir="${SAVE_ROOT}/A5_KvN_v2_Pruned128_ZeroShot_s${seed}"
        echo "=== Pruning ${model_id} (512 -> 128 dyadic modes, zero-shot) ==="
        python -c "
from pathlib import Path
import numpy as np, torch
from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.train.checkpointer import convert_float_dtype, patch_config_dtype
t2d = torch.from_numpy(np.load('${DATA_DIR}/t2d-${DRAFT_VOCAB_SIZE}.npy'))
d2t = torch.from_numpy(np.load('${DATA_DIR}/d2t-${DRAFT_VOCAB_SIZE}.npy'))
m = KvNDSparkDraftModel.from_pretrained('${ckpt_dir}', t2d=t2d, d2t=d2t)
m.prune_kvn_modes(128)
sd = convert_float_dtype(m.state_dict(), torch.bfloat16)
m.save_pretrained('${pruned_dir}', state_dict=sd)
patch_config_dtype(Path('${pruned_dir}') / 'config.json', torch.bfloat16)
"
        echo "=== Running 1-epoch post-pruning recalibration for seed ${seed} ==="
        mapfile -t recal_argv < <(emit_train_argv "A5_KvN_v2_Pruned128_Recal" "$seed")
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
            --standalone --nproc_per_node="$NPROC" \
            -m speculators.train "${recal_argv[@]}"
    fi
}

if [[ -n "${MODELS:-}" ]]; then
    read -r -a SELECTED_MODELS <<< "$MODELS"
    read -r -a SELECTED_SEEDS <<< "${SEEDS:-42}"
    for m in "${SELECTED_MODELS[@]}"; do
        for s in "${SELECTED_SEEDS[@]}"; do
            run_model_seed "$m" "$s"
        done
    done
else
    # 1. Core Headline Models across 3 independent random seeds (mean +/- std)
    for SEED in 42 1337 2026; do
        run_model_seed "B1_DSpark_SWA128" "$SEED"
        run_model_seed "A4_KvN_v2_Full128" "$SEED"
        run_model_seed "A5_KvN_v2_LASSO_512to128" "$SEED"
    done

    # 2. Orthogonal Single-Factor Ablations on canonical seed 42
    for ABL_MODEL in \
        "B2_DSpark_SWA512" \
        "B3_Hybrid_1SWA_4KvN" \
        "A0_KvN_v1_CausalFix" \
        "A1_KvN_ModalRFF_Comp" \
        "A2_KvN_Multilayer" \
        "A3_KvN_IntraBlock"; do
        run_model_seed "$ABL_MODEL" 42
    done
fi

echo "=== Phase C Complete: All checkpoints written to ${SAVE_ROOT} ==="
