#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Phase 3 — Master H200 Data Prep, 11-Model Orthogonal v3 Ablation
# Suite & Qwen3-32B Scale-Transfer Orchestrator
# Implements Phase B & Phase C of speculators/kvn_rff_rnn_mtp_train_phase_3.md
#
# Key Scientific & Systems Guarantees ([C1]-[C6] & [M1]-[M6] Verified):
#   1. Pre-flight /dev/shm >= 32 GB guard (--shm-size=64g recommended for 8k HS).
#   2. Conversation-ID disjoint 100k on-policy mix (45k unique UltraChat convs +
#      32k OpenMathInstruct-2 + 23k Magicoder), <think> truncation filter, and
#      train-only token_freq.pt (0.0% sibling prefix leakage).
#   3. Strict 6-epoch iso-compute across ALL models: V3_A6_ProxLASSO_512to128
#      runs 5 proximal Group-LASSO epochs + zero-shot prune + 1 warm recal epoch
#      (= 6 epochs total!), and --kvn-lasso-weight 0.0 on all non-pruned runs.
#   4. Includes explicit FFN-matched control (V3_FFNMatched_4p5, --kvn-ffn-mult 4.5)
#      matching Qwen3-8B SwiGLU 150,994,944 params/layer to the single parameter.
#
# Usage:
#   # Full cluster execution (2xH200 or 8xH200):
#   bash examples/train/kvn_dspark_phase3_h200_suite.sh
#
#   # Headline multi-seed execution:
#   MODELS="B1_DSpark_SWA128 V2_Fixed V3_A5_KrausTree V3_A6_ProxLASSO_512to128" \
#   SEEDS="42 1337 2026" SKIP_DATA_PREP=1 \
#       bash examples/train/kvn_dspark_phase3_h200_suite.sh
#
#   # Dry-run CLI verification (used by unit tests & Gate 2 verifier):
#   source examples/train/kvn_dspark_phase3_h200_suite.sh --dry-run-lib
#   emit_train_argv V3_A5_KrausTree 42
# ==============================================================================

VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-8B}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/kvn_phase3_artifacts}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/qwen3_8b_mix100k_seq8192}"
SAVE_ROOT="${SAVE_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
HS_CACHE_DIR="${HS_CACHE_DIR:-/dev/shm/kvn_phase3_hs_cache}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Stratified 100k Unique Source Conversations (95k train + 5k val by conversation_id)
ULTRACHAT_SAMPLES="${ULTRACHAT_SAMPLES:-45000}"
OPENMATH_SAMPLES="${OPENMATH_SAMPLES:-32000}"
MAGICODER_SAMPLES="${MAGICODER_SAMPLES:-23000}"

# Hardware Topology:
# - Default 2xH200: VLLM_GPUS="0", TRAIN_GPUS="1", NPROC=1, VLLM_DP=1
# - Recommended 8xH200: VLLM_GPUS="0,1", TRAIN_GPUS="2,3,4,5,6,7", NPROC=6, VLLM_DP=2
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_DP="${VLLM_DP:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
NPROC="${NPROC:-1}"

DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32768}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((SEQ_LENGTH + 4096))}"

# Strict Iso-Compute Epoch Budget:
# - Standard models: EPOCHS=6
# - Proximal LASSO pruned model (V3_A6): LASSO_EPOCHS=5 + RECAL_EPOCHS=1 = 6 total!
# - Proximal calibration: sum_t lr_t * PROX_LASSO_WEIGHT ~= 1.05 * ||w_g(0)||_RMS (~0.0142 on Qwen3-8B L=5 M=512)
EPOCHS="${EPOCHS:-6}"
LASSO_EPOCHS="${LASSO_EPOCHS:-$((EPOCHS - 1))}"
RECAL_EPOCHS="${RECAL_EPOCHS:-1}"
PROX_LASSO_WEIGHT="${PROX_LASSO_WEIGHT:-5.2e-3}"

TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
MIN_SHM_MB="${MIN_SHM_MB:-32768}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
SKIP_SHM_CHECK="${SKIP_SHM_CHECK:-0}"

emit_train_argv() {
    local model_id="$1"
    local seed="${2:-42}"
    local target_layers=()
    read -r -a target_layers <<< "$TARGET_LAYER_IDS"

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

    local scratch_common=(
        --lr 3e-4
        --muon-lr 2e-3
        --num-layers 5
        --markov-rank 256
        --enable-confidence-head
        --confidence-head-with-markov
        --kvn-state-dtype float32
    )

    local kvn_v2_base=(
        --kvn-backbone-type kvn
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
    )

    local specific_args=()
    case "$model_id" in
        B1_DSpark_SWA128)
            specific_args=(
                "${scratch_common[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
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
                --epochs "$EPOCHS"
                --markov-head-type rnn
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
                --epochs "$EPOCHS"
                --markov-head-type rnn
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
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V2_Fixed)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type gated
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --no-kvn-dyadic-rff
                --no-kvn-use-engram
                --no-kvn-symplectic-butterfly
                --no-kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.0
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A1_DyadicRFF_Engram)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-dyadic-rff
                --kvn-use-engram
                --no-kvn-symplectic-butterfly
                --no-kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.0
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A2_DEAO_SelfCond)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-dyadic-rff
                --kvn-use-engram
                --no-kvn-symplectic-butterfly
                --no-kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A3_SymplecticButterfly)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --no-kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A4_LindbladPurity)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --kvn-lindblad-purity
                --no-kvn-use-kraus-tree
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A5_KrausTree)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --kvn-lindblad-purity
                --kvn-use-kraus-tree
                --kvn-use-triton-scan
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_FFNMatched_4p5)
            # Reviewer-shield control ([M4]): --kvn-ffn-mult 4.5 gives 2 * 4096 * 18432 = 150,994,944
            # FFN params/layer, matching Qwen3-8B SwiGLU (3 * 4096 * 12288 = 150,994,944) identically!
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 128
                --kvn-ffn-mult 4.5
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --kvn-lindblad-purity
                --kvn-use-kraus-tree
                --kvn-use-triton-scan
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        V3_A6_ProxLASSO_512to128)
            specific_args=(
                "${scratch_common[@]}"
                "${kvn_v2_base[@]}"
                --epochs "$LASSO_EPOCHS"
                --markov-head-type rnn
                --sliding-window 128
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --kvn-n-modes 512
                --kvn-dyadic-rff
                --kvn-use-engram
                --kvn-symplectic-butterfly
                --kvn-lindblad-purity
                --kvn-use-kraus-tree
                --kvn-use-triton-scan
                --kvn-self-cond-prob 0.5
                --per-position-loss-weight deao
                --kvn-lasso-weight "$PROX_LASSO_WEIGHT"
            )
            ;;
        V3_A6_Pruned128_Recal)
            # Omit DECODER_SHAPING_FLAGS (--num-layers, --sliding-window) with --from-pretrained!
            specific_args=(
                --from-pretrained "${SAVE_ROOT}/V3_A6_Pruned128_ZeroShot_s${seed}"
                --save-path "${SAVE_ROOT}/${model_id}_s${seed}"
                --epochs "$RECAL_EPOCHS"
                --lr 5e-5
                --muon-lr 2e-4
                --per-position-loss-weight deao
                --kvn-lasso-weight 0.0
            )
            ;;
        *)
            echo "Unknown Phase 3 ablation model ID: $model_id" >&2
            return 1
            ;;
    esac

    printf "%s\n" "${base_args[@]}" "${specific_args[@]}"
}

if [[ "${1:-}" == "--dry-run-lib" ]]; then
    return 0 2>/dev/null || exit 0
fi

set -euo pipefail

# ==============================================================================
# Pre-Flight Guard: Verify /dev/shm Capacity ([M6] Docker 64MB SHM Trap Defense)
# Each seq_len=8192 BF16 hidden-state tensor (4 layers x 4096) is 268.4 MB in RAM!
# ==============================================================================
if [[ "$SKIP_SHM_CHECK" != "1" && -d "/dev/shm" ]]; then
    SHM_AVAIL_MB="$(df -Pm /dev/shm | awk 'NR==2 {print $4}')"
    if [[ -n "$SHM_AVAIL_MB" && "$SHM_AVAIL_MB" -lt "$MIN_SHM_MB" ]]; then
        echo "ERROR: /dev/shm has only ${SHM_AVAIL_MB} MB free (< MIN_SHM_MB=${MIN_SHM_MB} MB)." >&2
        echo "Relaunch container with '--shm-size=64g' (or point HS_CACHE_DIR to fast NVMe)." >&2
        exit 1
    fi
    echo "[Pre-Flight] /dev/shm check PASSED (${SHM_AVAIL_MB} MB available >= ${MIN_SHM_MB} MB)."
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
# Phase B: Launch vLLM Hidden-State Producer + Build Zero-Leak 100k Dataset
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
    RAW_JSONL="${ARTIFACT_ROOT}/raw_onpolicy_100k_conv_disjoint.jsonl"
    if [[ ! -f "$RAW_JSONL" ]]; then
        echo "=== Phase B.2a: Building 13-gram decontaminated, conversation_id-disjoint 100k mix ==="
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

    echo "=== Phase B.2b: Building conversation_id-disjoint Arrow dataset + train-only token_freq (seq_len=${SEQ_LENGTH}) ==="
    python scripts/build_decontaminated_mix.py \
        --prepare-only \
        --output-jsonl "$RAW_JSONL" \
        --prepare-arrow-dir "$DATA_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --train-data-ratio 0.95 \
        --seed 42
fi

# ==============================================================================
# Phase C: Orthogonal KvN-DSpark v3 Ablation Matrix (Strict 6-Epoch Iso-Compute)
# ==============================================================================
run_model_seed() {
    local model_id="$1"
    local seed="$2"
    echo "=== Training ${model_id} (seed=${seed}) on GPUs ${TRAIN_GPUS} ==="
    mapfile -t train_argv < <(emit_train_argv "$model_id" "$seed")
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
        --standalone --nproc_per_node="$NPROC" \
        -m speculators.train "${train_argv[@]}"

    if [[ "$model_id" == "V3_A6_ProxLASSO_512to128" ]]; then
        local last_ep=$((LASSO_EPOCHS - 1))
        local ckpt_dir="${SAVE_ROOT}/${model_id}_s${seed}/${last_ep}"
        local pruned_dir="${SAVE_ROOT}/V3_A6_Pruned128_ZeroShot_s${seed}"
        echo "=== Pruning ${model_id} (512 -> 128 modes after proximal Group-LASSO, zero-shot) ==="
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
        echo "=== Running ${RECAL_EPOCHS}-epoch warm recalibration (total = $((LASSO_EPOCHS + RECAL_EPOCHS)) iso-compute epochs) ==="
        mapfile -t recal_argv < <(emit_train_argv "V3_A6_Pruned128_Recal" "$seed")
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
    # 1. Core Headline Models across 3 independent seeds (mean +/- std for ICML/NeurIPS tables)
    for SEED in 42 1337 2026; do
        run_model_seed "B1_DSpark_SWA128" "$SEED"
        run_model_seed "V2_Fixed" "$SEED"
        run_model_seed "V3_A5_KrausTree" "$SEED"
        run_model_seed "V3_A6_ProxLASSO_512to128" "$SEED"
    done

    # 2. Orthogonal Single-Axis v3 Progression + Reviewer-Shield Controls on canonical seed 42
    for ABL_MODEL in \
        "B2_DSpark_SWA512" \
        "B3_Hybrid_1SWA_4KvN" \
        "V3_A1_DyadicRFF_Engram" \
        "V3_A2_DEAO_SelfCond" \
        "V3_A3_SymplecticButterfly" \
        "V3_A4_LindbladPurity" \
        "V3_FFNMatched_4p5"; do
        run_model_seed "$ABL_MODEL" 42
    done
fi

echo "=== Phase 3 Training Suite Complete: All checkpoints written to ${SAVE_ROOT} ==="
