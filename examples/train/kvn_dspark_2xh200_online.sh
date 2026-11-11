#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark v2 Zero-Drafter-KV-Cache Online Training & Dyadic Pruning Recipe
# Hardware Target: 2x NVIDIA H200 (2 x 141 GB HBM3e = 282 GB VRAM)
#   - GPU 0 (H200 #1): vLLM Server generating auxiliary hidden states in /dev/shm
#   - GPU 1 (H200 #2): torchrun speculators.train (--speculator-type kvn_dspark)
#
# Base Verifier LLMs supported out-of-the-box on 2xH200:
#   - Standard Speculators Benchmark: MODEL="Qwen/Qwen3-8B"   (TARGET_LAYER_IDS="2 18 33")
#   - High-Capacity H200 Scale-Up:    MODEL="Qwen/Qwen3-32B"  (TARGET_LAYER_IDS="4 32 60")
# ==============================================================================

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"
# Leave DATASET empty (default) to auto-build the 13-gram decontaminated multi-domain
# on-policy JSONL (45% UltraChat + 32% OpenMathInstruct-2 + 23% Magicoder) for $MODEL.
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/kvn_dspark_v2_qwen3_8b}"
HS_CACHE_DIR="${HS_CACHE_DIR:-/dev/shm/kvn_dspark_hs}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
ULTRACHAT_SAMPLES="${ULTRACHAT_SAMPLES:-$((MAX_SAMPLES * 45 / 100))}"
OPENMATH_SAMPLES="${OPENMATH_SAMPLES:-$((MAX_SAMPLES * 32 / 100))}"
MAGICODER_SAMPLES="${MAGICODER_SAMPLES:-$((MAX_SAMPLES - ULTRACHAT_SAMPLES - OPENMATH_SAMPLES))}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
EPOCHS="${EPOCHS:-6}"
RECAL_EPOCHS="${RECAL_EPOCHS:-1}"
LR="${LR:-3e-4}"
MUON_LR="${MUON_LR:-2e-3}"

# KvN-DSpark v2 Architecture & Dyadic Group-LASSO Scaling (M_train -> M_infer)
SPECULATOR_TYPE="kvn_dspark"
KVN_BACKBONE_TYPE="${KVN_BACKBONE_TYPE:-kvn}"   # kvn (2.56 KB/seq) | hybrid_1swa_2kvn | swa
KVN_N_MODES_TRAIN="${KVN_N_MODES_TRAIN:-512}"   # Over-parameterized RFF modes during training
KVN_N_MODES_INFER="${KVN_N_MODES_INFER:-128}"   # Pruned active RFF modes (1024x smaller than SWA-128)
KVN_REUPLOAD_LAYERS="${KVN_REUPLOAD_LAYERS:-2}"
KVN_REUPLOAD_DIM="${KVN_REUPLOAD_DIM:-512}"
KVN_ANCHOR_RANK="${KVN_ANCHOR_RANK:-512}"
KVN_LASSO_WEIGHT="${KVN_LASSO_WEIGHT:-3e-4}"

BLOCK_SIZE=8
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32768
MARKOV_RANK=256
MARKOV_HEAD_TYPE="gated"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# 2xH200 GPU Assignment
VLLM_GPUS="0"
TRAIN_GPUS="1"
NUM_TRAIN_GPUS=1
# =======================================

mkdir -p "$OUTPUT_DIR" "$HS_CACHE_DIR"

# Step 1: Launch live vLLM server on H200 GPU 0 (generates hidden states online in /dev/shm)
echo "=== Step 1: Launching vLLM server on GPU $VLLM_GPUS ($MODEL) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HS_CACHE_DIR" \
    -- --port "$VLLM_PORT" --max-model-len "$SEQ_LENGTH" --gpu-memory-utilization 0.85 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready on port ${VLLM_PORT}..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 2: Build decontaminated multi-domain on-policy JSONL (if DATASET not pre-supplied) & prepare Arrow dataset
if [[ -z "$DATASET" ]]; then
    DATASET="${OUTPUT_DIR}_raw_onpolicy_${MAX_SAMPLES}.jsonl"
    if [[ ! -f "$DATASET" ]]; then
        echo "=== Step 2a: Building 13-gram decontaminated on-policy corpus ($MAX_SAMPLES samples) for $MODEL ==="
        python scripts/build_decontaminated_mix.py \
            --verifier-model "$MODEL" \
            --ultrachat-samples "$ULTRACHAT_SAMPLES" \
            --openmath-samples "$OPENMATH_SAMPLES" \
            --magicoder-samples "$MAGICODER_SAMPLES" \
            --exclude-benchmarks mt_bench gsm8k math500 humaneval mbpp livecodebench \
            --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
            --concurrency 64 \
            --output-jsonl "$DATASET" \
            --val-samples 0 \
            --seed 42
    fi
fi

echo "=== Step 2b: Preparing token dataset ($DATASET) in $OUTPUT_DIR ==="
if [[ "$DATASET" == hf:* ]]; then
    speculators prepare-data \
        --model "$MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH" \
        --no-skip-token-freq
else
    speculators prepare-data \
        --model "$MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH" \
        --no-skip-token-freq \
        --render-endpoint "http://localhost:${VLLM_PORT}"
fi

# Step 3: Train KvN-DSpark v2 on H200 GPU 1
echo "=== Step 3: Training KvN-DSpark v2 (M_train=${KVN_N_MODES_TRAIN}, backbone=${KVN_BACKBONE_TYPE}) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --train-data-ratio "$TRAIN_DATA_RATIO" \
    --hidden-states-path "$HS_CACHE_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --muon-lr "$MUON_LR" \
    --optimizer muon \
    --scheduler-type cosine \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --kvn-backbone-type "$KVN_BACKBONE_TYPE" \
    --kvn-n-modes "$KVN_N_MODES_TRAIN" \
    --kvn-reupload-layers "$KVN_REUPLOAD_LAYERS" \
    --kvn-reupload-dim "$KVN_REUPLOAD_DIM" \
    --kvn-learnable-rff \
    --kvn-anchor-rank "$KVN_ANCHOR_RANK" \
    --kvn-enable-interferometer \
    --kvn-vacuum-born-norm \
    --kvn-multilayer-rollout \
    --kvn-intrablock-mixer \
    --kvn-strict-causal-shift \
    --kvn-lasso-weight "$KVN_LASSO_WEIGHT" \
    --kvn-use-born-confidence \
    --on-missing generate

# Step 4: Exact Dyadic Group-LASSO Mode Pruning (M_train -> M_infer) + Optional 1-Epoch Recalibration
if [[ "$KVN_BACKBONE_TYPE" != "swa" && "$KVN_N_MODES_INFER" -lt "$KVN_N_MODES_TRAIN" ]]; then
    echo "=== Step 4: Pruning dyadic RFF modes ${KVN_N_MODES_TRAIN} -> ${KVN_N_MODES_INFER} ==="
    LAST_EPOCH=$((EPOCHS - 1))
    CKPT_PATH="$OUTPUT_DIR/checkpoints/$LAST_EPOCH"
    PRUNED_PATH="$OUTPUT_DIR/checkpoints/pruned_zeroshot_m${KVN_N_MODES_INFER}"
    python -c "
from pathlib import Path
import numpy as np
import torch
from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.train.checkpointer import convert_float_dtype, patch_config_dtype
t2d = torch.from_numpy(np.load('$OUTPUT_DIR/t2d-${DRAFT_VOCAB_SIZE}.npy'))
d2t = torch.from_numpy(np.load('$OUTPUT_DIR/d2t-${DRAFT_VOCAB_SIZE}.npy'))
model = KvNDSparkDraftModel.from_pretrained('$CKPT_PATH', t2d=t2d, d2t=d2t)
model.prune_kvn_modes($KVN_N_MODES_INFER)
sd = convert_float_dtype(model.state_dict(), torch.bfloat16)
model.save_pretrained('$PRUNED_PATH', state_dict=sd)
patch_config_dtype(Path('$PRUNED_PATH') / 'config.json', torch.bfloat16)
print('Saved Zero-Shot pruned checkpoint to:', '$PRUNED_PATH', 'State bytes/seq:', model.config.estimated_state_bytes_per_seq())
"
    if [[ "$RECAL_EPOCHS" -gt 0 ]]; then
        echo "=== Step 5: Running ${RECAL_EPOCHS}-epoch post-pruning micro-recalibration ==="
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
            --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
            -m speculators.train \
            --verifier-name-or-path "$MODEL" \
            --from-pretrained "$PRUNED_PATH" \
            --data-path "$OUTPUT_DIR" \
            --train-data-ratio "$TRAIN_DATA_RATIO" \
            --hidden-states-path "$HS_CACHE_DIR" \
            --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
            --save-path "$OUTPUT_DIR/checkpoints/pruned_recal_m${KVN_N_MODES_INFER}" \
            --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
            --epochs "$RECAL_EPOCHS" \
            --lr 5e-5 \
            --muon-lr 2e-4 \
            --optimizer muon \
            --scheduler-type cosine \
            --total-seq-len "$SEQ_LENGTH" \
            --speculator-type "$SPECULATOR_TYPE" \
            --block-size "$BLOCK_SIZE" \
            --max-anchors "$MAX_ANCHORS" \
            --target-layer-ids $TARGET_LAYER_IDS \
            --loss-fn "$LOSS_FN" \
            --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
            --kvn-lasso-weight 0.0 \
            --on-missing generate
    fi
fi

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
