#!/usr/bin/env bash
# ==============================================================================
# KvN-DSpark Zero-KV-Cache Multi-Token Prediction (MTP) Online Training Script
# Hardware Target: 2x NVIDIA H200 (2 x 141 GB HBM3e = 282 GB VRAM)
#   - GPU 0 (H200 #1): vLLM Server generating auxiliary hidden states on-the-fly
#   - GPU 1 (H200 #2): torchrun speculators.train (--speculator-type kvn_dspark)
#
# Base Verifier LLMs supported out-of-the-box on 2xH200:
#   - Standard Speculators Benchmark: MODEL="Qwen/Qwen3-8B"   (TARGET_LAYER_IDS="2 18 33")
#   - High-Capacity H200 Scale-Up:    MODEL="Qwen/Qwen3-32B"  (TARGET_LAYER_IDS="2 32 61")
# ==============================================================================

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"
DATASET="${DATASET:-./ultrachat_qwen3.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/kvn_dspark_qwen3}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"

# KvN-DSpark Architecture & Barthe LASSO Scaling (M_train -> M_infer)
SPECULATOR_TYPE="kvn_dspark"
KVN_BACKBONE_TYPE="${KVN_BACKBONE_TYPE:-kvn}"   # kvn (Zero-KV) | hybrid_1swa_2kvn | swa
KVN_N_MODES_TRAIN="${KVN_N_MODES_TRAIN:-512}"   # Over-parameterized RFF modes during training
KVN_N_MODES_INFER="${KVN_N_MODES_INFER:-128}"   # Pruned active RFF modes for HBM Roofline inference
KVN_REUPLOAD_LAYERS="${KVN_REUPLOAD_LAYERS:-2}"
KVN_LASSO_WEIGHT="${KVN_LASSO_WEIGHT:-1e-4}"

BLOCK_SIZE=8
MAX_ANCHORS=1024
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
MARKOV_RANK=256
MARKOV_HEAD_TYPE="gated"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# 2xH200 GPU Assignment
VLLM_GPUS="0"
TRAIN_GPUS="1"
NUM_TRAIN_GPUS=1
# =======================================

# Step 0: Regenerate target responses if dataset JSONL does not exist yet
if [[ ! -f "$DATASET" ]]; then
    echo "=== Step 0: Regenerating target responses with vLLM on GPU $VLLM_GPUS ==="
    bash scripts/response_regeneration/run_all.sh \
        --model "$MODEL" --dataset ultrachat \
        --gpus "$VLLM_GPUS" --port "$VLLM_PORT" \
        --limit "$MAX_SAMPLES" --max-tokens "$SEQ_LENGTH" \
        --outfile "$DATASET" --resume
fi

# Step 1: Launch vLLM server on H200 GPU 0
echo "=== Step 1: Launching vLLM server on GPU $VLLM_GPUS ($MODEL) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" --gpu-memory-utilization 0.85 &
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

# Step 2: Prepare data & token frequency distribution
echo "=== Step 2: Preparing dataset in $OUTPUT_DIR ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --no-skip-token-freq \
    --render-endpoint "http://localhost:${VLLM_PORT}"

# Step 3: Train KvN-DSpark on H200 GPU 1
echo "=== Step 3: Training KvN-DSpark (M_train=${KVN_N_MODES_TRAIN}, backbone=${KVN_BACKBONE_TYPE}) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
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
    --kvn-lasso-weight "$KVN_LASSO_WEIGHT" \
    --kvn-use-born-confidence \
    --on-missing generate

# Step 4: Structural Barthe Theorem 5.6 LASSO Mode Pruning (M_train -> M_infer)
if [[ "$KVN_BACKBONE_TYPE" != "swa" && "$KVN_N_MODES_INFER" -lt "$KVN_N_MODES_TRAIN" ]]; then
    echo "=== Step 4: Pruning RFF modes ${KVN_N_MODES_TRAIN} -> ${KVN_N_MODES_INFER} for HBM Roofline Inference ==="
    LAST_EPOCH=$((EPOCHS - 1))
    CKPT_PATH="$OUTPUT_DIR/checkpoints/$LAST_EPOCH"
    PRUNED_PATH="$OUTPUT_DIR/checkpoints/pruned_m${KVN_N_MODES_INFER}"
    python -c "
import torch
from speculators.models.kvn_dspark import KvNDSparkDraftModel
model = KvNDSparkDraftModel.from_pretrained('$CKPT_PATH')
model.prune_kvn_modes($KVN_N_MODES_INFER)
model.save_pretrained('$PRUNED_PATH')
print('Saved pruned Zero-KV checkpoint to:', '$PRUNED_PATH', 'State bytes/seq:', model.config.estimated_state_bytes_per_seq())
"
fi

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
