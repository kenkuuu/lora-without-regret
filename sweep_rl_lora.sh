#!/bin/bash
# ============================================================
# rl_lora sweep: lora_r × lr grid search
#
# Usage:
#   bash sweep_rl_lora.sh
#
# Starts a fresh vLLM server for each run, then kills it.
# Logs are written to logs/vllm_r<R>_lr<LR>.log and
#                          logs/train_r<R>_lr<LR>.log
# ============================================================
set -uo pipefail

# ---- Sweep parameters (edit here) -------------------------
LR_VALUES=(1e-5 3e-5 1e-4 3e-4)
LORA_R_VALUES=(1 4 16 64)

# ---- GPU configuration ------------------------------------
# For single-GPU: TRAIN_GPUS=0, NUM_TRAIN_GPUS=1, VLLM_GPU=1
# For multi-GPU:  TRAIN_GPUS=0,1,2, NUM_TRAIN_GPUS=3, VLLM_GPU=3
TRAIN_GPUS=0
NUM_TRAIN_GPUS=1
VLLM_GPU=1

# ---- Model & training config ------------------------------
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_URL="http://localhost:8000"
VLLM_PORT=8000
VLLM_GPU_MEM_UTIL=0.5   # Leave room for training on same node
N_GRPO_STEPS=50
GROUP_SIZE=8
N_PROMPTS_PER_STEP=32
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=128
BASE_DIR="runs_sweep"
WANDB_PROJECT="math-grpo-sweep"
# Set to "--disable_wandb" to turn off W&B logging
WANDB_FLAG=""

# ============================================================

VLLM_PID=""
mkdir -p logs

# Kill vLLM on exit/interrupt
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[sweep] Killing vLLM (PID=$VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap cleanup EXIT INT TERM

wait_for_vllm() {
    local max_wait=300
    local elapsed=0
    echo -n "[sweep] Waiting for vLLM..."
    while ! curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo " TIMEOUT (${max_wait}s)"
            return 1
        fi
        echo -n "."
    done
    echo " ready"
}

start_vllm() {
    local log="$1"
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
        uv run vllm serve "$MODEL_ID" \
            --enable-lora \
            --port "$VLLM_PORT" \
            --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
            > "$log" 2>&1 &
    VLLM_PID=$!
    echo "[sweep] vLLM started (PID=$VLLM_PID, log=$log)"
    if ! wait_for_vllm; then
        echo "[sweep] ERROR: vLLM failed to start. Check $log"
        return 1
    fi
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        sleep 3  # wait for port release
    fi
}

# ---- Print sweep plan -------------------------------------
total_runs=$(( ${#LORA_R_VALUES[@]} * ${#LR_VALUES[@]} ))
echo "========================================"
echo "  rl_lora sweep"
echo "  model:    $MODEL_ID"
echo "  lora_r:   ${LORA_R_VALUES[*]}"
echo "  lr:       ${LR_VALUES[*]}"
echo "  total:    $total_runs runs"
echo "  train GPU(s): $TRAIN_GPUS (x$NUM_TRAIN_GPUS)"
echo "  vLLM GPU: $VLLM_GPU"
echo "========================================"
echo ""

# ---- Main sweep loop --------------------------------------
run_count=0

for lora_r in "${LORA_R_VALUES[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        run_count=$(( run_count + 1 ))
        tag="r${lora_r}_lr${lr}"

        echo ""
        echo "--- Run $run_count/$total_runs: lora_r=$lora_r  lr=$lr ---"

        start_vllm "logs/vllm_${tag}.log"

        # Build train command
        if [ "$NUM_TRAIN_GPUS" -eq 1 ]; then
            TRAIN_LAUNCHER=(uv run rl_lora.py)
        else
            TRAIN_LAUNCHER=(
                uv run accelerate launch
                --num_processes "$NUM_TRAIN_GPUS"
                rl_lora.py
            )
        fi

        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
            "${TRAIN_LAUNCHER[@]}" \
                --model_id         "$MODEL_ID" \
                --lr               "$lr" \
                --lora_r           "$lora_r" \
                --vllm_url         "$VLLM_URL" \
                --n_grpo_steps     "$N_GRPO_STEPS" \
                --group_size       "$GROUP_SIZE" \
                --n_prompts_per_step "$N_PROMPTS_PER_STEP" \
                --micro_batch_size "$MICRO_BATCH_SIZE" \
                --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
                --base_dir         "$BASE_DIR" \
                --wandb_project    "$WANDB_PROJECT" \
                --wandb_run_name   "$tag" \
                $WANDB_FLAG \
                2>&1 | tee "logs/train_${tag}.log"
        train_exit=${PIPESTATUS[0]}

        stop_vllm

        if [ "$train_exit" -ne 0 ]; then
            echo "[sweep] WARNING: run $tag exited with code $train_exit"
        else
            echo "[sweep] Run $tag completed successfully."
        fi
    done
done

echo ""
echo "========================================"
echo "  Sweep done: $run_count/$total_runs runs"
echo "========================================"
