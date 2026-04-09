#!/bin/bash
# lora_r × lr のグリッドサーチ
# vLLMサーバは起動済みを前提とする

LR_VALUES=(1e-5 3e-5 1e-4 3e-4)
LORA_R_VALUES=(1 4 16 64)

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"

for lora_r in "${LORA_R_VALUES[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        echo "=== lora_r=$lora_r  lr=$lr ==="
        CUDA_VISIBLE_DEVICES=0 uv run rl_lora.py \
            --model_id  "$MODEL_ID" \
            --lora_r    "$lora_r"   \
            --lr        "$lr"       \
            --wandb_run_name "r${lora_r}_lr${lr}"
    done
done
