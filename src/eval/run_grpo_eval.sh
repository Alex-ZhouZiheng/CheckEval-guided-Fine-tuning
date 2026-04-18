#!/usr/bin/env bash
# Evaluate a GRPO checkpoint on GSM8K via swift eval (vLLM infer backend).
#
# Usage:
#   bash src/eval/run_grpo_eval.sh <checkpoint_path>
#
# Example:
#   bash src/eval/run_grpo_eval.sh checkpoints/grpo_Qwen3.5-4B_gsm8k_lr1e-6/v0-xxxxxx/checkpoint-50

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash $0 <checkpoint_path>" >&2
  exit 1
fi

MODEL_PATH="$1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

swift eval \
    --model "${MODEL_PATH}" \
    --enable_thinking false \
    --eval_dataset gsm8k \
    --eval_backend Native --infer_backend vllm \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
