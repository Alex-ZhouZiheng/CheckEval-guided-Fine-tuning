#!/usr/bin/env bash
# Launch the frozen judge as an OpenAI-compatible vLLM server.
# Meant to be started on a DIFFERENT GPU than the GRPO generator trainer,
# so the two vLLM engines don't fight over memory.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash src/train/run_judge_vllm_serve.sh
#
# With a judge LoRA adapter:
#   CUDA_VISIBLE_DEVICES=1 JUDGE_ADAPTER_PATH=results/checkpoints/judge_sft_.../final_adapter \
#     bash src/train/run_judge_vllm_serve.sh
#
# Env:
#   PORT               (default 8000)
#   JUDGE_MODEL_PATH   (default cfg.JUDGE_MODEL_ID → models/Qwen3.5-9B)
#   JUDGE_ADAPTER_PATH (optional — registers LoRA module named 'judge')
#   MAX_MODEL_LEN      (default 16384)
#   VLLM_MEM           (default 0.90)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PORT="${PORT:-8000}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-9B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_MEM="${VLLM_MEM:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARGS=(
  --model "${JUDGE_MODEL_PATH}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_MEM}"
  --dtype bfloat16
  --trust-remote-code
)

if [[ -n "${JUDGE_ADAPTER_PATH:-}" ]]; then
  ARGS+=(
    --enable-lora
    --lora-modules "judge=${JUDGE_ADAPTER_PATH}"
    --max-lora-rank 16
  )
  echo "[judge-serve] served model name for client: 'judge' (LoRA adapter)"
else
  echo "[judge-serve] served model name for client: '$(basename "${JUDGE_MODEL_PATH}")'"
fi

exec vllm serve "${ARGS[@]}"
