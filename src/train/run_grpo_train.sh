#!/usr/bin/env bash
# Generator GRPO via ms-swift (Qwen3.5 Dense, full-param).
# Follows the official Qwen3.5 Dense GRPO best-practice recipe:
# gsm8k_accuracy + gsm8k_format rewards, vLLM colocate rollout, DeepSpeed ZeRO-2.
#
# Usage (single GPU, default):
#   bash src/train/run_grpo_train.sh
#
# Usage (4 GPU, matches the official recipe):
#   NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash src/train/run_grpo_train.sh
#
# Env overrides (all optional):
#   MODEL=/path/to/weights-or-hf-id
#   TAG=exp1
#   LR=1e-6 EPOCHS=1 BATCH_SIZE=4 GRAD_ACCUM=4
#   NUM_GENERATIONS=8 MAX_COMPLETION_LEN=8192 MAX_LEN=2048
#   VLLM_MEM=0.4 VLLM_MAX_LEN=10240

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${MODEL:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
TAG="${TAG:-gsm8k}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_LEN="${MAX_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-8192}"
VLLM_MEM="${VLLM_MEM:-0.4}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-10240}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
RESUME="${RESUME:-}"

MODEL_TAG="$(basename "${MODEL}")"
RUN_NAME="grpo_${MODEL_TAG}_${TAG}_lr${LR}"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
PLUGIN_PATH="${PROJECT_ROOT}/src/train/plugin/gsm8k_plugin.py"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.'

RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" == "auto" ]]; then
        LATEST_CKPT="$(ls -1dt "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -n 1 || true)"
        if [[ -z "${LATEST_CKPT}" ]]; then
            echo "[resume] no checkpoint found under ${OUTPUT_DIR}, starting fresh" >&2
        else
            echo "[resume] resuming from ${LATEST_CKPT}" >&2
            RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CKPT}")
        fi
    else
        echo "[resume] resuming from ${RESUME}" >&2
        RESUME_ARGS=(--resume_from_checkpoint "${RESUME}")
    fi
fi

swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --columns '{"answer": "solution"}' \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_MEM}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${VLLM_MAX_LEN}" \
    --sleep_level 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset 'modelscope/gsm8k' \
    --load_from_cache_file true \
    --max_length "${MAX_LEN}" \
    --max_completion_length "${MAX_COMPLETION_LEN}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --lr_scheduler_type cosine \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --system "${SYSTEM_PROMPT}" \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --output_dir "${OUTPUT_DIR}" \
    "${RESUME_ARGS[@]}"
