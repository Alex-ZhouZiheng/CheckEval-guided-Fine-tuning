#!/usr/bin/env bash
# GRPO fine-tune the self-checklist judge with verifier rewards.
#
# Reward = correctness of parsed "Winner: X" against gold winner from
# HelpSteer3 + format signal (think tags, checklist/verdict counts).
# No external judge model required (verifier is a pure parse + compare).
#
# Prereqs:
#   1. JSONL dataset (build first):
#        python -m src.data_process.prepare_judge_grpo --tier tier_10k
#   2. Optional: warm-start from SFT adapter via RESUME_ADAPTER (--adapters).
#
# Single GPU (5090, 32GB) example:
#   bash src/train/run_judge_grpo_swift.sh
#
# Common env overrides:
#   MODEL_NAME      Qwen3.5-4B (default) | Qwen3.5-9B
#   TIER            tier_10k (default) | debug_5k | train_debug_5k_selfcheck
#   QUANT_BITS      0 (BF16 LoRA, default) | 4 (QLoRA NF4)
#   RESUME_ADAPTER  path to SFT final adapter to warm-start
#   NUM_GENERATIONS 4 (default)  -- bigger = better signal, more memory
#   MAX_LEN         6144   -- prompt + completion budget
#   MAX_COMPLETION_LEN 4096
#   LR              5e-6
#   EPOCHS          1
#   BATCH_SIZE      1   per_device_train_batch_size
#   GRAD_ACCUM      16
#   BETA            0.04  KL coefficient
#   TEMPERATURE     1.0   rollout sampling
#   REWARD_FUNCS    "judge_selfcheck_winner judge_selfcheck_format"
#   REWARD_WEIGHTS  "0.85 0.15"
#   ENABLE_THINKING true (default) | false

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/${MODEL_NAME}}"
TIER="${TIER:-tier_10k}"
TAG="${TAG:-grpo}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/judge_sft/grpo_${TIER}_selfcheck.jsonl}"
PLUGIN_PATH="${PROJECT_ROOT}/src/train/plugin/judge_selfcheck_reward.py"
EVAL_PLUGIN_PATH="${PROJECT_ROOT}/src/train/plugin/judge_selfcheck_eval_callback.py"

QUANT_BITS="${QUANT_BITS:-0}"
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_LEN="${MAX_LEN:-9126}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-4096}"
BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

REWARD_FUNCS="${REWARD_FUNCS:-judge_selfcheck_winner}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0}"

# In-training self-checklist eval (callback plugin). Set EVAL_STEPS=0 to disable.
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-dev_600}"
EVAL_SUBSET="${EVAL_SUBSET:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4048}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-${ENABLE_THINKING}}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"
EVAL_LABEL_PREFIX="${EVAL_LABEL_PREFIX:-swift_grpo}"

export SELFCHECK_EVAL_STEPS="${EVAL_STEPS}"
export SELFCHECK_EVAL_SPLIT="${EVAL_SPLIT}"
export SELFCHECK_EVAL_SUBSET="${EVAL_SUBSET}"
export SELFCHECK_EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES}"
export SELFCHECK_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}"
export SELFCHECK_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}"
export SELFCHECK_EVAL_TEMPERATURE="${EVAL_TEMPERATURE}"
export SELFCHECK_EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING}"
export SELFCHECK_EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN}"
export SELFCHECK_EVAL_LABEL_PREFIX="${EVAL_LABEL_PREFIX}"

EXTERNAL_PLUGINS=("${PLUGIN_PATH}")
if [[ "${EVAL_STEPS}" != "0" || "${EVAL_BEFORE_TRAIN}" == "true" ]]; then
  EXTERNAL_PLUGINS+=("${EVAL_PLUGIN_PATH}")
fi

# vLLM colocate rollout (frees memory during gradient step via sleep_level).
USE_VLLM="${USE_VLLM:-true}"
VLLM_MEM="${VLLM_MEM:-0.35}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-${MAX_LEN}}"

[[ -f "${DATASET_PATH}" ]] || {
  PREPARE_THINKING_FLAG="--enable-thinking"
  if [[ "${ENABLE_THINKING}" != "true" ]]; then
    PREPARE_THINKING_FLAG="--no-thinking"
  fi
  echo "[grpo-judge] dataset not found: ${DATASET_PATH}" >&2
  echo "  build it: python -m src.data_process.prepare_judge_grpo --tier ${TIER} ${PREPARE_THINKING_FLAG}" >&2
  exit 1
}

RUN_NAME="judge_grpo_${MODEL_NAME}_${TIER}_${TAG}_lr${LR}_b${BETA}"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

QUANT_FLAGS=()
if [[ "${QUANT_BITS}" != "0" ]]; then
  QUANT_FLAGS=(
    --quant_bits "${QUANT_BITS}"
    --quant_method bnb
    --bnb_4bit_compute_dtype bfloat16
    --bnb_4bit_quant_type nf4
  )
fi

RESUME_FLAGS=()
if [[ -n "${RESUME_ADAPTER:-}" ]]; then
  RESUME_FLAGS=(--adapters "${RESUME_ADAPTER}")
fi

VLLM_FLAGS=()
if [[ "${USE_VLLM}" == "true" ]]; then
  VLLM_FLAGS=(
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_MEM}"
    --vllm_tensor_parallel_size 1
    --vllm_max_model_len "${VLLM_MAX_LEN}"
    --sleep_level 1
  )
fi

# Native Qwen3 thinking mode toggle. Same logic as run_judge_sft_swift.sh.
THINK_FLAGS=()
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  THINK_FLAGS=(--add_non_thinking_prefix false)
else
  THINK_FLAGS=(--add_non_thinking_prefix true)
fi

# shellcheck disable=SC2086
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --external_plugins "${EXTERNAL_PLUGINS[@]}" \
    --reward_funcs ${REWARD_FUNCS} \
    --reward_weights ${REWARD_WEIGHTS} \
    "${VLLM_FLAGS[@]}" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    "${QUANT_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_PATH}" \
    --load_from_cache_file true \
    --max_length "${MAX_LEN}" \
    --max_completion_length "${MAX_COMPLETION_LEN}" \
    --truncation_strategy left \
    "${THINK_FLAGS[@]}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --lr_scheduler_type cosine \
    --warmup_steps "${WARMUP_RATIO}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --beta "${BETA}" \
    --use_liger_kernel false \
    --attn_impl flash_attn \
    --gradient_checkpointing false \
    --optim adamw_torch_fused \
    --log_completions true \
    --report_to tensorboard wandb\
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}"\
