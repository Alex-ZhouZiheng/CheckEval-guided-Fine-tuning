#!/usr/bin/env bash
# GRPO fine-tune the self-checklist judge with Unsloth + TRL.
#
# This is the Qwen3.5 path when vLLM rollout is unsupported or unstable:
# FastLanguageModel.from_pretrained(..., fast_inference=False).
#
# Prereq:
#   python -m src.data_process.prepare_judge_grpo --tier tier_10k
#
# Usage:
#   bash src/train/run_judge_grpo_unsloth.sh
#
# Common env overrides:
#   MODEL_NAME=Qwen3.5-4B
#   MODEL_PATH=/path/to/model_or_merged_checkpoint
#   TIER=tier_10k
#   LOAD_IN_4BIT=false
#   FAST_INFERENCE=false
#   NUM_GENERATIONS=4
#   MAX_SEQ_LENGTH=10240
#   MAX_PROMPT_LENGTH=4096
#   MAX_COMPLETION_LENGTH=6144
#   LR=5e-6
#   EPOCHS=1
#   BATCH_SIZE=1
#   GRAD_ACCUM=16
#   REWARD_FUNCS="winner format"
#   SAVE_MERGED_16BIT=false
#   EVAL_SPLIT=dev_600 EVAL_MAX_SAMPLES=200
#   EVAL_STEPS=0              # set >0 for in-training self-check eval
#   NO_FINAL_EVAL=false

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/${MODEL_NAME}}"
TIER="${TIER:-tier_10k}"
TAG="${TAG:-unsloth_grpo}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/judge_sft/grpo_${TIER}_selfcheck.jsonl}"
if [[ ! -f "${DATASET_PATH}" && -f "${PROJECT_ROOT}/data/judge_sft/grpo_train_${TIER}_selfcheck.jsonl" ]]; then
  DATASET_PATH="${PROJECT_ROOT}/data/judge_sft/grpo_train_${TIER}_selfcheck.jsonl"
fi

LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
FAST_INFERENCE="${FAST_INFERENCE:-false}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-10240}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-6144}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BETA="${BETA:-0.04}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REWARD_FUNCS="${REWARD_FUNCS:-winner}"
REPORT_TO="${REPORT_TO:-tensorboard}"
SAVE_MERGED_16BIT="${SAVE_MERGED_16BIT:-false}"
NO_FINAL_EVAL="${NO_FINAL_EVAL:-false}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-dev_600}"
EVAL_SUBSET="${EVAL_SUBSET:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-true}"

[[ -f "${DATASET_PATH}" ]] || {
  echo "[unsloth-grpo-judge] dataset not found: ${DATASET_PATH}" >&2
  echo "  build it: python -m src.data_process.prepare_judge_grpo --tier ${TIER}" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

FLAGS=()
if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  FLAGS+=(--load-in-4bit)
fi
if [[ "${FAST_INFERENCE}" == "true" ]]; then
  FLAGS+=(--fast-inference)
fi
if [[ "${SAVE_MERGED_16BIT}" == "true" ]]; then
  FLAGS+=(--save-merged-16bit)
fi
if [[ "${NO_FINAL_EVAL}" == "true" ]]; then
  FLAGS+=(--no-final-eval)
fi
if [[ "${EVAL_BEFORE_TRAIN}" == "true" ]]; then
  FLAGS+=(--eval-before-train)
fi
if [[ "${EVAL_ENABLE_THINKING}" == "true" ]]; then
  FLAGS+=(--eval-enable-thinking)
else
  FLAGS+=(--eval-no-thinking)
fi
if [[ -n "${EVAL_SUBSET}" ]]; then
  FLAGS+=(--eval-subset "${EVAL_SUBSET}")
fi

# shellcheck disable=SC2206
REWARD_FUNC_ARGS=(${REWARD_FUNCS})
# shellcheck disable=SC2206
REPORT_TO_ARGS=(${REPORT_TO})

python -m src.train.run_judge_grpo_unsloth \
  --tier "${TIER}" \
  --model-name "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --tag "${TAG}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --learning-rate "${LR}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --num-generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --beta "${BETA}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --reward-funcs "${REWARD_FUNC_ARGS[@]}" \
  --report-to "${REPORT_TO_ARGS[@]}" \
  --eval-steps "${EVAL_STEPS}" \
  --eval-split "${EVAL_SPLIT}" \
  --eval-max-samples "${EVAL_MAX_SAMPLES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --eval-temperature "${EVAL_TEMPERATURE}" \
  "${FLAGS[@]}"
