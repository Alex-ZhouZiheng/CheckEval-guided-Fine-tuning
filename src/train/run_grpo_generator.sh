#!/usr/bin/env bash
# GRPO fine-tune the checklist generator using a frozen judge as the reward.
#
# Prereqs:
#   1. JSONL dataset exists:
#        python -m src.data_process.prepare_grpo_pairwise --tier tier_10k
#   2. Judge deployment:
#        JUDGE_MODE=http (default, required for pipeline_eval):
#          CUDA_VISIBLE_DEVICES=1 bash src/train/run_judge_vllm_serve.sh
#        JUDGE_MODE=hf (same card, slow, smoke-test only, no pipeline_eval):
#          export JUDGE_MODE=hf JUDGE_MODEL_PATH=... JUDGE_ADAPTER_PATH=...
#
# Expected GPU topology (single-node, 2 cards):
#   GPU 0: training (HF) + colocate rollout vLLM. During pipeline_eval the
#          rollout engine sleeps (level>=1) and GPU 0 hosts the eval
#          generator vLLM.
#   GPU 1: resident judge vLLM server. Both training-time reward and
#          pipeline_eval reach it over HTTP via JUDGE_URL.
#
# Usage:
#   bash src/train/run_grpo_generator.sh
#
# Env overrides:
#   MODEL             generator init (default: generator SFT checkpoint)
#   TIER              data tier for JSONL (default tier_10k)
#   TAG               run-name suffix (default judge_rl)
#   JUDGE_MODE        http | hf
#   JUDGE_URL         http://127.0.0.1:8000/v1
#   JUDGE_MODEL       model name registered at the vLLM server
#   LR, EPOCHS, BATCH_SIZE, GRAD_ACCUM, NUM_GENERATIONS,
#   MAX_LEN, MAX_COMPLETION_LEN, VLLM_MEM, VLLM_MAX_LEN, SAVE_STEPS

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default: continue from the latest generator SFT final_adapter-merged checkpoint.
MODEL="${MODEL:-${PROJECT_ROOT}/results/checkpoints/generator_sft_debug_5k_r16_lr2e-05/final_merged}"

TIER="${TIER:-tier_10k}"
TAG="${TAG:-judge_rl}"
DATASET_PATH="${PROJECT_ROOT}/data/generator_sft/grpo_${TIER}.jsonl"
PLUGIN_PATH="${PROJECT_ROOT}/src/train/plugin/checkeval_reward.py"

LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1024}"
VLLM_MEM="${VLLM_MEM:-0.4}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-50}"

# Reward-side env consumed by checkeval_reward.py at runtime.
export JUDGE_MODE="${JUDGE_MODE:-http}"
export JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8000/v1}"
if [[ "${JUDGE_MODE}" == "http" && -z "${JUDGE_MODEL:-}" ]]; then
  echo "[grpo] JUDGE_MODE=http requires JUDGE_MODEL env var (model name at the vLLM server)." >&2
  exit 1
fi
export JUDGE_MODEL="${JUDGE_MODEL:-}"
export JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
export CHECKEVAL_NA_POLICY="${CHECKEVAL_NA_POLICY:-as_no}"
export CHECKEVAL_COVERAGE_THRESHOLD="${CHECKEVAL_COVERAGE_THRESHOLD:-0.8}"
export CHECKEVAL_TIE_DELTA="${CHECKEVAL_TIE_DELTA:-0.05}"
export CHECKEVAL_DIR_SCALE="${CHECKEVAL_DIR_SCALE:-0.2}"
export CHECKEVAL_MARGIN_WEIGHT="${CHECKEVAL_MARGIN_WEIGHT:-0.2}"
export CHECKEVAL_MARGIN_CAP="${CHECKEVAL_MARGIN_CAP:-0.2}"
export CHECKEVAL_COV_PEN_WEIGHT="${CHECKEVAL_COV_PEN_WEIGHT:-0.3}"

# Periodic pipeline-eval (dev_600) via PipelineEvalCallback.
export PIPELINE_EVAL_STEPS="${PIPELINE_EVAL_STEPS:-100}"
export PIPELINE_EVAL_SUBSET="${PIPELINE_EVAL_SUBSET:-dev_600}"
export PIPELINE_EVAL_SPLIT="${PIPELINE_EVAL_SPLIT:-dev}"
export PIPELINE_EVAL_BATCH_SIZE="${PIPELINE_EVAL_BATCH_SIZE:-16}"
export PIPELINE_EVAL_TIE_DELTA="${PIPELINE_EVAL_TIE_DELTA:-0.05}"
export PIPELINE_EVAL_CUDA_DEVICES="${PIPELINE_EVAL_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export PIPELINE_EVAL_JUDGE_ADAPTER="${PIPELINE_EVAL_JUDGE_ADAPTER:-}"

[[ -f "${DATASET_PATH}" ]] || {
  echo "[grpo] dataset not found: ${DATASET_PATH}" >&2
  echo "       build it with: python -m src.data_process.prepare_grpo_pairwise --tier ${TIER}" >&2
  exit 1
}

MODEL_TAG="$(basename "${MODEL}")"
RUN_NAME="grpo_${MODEL_TAG}_${TIER}_${TAG}_lr${LR}"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs checkeval_pairwise_r1 \
    --reward_weights 1.0 \
    --callbacks pipeline_eval \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_MEM}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${VLLM_MAX_LEN}" \
    --sleep_level 1 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_PATH}" \
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
    --warmup_steps 0.03 \
    --dataloader_num_workers 2 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard wandb \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --output_dir "${OUTPUT_DIR}"
