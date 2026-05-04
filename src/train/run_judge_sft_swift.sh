#!/usr/bin/env bash
# Usage:
#   bash src/train/run_judge_sft_swift.sh debug_5k_teacher
#   MODEL_NAME=Qwen3.5-4B QUANT_BITS=0 MAX_LEN=6144 \
#       bash src/train/run_judge_sft_swift.sh train_debug_5k_selfcheck
#
# Env overrides (all optional):
#   MODEL_NAME=Qwen3.5-9B|Qwen3.5-4B   (default: Qwen3.5-9B)
#   QUANT_BITS=4|0                     (4 = QLoRA NF4, 0 = BF16 LoRA. default: 4)
#   MAX_LEN=3072 LORA_RANK=16 LR=2e-5 EPOCHS=3 GRAD_ACCUM=16
#   ENABLE_THINKING=true|false         (Qwen3 native think mode. default: false)

set -euo pipefail

TIER="${1:-debug_5k_teacher}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
QUANT_BITS="${QUANT_BITS:-4}"
MAX_LEN="${MAX_LEN:-3072}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"
PARQUET_PATH="${PROJECT_ROOT}/data/judge_sft/train_${TIER}.parquet"
JSONL_PATH="${PROJECT_ROOT}/data/judge_sft/train_${TIER}.messages.jsonl"
RUN_NAME="judge_sft_swift_${MODEL_NAME}_${TIER}_r${LORA_RANK}_lr${LR}"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"

# ── 1. Convert parquet → JSONL (swift 'messages' format) ──
if [[ ! -f "$JSONL_PATH" || "$PARQUET_PATH" -nt "$JSONL_PATH" ]]; then
  echo "[swift-sft] Converting parquet → jsonl …"
  python - <<PY
import json, pandas as pd
df = pd.read_parquet("${PARQUET_PATH}")
with open("${JSONL_PATH}", "w", encoding="utf-8") as f:
    for _, r in df.iterrows():
        msgs = r["messages"]
        msgs = json.loads(msgs) if isinstance(msgs, str) else list(msgs)
        msgs = list(msgs) + [{"role": "assistant", "content": r["target_output"]}]
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
print(f"[swift-sft] wrote {len(df)} rows → ${JSONL_PATH}")
PY
fi

# ── 2. Launch swift sft ──
# Key flags solving the earlier OOMs:
#   --packing true                 : concat short samples → fewer seq-len buckets, autotune cache hit
#   --lazy_tokenize true           : stream-tokenize, avoid peak RAM
#   --use_liger_kernel true        : fused RoPE/RMSNorm/SwiGLU + FLCE
#   --quant_bits 4 --quant_method bnb : QLoRA (4-bit NF4)
#   --gradient_checkpointing true  : trade compute for mem
#   --attn_impl flash_attn
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# QLoRA flags: only injected when QUANT_BITS != 0. Qwen3.5 4B in BF16 fits a
# 32GB 5090 at MAX_LEN=6144; QLoRA adds quant noise unsloth flagged as worse
# than usual on this arch.
QUANT_FLAGS=()
if [[ "${QUANT_BITS}" != "0" ]]; then
  QUANT_FLAGS=(
    --quant_bits "${QUANT_BITS}"
    --quant_method bnb
    --bnb_4bit_compute_dtype bfloat16
    --bnb_4bit_quant_type nf4
  )
fi

# enable_thinking forwards to Qwen3 chat template. Self-checklist data has
# <think>...</think> already in target_output → must NOT add another prefix
# (--add_non_thinking_prefix false), otherwise we get nested <think> blocks.
THINK_FLAGS=()
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  THINK_FLAGS=(
    --template_kwargs '{"enable_thinking": true}'
    --add_non_thinking_prefix false
  )
else
  THINK_FLAGS=(
    --add_non_thinking_prefix true
  )
fi

swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${JSONL_PATH}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  "${QUANT_FLAGS[@]}" \
  --torch_dtype bfloat16 \
  --max_length "${MAX_LEN}" \
  --truncation_strategy left \
  --group_by_length true \
  "${THINK_FLAGS[@]}" \
  --loss_scale ignore_empty_think \
  --use_liger_kernel true \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --optim paged_adamw_8bit \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --load_from_cache_file true \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --seed 42
