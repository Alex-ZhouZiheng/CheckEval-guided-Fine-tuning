#!/usr/bin/env bash
# Judge SFT via ms-swift (Qwen3.5-9B + QLoRA).
# Handles the Triton-autotune / long-seq OOMs that TRL couldn't, by using
# swift's packing + length-sorted sampler + Liger integration.
#
# Usage:
#   bash src/train/run_judge_sft_swift.sh debug_5k_teacher
#
# Env overrides (all optional):
#   MAX_LEN=3072 LORA_RANK=16 LR=2e-5 EPOCHS=3 GRAD_ACCUM=16 \
#   bash src/train/run_judge_sft_swift.sh debug_5k_teacher

set -euo pipefail

TIER="${1:-debug_5k_teacher}"
MAX_LEN="${MAX_LEN:-3072}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen3.5-9B"
PARQUET_PATH="${PROJECT_ROOT}/data/judge_sft/train_${TIER}.parquet"
JSONL_PATH="${PROJECT_ROOT}/data/judge_sft/train_${TIER}.messages.jsonl"
RUN_NAME="judge_sft_swift_${TIER}_r${LORA_RANK}_lr${LR}"
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

swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_5 \
  --dataset "${JSONL_PATH}" \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --quant_bits 4 \
  --quant_method bnb \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 \
  --torch_dtype bfloat16 \
  --max_length "${MAX_LEN}" \
  --truncation_strategy left \
  --packing true \
  --lazy_tokenize true \
  --use_liger_kernel true \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --optim paged_adamw_8bit \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --seed 42
