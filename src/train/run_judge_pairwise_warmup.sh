#!/usr/bin/env bash
# 2-GPU DDP warm-up run. Adjust --tier / --resume-adapter as needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

NPROC=${NPROC:-2}
TIER=${TIER:-tier_10k}
RESUME=${RESUME:-}

EXTRA=()
if [[ -n "$RESUME" ]]; then
  EXTRA+=(--resume-adapter "$RESUME")
fi

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

torchrun --standalone --nproc_per_node="$NPROC" \
  src/train/run_judge_pairwise_warmup.py \
  --tier "$TIER" \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 5e-6 \
  --eval-every 100 \
  "${EXTRA[@]}" \
  "$@"
