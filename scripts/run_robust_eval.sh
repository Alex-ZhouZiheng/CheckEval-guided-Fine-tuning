#!/usr/bin/env bash
# Run 4 judges (vanilla, finetuned-vanilla, pipeline-cmp, selfcheck-265-clean)
# across 4 splits (dev_600 + 3 perturbed). 5090-tuned vLLM params.
#
# Outputs prediction-parquet path index to results/robustness/preds_index.tsv
# Format: judge_tag<TAB>split<TAB>predictions.parquet<TAB>metrics.json

set -euo pipefail
cd /root/autodl-tmp/Thesis

PY=.venvmerge/bin/python
BASE9=/root/autodl-tmp/Thesis/models/Qwen3.5-9B
BASE4=/root/autodl-tmp/Thesis/models/Qwen3.5-4B

ADAPTER_FT_VANILLA=/root/autodl-tmp/Thesis/results/checkpoints/judge_warmup_tier_5k_r16_lr5e-06/final_adapter
ADAPTER_SELFCHECK_265=/root/autodl-tmp/Thesis/checkpoints/judge_sft_swift_Qwen3.5-4B_train_debug_5k_selfcheck_clean_r16_lr2e-5/v0-20260506-200134/checkpoint-265

SPLITS=(dev_600 dev_600_swap dev_600_verbose dev_600_format)
if [[ -n "${ROBUST_SPLITS:-}" ]]; then
  read -ra SPLITS <<< "$ROBUST_SPLITS"
fi

mkdir -p results/robustness logs/robust
INDEX=results/robustness/preds_index.tsv
: > "$INDEX"

stage() { echo "===== $* $(date -Is) ====="; }

run_vanilla() {
  local split=$1
  local sfx="robust_${split}"
  stage "vanilla $split"
  $PY src/evaluation/run_zeroshot.py \
      --split "$split" --suffix "$sfx" --backend vllm --model-id "$BASE9" \
      --batch-size 16 --max-model-len 8192 --gpu-memory-utilization 0.90
  local pred=$(ls -t results/vanilla_judge_${split}_${sfx}_predictions.parquet 2>/dev/null | head -1)
  local met=${pred%_predictions.parquet}_metrics.json
  echo -e "vanilla\t${split}\t${pred}\t${met}" >> "$INDEX"
}

run_ft_vanilla() {
  local split=$1
  stage "ft_vanilla $split"
  $PY src/evaluation/run_eval_finetuned.py \
      --adapter-path "$ADAPTER_FT_VANILLA" \
      --eval-split "$split" --eval-mode vanilla --base-model "$BASE9" \
      --backend vllm --batch-size 16 --max-model-len 8192 --gpu-memory-utilization 0.90
  local pred=$(ls -t results/finetuned_vanilla_final_adapter_${split}_*_predictions.parquet 2>/dev/null | head -1)
  local met=${pred%_predictions.parquet}_metrics.json
  echo -e "ft_vanilla\t${split}\t${pred}\t${met}" >> "$INDEX"
}

run_pipeline_cmp() {
  local split=$1
  local sfx="robust_${split}"
  stage "pipeline_cmp $split"
  env CHECKEVAL_NA_POLICY=as_no \
  $PY src/evaluation/run_pipeline_eval.py \
      --generator-base "$BASE4" --judge-base "$BASE4" \
      --eval-split "$split" --batch-size 16 --tie-delta 0.05 \
      --ab-aware --experiment-suffix "$sfx"
  local pred=$(ls -t results/pipeline_judge_cmp_base_${split}_${sfx}_predictions.parquet 2>/dev/null | head -1)
  local met=${pred%_predictions.parquet}_metrics.json
  echo -e "pipeline_cmp\t${split}\t${pred}\t${met}" >> "$INDEX"
}

run_selfcheck_265() {
  local split=$1
  local sfx="robust_${split}_ckpt265_clean"
  stage "selfcheck_265 $split"
  $PY src/evaluation/run_self_checklist_eval.py \
      --judge-adapter "$ADAPTER_SELFCHECK_265" \
      --base-model "$BASE4" --eval-split "$split" --backend vllm \
      --batch-size 16 --max-new-tokens 6144 --max-model-len 12288 --gpu-memory-utilization 0.90 \
      --experiment-suffix "$sfx"
  local pred=$(ls -t results/selfchecklist_checkpoint-265_${split}_${sfx}_predictions.parquet 2>/dev/null | head -1)
  [[ -z "$pred" ]] && pred=$(ls -t results/selfchecklist_checkpoint-265_${split}_*_predictions.parquet 2>/dev/null | head -1)
  local met=${pred%_predictions.parquet}_metrics.json
  echo -e "selfcheck_265\t${split}\t${pred}\t${met}" >> "$INDEX"
}

for sp in "${SPLITS[@]}"; do
  run_vanilla       "$sp" 2>&1 | tee -a "logs/robust/eval_${sp}.log"
  run_ft_vanilla    "$sp" 2>&1 | tee -a "logs/robust/eval_${sp}.log"
  run_pipeline_cmp  "$sp" 2>&1 | tee -a "logs/robust/eval_${sp}.log"
  run_selfcheck_265 "$sp" 2>&1 | tee -a "logs/robust/eval_${sp}.log"
done

echo "[done] $(date -Is)"
cat "$INDEX"
