#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Thesis

PY=.venvmerge/bin/python
BASE9=/root/autodl-tmp/Thesis/models/Qwen3.5-9B
BASE4=/root/autodl-tmp/Thesis/models/Qwen3.5-4B
OUTDIR=results/unified_test_20260509

echo "[start] $(date -Is)"
echo "[split] data/splits/test.parquet"
$PY - <<'PYEOF'
import pandas as pd
print("test_n_total", len(pd.read_parquet("data/splits/test.parquet")))
PYEOF

run_step() {
  name="$1"
  shift
  echo "===== ${name} START $(date -Is) ====="
  "$@"
  echo "===== ${name} DONE $(date -Is) ====="
}

run_step vanilla_judge_test \
  $PY -c "from pathlib import Path; g={'__file__':'src/evaluation/run_zeroshot.py','__name__':'__main__','Path':Path}; exec(compile(open('src/evaluation/run_zeroshot.py','rb').read(),'src/evaluation/run_zeroshot.py','exec'), g)" \
    --split test --suffix unified_20260509 --backend vllm --model-id "$BASE9" \
    --batch-size 16 --max-model-len 8192 --gpu-memory-utilization 0.90

run_step finetuned_vanilla_adapter_test \
  $PY src/evaluation/run_eval_finetuned.py \
    --adapter-path /root/autodl-tmp/Thesis/results/checkpoints/judge_warmup_tier_5k_r16_lr5e-06/final_adapter \
    --eval-split test --eval-mode vanilla --base-model "$BASE9" --backend vllm \
    --batch-size 16 --max-model-len 8192 --gpu-memory-utilization 0.90

run_step finetuned_checkeval_adapter_test \
  $PY src/evaluation/run_eval_finetuned.py \
    --adapter-path /root/autodl-tmp/Thesis/results/checkpoints/joint_dpo_tier_10k_r16_b0.1_lr1e-06_lam1.0/final_adapter \
    --eval-split test --eval-mode checkeval --run-name unified_20260509 --base-model "$BASE9" --backend vllm \
    --batch-size 8 --max-model-len 8192 --gpu-memory-utilization 0.90 --tie-delta 0.0

run_step pipeline_comparative_checklist_test \
  $PY scripts/run_comparative_eval.py \
    --question-source generated --aggregation weighted \
    --output results/comparative/gen_comparative_weighted_test_unified_20260509 \
    --split test --backend vllm --model-id "$BASE9" \
    --batch-size 8 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90

run_step selfchecklist_checkpoint_200_test \
  $PY src/evaluation/run_self_checklist_eval.py \
    --judge-adapter /root/autodl-tmp/Thesis/checkpoints/judge_sft_swift_Qwen3.5-4B_train_debug_5k_selfcheck_r16_lr3e-5/v1-20260506-170128/checkpoint-200 \
    --base-model "$BASE4" --eval-split test --backend vllm \
    --batch-size 16 --max-new-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --experiment-suffix unified_20260509_ckpt200

run_step selfchecklist_checkpoint_265_clean_test \
  $PY src/evaluation/run_self_checklist_eval.py \
    --judge-adapter /root/autodl-tmp/Thesis/checkpoints/judge_sft_swift_Qwen3.5-4B_train_debug_5k_selfcheck_clean_r16_lr2e-5/v0-20260506-200134/checkpoint-265 \
    --base-model "$BASE4" --eval-split test --backend vllm \
    --batch-size 16 --max-new-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --experiment-suffix unified_20260509_ckpt265_clean

run_step selfchecklist_checkpoint_558_4b_sft_test \
  $PY src/evaluation/run_self_checklist_eval.py \
    --judge-adapter /root/autodl-tmp/Thesis/checkpoints/judge_sft_swift_Qwen3.5-4B_train_debug_5k_selfcheck_r8_lr1e-4/v3-20260504-233716/checkpoint-558 \
    --base-model "$BASE4" --eval-split test --backend vllm \
    --batch-size 16 --max-new-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --experiment-suffix unified_20260509_ckpt558_4b_sft

$PY - <<'PYEOF'
import csv
import glob
import json
from pathlib import Path

out = Path("results/unified_test_20260509")
rows_spec = [
    ("Vanilla judge", "results/vanilla_judge_test_unified_20260509_metrics.json"),
    ("Finetuned vanilla adapter", sorted(glob.glob("results/finetuned_vanilla_final_adapter_test_2026-05-09_metrics.json"))[-1]),
    ("Finetuned CheckEval adapter", sorted(glob.glob("results/finetuned_checkeval_final_adapter_test_unified_20260509_2026-05-09_metrics.json"))[-1]),
    ("Pipeline comparative checklist", "results/comparative/gen_comparative_weighted_test_unified_20260509/metrics.json"),
    ("Selfchecklist checkpoint-200", "results/selfchecklist_checkpoint-200_test_unified_20260509_ckpt200_metrics.json"),
    ("Selfchecklist checkpoint-265 clean data", "results/selfchecklist_checkpoint-265_test_unified_20260509_ckpt265_clean_metrics.json"),
    ("Selfchecklist checkpoint-558 4B SFT", "results/selfchecklist_checkpoint-558_test_unified_20260509_ckpt558_4b_sft_metrics.json"),
]
rows = []
for method, path in rows_spec:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    n_valid = int(d.get("n_valid", 0))
    n_total = int(d.get("n_total", d.get("n_samples_total", 0)))
    valid_acc = d.get("valid_accuracy", d.get("accuracy"))
    real_acc = d.get("real_accuracy")
    if real_acc is None:
        real_acc = (float(valid_acc) * n_valid / n_total) if n_total else 0.0
    parse = d.get("parse_rate", d.get("parse_ok_rate", (n_valid / n_total if n_total else 0.0)))
    rows.append({
        "Method": method,
        "Real acc": real_acc,
        "Valid acc": float(valid_acc),
        "Parse": float(parse),
        "n_valid": n_valid,
        "n_total": n_total,
        "path": path,
    })

with open(out / "main_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Method", "Real acc", "Valid acc", "Parse", "n_valid", "n_total", "path"])
    w.writeheader()
    w.writerows(rows)

with open(out / "main_table.md", "w", encoding="utf-8") as f:
    f.write("| Method | Real acc | Valid acc | Parse | n_valid | n_total |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['Method']} | {r['Real acc']:.4f} | {r['Valid acc']:.4f} | {r['Parse']:.4f} | {r['n_valid']} | {r['n_total']} |\n")

with open(out / "main_table_with_paths.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print((out / "main_table.md").read_text(encoding="utf-8"))
PYEOF

echo "[done] $(date -Is)"
