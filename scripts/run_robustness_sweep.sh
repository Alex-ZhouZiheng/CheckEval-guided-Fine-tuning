#!/usr/bin/env bash
# Robustness sweep: build perturbed dev_600 splits and evaluate any judge on them.
#
# Usage:
#   bash scripts/run_robustness_sweep.sh build              # build splits only
#   bash scripts/run_robustness_sweep.sh eval_zeroshot      # run zeroshot judge on all splits
#   bash scripts/run_robustness_sweep.sh eval_pipeline      # run full pipeline judge
#   bash scripts/run_robustness_sweep.sh metrics            # compute robustness JSONs
#   bash scripts/run_robustness_sweep.sh all                # everything (default)
#
# Env knobs:
#   IN_SPLIT          base split name (default: dev_600)
#   CONCURRENCY       DeepSeek concurrency (default: 16)
#   GENERATOR_ADAPTER (optional) for pipeline eval
#   JUDGE_ADAPTER     (optional) for pipeline eval
#   ORIG_PRED         path to original-split predictions parquet (required for `metrics`)
#   PRED_TAG          short tag for output JSON filenames (default: judge)
set -euo pipefail

IN_SPLIT=${IN_SPLIT:-dev_600}
CONCURRENCY=${CONCURRENCY:-16}
PRED_TAG=${PRED_TAG:-judge}

SWAP_SPLIT="${IN_SPLIT}_swap"
VERB_SPLIT="${IN_SPLIT}_verbose"
FMT_SPLIT="${IN_SPLIT}_format"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

cmd=${1:-all}

build() {
  python src/data_process/build_robust_swap.py \
      --in-split "$IN_SPLIT" --out-split "$SWAP_SPLIT"

  python src/data_process/build_robust_verbose.py \
      --in-split "$IN_SPLIT" --out-split "$VERB_SPLIT" \
      --concurrency "$CONCURRENCY" \
      --audit-out "data/splits/${VERB_SPLIT}_audit.parquet"

  python src/data_process/build_robust_format.py \
      --in-split "$IN_SPLIT" --out-split "$FMT_SPLIT" \
      --concurrency "$CONCURRENCY" \
      --audit-dir "data/splits/_format_audit"
}

eval_zeroshot() {
  for sp in "$IN_SPLIT" "$SWAP_SPLIT" "$VERB_SPLIT" "$FMT_SPLIT"; do
    echo "==> zeroshot on $sp"
    python src/evaluation/run_zeroshot.py --eval-split "$sp"
  done
}

eval_pipeline() {
  local args=()
  [[ -n "${GENERATOR_ADAPTER:-}" ]] && args+=(--generator-adapter "$GENERATOR_ADAPTER")
  [[ -n "${JUDGE_ADAPTER:-}"     ]] && args+=(--judge-adapter     "$JUDGE_ADAPTER")
  for sp in "$IN_SPLIT" "$SWAP_SPLIT" "$VERB_SPLIT" "$FMT_SPLIT"; do
    echo "==> pipeline on $sp"
    python src/evaluation/run_pipeline_eval.py \
        --eval-split "$sp" "${args[@]}"
  done
}

metrics() {
  : "${ORIG_PRED:?ORIG_PRED env var required (path to original predictions parquet)}"
  : "${SWAP_PRED:?SWAP_PRED env var required}"
  : "${VERB_PRED:?VERB_PRED env var required}"
  : "${FMT_PRED:?FMT_PRED env var required}"
  mkdir -p results/robustness
  python src/evaluation/compute_robustness.py \
      --orig-pred "$ORIG_PRED" --pert-pred "$SWAP_PRED" \
      --mode swap --per-domain \
      --out "results/robustness/${PRED_TAG}_swap.json"
  python src/evaluation/compute_robustness.py \
      --orig-pred "$ORIG_PRED" --pert-pred "$VERB_PRED" \
      --mode verbose --per-domain \
      --out "results/robustness/${PRED_TAG}_verbose.json"
  python src/evaluation/compute_robustness.py \
      --orig-pred "$ORIG_PRED" --pert-pred "$FMT_PRED" \
      --mode format --per-domain \
      --out "results/robustness/${PRED_TAG}_format.json"
}

case "$cmd" in
  build) build ;;
  eval_zeroshot) eval_zeroshot ;;
  eval_pipeline) eval_pipeline ;;
  metrics) metrics ;;
  all) build; eval_zeroshot; echo "Set ORIG_PRED/SWAP_PRED/VERB_PRED/FMT_PRED then: bash $0 metrics" ;;
  *) echo "unknown subcmd: $cmd" >&2; exit 2 ;;
esac
