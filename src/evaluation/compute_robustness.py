#!/usr/bin/env python3
"""Compute robustness metrics from two prediction parquets.

Inputs are `*_predictions.parquet` files produced by run_zeroshot.py /
run_judge_eval.py / run_pipeline_eval.py / etc. They must contain
`prompt_id`, `winner` (gold), and `predicted_winner` columns.

Joins original-vs-perturbed on prompt_id and reports:
  - n_overlap
  - acc_orig      accuracy on rows present in both (Tie/unparseable=wrong)
  - acc_pert      accuracy on perturbed predictions
  - acc_drop      acc_orig - acc_pert
  - invariance    fraction with identical pred (after AB flip if --swap)
  - flip_rate     1 - invariance

For --swap mode: original predictions are flipped (A<->B) before comparison
because the perturbed split has A/B swapped. Gold winner in the perturbed
parquet is already flipped, so acc_pert is computed against that gold.

Usage:
    python src/evaluation/compute_robustness.py \
        --orig-pred results/zeroshot_dev_600_predictions.parquet \
        --pert-pred results/zeroshot_dev_600_swap_predictions.parquet \
        --mode swap --out results/robustness_zeroshot_swap.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _flip(p: str | None) -> str | None:
    if p == "A":
        return "B"
    if p == "B":
        return "A"
    return p


def _acc(gold: pd.Series, pred: pd.Series) -> float:
    if len(gold) == 0:
        return 0.0
    return float((gold.values == pred.values).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-pred", required=True)
    p.add_argument("--pert-pred", required=True)
    p.add_argument("--mode", choices=["swap", "verbose", "format"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-domain", action="store_true")
    args = p.parse_args()

    o = pd.read_parquet(args.orig_pred)
    q = pd.read_parquet(args.pert_pred)
    for df, name in [(o, "orig"), (q, "pert")]:
        if "prompt_id" not in df.columns and "sample_id" in df.columns:
            df.rename(columns={"sample_id": "prompt_id"}, inplace=True)
        for col in ("prompt_id", "winner", "predicted_winner"):
            if col not in df.columns:
                raise SystemExit(f"{name} missing column: {col}")

    o = o[["prompt_id", "domain", "winner", "predicted_winner"]] \
        if "domain" in o.columns else o[["prompt_id", "winner", "predicted_winner"]]
    q = q[["prompt_id", "winner", "predicted_winner"]]

    o = o.rename(columns={"winner": "gold_orig", "predicted_winner": "pred_orig"})
    q = q.rename(columns={"winner": "gold_pert", "predicted_winner": "pred_pert"})

    m = o.merge(q, on="prompt_id", how="inner")
    n_overlap = len(m)
    if n_overlap == 0:
        raise SystemExit("no overlapping prompt_ids between the two prediction files")

    pred_orig_for_compare = m["pred_orig"].map(_flip) if args.mode == "swap" else m["pred_orig"]

    invariance = float((pred_orig_for_compare.values == m["pred_pert"].values).mean())
    acc_orig = _acc(m["gold_orig"], m["pred_orig"])
    acc_pert = _acc(m["gold_pert"], m["pred_pert"])

    out = {
        "mode": args.mode,
        "orig_pred_file": str(args.orig_pred),
        "pert_pred_file": str(args.pert_pred),
        "n_orig": int(len(o)),
        "n_pert": int(len(q)),
        "n_overlap": int(n_overlap),
        "acc_orig": acc_orig,
        "acc_pert": acc_pert,
        "acc_drop": acc_orig - acc_pert,
        "invariance_rate": invariance,
        "flip_rate": 1.0 - invariance,
    }

    if args.per_domain and "domain" in m.columns:
        per: dict[str, dict[str, float]] = {}
        for dom, g in m.groupby("domain"):
            pog = g["pred_orig"].map(_flip) if args.mode == "swap" else g["pred_orig"]
            per[str(dom)] = {
                "n": int(len(g)),
                "acc_orig": _acc(g["gold_orig"], g["pred_orig"]),
                "acc_pert": _acc(g["gold_pert"], g["pred_pert"]),
                "invariance_rate": float((pog.values == g["pred_pert"].values).mean()),
            }
        out["per_domain"] = per

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
