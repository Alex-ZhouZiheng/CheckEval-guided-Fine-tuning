#!/usr/bin/env python3
"""Plot accuracy-vs-budget Pareto curve from dynamic-eval metrics.

Usage:
    python src/analysis/pareto_plot.py \
        --inputs results/dynamic_dev_600/p0 results/dynamic_dev_600/p3_k20 \
        --out results/dynamic_dev_600/pareto.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _resolve_metric_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            m = p / "metrics.json"
            if m.exists():
                paths.append(m)
        elif p.is_file() and p.name.endswith(".json"):
            paths.append(p)
    return paths


def _load_rows(metric_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)

        if "accuracy" not in m:
            continue

        rows.append(
            {
                "policy": str(m.get("policy", path.parent.name)),
                "accuracy": float(m.get("accuracy", 0.0)),
                "avg_k": float(m.get("avg_k", m.get("k", 0.0))),
                "tie_rate": float(m.get("tie_rate", 0.0)),
                "samples_per_second": m.get("samples_per_second_estimate", None),
                "path": str(path),
            }
        )

    if not rows:
        raise SystemExit("No usable metrics with 'accuracy' found.")
    return pd.DataFrame(rows)


def _compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["avg_k", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    keep = []
    best_acc = -1.0
    for _, row in work.iterrows():
        acc = float(row["accuracy"])
        if acc >= best_acc:
            keep.append(True)
            best_acc = acc
        else:
            keep.append(False)
    return work[pd.Series(keep)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[Path("results/dynamic_dev_600")],
        help="Metrics json files or directories containing metrics.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Dynamic CheckEval Pareto: Accuracy vs Avg Questions")
    args = parser.parse_args()

    metric_paths = _resolve_metric_paths(args.inputs)
    if not metric_paths:
        raise SystemExit("No metrics.json found from --inputs")

    df = _load_rows(metric_paths)
    pareto = _compute_pareto(df)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for pareto_plot.py. Install it first."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["avg_k"], df["accuracy"], s=60, alpha=0.85, label="Policies")

    for _, r in df.iterrows():
        ax.annotate(
            str(r["policy"]),
            (float(r["avg_k"]), float(r["accuracy"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )

    p = pareto.sort_values("avg_k")
    ax.plot(p["avg_k"], p["accuracy"], linestyle="--", linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Average Questions per Sample (avg_k)")
    ax.set_ylabel("Pairwise Accuracy")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    csv_out = args.out.with_suffix(".csv")
    df.sort_values(["avg_k", "accuracy"], ascending=[True, False]).to_csv(csv_out, index=False)

    log.info("Saved plot -> %s", args.out)
    log.info("Saved table -> %s", csv_out)


if __name__ == "__main__":
    main()
