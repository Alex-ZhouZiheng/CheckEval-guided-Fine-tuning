#!/usr/bin/env python3
"""Post-training component audit for judge_selfcheck_quality reward."""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SRC_ROOT / "data_process") not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT / "data_process"))

from analysis.test_quality_reward import (  # noqa: E402
    _StubEncoder,
    _get_value,
    _infer_col,
    _read_records,
    score_one,
)
from train.plugin.judge_selfcheck_reward import JudgeSelfCheckQuality, _DiversityScorer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    return num / (den_x * den_y) if den_x > 0 and den_y > 0 else float("nan")


def _summary_line(name: str, vals: list[float]) -> str:
    return (
        f"{name:15s} mean={statistics.mean(vals):+.4f} "
        f"std={statistics.stdev(vals) if len(vals) > 1 else 0.0:.4f} "
        f"min={min(vals):+.4f} max={max(vals):+.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--completion-col", default="auto")
    parser.add_argument("--winner-col", default="auto")
    parser.add_argument("--prompt-id-col", default="prompt_id")
    parser.add_argument(
        "--eval-metrics",
        type=Path,
        default=None,
        help="Optional parquet/jsonl with prompt_id and pred_correct.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--stub-encoder",
        action="store_true",
        help="Use a deterministic local stub encoder for script smoke tests only.",
    )
    args = parser.parse_args(argv)

    records = _read_records(args.rollouts)
    if not records:
        raise RuntimeError(f"No rows found in {args.rollouts}")
    if args.stub_encoder:
        _DiversityScorer._model = _StubEncoder()
    completion_col = _infer_col(
        records,
        args.completion_col,
        ["completion", "target", "target_output", "output", "response", "raw_output"],
    )
    winner_col = _infer_col(records, args.winner_col, ["winner", "gold_winner", "winner_label"])
    fn = JudgeSelfCheckQuality()

    rows = []
    for row in records:
        comp = str(_get_value(row, completion_col) or "")
        gold = str(_get_value(row, winner_col) or "")
        comps = score_one(fn, comp, gold)
        rows.append(
            {
                "prompt_id": row.get(args.prompt_id_col, ""),
                "gold": gold,
                **comps,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s (n=%d)", args.out, len(rows))

    lines = []
    for col in ["parse_ok", "discriminative", "diversity", "winner", "R"]:
        lines.append(_summary_line(col, [float(r[col]) for r in rows]))

    if args.eval_metrics is not None:
        eval_records = _read_records(args.eval_metrics)
        pred_by_id = {
            str(r.get("prompt_id")): float(r.get("pred_correct"))
            for r in eval_records
            if r.get("prompt_id") is not None and r.get("pred_correct") is not None
        }
        paired = [(float(r["R"]), pred_by_id[str(r["prompt_id"])]) for r in rows if str(r["prompt_id"]) in pred_by_id]
        if paired:
            corr = _pearson([p[0] for p in paired], [p[1] for p in paired])
            lines.append(f"corr(R, eval_correct) = {corr:+.4f} (n={len(paired)})")
        else:
            lines.append("corr(R, eval_correct) = N/A (no prompt_id overlap)")

    summary_text = "\n".join(lines)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
