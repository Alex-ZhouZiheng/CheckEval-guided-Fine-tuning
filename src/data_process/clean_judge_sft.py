#!/usr/bin/env python3
"""
Clean a judge SFT parquet produced by prepare_judge_sft.py:
drop any row whose ``target_output`` cannot be parsed as exactly
``n_questions`` binary ``Q{i}: yes|no`` lines.

Usage:
    python -m src.data_process.clean_judge_sft \\
        --input data/judge_sft/train_tier_10k_teacher.parquet

    # write to a custom path instead of overwriting
    python -m src.data_process.clean_judge_sft \\
        --input data/judge_sft/train_tier_10k_teacher.parquet \\
        --output data/judge_sft/train_tier_10k_teacher_clean.parquet
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

_LINE_RE = re.compile(r"^\s*Q(\d+):\s*(yes|no)\s*$", re.IGNORECASE)


def validate_target(target: str, n_q: int) -> tuple[bool, str]:
    """Return (ok, reason). Ok iff target contains exactly n_q unique
    ``Q1..Qn`` binary lines in order with no extra/missing/garbage lines."""
    if not isinstance(target, str) or not target.strip():
        return False, "empty_target"

    lines = [ln for ln in target.splitlines() if ln.strip()]
    if len(lines) != n_q:
        return False, f"line_count_mismatch({len(lines)}!={n_q})"

    seen: set[int] = set()
    for i, ln in enumerate(lines, 1):
        m = _LINE_RE.match(ln)
        if not m:
            return False, f"bad_line:{ln[:60]!r}"
        q = int(m.group(1))
        if q != i:
            return False, f"out_of_order(expected Q{i} got Q{q})"
        if q in seen:
            return False, f"duplicate_Q{q}"
        seen.add(q)
    return True, "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="Output parquet (default: overwrite --input)")
    p.add_argument("--stats-out", type=Path, default=None,
                   help="JSON with cleaning stats (default: <output>.clean_stats.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be dropped; do not write output")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found")

    df = pd.read_parquet(args.input)
    n_before = len(df)
    console.print(f"Loaded {n_before:,} rows from {args.input}")

    required = {"target_output", "n_questions"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input missing columns: {sorted(missing)}")

    # validate
    reasons: list[str] = []
    ok_flags: list[bool] = []
    for _, r in df.iterrows():
        ok, reason = validate_target(r["target_output"], int(r["n_questions"]))
        ok_flags.append(ok)
        reasons.append(reason)

    df = df.assign(_ok=ok_flags, _reason=reasons)
    dropped = df[~df["_ok"]]
    kept = df[df["_ok"]].drop(columns=["_ok", "_reason"]).reset_index(drop=True)

    reason_counts = Counter(dropped["_reason"].tolist())
    n_dropped = int(len(dropped))

    # summary
    t = Table(title="Judge SFT Clean — Drop Summary")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Rows before", f"{n_before:,}")
    t.add_row("Rows dropped", f"{n_dropped:,}")
    t.add_row("Rows kept", f"{len(kept):,}")
    t.add_row("Drop rate", f"{n_dropped / max(n_before, 1):.2%}")
    console.print(t)

    if reason_counts:
        t2 = Table(title="Drop reasons (top 15)")
        t2.add_column("Reason", overflow="fold")
        t2.add_column("Count", justify="right")
        for reason, cnt in reason_counts.most_common(15):
            t2.add_row(reason, f"{cnt:,}")
        console.print(t2)

    if args.dry_run:
        console.print("\n[yellow]--dry-run: not writing output.[/yellow]")
        return

    out_path = args.output or args.input
    kept.to_parquet(out_path, index=False)
    console.print(f"\n[bold green]Wrote {len(kept):,} rows -> {out_path}[/bold green]")

    stats_out = args.stats_out or out_path.with_suffix(".clean_stats.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump({
            "input": str(args.input),
            "output": str(out_path),
            "n_before": n_before,
            "n_dropped": n_dropped,
            "n_kept": int(len(kept)),
            "drop_reasons": dict(reason_counts),
        }, f, indent=2, ensure_ascii=False)
    console.print(f"stats  -> {stats_out}")


if __name__ == "__main__":
    main()
