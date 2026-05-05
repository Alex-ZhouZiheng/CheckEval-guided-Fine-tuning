#!/usr/bin/env python3
"""
Build a JSONL dataset for ms-swift GRPO fine-tuning of the **self-checklist
judge**. Each row carries:

  * messages      - [{"role": "user", "content": <student prompt>}]
                    The student prompt asks the model to produce a self-
                    checklist + per-item verdicts + final winner. Identical
                    format to the SFT student prompt (native thinking mode).
  * winner        - gold winner ("A" / "B" / "Tie") used by the reward fn
  * sample_id, domain, preference_strength

ms-swift forwards every non-`messages` column as **kwargs to the registered
reward function. The reward parses the completion's "Winner: X" line and
compares against `winner`.

Usage:
    python -m src.data_process.prepare_judge_grpo --tier tier_10k
    python -m src.data_process.prepare_judge_grpo --tier debug_5k --max 1500
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import config as cfg
from prepare_self_checklist_sft import (
    build_self_checklist_student_prompt,
    load_pairs,
    stratified_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default="tier_10k", choices=list(cfg.TIER_SIZES))
    p.add_argument("--max", type=int, default=None, help="Optional row cap.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="Use Qwen3 native thinking student prompt (default on).")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--drop-tie", action="store_true", default=True,
                   help="Drop rows where gold winner is Tie (default on).")
    p.add_argument("--keep-tie", dest="drop_tie", action="store_false")
    p.add_argument("--exclude-sft", type=str, default=None,
                   help="Path to SFT parquet whose sample_ids should be excluded "
                        "(prevents memorisation overlap). Pass the same parquet "
                        "used for run_judge_sft_swift.sh.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL (default: data/judge_sft/grpo_<tier>_selfcheck.jsonl)")
    args = p.parse_args()

    df = load_pairs(args.tier)
    log.info("Loaded %d pairs from %s", len(df), args.tier)

    if args.drop_tie:
        n0 = len(df)
        df = df[df["winner"].astype(str).str.upper().isin(["A", "B"])].reset_index(drop=True)
        log.info("Dropped %d Tie rows (kept %d)", n0 - len(df), len(df))

    if args.exclude_sft:
        sft_path = Path(args.exclude_sft)
        if not sft_path.exists():
            raise FileNotFoundError(f"--exclude-sft path missing: {sft_path}")
        sft_df = pd.read_parquet(sft_path)
        if "sample_id" not in sft_df.columns:
            raise ValueError(f"{sft_path} has no sample_id column")
        sft_ids = set(sft_df["sample_id"].astype(str))
        n0 = len(df)
        df = df[~df["sample_id"].astype(str).isin(sft_ids)].reset_index(drop=True)
        log.info("Excluded %d SFT-overlap rows from %s (kept %d)",
                 n0 - len(df), sft_path.name, len(df))

    df = stratified_sample(df, args.max, seed=args.seed)
    log.info("After sample: %d rows", len(df))

    out_path = (
        Path(args.output)
        if args.output
        else cfg.JUDGE_SFT_DIR / f"grpo_{args.tier}_selfcheck.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            winner = str(row["winner"]).strip().upper()
            if winner == "TIE":
                winner = "Tie"
            if winner not in ("A", "B", "Tie"):
                n_skipped += 1
                continue
            prompt = build_self_checklist_student_prompt(row, thinking=args.enable_thinking)
            strength = int(row.get("preference_strength", 1) or 1)
            rec = {
                "messages": [{"role": "user", "content": prompt}],
                "winner": winner,
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "preference_strength": abs(strength),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    log.info("Wrote %d rows (skipped %d) -> %s", n_written, n_skipped, out_path)


if __name__ == "__main__":
    main()
