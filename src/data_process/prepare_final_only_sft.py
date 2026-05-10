#!/usr/bin/env python3
"""
Final-only SFT data prep.

Input  : data/splits/train_<tier>.parquet (pairwise: context, response_a, response_b, winner)
Output : data/judge_sft/train_<tier>_finalonly.parquet with columns:
            messages       - JSON-encoded [system, user] using VANILLA_JUDGE_PROMPT
            target_output  - "A" or "B"

Usage:
    python -m src.data_process.prepare_final_only_sft --tier debug_5k
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

import config as cfg
from utils import build_vanilla_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an impartial judge."


def build_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = Counter()
    for _, r in df.iterrows():
        w = str(r["winner"]).strip()
        if w not in ("A", "B"):
            skipped[w or "<empty>"] += 1
            continue
        prompt = build_vanilla_prompt(r)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rows.append({
            "prompt_id": r.get("prompt_id"),
            "domain": r.get("domain"),
            "winner": w,
            "messages": json.dumps(messages, ensure_ascii=False),
            "target_output": w,
        })
    log.info("Built %d rows; skipped (non-A/B winners): %s", len(rows), dict(skipped))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", type=str, default="debug_5k")
    p.add_argument("--input-path", type=str, default=None)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    in_path = Path(args.input_path) if args.input_path else cfg.SPLITS_DIR / f"train_{args.tier}.parquet"
    out_path = Path(args.output_path) if args.output_path else cfg.JUDGE_SFT_DIR / f"train_{args.tier}_finalonly.parquet"

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path)
    required = {"context", "response_a", "response_b", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    out = build_rows(df)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    log.info("winner balance: %s", dict(Counter(out["winner"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Saved %d rows -> %s", len(out), out_path)


if __name__ == "__main__":
    main()
