#!/usr/bin/env python3
"""
Export a pairwise train split to a JSONL dataset consumable by ms-swift GRPO.

Each row carries:
  * messages       - generator's system+user prompt (ready for the chat template)
  * winner, context, response_a, response_b, domain, sample_id

ms-swift GRPO forwards every non-`messages` column into the reward plugin as
**kwargs, so the CheckEval reward can rebuild the judge prompts per-completion.

Usage:
    python -m src.data_process.prepare_grpo_pairwise --tier tier_10k
    python -m src.data_process.prepare_grpo_pairwise --tier debug_5k --max 1000
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
from prepare_data_reasoning import make_sample_id
from prepare_generator_sft import build_generator_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_pairwise(tier: str) -> pd.DataFrame:
    path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    df = pd.read_parquet(path)
    required = {"prompt_id", "context", "response_a", "response_b", "domain", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairwise parquet missing columns: {sorted(missing)}")
    if "sample_id" not in df.columns:
        df["sample_id"] = df.apply(
            lambda r: make_sample_id(
                prompt_id=r["prompt_id"],
                response_a=str(r["response_a"]),
                response_b=str(r["response_b"]),
                winner=str(r["winner"]),
            ),
            axis=1,
        )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default="tier_10k", choices=list(cfg.TIER_SIZES))
    p.add_argument("--max", type=int, default=None, help="Optional row cap.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path (default: data/generator_sft/grpo_<tier>.jsonl)")
    args = p.parse_args()

    df = load_pairwise(args.tier)
    if args.max:
        df = df.head(args.max).reset_index(drop=True)

    out_path = (
        Path(args.output)
        if args.output
        else cfg.GENERATOR_SFT_DIR / f"grpo_{args.tier}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            winner = str(row["winner"]).strip().upper()
            if winner not in ("A", "B"):
                n_skipped += 1
                continue
            msgs = build_generator_messages(row)
            rec = {
                "messages": msgs,
                "winner": winner,
                "context": row["context"],
                "response_a": row["response_a"],
                "response_b": row["response_b"],
                "domain": row["domain"],
                "sample_id": row["sample_id"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    log.info("Wrote %d rows (skipped %d non-A/B) → %s", n_written, n_skipped, out_path)


if __name__ == "__main__":
    main()
