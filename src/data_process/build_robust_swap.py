#!/usr/bin/env python3
"""Build A/B-swap robustness split.

Reads a base split (default: dev_600.parquet), swaps response_a <-> response_b,
flips the gold winner (A<->B; Tie unchanged), and writes a new parquet whose
schema matches the original so existing eval scripts work unchanged.

Usage:
    python src/data_process/build_robust_swap.py \
        --in-split dev_600 --out-split dev_600_swap
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import pandas as pd

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _flip_winner(w: str) -> str:
    if w == "A":
        return "B"
    if w == "B":
        return "A"
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-split", default="dev_600")
    p.add_argument("--out-split", default="dev_600_swap")
    args = p.parse_args()

    src = cfg.SPLITS_DIR / f"{args.in_split}.parquet"
    dst = cfg.SPLITS_DIR / f"{args.out_split}.parquet"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_parquet(src).copy()
    log.info("Loaded %d rows from %s", len(df), src.name)

    df["response_a"], df["response_b"] = df["response_b"].copy(), df["response_a"].copy()
    df["winner"] = df["winner"].map(_flip_winner)
    df.attrs["source_split"] = args.in_split
    df.attrs["perturbation"] = "ab_swap"

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    log.info("Wrote %d rows -> %s", len(df), dst)


if __name__ == "__main__":
    main()
