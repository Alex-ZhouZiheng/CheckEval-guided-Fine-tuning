#!/usr/bin/env python3
"""Extract 595 tie/error review samples into a standalone parquet consumable by build_oracle_labels.py.

Reads: results/review/hroracle_weighted_tie_error_review_with_weights.parquet
Writes: data/subsets/tie_error_595.parquet

Keeps only columns required by _load_pairs() plus error_category / _review_split for downstream analysis.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

SRC = Path(__file__).resolve().parent.parent


def main() -> None:
    review_path = SRC / "results/review/hroracle_weighted_tie_error_review_with_weights.parquet"
    review = pd.read_parquet(review_path)

    required = {"prompt_id", "domain", "context", "response_a", "response_b", "winner"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review parquet missing required columns: {sorted(missing)}")

    carry = ["sample_id"] + list(required) + ["error_category", "_review_split"]
    keep = [c for c in carry if c in review.columns]
    subset = review[keep].copy()

    # Ensure sample_id exists.
    if "sample_id" not in subset.columns:
        import hashlib

        def _make_sid(r):
            raw = f"{r['prompt_id']}|{r['response_a']}|{r['response_b']}|{r['winner']}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]

        subset["sample_id"] = subset.apply(_make_sid, axis=1)

    n = len(subset)
    out = SRC / "data/subsets/tie_error_595.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)

    print(f"Subset: {n} rows")
    for col in carry:
        if col in subset.columns:
            if col == "error_category":
                print(f"  {col} value_counts: {dict(subset[col].value_counts())}")
            elif col == "_review_split":
                print(f"  {col} value_counts: {dict(subset[col].value_counts())}")

    assert n == 595, f"Expected 595 rows, got {n}"
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
