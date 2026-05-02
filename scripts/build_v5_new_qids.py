#!/usr/bin/env python3
"""Diff v5_frozen vs v4_frozen bank_index.parquet → emit parquet of NEW qids only.

Usage:
    python scripts/build_v5_new_qids.py
    # writes data/oracle/v5_new_qids.parquet (74 rows expected)
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

SRC = Path(__file__).resolve().parent.parent


def main() -> None:
    v5 = pd.read_parquet(SRC / "checklists/v5_frozen/bank_index.parquet")
    v4 = pd.read_parquet(SRC / "checklists/v4_frozen/bank_index.parquet")

    # Canonical text set from v4 (stable order irrelevant for set membership).
    v4_texts: set[str] = set(v4["question_text"].str.strip().tolist())

    new_mask = ~v5["question_text"].str.strip().isin(v4_texts)
    new_df = v5[new_mask].copy()

    out = SRC / "data/oracle/v5_new_qids.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(out, index=False)

    print(f"v4: {len(v4)} questions, v5: {len(v5)} questions, NEW: {len(new_df)}")
    print(f"\nPer-dimension breakdown:")
    for dim, cnt in new_df["dimension"].value_counts().sort_index().items():
        print(f"  {dim}: {cnt}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
