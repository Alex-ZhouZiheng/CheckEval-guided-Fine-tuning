#!/usr/bin/env python3
"""
Create a 600-sample dev subset from an existing dev.parquet.

Usage:
    python src/make_dev600.py
    python src/make_dev600.py --n-dev 600
    python src/make_dev600.py --source dev.parquet --out-name dev_600.parquet
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


def exact_stratified_sample(
    df: pd.DataFrame,
    n: int,
    strat_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    """
    Sample exactly n rows while approximately preserving the distribution
    over `strat_cols`.

    Strategy:
    1. Compute ideal fractional allocations by stratum.
    2. Floor them.
    3. Distribute remaining rows by largest remainder.
    4. Sample within each stratum without replacement.
    """
    if n > len(df):
        raise ValueError(f"Requested n={n}, but source only has {len(df)} rows.")

    rng = np.random.RandomState(seed)

    work = df.copy()
    work["_stratum"] = work[strat_cols].astype(str).agg("||".join, axis=1)

    counts = work["_stratum"].value_counts().sort_index()
    proportions = counts / counts.sum()
    ideal = proportions * n
    alloc = np.floor(ideal).astype(int)

    remainder = n - alloc.sum()
    if remainder > 0:
        frac = (ideal - alloc).sort_values(ascending=False)
        for stratum in frac.index[:remainder]:
            alloc.loc[stratum] += 1

    # guard: do not allocate more than available in a stratum
    overflow = alloc - counts
    if (overflow > 0).any():
        extra = int(overflow[overflow > 0].sum())
        alloc = np.minimum(alloc, counts)

        # redistribute leftover rows to strata with spare capacity
        spare = counts - alloc
        spare = spare[spare > 0].sort_values(ascending=False)
        for stratum in spare.index:
            if extra == 0:
                break
            take = min(int(spare.loc[stratum]), extra)
            alloc.loc[stratum] += take
            extra -= take

    assert int(alloc.sum()) == n, f"Allocation sum {alloc.sum()} != requested {n}"

    parts = []
    for stratum, k in alloc.items():
        if k <= 0:
            continue
        sdf = work[work["_stratum"] == stratum]
        idx = rng.choice(len(sdf), size=int(k), replace=False)
        parts.append(sdf.iloc[idx])

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.drop(columns=["_stratum"])


def print_summary(full_df: pd.DataFrame, subset_df: pd.DataFrame, name: str) -> None:
    table = Table(title=f"{name} Summary")
    table.add_column("Split", style="bold")
    table.add_column("Total", justify="right")
    table.add_column("General", justify="right")
    table.add_column("STEM", justify="right")
    table.add_column("Code", justify="right")
    table.add_column("Winner=A", justify="right")
    table.add_column("Winner=B", justify="right")
    table.add_column("Tie", justify="right")

    for split_name, sdf in [("source", full_df), (name, subset_df)]:
        dom = sdf["domain"].value_counts()
        win = sdf["winner"].value_counts()
        table.add_row(
            split_name,
            f"{len(sdf):,}",
            f"{dom.get('general', 0):,}",
            f"{dom.get('stem', 0):,}",
            f"{dom.get('code', 0):,}",
            f"{win.get('A', 0):,}",
            f"{win.get('B', 0):,}",
            f"{win.get('Tie', 0):,}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Create a 600-row dev subset")
    parser.add_argument("--n-dev", type=int, default=600, help="Target subset size")
    parser.add_argument(
        "--source",
        type=str,
        default="dev.parquet",
        help="Source parquet under cfg.SPLITS_DIR",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="dev_600.parquet",
        help="Output parquet filename under cfg.SPLITS_DIR",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats, do not write files",
    )
    args = parser.parse_args()

    source_path = cfg.SPLITS_DIR / args.source
    if not source_path.exists():
        raise FileNotFoundError(f"{source_path} not found.")

    df = pd.read_parquet(source_path)
    log.info("Loaded source dev split: %s rows from %s", len(df), source_path)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        log.warning("Dropped %d exact duplicate rows from source before sampling.", before - len(df))

    required_cols = {"domain", "winner"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subset = exact_stratified_sample(
        df=df,
        n=args.n_dev,
        strat_cols=["domain", "winner"],
        seed=args.seed,
    )

    print_summary(df, subset, Path(args.out_name).stem)

    if args.dry_run:
        log.info("Dry run; not writing files.")
        return

    out_path = cfg.SPLITS_DIR / args.out_name
    subset.to_parquet(out_path, index=False)
    log.info("Saved subset -> %s", out_path)

    meta = {
        "source": str(source_path),
        "output": str(out_path),
        "seed": args.seed,
        "n_source": int(len(df)),
        "n_subset": int(len(subset)),
        "domain_counts": subset["domain"].value_counts().to_dict(),
        "winner_counts": subset["winner"].value_counts().to_dict(),
        "prompt_ids": int(subset["prompt_id"].nunique()) if "prompt_id" in subset.columns else None,
        "sampling": "exact stratified by domain × winner",
    }

    meta_path = cfg.SPLITS_DIR / args.out_name.replace(".parquet", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved metadata -> %s", meta_path)
    console.print(f"\n[bold green]✓ Wrote {args.n_dev}-row dev subset.[/bold green]")


if __name__ == "__main__":
    main()