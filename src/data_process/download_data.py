#!/usr/bin/env python3
"""
Download 10k samples from HelpSteer3, stratified by domain proportion.
Excludes multilingual domain (only keeps general / stem / code).

Usage:
    python download_data.py
    python download_data.py --cache-dir /tmp
    python download_data.py -n 10000
    python download_data.py --dry-run
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
from rich.table import Table

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

TRAIN_OUT = cfg.RAW_DIR / "helpsteer3_train.parquet"
TEST_OUT = cfg.RAW_DIR / "helpsteer3_test.parquet"

def _filter(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Apply domain + preference filters to a dataframe."""
    df["domain_lower"] = df["domain"].str.strip().str.lower()

    before = len(df)
    df = df[df["domain_lower"].isin(cfg.KEEP_DOMAINS)].copy()
    log.info(
        f"  [{split_name}] After domain filter ({cfg.KEEP_DOMAINS}): {len(df):,}  "
        f"(dropped {before - len(df):,})"
    )

    before = len(df)
    df = df[df["overall_preference"].isin(cfg.KEEP_PREFERENCES)].copy()
    log.info(
        f"  [{split_name}] After preference filter ({cfg.KEEP_PREFERENCES}): {len(df):,}  "
        f"(dropped {before - len(df):,})"
    )
    return df

def download_and_filter(cache_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.HF_TOKEN:
        login(cfg.HF_TOKEN)

    log.info("Downloading...")
    ds = load_dataset(cfg.HF_DATASET_ID, cache_dir=cache_dir)

    train_df = ds["train"].to_pandas()
    log.info(f"  Raw train rows: {len(train_df):,}")
    train_df = _filter(train_df, "train")

    val_df = ds["validation"].to_pandas()
    log.info(f"  Raw validation rows: {len(val_df):,}")
    val_df = _filter(val_df, "validation/test")

    return train_df, val_df



def print_summary(df: pd.DataFrame,title: str = "Filtered HelpSteer3") -> None:
    table = Table(title=f"{title} (n={len(df):,})")
    table.add_column("Domain", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Proportion", justify="right")

    for domain, cnt in df["domain_lower"].value_counts().items():
        table.add_row(domain, f"{cnt:,}", f"{cnt/len(df):.1%}")
    console.print(table)

    table2 = Table(title=f"{title} — Preference Distribution")
    table2.add_column("overall_preference", style="bold")
    table2.add_column("Count", justify="right")
    for pref, cnt in df["overall_preference"].value_counts().sort_index().items():
        table2.add_row(str(pref), f"{cnt:,}")
    console.print(table2)


def main():
    parser = argparse.ArgumentParser(description="Download HelpSteer3 sample")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("-n", type=int,default=20000)
    args = parser.parse_args()

    train_df, test_df = download_and_filter(cache_dir=args.cache_dir)
    print_summary(train_df, title="Train split")
    print_summary(test_df, title="Test split (validation)")

    if args.dry_run:
        log.info("Dry run -- not writing files.")
        return

    train_df.to_parquet(TRAIN_OUT, index=False)
    log.info(f"Saved {len(train_df):,} rows -> {TRAIN_OUT}")

    test_df.to_parquet(TEST_OUT, index=False)
    log.info(f"Saved {len(test_df):,} rows -> {TEST_OUT}")

    meta = {
        "train_n": len(train_df),
        "test_n": len(test_df),
        "seed": cfg.SEED,
        "keep_domains": sorted(cfg.KEEP_DOMAINS),
        "keep_preferences": sorted(cfg.KEEP_PREFERENCES),
        "train_domain_counts": train_df["domain_lower"].value_counts().to_dict(),
        "test_domain_counts": test_df["domain_lower"].value_counts().to_dict(),
    }
    meta_path = cfg.RAW_DIR / "helpsteer3_filtered_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"Saved metadata -> {meta_path}")

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
