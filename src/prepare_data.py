#!/usr/bin/env python3
"""
    python prepare_data.py                    # full pipeline
    python prepare_data.py --dry-run          # just print stats, don't write
    python prepare_data.py --cache-dir /tmp   # custom HF cache
"""

from __future__ import annotations

import argparse
import hashlib
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

# ────────────────────────── helpers ──────────────────────────


def context_to_text(context: list[dict]) -> str:
    """Serialise a multi-turn conversation list into a single string."""
    parts = []
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def make_prompt_id(text: str) -> str:
    """Deterministic 16-char hex hash of a prompt string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def preference_to_winner(score: int) -> str:
    """
    Map HelpSteer3 overall_preference to A/B/Tie.

    Negative → response1 (A) preferred
    Positive → response2 (B) preferred
    Zero     → tie
    """
    if score < 0:
        return "A"
    elif score > 0:
        return "B"
    return "Tie"


# ────────────────────────── main pipeline ──────────────────────


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-filtered train and test parquet files from RAW_DIR."""
    train_path = cfg.RAW_DIR / "helpsteer3_train.parquet"
    test_path = cfg.RAW_DIR / "helpsteer3_test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. Run download_data.py first."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    log.info(f"  Loaded train: {len(train_df):,} rows, test: {len(test_df):,} rows")
    return train_df, test_df


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rows to the canonical pairwise-judge format."""
    records = []
    for _, row in df.iterrows():
        winner = preference_to_winner(row["overall_preference"])

        context_text = context_to_text(row["context"])
        prompt_id = make_prompt_id(context_text)

        # Preference strength (absolute value) for optional filtering later
        strength = abs(row["overall_preference"])

        records.append(
            {
                "prompt_id": prompt_id,
                "domain": row["domain_lower"],
                "context": context_text,
                "response_a": row["response1"],
                "response_b": row["response2"],
                "winner": winner,
                "preference_strength": strength,
            }
        )

    out = pd.DataFrame(records)
    log.info(f"  Pairwise pairs (including ties): {len(out):,}")
    return out


def split_train_dev(df: pd.DataFrame, dev_ratio: float = cfg.DEV_RATIO, seed: int = cfg.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train data into train / dev by *unique prompt_id* to prevent leakage.
    """
    rng = np.random.RandomState(seed)

    prompt_ids = df["prompt_id"].unique().to_numpy()
    rng.shuffle(prompt_ids)

    n_dev = int(len(prompt_ids) * dev_ratio)
    dev_ids = set(prompt_ids[:n_dev])
    train_ids = set(prompt_ids[n_dev:])

    train_df = df[df["prompt_id"].isin(train_ids)].copy()
    dev_df = df[df["prompt_id"].isin(dev_ids)].copy()

    assert train_ids.isdisjoint(dev_ids)

    log.info(f"  train : {len(train_df):>6,} pairs  ({len(train_ids):,} prompts)")
    log.info(f"  dev   : {len(dev_df):>6,} pairs  ({len(dev_ids):,} prompts)")

    return train_df, dev_df


def save_tiered_subsets(train_df: pd.DataFrame, seed: int = cfg.SEED) -> None:
    """Sample debug-5k / 10k / 20k subsets from the training split."""
    rng = np.random.RandomState(seed)
    n_train = len(train_df)

    for tier_name, tier_size in cfg.TIER_SIZES.items():
        if tier_size > n_train:
            log.warning(
                f"  Tier '{tier_name}' requests {tier_size:,} but train only has "
                f"{n_train:,} – saving full train instead."
            )
            subset = train_df.copy()
        else:
            idx = rng.choice(n_train, size=tier_size, replace=False)
            subset = train_df.iloc[sorted(idx)].copy()

        out_path = cfg.SPLITS_DIR / f"train_{tier_name}.parquet"
        subset.to_parquet(out_path, index=False)
        log.info(f"  Saved {tier_name}: {len(subset):,} pairs → {out_path}")


def print_summary(splits: dict[str, pd.DataFrame]) -> None:
    """Pretty-print domain × split statistics."""
    table = Table(title="HelpSteer3 Pairwise Splits Summary")
    table.add_column("Split", style="bold")
    table.add_column("Total", justify="right")
    table.add_column("General", justify="right")
    table.add_column("STEM", justify="right")
    table.add_column("Code", justify="right")
    table.add_column("Winner=A", justify="right")
    table.add_column("Winner=B", justify="right")
    table.add_column("Tie", justify="right")

    for name, sdf in splits.items():
        dom = sdf["domain"].value_counts()
        win = sdf["winner"].value_counts()
        table.add_row(
            name,
            f"{len(sdf):,}",
            f"{dom.get('general', 0):,}",
            f"{dom.get('stem', 0):,}",
            f"{dom.get('code', 0):,}",
            f"{win.get('A', 0):,}",
            f"{win.get('B', 0):,}",
            f"{win.get('Tie', 0):,}",
        )
    console.print(table)


# ────────────────────────── CLI ──────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare HelpSteer3 pairwise splits")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    # 1. Load pre-filtered data
    train_raw, test_raw = load_raw()

    # 2. Build pairwise format
    log.info("Building pairwise format for train ...")
    train_pairs = build_pairwise(train_raw)
    log.info("Building pairwise format for test ...")
    test_pairs = build_pairwise(test_raw)

    # 3. Split train into train / dev
    train_df, dev_df = split_train_dev(train_pairs)

    splits = {"train": train_df, "dev": dev_df, "test": test_pairs}

    # 4. Summary
    print_summary(splits)

    if args.dry_run:
        log.info("Dry run – not writing files.")
        return

    # 5. Save splits
    for name, sdf in splits.items():
        path = cfg.SPLITS_DIR / f"{name}.parquet"
        sdf.to_parquet(path, index=False)
        log.info(f"Saved {name} → {path}")

    # 6. Save tiered training subsets
    save_tiered_subsets(train_df)

    # 7. Save split metadata for reproducibility
    meta = {
        "seed": cfg.SEED,
        "dev_ratio": cfg.DEV_RATIO,
        "keep_domains": sorted(cfg.KEEP_DOMAINS),
        "keep_preferences": sorted(cfg.KEEP_PREFERENCES),
        "test_source": "HelpSteer3 validation split",
        "n_pairs": {k: len(v) for k, v in splits.items()},
        "n_prompts": {k: int(v["prompt_id"].nunique()) for k, v in splits.items()},
        "domain_counts": {
            k: v["domain"].value_counts().to_dict() for k, v in splits.items()
        },
    }
    meta_path = cfg.SPLITS_DIR / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"Saved split metadata → {meta_path}")

    console.print("\n[bold green]✓ Data preparation complete.[/bold green]")


if __name__ == "__main__":
    main()
