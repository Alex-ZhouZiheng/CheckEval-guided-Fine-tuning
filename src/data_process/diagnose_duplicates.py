#!/usr/bin/env python3
"""
Diagnose duplicate rows in a pairwise split parquet file.

Checks three levels of duplication:
  1. Exact row duplicates (all columns identical)
  2. KEY_COLS duplicates  (prompt_id + domain + context + response_a + response_b + winner)
  3. Prompt-level duplicates (same prompt_id appears more than once)

Usage:
    python src/data_process/diagnose_duplicates.py
    python src/data_process/diagnose_duplicates.py --split dev_600
    python src/data_process/diagnose_duplicates.py --input-path data/splits/dev_600.parquet
    python src/data_process/diagnose_duplicates.py --fix          # write de-duped parquet in place
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

KEY_COLS = ["prompt_id", "domain", "context", "response_a", "response_b", "winner"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int = 60) -> str:
    s = str(s).replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def _section(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


# ──────────────────────────────────────────────────────────────────────────────
# Level-1: Exact duplicates
# ──────────────────────────────────────────────────────────────────────────────

def _hashable_cols(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """Return subset of cols (default: all) whose values are hashable by pandas."""
    candidates = cols if cols is not None else list(df.columns)
    hashable = []
    for c in candidates:
        if c not in df.columns:
            continue
        try:
            df[c].duplicated()
            hashable.append(c)
        except TypeError:
            log.warning("Column %r contains unhashable values (e.g. arrays) — skipped in duplicate check.", c)
    return hashable


def check_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are exact duplicates across all hashable columns."""
    cols = _hashable_cols(df)
    if not cols:
        log.warning("No hashable columns available for exact-duplicate check.")
        return pd.DataFrame()
    mask = df.duplicated(subset=cols, keep=False)
    return df[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Level-2: KEY_COLS duplicates
# ──────────────────────────────────────────────────────────────────────────────

def check_key_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with duplicate KEY_COLS combinations (hashable ones only)."""
    available = _hashable_cols(df, [c for c in KEY_COLS if c in df.columns])
    if not available:
        log.warning("No hashable KEY_COLS available for duplicate check.")
        return pd.DataFrame()
    mask = df.duplicated(subset=available, keep=False)
    return df[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Level-3: prompt_id frequency
# ──────────────────────────────────────────────────────────────────────────────

def check_prompt_id_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Return prompt_ids that appear more than once."""
    if "prompt_id" not in df.columns:
        return pd.DataFrame()
    counts = df["prompt_id"].value_counts()
    repeated = counts[counts > 1].reset_index()
    repeated.columns = ["prompt_id", "count"]
    return repeated.sort_values("count", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def report_exact(df: pd.DataFrame, exact_dups: pd.DataFrame) -> None:
    _section("Level 1 — Exact row duplicates")
    if exact_dups.empty:
        console.print("[green]No exact duplicates found.[/green]")
        return

    console.print(f"[red]{len(exact_dups)} rows[/red] are involved in exact duplications "
                  f"(out of {len(df)} total).\n")

    # group and show first few groups
    show_cols = [c for c in ["prompt_id", "domain", "winner"] if c in exact_dups.columns]
    grouped = exact_dups.groupby(show_cols, dropna=False).size().reset_index(name="count")
    grouped = grouped.sort_values("count", ascending=False).head(10)

    t = Table(title="Top exact-duplicate groups (up to 10)", box=box.SIMPLE_HEAVY)
    for col in show_cols:
        t.add_column(col)
    t.add_column("count", justify="right")
    for _, row in grouped.iterrows():
        t.add_row(*[str(row[c]) for c in show_cols], str(row["count"]))
    console.print(t)


def report_key(df: pd.DataFrame, key_dups: pd.DataFrame) -> None:
    _section("Level 2 — KEY_COLS duplicates")
    available = [c for c in KEY_COLS if c in df.columns]
    console.print(f"Key columns used: {available}\n")

    if key_dups.empty:
        console.print("[green]No KEY_COLS duplicates found.[/green]")
        return

    console.print(f"[red]{len(key_dups)} rows[/red] share a KEY_COLS combination with at least one other row.\n")

    # Show sample duplicate groups
    key_dups = key_dups.copy()
    key_dups["_key"] = key_dups[available].astype(str).agg("|".join, axis=1)
    groups = key_dups.groupby("_key")

    shown = 0
    for key, grp in groups:
        if shown >= 5:
            console.print(f"  … and {len(groups) - shown} more groups (use --verbose to see all).")
            break
        pid = grp["prompt_id"].iloc[0] if "prompt_id" in grp.columns else "?"
        dom = grp["domain"].iloc[0] if "domain" in grp.columns else "?"
        win = grp["winner"].iloc[0] if "winner" in grp.columns else "?"
        console.print(f"[yellow]Group {shown+1}[/yellow]: prompt_id={pid}  domain={dom}  winner={win}  ({len(grp)} rows)")

        # Show how the rows differ (if at all)
        extra_cols = [c for c in grp.columns if c not in available and c != "_key"]
        if extra_cols:
            diff_cols = [c for c in extra_cols if grp[c].nunique() > 1]
            if diff_cols:
                console.print(f"  Differs in: {diff_cols}")
                for col in diff_cols[:3]:
                    vals = grp[col].tolist()
                    console.print(f"    {col}: {[_truncate(v) for v in vals]}")
            else:
                console.print("  [dim]Rows are identical except for row index.[/dim]")
        shown += 1
    console.print()


def report_prompt_freq(df: pd.DataFrame, repeated: pd.DataFrame) -> None:
    _section("Level 3 — prompt_id frequency")
    if repeated.empty:
        console.print("[green]Every prompt_id is unique.[/green]")
        return

    console.print(f"[yellow]{len(repeated)} prompt_ids[/yellow] appear more than once.\n")

    t = Table(title="prompt_ids with highest repetition (top 10)", box=box.SIMPLE_HEAVY)
    t.add_column("prompt_id")
    t.add_column("count", justify="right")
    for _, row in repeated.head(10).iterrows():
        t.add_row(str(row["prompt_id"]), str(row["count"]))
    console.print(t)

    # Show detail for the worst offender
    worst_pid = repeated.iloc[0]["prompt_id"]
    worst_rows = df[df["prompt_id"] == worst_pid]
    console.print(f"\n[bold]Detail for prompt_id={worst_pid}[/bold] ({len(worst_rows)} rows):")
    show = [c for c in ["domain", "winner", "response_a", "response_b"] if c in worst_rows.columns]
    for i, (_, row) in enumerate(worst_rows.iterrows()):
        parts = {c: _truncate(row[c]) for c in show}
        console.print(f"  row {i}: {parts}")


def summary_table(df: pd.DataFrame, exact_dups: pd.DataFrame,
                  key_dups: pd.DataFrame, repeated: pd.DataFrame) -> None:
    _section("Summary")
    t = Table(box=box.ROUNDED)
    t.add_column("Check")
    t.add_column("Affected rows", justify="right")
    t.add_column("Status")

    def status(n: int) -> str:
        return "[green]OK[/green]" if n == 0 else "[red]FAIL[/red]"

    t.add_row("Total rows", str(len(df)), "")
    t.add_row("Exact duplicates", str(len(exact_dups)), status(len(exact_dups)))
    t.add_row("KEY_COLS duplicates", str(len(key_dups)), status(len(key_dups)))
    t.add_row("prompt_ids appearing >1×", str(len(repeated)), status(len(repeated)))
    console.print(t)


# ──────────────────────────────────────────────────────────────────────────────
# Fix
# ──────────────────────────────────────────────────────────────────────────────

def fix_and_save(df: pd.DataFrame, path: Path) -> None:
    available = [c for c in KEY_COLS if c in df.columns]
    before = len(df)
    df_clean = df.drop_duplicates(subset=available, keep="first").reset_index(drop=True)
    after = len(df_clean)
    dropped = before - after

    if dropped == 0:
        console.print("[green]No KEY_COLS duplicates to remove — file unchanged.[/green]")
        return

    backup = path.with_suffix(".bak.parquet")
    import shutil
    shutil.copy2(path, backup)
    log.info("Backed up original -> %s", backup)

    df_clean.to_parquet(path, index=False)
    console.print(f"[bold green]Fixed:[/bold green] dropped {dropped} duplicate rows "
                  f"({before} -> {after}). Saved to {path}.")
    console.print(f"[dim]Original backed up to {backup}[/dim]")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose duplicates in a pairwise split parquet")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--split", type=str, default="dev_600",
                     help="Split name under cfg.SPLITS_DIR (default: dev_600)")
    grp.add_argument("--input-path", type=str, default=None,
                     help="Explicit path to a parquet file")
    parser.add_argument("--fix", action="store_true",
                        help="Drop KEY_COLS duplicates and overwrite the file (backs up original first)")
    args = parser.parse_args()

    if args.input_path:
        path = Path(args.input_path)
    else:
        path = cfg.SPLITS_DIR / f"{args.split}.parquet"

    if not path.exists():
        log.error("File not found: %s", path)
        raise SystemExit(1)

    console.print(f"\n[bold]Loading:[/bold] {path}")
    df = pd.read_parquet(path)
    console.print(f"Shape: {df.shape}   Columns: {list(df.columns)}\n")

    exact_dups = check_exact_duplicates(df)
    key_dups   = check_key_duplicates(df)
    repeated   = check_prompt_id_frequency(df)

    report_exact(df, exact_dups)
    report_key(df, key_dups)
    report_prompt_freq(df, repeated)
    summary_table(df, exact_dups, key_dups, repeated)

    if args.fix:
        fix_and_save(df, path)


if __name__ == "__main__":
    main()
