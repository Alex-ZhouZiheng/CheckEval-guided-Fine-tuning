#!/usr/bin/env python3
"""
Training set statistics for DPO bias diagnostics.

Reports:
  1. A-win / B-win ratio (overall + per domain)
  2. Winner-in-first-position (response_a) ratio — proxy for presentation order bias
  3. Whether any A/B random swap was applied during data prep
  4. Token-length imbalance between chosen/rejected (proxy for length exploitation)

Usage:
    python analyze_train_stats.py
    python analyze_train_stats.py --split dev
    python analyze_train_stats.py --tier train_debug_5k
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── helpers ──────────────────────────


def token_len_approx(text: str) -> int:
    """Whitespace-split token count (language-agnostic approximation)."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def load_split(name: str) -> pd.DataFrame:
    path = cfg.SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_parquet(path)


# ────────────────────────── analysis ─────────────────────────


def report_win_ratio(df: pd.DataFrame) -> None:
    """1. A-win / B-win ratio."""
    table = Table(title="Winner Distribution")
    table.add_column("Subset", style="bold")
    table.add_column("Total")
    table.add_column("A wins")
    table.add_column("B wins")
    table.add_column("Tie")
    table.add_column("A%", justify="right")
    table.add_column("B%", justify="right")
    table.add_column("A/B ratio", justify="right")

    def _add_row(label: str, sub: pd.DataFrame):
        vc = sub["winner"].value_counts()
        a, b = vc.get("A", 0), vc.get("B", 0)
        tie = vc.get("Tie", 0)
        total = len(sub)
        a_pct = 100 * a / total if total else 0
        b_pct = 100 * b / total if total else 0
        ratio = a / b if b else float("inf")
        table.add_row(
            label,
            str(total),
            str(a),
            str(b),
            str(tie),
            f"{a_pct:.1f}%",
            f"{b_pct:.1f}%",
            f"{ratio:.3f}",
        )

    _add_row("Overall", df)
    for domain in sorted(df["domain"].unique()):
        _add_row(f"  {domain}", df[df["domain"] == domain])

    console.print(table)


def report_position_bias(df: pd.DataFrame) -> None:
    """2. Winner-in-first-position ratio.

    If winner == 'A', the preferred response is already in the first
    (response_a) slot.  A ratio significantly > 0.5 indicates the
    original dataset may have positioned the better response first.
    """
    non_tie = df[df["winner"].isin(["A", "B"])]
    first_pos = (non_tie["winner"] == "A").sum()
    total = len(non_tie)
    ratio = first_pos / total if total else 0

    table = Table(title="Position Bias: Winner in First Position (response_a)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Winner == A (first position)", str(first_pos))
    table.add_row("Winner == B (second position)", str(total - first_pos))
    table.add_row("Total (excl. ties)", str(total))
    table.add_row("P(winner in pos-1)", f"{ratio:.4f}")
    table.add_row(
        "Status",
        "[green]Balanced[/green]" if 0.45 <= ratio <= 0.55
        else "[yellow]Mild bias[/yellow]" if 0.40 <= ratio <= 0.60
        else "[red]Significant bias[/red]",
    )
    console.print(table)


def report_swap_detection(df: pd.DataFrame) -> None:
    """3. Check whether A/B responses were randomly swapped.

    Heuristic: If the dataset applied random swaps, the correlation
    between original preference direction and winner should break.
    We check if the preference_strength sign pattern is consistent
    with the winner label.
    """
    table = Table(title="A/B Swap Detection")
    table.add_column("Check")
    table.add_column("Result")

    # If winner distribution is close to 50/50, swaps were likely applied
    non_tie = df[df["winner"].isin(["A", "B"])]
    a_frac = (non_tie["winner"] == "A").mean()
    balanced = 0.45 <= a_frac <= 0.55

    table.add_row(
        "A-win fraction",
        f"{a_frac:.4f} — {'balanced (swap likely applied)' if balanced else 'imbalanced (may lack swaps)'}",
    )

    # Check if prompt_id has duplicates (would indicate augmented swaps)
    n_unique_prompts = df["prompt_id"].nunique()
    n_rows = len(df)
    has_duplicates = n_unique_prompts < n_rows
    table.add_row(
        "Duplicate prompt_ids",
        f"{n_rows - n_unique_prompts} duplicates out of {n_rows} rows — "
        + ("augmented swap pairs detected" if has_duplicates else "no duplicates, single instance per prompt"),
    )

    # Check if for the same prompt_id we see both A and B as winners
    if has_duplicates:
        multi = df.groupby("prompt_id")["winner"].nunique()
        both_labels = (multi > 1).sum()
        table.add_row(
            "Prompts with both A/B labels",
            f"{both_labels} — {'swap augmentation confirmed' if both_labels > 0 else 'no contradictory labels'}",
        )

    console.print(table)


def report_token_balance(df: pd.DataFrame) -> None:
    """4. Token-length imbalance between chosen and rejected responses.

    DPO can exploit length shortcuts if chosen is systematically longer
    or shorter than rejected.
    """
    non_tie = df[df["winner"].isin(["A", "B"])].copy()

    non_tie["chosen_tokens"] = non_tie.apply(
        lambda r: token_len_approx(r["response_a"] if r["winner"] == "A" else r["response_b"]),
        axis=1,
    )
    non_tie["rejected_tokens"] = non_tie.apply(
        lambda r: token_len_approx(r["response_b"] if r["winner"] == "A" else r["response_a"]),
        axis=1,
    )
    non_tie["len_ratio"] = non_tie["chosen_tokens"] / non_tie["rejected_tokens"].clip(lower=1)
    non_tie["len_diff"] = non_tie["chosen_tokens"] - non_tie["rejected_tokens"]

    table = Table(title="Token Length Balance: Chosen vs Rejected")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Chosen tokens (mean)", f"{non_tie['chosen_tokens'].mean():.1f}")
    table.add_row("Rejected tokens (mean)", f"{non_tie['rejected_tokens'].mean():.1f}")
    table.add_row("Chosen tokens (median)", f"{non_tie['chosen_tokens'].median():.1f}")
    table.add_row("Rejected tokens (median)", f"{non_tie['rejected_tokens'].median():.1f}")
    table.add_row("Len ratio chosen/rejected (mean)", f"{non_tie['len_ratio'].mean():.3f}")
    table.add_row("Len ratio chosen/rejected (median)", f"{non_tie['len_ratio'].median():.3f}")
    table.add_row("Len diff chosen-rejected (mean)", f"{non_tie['len_diff'].mean():.1f}")
    table.add_row("Len diff chosen-rejected (std)", f"{non_tie['len_diff'].std():.1f}")

    # Per-domain breakdown
    console.print(table)

    domain_table = Table(title="Token Length Balance by Domain")
    domain_table.add_column("Domain", style="bold")
    domain_table.add_column("Chosen (mean)", justify="right")
    domain_table.add_column("Rejected (mean)", justify="right")
    domain_table.add_column("Ratio (mean)", justify="right")
    domain_table.add_column("Diff (mean)", justify="right")

    for domain in sorted(non_tie["domain"].unique()):
        sub = non_tie[non_tie["domain"] == domain]
        domain_table.add_row(
            domain,
            f"{sub['chosen_tokens'].mean():.1f}",
            f"{sub['rejected_tokens'].mean():.1f}",
            f"{sub['len_ratio'].mean():.3f}",
            f"{sub['len_diff'].mean():.1f}",
        )
    console.print(domain_table)

    # Severity assessment
    mean_ratio = non_tie["len_ratio"].mean()
    if 0.85 <= mean_ratio <= 1.15:
        console.print("[green]Length balance: OK (ratio within 15%)[/green]")
    elif 0.70 <= mean_ratio <= 1.30:
        console.print("[yellow]Length balance: Mild imbalance (ratio 15-30%)[/yellow]")
    else:
        console.print("[red]Length balance: Severe imbalance — DPO may exploit length shortcut[/red]")


# ────────────────────────── main ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Training set statistics for DPO bias diagnostics")
    parser.add_argument("--split", default="train", help="Split to analyze (default: train)")
    parser.add_argument("--tier", default=None, help="Tier subset (e.g. train_debug_5k)")
    args = parser.parse_args()

    name = args.tier or args.split
    log.info(f"Loading {name}...")
    df = load_split(name)
    log.info(f"Loaded {len(df)} samples")

    console.rule(f"[bold]Training Set Statistics: {name}[/bold]")
    report_win_ratio(df)
    console.print()
    report_position_bias(df)
    console.print()
    report_swap_detection(df)
    console.print()
    report_token_balance(df)


if __name__ == "__main__":
    main()
