#!/usr/bin/env python3
"""
Swap test for position bias detection (INV test).

For every sample in the eval set, swap response_a ↔ response_b and
flip the ground-truth label.  Run inference on both original and swapped
versions.  A robust model should produce consistent (flipped) predictions.

Metrics reported:
  1. Swap consistency rate — fraction of samples where prediction flips
     correctly after swap (the core INV metric)
  2. Position bias score — fraction that always predict A (or B)
     regardless of content
  3. Per-domain and per-class breakdown
  4. Consistency vs preference strength (are easy samples more consistent?)

Based on Ribeiro et al. INV/DIR invariance testing framework.

Usage:
    # Test base model
    python analyze_swap_test.py --base-only --eval-split test

    # Test DPO model
    python analyze_swap_test.py \\
        --adapter-path results/checkpoints/dpo_.../final_adapter \\
        --eval-split test

    # Both models compared
    python analyze_swap_test.py \\
        --adapter-path results/checkpoints/dpo_.../final_adapter \\
        --compare-base --eval-split test
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from utils import (
    build_vanilla_prompt,
    compute_metrics,
    generate_batch,
    load_eval_data,
    load_judge_model,
    parse_winner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── swap logic ───────────────────────


def create_swapped_df(df: pd.DataFrame) -> pd.DataFrame:
    """Swap response_a ↔ response_b and flip the ground truth winner."""
    swapped = df.copy()
    swapped["response_a"] = df["response_b"]
    swapped["response_b"] = df["response_a"]

    flip_map = {"A": "B", "B": "A", "Tie": "Tie"}
    swapped["winner"] = df["winner"].map(flip_map)
    return swapped


def run_inference(model, df: pd.DataFrame, batch_size: int) -> list[str | None]:
    """Run vanilla judge and return raw predictions."""
    all_messages = []
    for _, row in df.iterrows():
        prompt_text = build_vanilla_prompt(row)
        all_messages.append(
            [
                {"role": "system", "content": "You are an impartial judge."},
                {"role": "user", "content": prompt_text},
            ]
        )
    raw_outputs = generate_batch(model, all_messages, batch_size=batch_size)
    return [parse_winner(raw) for raw in raw_outputs]


# ────────────────────────── analysis ─────────────────────────


def analyze_swap_consistency(
    label: str,
    df: pd.DataFrame,
    orig_preds: list[str | None],
    swap_preds: list[str | None],
) -> dict:
    """Core swap consistency analysis."""
    flip_map = {"A": "B", "B": "A"}
    results = []

    for i, (row_idx, row) in enumerate(df.iterrows()):
        orig_p = orig_preds[i]
        swap_p = swap_preds[i]
        gt = row["winner"]

        # Valid only if both predictions are A or B
        if orig_p not in ("A", "B") or swap_p not in ("A", "B"):
            category = "unparseable"
        elif swap_p == flip_map.get(orig_p):
            category = "consistent"  # prediction correctly flipped
        elif orig_p == swap_p:
            category = "position_locked"  # always same position regardless of content
        else:
            category = "inconsistent"  # flipped but not in expected direction

        results.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "winner": gt,
            "preference_strength": row.get("preference_strength"),
            "orig_pred": orig_p,
            "swap_pred": swap_p,
            "expected_swap_pred": flip_map.get(orig_p),
            "category": category,
            "orig_correct": orig_p == gt if orig_p in ("A", "B") and gt in ("A", "B") else None,
        })

    rdf = pd.DataFrame(results)
    valid = rdf[rdf["category"] != "unparseable"]

    # ── Summary table ──
    n_valid = len(valid)
    counts = valid["category"].value_counts()
    consistent = counts.get("consistent", 0)
    pos_locked = counts.get("position_locked", 0)

    summary = Table(title=f"Swap Test Results — {label}")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total samples", str(len(rdf)))
    summary.add_row("Valid (both parsed)", str(n_valid))
    summary.add_row("Consistent (pred flips correctly)", f"{consistent}  ({100*consistent/n_valid:.1f}%)" if n_valid else "N/A")
    summary.add_row("Position-locked (same pred after swap)", f"{pos_locked}  ({100*pos_locked/n_valid:.1f}%)" if n_valid else "N/A")
    summary.add_row(
        "Swap Consistency Rate (INV score)",
        f"[bold]{consistent/n_valid:.4f}[/bold]" if n_valid else "N/A",
    )
    summary.add_row(
        "Position Bias Rate",
        f"{pos_locked/n_valid:.4f}" if n_valid else "N/A",
    )
    console.print(summary)

    # ── Which position is locked? ──
    if pos_locked > 0:
        locked = valid[valid["category"] == "position_locked"]
        locked_to_a = (locked["orig_pred"] == "A").sum()
        locked_to_b = (locked["swap_pred"] == "A").sum()  # same as orig since locked
        bias_table = Table(title=f"Position Lock Direction — {label}")
        bias_table.add_column("Always predicts")
        bias_table.add_column("Count", justify="right")
        bias_table.add_column("Fraction of locked", justify="right")
        bias_table.add_row("A (first position)", str(locked_to_a), f"{100*locked_to_a/pos_locked:.1f}%")
        bias_table.add_row("B (second position)", str(pos_locked - locked_to_a), f"{100*(pos_locked-locked_to_a)/pos_locked:.1f}%")
        console.print(bias_table)

    # ── Per-domain breakdown ──
    domain_table = Table(title=f"Swap Consistency by Domain — {label}")
    domain_table.add_column("Domain", style="bold")
    domain_table.add_column("N")
    domain_table.add_column("Consistent", justify="right")
    domain_table.add_column("Pos-Locked", justify="right")
    domain_table.add_column("INV Score", justify="right")

    for domain in sorted(valid["domain"].unique()):
        sub = valid[valid["domain"] == domain]
        n = len(sub)
        c = (sub["category"] == "consistent").sum()
        p = (sub["category"] == "position_locked").sum()
        domain_table.add_row(
            domain, str(n), str(c), str(p),
            f"{c/n:.4f}" if n else "N/A",
        )
    console.print(domain_table)

    # ── Per ground-truth class ──
    class_table = Table(title=f"Swap Consistency by Ground Truth — {label}")
    class_table.add_column("GT Class", style="bold")
    class_table.add_column("N")
    class_table.add_column("INV Score", justify="right")
    class_table.add_column("Pos-Locked Rate", justify="right")

    for gt in ["A", "B"]:
        sub = valid[valid["winner"] == gt]
        n = len(sub)
        c = (sub["category"] == "consistent").sum()
        p = (sub["category"] == "position_locked").sum()
        class_table.add_row(
            gt, str(n),
            f"{c/n:.4f}" if n else "N/A",
            f"{p/n:.4f}" if n else "N/A",
        )
    console.print(class_table)

    # ── Consistency vs preference strength ──
    if "preference_strength" in valid.columns and valid["preference_strength"].notna().any():
        strength_table = Table(title=f"Consistency vs Preference Strength — {label}")
        strength_table.add_column("Pref Strength")
        strength_table.add_column("N")
        strength_table.add_column("INV Score", justify="right")
        strength_table.add_column("Pos-Locked Rate", justify="right")

        for strength in sorted(valid["preference_strength"].dropna().unique()):
            sub = valid[valid["preference_strength"] == strength]
            n = len(sub)
            c = (sub["category"] == "consistent").sum()
            p = (sub["category"] == "position_locked").sum()
            strength_table.add_row(
                str(int(strength)), str(n),
                f"{c/n:.4f}" if n else "N/A",
                f"{p/n:.4f}" if n else "N/A",
            )
        console.print(strength_table)

    return {
        "label": label,
        "n_valid": n_valid,
        "consistent": consistent,
        "position_locked": pos_locked,
        "inv_score": consistent / n_valid if n_valid else None,
        "position_bias_rate": pos_locked / n_valid if n_valid else None,
        "details_df": rdf,
    }


# ────────────────────────── main ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Swap test for position bias (INV invariance test)"
    )
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--compare-base", action="store_true",
                        help="Run swap test on both base and DPO model")
    parser.add_argument("--eval-split", default="test", choices=["test", "dev"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-details", action="store_true",
                        help="Save per-sample swap details to parquet")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    args = parser.parse_args()

    if not args.base_only and args.adapter_path is None:
        parser.error("--adapter-path required unless --base-only")

    # Load eval data
    df = load_eval_data(args.eval_split)
    if args.max_samples:
        df = df.head(args.max_samples)
    log.info(f"Eval set: {len(df)} samples from {args.eval_split}")

    # Create swapped version
    swapped_df = create_swapped_df(df)

    vllm_kwargs = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    all_results = {}

    # ── Base model ──
    if args.base_only or args.compare_base:
        console.rule("[bold]Swap Test — Base Model[/bold]")
        log.info("Loading base model: %s", cfg.JUDGE_MODEL_ID)
        model = load_judge_model(model_id=cfg.JUDGE_MODEL_ID, **vllm_kwargs)

        t0 = time.time()
        orig_preds = run_inference(model, df, args.batch_size)
        swap_preds = run_inference(model, swapped_df, args.batch_size)
        elapsed = time.time() - t0
        log.info("Base swap test: %.1fs (2x %d samples)", elapsed, len(df))

        base_result = analyze_swap_consistency("Base", df, orig_preds, swap_preds)
        all_results["base"] = base_result

        # Also report standard accuracy on original
        orig_metrics = compute_metrics(
            y_true=df["winner"].tolist(),
            y_pred=orig_preds,
            domains=df["domain"].tolist(),
        )
        console.print(f"\nBase accuracy (original order): {orig_metrics.get('accuracy', 'N/A'):.4f}")

        del model

    # ── DPO model ──
    if not args.base_only:
        console.rule("[bold]Swap Test — DPO Model[/bold]")
        adapter_path = Path(args.adapter_path)

        if args.skip_merge:
            model_path = str(adapter_path)
        else:
            from run_eval_finetuned import merge_adapter
            model_path = str(merge_adapter(adapter_path))

        log.info("Loading DPO model: %s", model_path)
        model = load_judge_model(model_id=model_path, **vllm_kwargs)

        t0 = time.time()
        orig_preds = run_inference(model, df, args.batch_size)
        swap_preds = run_inference(model, swapped_df, args.batch_size)
        elapsed = time.time() - t0
        log.info("DPO swap test: %.1fs (2x %d samples)", elapsed, len(df))

        dpo_result = analyze_swap_consistency("DPO", df, orig_preds, swap_preds)
        all_results["dpo"] = dpo_result

        orig_metrics = compute_metrics(
            y_true=df["winner"].tolist(),
            y_pred=orig_preds,
            domains=df["domain"].tolist(),
        )
        console.print(f"\nDPO accuracy (original order): {orig_metrics.get('accuracy', 'N/A'):.4f}")

        del model

    # ── Comparison ──
    if "base" in all_results and "dpo" in all_results:
        console.rule("[bold]Swap Test Comparison: Base vs DPO[/bold]")
        cmp_table = Table(title="INV Score Comparison")
        cmp_table.add_column("Model", style="bold")
        cmp_table.add_column("INV Score", justify="right")
        cmp_table.add_column("Position Bias Rate", justify="right")

        for key in ["base", "dpo"]:
            r = all_results[key]
            cmp_table.add_row(
                r["label"],
                f"{r['inv_score']:.4f}" if r["inv_score"] is not None else "N/A",
                f"{r['position_bias_rate']:.4f}" if r["position_bias_rate"] is not None else "N/A",
            )

        console.print(cmp_table)

        base_inv = all_results["base"]["inv_score"] or 0
        dpo_inv = all_results["dpo"]["inv_score"] or 0
        delta = dpo_inv - base_inv

        if delta > 0.02:
            console.print(f"[green]DPO improved swap consistency by {delta:+.4f}[/green]")
        elif delta < -0.02:
            console.print(f"[red]DPO worsened swap consistency by {delta:+.4f} — position bias increased[/red]")
        else:
            console.print(f"[yellow]Swap consistency roughly unchanged ({delta:+.4f})[/yellow]")

    # Save details
    if args.save_details:
        for key, r in all_results.items():
            out = cfg.RESULTS_DIR / f"swap_test_{key}_{args.eval_split}.parquet"
            r["details_df"].to_parquet(out, index=False)
            log.info("Saved swap details to %s", out)

    log.info("Done.")


if __name__ == "__main__":
    main()
