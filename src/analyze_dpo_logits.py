#!/usr/bin/env python3
"""
DPO before/after logits & confusion analysis.

Compares base model vs DPO fine-tuned model on eval set:
  1. P(A) mean shift — does the model globally favor A after DPO?
  2. Per-class confidence — is the model less confident on B samples?
  3. Error migration — do errors concentrate on "already hard" or "previously easy" samples?
  4. Confusion matrix diff (base vs DPO)

Requires vLLM for fast batched inference.

Usage:
    python analyze_dpo_logits.py \\
        --adapter-path results/checkpoints/dpo_.../final_adapter \\
        --eval-split test

    # With pre-merged model
    python analyze_dpo_logits.py \\
        --adapter-path /path/to/merged --skip-merge \\
        --eval-split test

    # Base model only (no adapter)
    python analyze_dpo_logits.py --base-only --eval-split test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix, accuracy_score

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


# ────────────────────────── inference ────────────────────────


def run_inference(
    model, df: pd.DataFrame, batch_size: int
) -> pd.DataFrame:
    """Run vanilla pairwise judge and return predictions."""
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
    predicted = [parse_winner(raw) for raw in raw_outputs]

    results = df.copy()
    results["raw_output"] = raw_outputs
    results["predicted_winner"] = predicted
    return results


# ────────────────────────── analysis ─────────────────────────


def report_prediction_distribution(
    label: str, results: pd.DataFrame
) -> None:
    """Show P(A), P(B), unparseable rate."""
    preds = results["predicted_winner"]
    total = len(preds)
    counts = Counter(preds)
    a_count = counts.get("A", 0)
    b_count = counts.get("B", 0)
    none_count = counts.get(None, 0)

    table = Table(title=f"Prediction Distribution — {label}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total samples", str(total))
    table.add_row("Predict A", f"{a_count}  ({100*a_count/total:.1f}%)")
    table.add_row("Predict B", f"{b_count}  ({100*b_count/total:.1f}%)")
    table.add_row("Unparseable", f"{none_count}  ({100*none_count/total:.1f}%)")
    table.add_row("P(A) overall", f"{a_count/(a_count+b_count):.4f}" if (a_count+b_count) else "N/A")
    console.print(table)


def report_per_class_confidence(
    label: str, results: pd.DataFrame
) -> None:
    """On samples where ground truth is A vs B, what fraction does the model get right?"""
    valid = results[results["predicted_winner"].isin(["A", "B"])].copy()

    table = Table(title=f"Per-Class Accuracy — {label}")
    table.add_column("Ground Truth")
    table.add_column("N")
    table.add_column("Correct")
    table.add_column("Accuracy", justify="right")
    table.add_column("Predicts A", justify="right")
    table.add_column("Predicts B", justify="right")

    for gt_class in ["A", "B"]:
        sub = valid[valid["winner"] == gt_class]
        n = len(sub)
        if n == 0:
            continue
        correct = (sub["predicted_winner"] == gt_class).sum()
        pred_a = (sub["predicted_winner"] == "A").sum()
        pred_b = (sub["predicted_winner"] == "B").sum()
        table.add_row(
            f"GT={gt_class}",
            str(n),
            str(correct),
            f"{100*correct/n:.1f}%",
            f"{pred_a} ({100*pred_a/n:.1f}%)",
            f"{pred_b} ({100*pred_b/n:.1f}%)",
        )

    # Per domain per class
    console.print(table)

    domain_table = Table(title=f"Per-Domain Per-Class Accuracy — {label}")
    domain_table.add_column("Domain")
    domain_table.add_column("GT")
    domain_table.add_column("N")
    domain_table.add_column("Accuracy", justify="right")

    for domain in sorted(valid["domain"].unique()):
        for gt_class in ["A", "B"]:
            sub = valid[(valid["domain"] == domain) & (valid["winner"] == gt_class)]
            n = len(sub)
            if n == 0:
                continue
            correct = (sub["predicted_winner"] == gt_class).sum()
            domain_table.add_row(domain, gt_class, str(n), f"{100*correct/n:.1f}%")

    console.print(domain_table)


def report_confusion(label: str, results: pd.DataFrame) -> None:
    """Print confusion matrix."""
    valid = results[
        results["predicted_winner"].isin(["A", "B"])
        & results["winner"].isin(["A", "B"])
    ]
    if len(valid) == 0:
        return

    labels = ["A", "B"]
    cm = confusion_matrix(valid["winner"], valid["predicted_winner"], labels=labels)

    table = Table(title=f"Confusion Matrix — {label}")
    table.add_column("", style="bold")
    table.add_column("Pred A", justify="right")
    table.add_column("Pred B", justify="right")
    table.add_row("GT A", str(cm[0, 0]), str(cm[0, 1]))
    table.add_row("GT B", str(cm[1, 0]), str(cm[1, 1]))
    console.print(table)


def report_error_migration(
    base_results: pd.DataFrame, dpo_results: pd.DataFrame
) -> None:
    """Analyze where DPO errors come from relative to base model.

    Categories:
    - Still correct: base correct → DPO correct
    - Fixed: base wrong → DPO correct
    - Regressed: base correct → DPO wrong
    - Still wrong: base wrong → DPO wrong
    """
    # Align on prompt_id
    merged = base_results[["prompt_id", "winner", "predicted_winner"]].rename(
        columns={"predicted_winner": "base_pred"}
    ).merge(
        dpo_results[["prompt_id", "predicted_winner"]].rename(
            columns={"predicted_winner": "dpo_pred"}
        ),
        on="prompt_id",
        how="inner",
    )

    # Filter to valid predictions in both
    merged = merged[
        merged["base_pred"].isin(["A", "B"])
        & merged["dpo_pred"].isin(["A", "B"])
        & merged["winner"].isin(["A", "B"])
    ]

    base_correct = merged["base_pred"] == merged["winner"]
    dpo_correct = merged["dpo_pred"] == merged["winner"]

    still_correct = (base_correct & dpo_correct).sum()
    fixed = (~base_correct & dpo_correct).sum()
    regressed = (base_correct & ~dpo_correct).sum()
    still_wrong = (~base_correct & ~dpo_correct).sum()
    total = len(merged)

    table = Table(title="Error Migration: Base → DPO")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")
    table.add_row("Still correct (✓→✓)", str(still_correct), f"{100*still_correct/total:.1f}%")
    table.add_row("Fixed (✗→✓)", str(fixed), f"{100*fixed/total:.1f}%")
    table.add_row("Regressed (✓→✗)", str(regressed), f"{100*regressed/total:.1f}%")
    table.add_row("Still wrong (✗→✗)", str(still_wrong), f"{100*still_wrong/total:.1f}%")
    table.add_row("Total", str(total), "")
    console.print(table)

    # Net improvement
    net = fixed - regressed
    console.print(
        f"\nNet improvement: {net:+d} samples "
        f"(fixed {fixed} − regressed {regressed})"
    )

    # Are regressions on easy or hard samples?
    if regressed > 0:
        reg_mask = base_correct & ~dpo_correct
        reg_samples = merged[reg_mask]

        # Check preference strength distribution of regressed samples
        if "preference_strength" in base_results.columns:
            reg_with_strength = reg_samples.merge(
                base_results[["prompt_id", "preference_strength"]],
                on="prompt_id",
                how="left",
            )
            all_with_strength = merged.merge(
                base_results[["prompt_id", "preference_strength"]],
                on="prompt_id",
                how="left",
            )

            reg_table = Table(title="Regressed Samples: Preference Strength Distribution")
            reg_table.add_column("Subset")
            reg_table.add_column("Mean Pref Strength", justify="right")
            reg_table.add_column("Median Pref Strength", justify="right")
            reg_table.add_row(
                "Regressed samples",
                f"{reg_with_strength['preference_strength'].mean():.2f}",
                f"{reg_with_strength['preference_strength'].median():.1f}",
            )
            reg_table.add_row(
                "All samples",
                f"{all_with_strength['preference_strength'].mean():.2f}",
                f"{all_with_strength['preference_strength'].median():.1f}",
            )
            console.print(reg_table)

            if reg_with_strength["preference_strength"].mean() > all_with_strength["preference_strength"].mean():
                console.print(
                    "[yellow]Regressions tend to be on EASIER samples "
                    "(higher pref strength) — concerning[/yellow]"
                )
            else:
                console.print(
                    "[green]Regressions tend to be on HARDER samples "
                    "(lower pref strength) — less concerning[/green]"
                )

    # Breakdown by ground truth class
    gt_table = Table(title="Regression Breakdown by Ground Truth")
    gt_table.add_column("GT Class")
    gt_table.add_column("Regressed", justify="right")
    gt_table.add_column("Total in class", justify="right")
    gt_table.add_column("Regression rate", justify="right")

    for gt in ["A", "B"]:
        class_mask = merged["winner"] == gt
        class_reg = (base_correct & ~dpo_correct & class_mask).sum()
        class_total = class_mask.sum()
        gt_table.add_row(
            gt, str(class_reg), str(class_total),
            f"{100*class_reg/class_total:.1f}%" if class_total else "N/A",
        )
    console.print(gt_table)


def report_pa_shift(
    base_results: pd.DataFrame, dpo_results: pd.DataFrame
) -> None:
    """Compare P(A) between base and DPO model."""
    def _pa(df):
        valid = df[df["predicted_winner"].isin(["A", "B"])]
        if len(valid) == 0:
            return float("nan")
        return (valid["predicted_winner"] == "A").mean()

    base_pa = _pa(base_results)
    dpo_pa = _pa(dpo_results)

    table = Table(title="P(A) Shift: Base → DPO")
    table.add_column("Model")
    table.add_column("P(A)", justify="right")
    table.add_column("P(B)", justify="right")
    table.add_row("Base", f"{base_pa:.4f}", f"{1-base_pa:.4f}")
    table.add_row("DPO", f"{dpo_pa:.4f}", f"{1-dpo_pa:.4f}")
    table.add_row("Shift", f"{dpo_pa - base_pa:+.4f}", "")
    console.print(table)

    if abs(dpo_pa - base_pa) > 0.05:
        direction = "A" if dpo_pa > base_pa else "B"
        console.print(
            f"[yellow]Warning: DPO shifted P(A) by {dpo_pa-base_pa:+.4f} — "
            f"model now favors {direction}[/yellow]"
        )


# ────────────────────────── main ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="DPO before/after logits & confusion analysis"
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Path to LoRA adapter (or merged model with --skip-merge)",
    )
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--base-only", action="store_true",
                        help="Only run base model analysis (no DPO comparison)")
    parser.add_argument("--eval-split", default="test", choices=["test", "dev"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save per-sample predictions to parquet")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    args = parser.parse_args()

    if not args.base_only and args.adapter_path is None:
        parser.error("--adapter-path is required unless --base-only is set")

    # Load eval data
    df = load_eval_data(args.eval_split)
    if args.max_samples:
        df = df.head(args.max_samples)
    log.info(f"Eval set: {len(df)} samples from {args.eval_split}")

    vllm_kwargs = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # ── Base model ──
    console.rule("[bold]Base Model Analysis[/bold]")
    log.info("Loading base model: %s", cfg.JUDGE_MODEL_ID)
    base_model = load_judge_model(model_id=cfg.JUDGE_MODEL_ID, **vllm_kwargs)

    t0 = time.time()
    base_results = run_inference(base_model, df, args.batch_size)
    base_time = time.time() - t0
    log.info("Base inference: %.1fs", base_time)

    report_prediction_distribution("Base", base_results)
    report_per_class_confidence("Base", base_results)
    report_confusion("Base", base_results)

    base_metrics = compute_metrics(
        y_true=base_results["winner"].tolist(),
        y_pred=base_results["predicted_winner"].tolist(),
        domains=base_results["domain"].tolist(),
    )
    console.print(f"\nBase accuracy: {base_metrics.get('accuracy', 'N/A'):.4f}")
    console.print(f"Base macro-F1: {base_metrics.get('macro_f1', 'N/A'):.4f}")

    # Free GPU memory
    del base_model

    if args.base_only:
        if args.save_predictions:
            out = cfg.RESULTS_DIR / f"logits_base_{args.eval_split}.parquet"
            base_results.to_parquet(out, index=False)
            log.info("Saved base predictions to %s", out)
        return

    # ── DPO model ──
    console.rule("[bold]DPO Model Analysis[/bold]")
    adapter_path = Path(args.adapter_path)

    if args.skip_merge:
        model_path = str(adapter_path)
    else:
        from run_eval_finetuned import merge_adapter
        model_path = str(merge_adapter(adapter_path))

    log.info("Loading DPO model: %s", model_path)
    dpo_model = load_judge_model(model_id=model_path, **vllm_kwargs)

    t0 = time.time()
    dpo_results = run_inference(dpo_model, df, args.batch_size)
    dpo_time = time.time() - t0
    log.info("DPO inference: %.1fs", dpo_time)

    report_prediction_distribution("DPO", dpo_results)
    report_per_class_confidence("DPO", dpo_results)
    report_confusion("DPO", dpo_results)

    dpo_metrics = compute_metrics(
        y_true=dpo_results["winner"].tolist(),
        y_pred=dpo_results["predicted_winner"].tolist(),
        domains=dpo_results["domain"].tolist(),
    )
    console.print(f"\nDPO accuracy: {dpo_metrics.get('accuracy', 'N/A'):.4f}")
    console.print(f"DPO macro-F1: {dpo_metrics.get('macro_f1', 'N/A'):.4f}")

    del dpo_model

    # ── Comparative analysis ──
    console.rule("[bold]Comparative Analysis[/bold]")
    report_pa_shift(base_results, dpo_results)
    console.print()
    report_error_migration(base_results, dpo_results)

    # Save predictions
    if args.save_predictions:
        out_base = cfg.RESULTS_DIR / f"logits_base_{args.eval_split}.parquet"
        out_dpo = cfg.RESULTS_DIR / f"logits_dpo_{args.eval_split}.parquet"
        base_results.to_parquet(out_base, index=False)
        dpo_results.to_parquet(out_dpo, index=False)
        log.info("Saved predictions to %s and %s", out_base, out_dpo)


if __name__ == "__main__":
    main()
