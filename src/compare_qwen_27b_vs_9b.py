#!/usr/bin/env python3
"""
Compare Qwen3.5-27B (4-bit quantized) vs Qwen3.5-9B on dev_600 checklist evaluation.

Goal: determine the more stable / accurate model to use as teacher.

Usage:
    # Run both models sequentially
    python src/compare_qwen_27b_vs_9b.py

    # Run only 27B (if 9B results already cached)
    python src/compare_qwen_27b_vs_9b.py --only 27b

    # Quick test with fewer samples
    python src/compare_qwen_27b_vs_9b.py --max-samples 50

    # Custom model path for 27B
    python src/compare_qwen_27b_vs_9b.py --model-27b /path/to/Qwen3.5-27B-AWQ
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from utils import (
    build_checkeval_prompt,
    compare_checklists_pairwise,
    expected_question_count,
    generate_batch,
    load_checklists,
    load_judge_model,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

RESULTS_DIR = cfg.RESULTS_DIR / "teacher_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────── inference ─────────────────────────


def run_qwen_checklist(
    prompts: list[str],
    model_id: str,
    batch_size: int = 16,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.92,
    max_num_seqs: int = 16,
    quantization: str | None = None,
) -> list[str]:
    """Run checklist evaluation prompts through a Qwen model via vLLM."""
    log.info("Loading model: %s (quantization=%s)", model_id, quantization)

    # Build engine kwargs, allowing quantization override
    extra_kwargs = {}
    if quantization:
        extra_kwargs["quantization"] = quantization

    from vllm import LLM, SamplingParams

    engine_kwargs = dict(cfg.VLLM_ENGINE_KWARGS)
    engine_kwargs.update({
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": True,
        "max_num_batched_tokens": max_model_len,
    })
    engine_kwargs.update(extra_kwargs)

    t0 = time.time()
    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        **engine_kwargs,
    )
    log.info("Model loaded in %.1fs", time.time() - t0)

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    outputs = generate_batch(llm, messages_list, batch_size=batch_size, max_new_tokens=2048)

    # Free GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    return outputs


# ────────────────────────── evaluation ─────────────────────────


def evaluate_teacher(
    raw_outputs_a: list[str],
    raw_outputs_b: list[str],
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    label: str,
) -> dict:
    """Evaluate teacher model outputs against ground truth."""
    n = len(df)
    assert len(raw_outputs_a) == n and len(raw_outputs_b) == n

    predicted_winners = []
    parse_ok = 0
    total_na_a, total_na_b = 0, 0
    chosen_scores, rejected_scores = [], []
    all_parsed_a, all_parsed_b = [], []

    for i, (_, row) in enumerate(df.iterrows()):
        domain = row["domain"]
        winner = row["winner"]
        n_q = expected_question_count(domain, checklists)

        parsed_a = parse_checkeval_output(raw_outputs_a[i], expected_n=n_q)
        parsed_b = parse_checkeval_output(raw_outputs_b[i], expected_n=n_q)
        all_parsed_a.append(parsed_a)
        all_parsed_b.append(parsed_b)

        total_na_a += parsed_a.get("n_na", 0)
        total_na_b += parsed_b.get("n_na", 0)

        pw = compare_checklists_pairwise(parsed_a, parsed_b, n_q, tie_delta=0.05)
        if pw is not None:
            predicted_winners.append(pw["winner"])
            parse_ok += 1

            score_a = parsed_a.get("score", 0)
            score_b = parsed_b.get("score", 0)
            if winner == "A":
                chosen_scores.append(score_a)
                rejected_scores.append(score_b)
            elif winner == "B":
                chosen_scores.append(score_b)
                rejected_scores.append(score_a)
        else:
            predicted_winners.append(None)

    # Accuracy (excluding unparseable and ties)
    valid_mask = [
        i for i, (pw, gt) in enumerate(zip(predicted_winners, df["winner"]))
        if pw is not None and pw != "Tie"
    ]
    gt_valid = [df.iloc[i]["winner"] for i in valid_mask]
    pred_valid = [predicted_winners[i] for i in valid_mask]
    accuracy = (
        sum(p == g for p, g in zip(pred_valid, gt_valid)) / len(gt_valid)
        if gt_valid else 0
    )

    # Full accuracy (including ties as wrong)
    full_valid_mask = [i for i, pw in enumerate(predicted_winners) if pw is not None]
    gt_full = [df.iloc[i]["winner"] for i in full_valid_mask]
    pred_full = [predicted_winners[i] for i in full_valid_mask]
    accuracy_full = (
        sum(p == g for p, g in zip(pred_full, gt_full)) / len(gt_full)
        if gt_full else 0
    )

    n_ties = sum(1 for pw in predicted_winners if pw == "Tie")
    n_unparseable = sum(1 for pw in predicted_winners if pw is None)

    parse_a_ok = sum(1 for p in all_parsed_a if not p.get("_raw_fallback", False))
    parse_b_ok = sum(1 for p in all_parsed_b if not p.get("_raw_fallback", False))

    # Per-domain accuracy
    per_domain = {}
    for domain in sorted(df["domain"].unique()):
        domain_idx = [i for i in valid_mask if df.iloc[i]["domain"] == domain]
        if domain_idx:
            dgt = [df.iloc[i]["winner"] for i in domain_idx]
            dpred = [predicted_winners[i] for i in domain_idx]
            per_domain[domain] = {
                "n": len(domain_idx),
                "accuracy": sum(p == g for p, g in zip(dpred, dgt)) / len(dgt),
            }

    return {
        "label": label,
        "n_total": n,
        "parse_rate_a": parse_a_ok / n,
        "parse_rate_b": parse_b_ok / n,
        "pairwise_parse_rate": parse_ok / n,
        "n_ties": n_ties,
        "n_unparseable": n_unparseable,
        "accuracy_excl_ties": accuracy,
        "accuracy_incl_ties": accuracy_full,
        "n_evaluated": len(gt_valid),
        "avg_na_a": total_na_a / n,
        "avg_na_b": total_na_b / n,
        "avg_chosen_score": float(np.mean(chosen_scores)) if chosen_scores else None,
        "avg_rejected_score": float(np.mean(rejected_scores)) if rejected_scores else None,
        "score_gap": (
            float(np.mean(chosen_scores) - np.mean(rejected_scores))
            if chosen_scores else None
        ),
        "per_domain": per_domain,
    }


# ────────────────────────── display ────────────────────────────


def print_comparison(results: list[dict]) -> None:
    table = Table(title="Qwen 27B-4bit vs 9B — Checklist Teacher Comparison")
    table.add_column("Metric", style="bold")
    for r in results:
        table.add_column(r["label"], justify="right")

    metrics = [
        ("Samples", "n_total"),
        ("Parse rate (A)", "parse_rate_a"),
        ("Parse rate (B)", "parse_rate_b"),
        ("Pairwise parse rate", "pairwise_parse_rate"),
        ("Ties", "n_ties"),
        ("Unparseable", "n_unparseable"),
        ("Accuracy (excl ties)", "accuracy_excl_ties"),
        ("Accuracy (incl ties)", "accuracy_incl_ties"),
        ("# Evaluated", "n_evaluated"),
        ("Avg N/A (A)", "avg_na_a"),
        ("Avg N/A (B)", "avg_na_b"),
        ("Avg chosen score", "avg_chosen_score"),
        ("Avg rejected score", "avg_rejected_score"),
        ("Score gap (chosen-rej)", "score_gap"),
        ("Runtime (s)", "time_s"),
    ]

    for name, key in metrics:
        vals = []
        for r in results:
            v = r.get(key)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        table.add_row(name, *vals)

    # Per-domain accuracy
    all_domains = set()
    for r in results:
        all_domains.update(r.get("per_domain", {}).keys())
    for domain in sorted(all_domains):
        vals = []
        for r in results:
            d = r.get("per_domain", {}).get(domain)
            if d:
                vals.append(f"{d['accuracy']:.4f} (n={d['n']})")
            else:
                vals.append("—")
        table.add_row(f"  {domain}", *vals)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Qwen3.5-27B-4bit vs Qwen3.5-9B as teacher model"
    )
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--only", type=str, default=None, choices=["27b", "9b"],
                        help="Run only one model (use cached results for the other)")
    parser.add_argument(
        "--model-27b", type=str,
        default="Qwen/Qwen3.5-27B-GPTQ-Int4",
        help="Model ID or local path for 27B 4-bit model "
             "(default: Qwen/Qwen3.5-27B-GPTQ-Int4 from HuggingFace)",
    )
    parser.add_argument(
        "--model-9b", type=str,
        default=str(cfg.JUDGE_MODEL_ID),
        help="Model ID or local path for 9B model",
    )
    parser.add_argument("--quantization-27b", type=str, default=None,
                        help="Quantization method for 27B (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len-27b", type=int, default=16384,
                        help="Max context length for 27B model")
    parser.add_argument("--gpu-mem-27b", type=float, default=0.95,
                        help="GPU memory utilization for 27B (default: 0.95)")
    parser.add_argument("--max-num-seqs-27b", type=int, default=8,
                        help="Max concurrent sequences for 27B (lower = less OOM risk)")
    parser.add_argument("--save-raw", action="store_true",
                        help="Save raw model outputs for analysis")
    args = parser.parse_args()

    # ── Load checklists & data ──
    checklists, definitions = load_checklists(cfg.CHECKLISTS_DIR)
    log.info("Checklists: %d dims, %d total questions",
             len(checklists), sum(len(q) for q in checklists.values()))

    split_path = cfg.SPLITS_DIR / f"{args.split}.parquet"
    if not split_path.exists():
        log.error("Eval data not found: %s", split_path)
        return

    df = pd.read_parquet(split_path)
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    log.info("Loaded %d samples from %s", len(df), split_path.name)

    # ── Build prompts (2 per sample: side A and side B) ──
    prompts_a, prompts_b = [], []
    for _, row in df.iterrows():
        prompts_a.append(build_checkeval_prompt(row, checklists, definitions, side="A"))
        prompts_b.append(build_checkeval_prompt(row, checklists, definitions, side="B"))

    all_prompts = prompts_a + prompts_b
    n = len(df)
    all_results = []

    # ── Qwen3.5-27B 4-bit ──
    if args.only != "9b":
        log.info("=" * 60)
        log.info("Running Qwen3.5-27B-4bit: %s", args.model_27b)
        log.info("  max_model_len=%d  gpu_mem=%.2f  max_num_seqs=%d",
                 args.max_model_len_27b, args.gpu_mem_27b, args.max_num_seqs_27b)
        log.info("=" * 60)

        t0 = time.time()
        outputs_27b = run_qwen_checklist(
            all_prompts,
            model_id=args.model_27b,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len_27b,
            gpu_memory_utilization=args.gpu_mem_27b,
            max_num_seqs=args.max_num_seqs_27b,
            quantization=args.quantization_27b,
        )
        time_27b = time.time() - t0
        log.info("27B done in %.1fs (%.2fs/sample)", time_27b, time_27b / n)

        eval_27b = evaluate_teacher(
            outputs_27b[:n], outputs_27b[n:], df, checklists,
            label="Qwen3.5-27B-4bit",
        )
        eval_27b["time_s"] = time_27b
        eval_27b["model_id"] = args.model_27b
        all_results.append(eval_27b)

        if args.save_raw:
            raw_path = RESULTS_DIR / f"qwen27b_4bit_{args.split}_raw.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": args.model_27b,
                    "outputs_a": outputs_27b[:n],
                    "outputs_b": outputs_27b[n:],
                }, f, ensure_ascii=False)
            log.info("Saved raw 27B outputs to %s", raw_path)

    # ── Qwen3.5-9B ──
    if args.only != "27b":
        log.info("=" * 60)
        log.info("Running Qwen3.5-9B: %s", args.model_9b)
        log.info("=" * 60)

        t0 = time.time()
        outputs_9b = run_qwen_checklist(
            all_prompts,
            model_id=args.model_9b,
            batch_size=args.batch_size,
        )
        time_9b = time.time() - t0
        log.info("9B done in %.1fs (%.2fs/sample)", time_9b, time_9b / n)

        eval_9b = evaluate_teacher(
            outputs_9b[:n], outputs_9b[n:], df, checklists,
            label="Qwen3.5-9B",
        )
        eval_9b["time_s"] = time_9b
        eval_9b["model_id"] = args.model_9b
        all_results.append(eval_9b)

        if args.save_raw:
            raw_path = RESULTS_DIR / f"qwen9b_{args.split}_raw.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": args.model_9b,
                    "outputs_a": outputs_9b[:n],
                    "outputs_b": outputs_9b[n:],
                }, f, ensure_ascii=False)
            log.info("Saved raw 9B outputs to %s", raw_path)

    # ── Compare & recommend ──
    if all_results:
        print_comparison(all_results)

        # Rank by: accuracy (excl ties) > parse rate > score gap
        ranked = sorted(
            all_results,
            key=lambda r: (
                r.get("accuracy_excl_ties", -1),
                r.get("pairwise_parse_rate", -1),
                r.get("score_gap", float("-inf")) if r.get("score_gap") is not None else float("-inf"),
            ),
            reverse=True,
        )
        best = ranked[0]

        console.print()
        console.print(f"[bold green]Recommended teacher model: {best['label']}[/bold green]")
        console.print(
            f"  accuracy_excl_ties = {best.get('accuracy_excl_ties', 0):.4f}  |  "
            f"parse_rate = {best.get('pairwise_parse_rate', 0):.4f}  |  "
            f"score_gap = {best.get('score_gap', 0) or 0:.4f}"
        )

        # Save comparison
        out_path = RESULTS_DIR / f"qwen_27b_vs_9b_{args.split}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info("Saved comparison metrics to %s", out_path)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
