#!/usr/bin/env python3
"""
Dimension-ablation wrapper around the pairwise CheckEval judge.

Drops specified dimensions from the checklist, then runs the same
inference + pairwise scoring pipeline.  Results go to results/ablation/.

Usage
-----
# Version A: drop instruction_following
python src/run_ablation_dims.py --eval-split dev_600 \
    --drop-dims instruction_following

# Version B: drop instruction_following + stem
python src/run_ablation_dims.py --eval-split dev_600 \
    --drop-dims instruction_following stem
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import json
import logging
import time
import platform
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as cfg
from utils import (
    aggregate_checklist_score,
    build_checkeval_prompt,
    compare_checklists_pairwise,
    compute_dimension_accuracy,
    compute_metrics,
    compute_question_diagnostics,
    expected_question_count,
    generate_batch,
    load_checklists,
    load_eval_data,
    load_judge_model,
    parse_checkeval_output,
    save_results,
)
from run_checkeval_judge import run_checkeval_judge, TIE_DELTA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ABLATION_DIR = cfg.RESULTS_DIR / "ablation"


def main():
    parser = argparse.ArgumentParser(
        description="Dimension-ablation pairwise CheckEval judge",
    )
    parser.add_argument("--eval-split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"],
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"],
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"],
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_num_seqs"],
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--tie-delta", type=float, default=TIE_DELTA,
        help="Pairwise margin threshold for Tie (default: 0.05)",
    )
    parser.add_argument(
        "--na-policy", type=str, default="strict",
        choices=["strict", "as_no", "skip", "partial"],
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument(
        "--drop-dims",
        nargs="+",
        required=True,
        help="Dimension names to drop (e.g. instruction_following stem)",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Auto shutdown the system 60s after task completes",
    )
    parser.add_argument("--backend", type=str, default=None,
                        choices=["llamacpp", "vllm"],
                        help="Inference backend; defaults to cfg.INFERENCE_BACKEND.")
    args = parser.parse_args()

    # ── load & filter checklists ──
    checklists, definitions = load_checklists()
    drop_set = set(args.drop_dims)

    removed = {d: checklists.pop(d) for d in list(drop_set) if d in checklists}
    for d in drop_set:
        definitions.pop(d, None)

    if not removed:
        log.warning("None of %s found in checklists — running with ALL dims", drop_set)
    for d, qs in removed.items():
        log.info("DROPPED dimension '%s' (%d questions)", d, len(qs))

    total_q = sum(len(qs) for qs in checklists.values())
    log.info("Remaining: %d dimensions, %d total questions", len(checklists), total_q)
    for dim_name, questions in checklists.items():
        log.info("  %s: %s questions", dim_name, len(questions))

    # ── load data ──
    df = load_eval_data(args.eval_split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples)
        log.info("Capped to %s samples", len(df))

    # ── load model ──
    model = load_judge_model(
        model_id=args.model_id,
        backend=args.backend,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    # ── inference ──
    t0 = time.time()
    results, parsed_a_list, parsed_b_list = run_checkeval_judge(
        df,
        model,
        checklists,
        definitions,
        batch_size=args.batch_size,
        tie_delta=args.tie_delta,
        na_policy=args.na_policy,
        coverage_threshold=args.coverage_threshold,
    )
    elapsed = time.time() - t0
    log.info(
        "Inference complete in %.1fs  (%.2fs/sample, 2 calls each)",
        elapsed, elapsed / len(df),
    )

    # ── metrics ──
    metrics = compute_metrics(
        y_true=results["winner"].tolist(),
        y_pred=results["predicted_winner"].tolist(),
        domains=results["domain"].tolist(),
        scores_a=results["score_a"].tolist(),
        scores_b=results["score_b"].tolist(),
        n_answered_a=results["n_answered_a"].tolist(),
        n_answered_b=results["n_answered_b"].tolist(),
    )
    metrics["inference_time_s"] = elapsed
    metrics["samples_per_second"] = len(df) / elapsed
    metrics["model_id"] = args.model_id
    metrics["backend"] = "vllm"
    metrics["tensor_parallel_size"] = args.tensor_parallel_size
    metrics["max_model_len"] = args.max_model_len
    metrics["checklist_parse_rate"] = results["checklist_parsed"].mean()
    metrics["n_checklist_questions"] = total_q
    metrics["tie_delta"] = args.tie_delta
    metrics["na_policy"] = args.na_policy
    metrics["scoring_method"] = "pairwise_naaware"
    metrics["dropped_dims"] = sorted(drop_set)
    metrics["remaining_dims"] = sorted(checklists.keys())
    metrics["avg_pairwise_margin"] = results["pairwise_margin"].dropna().mean()
    metrics["avg_abs_pairwise_margin"] = results["pairwise_margin"].dropna().abs().mean()
    metrics["avg_n_aligned"] = results["pairwise_n_aligned"].dropna().mean()

    # N/A stats
    metrics["pct_a_with_na"] = (results["n_na_a"] > 0).mean()
    metrics["pct_b_with_na"] = (results["n_na_b"] > 0).mean()
    metrics["pct_either_with_na"] = ((results["n_na_a"] > 0) | (results["n_na_b"] > 0)).mean()
    metrics["avg_n_na_a"] = results["n_na_a"].mean()
    metrics["avg_n_na_b"] = results["n_na_b"].mean()

    # ── experiment naming & save ──
    split_tag = args.subset or args.eval_split
    drop_tag = "_".join(sorted(drop_set))
    experiment_name = f"ablation_drop_{drop_tag}_{split_tag}"
    save_results(results, metrics, experiment_name, output_dir=ABLATION_DIR)

    # ── per-question diagnostics ──
    q_diag = compute_question_diagnostics(
        parsed_a_list, parsed_b_list,
        results["domain"].tolist(), checklists,
    )
    if q_diag:
        import csv
        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        q_diag_path = ABLATION_DIR / f"{experiment_name}_question_diagnostics.csv"
        with open(q_diag_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=q_diag[0].keys())
            writer.writeheader()
            writer.writerows(q_diag)
        log.info("Saved question diagnostics -> %s", q_diag_path)

    # ── per-dimension accuracy ──
    dim_acc = compute_dimension_accuracy(
        results, checklists, parsed_a_list, parsed_b_list,
    )
    if dim_acc:
        dim_path = ABLATION_DIR / f"{experiment_name}_dimension_diagnostics.json"
        with open(dim_path, "w", encoding="utf-8") as f:
            json.dump(dim_acc, f, indent=2, ensure_ascii=False)
        log.info("Saved dimension diagnostics -> %s", dim_path)
        for dim_name, d in dim_acc.items():
            log.info("  [%s] acc=%.4f  avg_A=%.4f  avg_B=%.4f  na_rate=%.4f  n=%d",
                     dim_name, d["dimension_accuracy"],
                     d["avg_score_a"], d["avg_score_b"],
                     d["avg_na_rate"], d["n_samples"])

    log.info("Done.  experiment=%s", experiment_name)

    if args.shutdown:
        import os
        log.info("System will shutdown in 60 seconds.")
        if platform.system() == "Windows":
            os.system("shutdown /s /t 60")
        else:
            os.system("sudo shutdown -h +1")


if __name__ == "__main__":
    main()
