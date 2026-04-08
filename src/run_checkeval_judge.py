#!/usr/bin/env python3
"""
Pairwise CheckEval Qwen Judge (NA-aware).

For each sample, constructs two independent pointwise prompts (one for
Response A, one for Response B), then compares per-question labels via
pairwise margin to determine a winner.  Aggregate scores are retained
as diagnostics only.
"""

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
    build_question_index,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Tie threshold: if |score_A - score_B| <= TIE_DELTA, predict Tie
TIE_DELTA = 0.05  # pairwise margin: ~1 question tolerance


def run_checkeval_judge(
    df: pd.DataFrame,
    model,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    batch_size,
    tie_delta: float = TIE_DELTA,
    na_policy: str = "strict",
    coverage_threshold: float = 0.8,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Run pointwise CheckEval judge: evaluate A and B independently.

    Returns (results_df, parsed_a_list, parsed_b_list) so callers can
    run per-question / per-dimension diagnostics on the parsed data.
    """
    results = df.copy()

    # ── build two prompts per sample ──
    messages_a: list[list[dict]] = []
    messages_b: list[list[dict]] = []
    for _, row in df.iterrows():
        prompt_a = build_checkeval_prompt(row, checklists, definitions, side="A")
        prompt_b = build_checkeval_prompt(row, checklists, definitions, side="B")
        messages_a.append([{"role": "user", "content": prompt_a}])
        messages_b.append([{"role": "user", "content": prompt_b}])

    # ── batch inference: A prompts then B prompts ──
    all_messages = messages_a + messages_b
    all_raw = generate_batch(
        model,
        all_messages,
        batch_size=batch_size,
        max_new_tokens=2048,
    )
    n = len(df)
    raw_a = all_raw[:n]
    raw_b = all_raw[n:]

    # ── parse and compare ──
    predicted_winners = []
    scores_a_list = []
    scores_b_list = []
    n_yes_a_list = []
    n_yes_b_list = []
    n_answered_a_list = []
    n_answered_b_list = []
    n_na_a_list = []
    n_na_b_list = []
    na_qnums_a_list = []
    na_qnums_b_list = []
    stop_reason_a_list = []
    stop_reason_b_list = []
    score_gaps = []
    parse_successes = []
    pairwise_margins = []
    pairwise_n_aligned_list = []
    expected_n_list = []
    all_parsed_a = []
    all_parsed_b = []

    for (_, row), ra, rb in tqdm(zip(df.iterrows(), raw_a, raw_b), total=n, desc="CheckEval Parse"):
        expected_n = expected_question_count(row["domain"], checklists)

        parsed_a = parse_checkeval_output(ra, expected_n=expected_n)
        parsed_b = parse_checkeval_output(rb, expected_n=expected_n)
        all_parsed_a.append(parsed_a)
        all_parsed_b.append(parsed_b)

        agg_a = aggregate_checklist_score(
            parsed_a, na_policy=na_policy,
            coverage_threshold=coverage_threshold, expected_n=expected_n,
        )
        agg_b = aggregate_checklist_score(
            parsed_b, na_policy=na_policy,
            coverage_threshold=coverage_threshold, expected_n=expected_n,
        )

        ok_a = agg_a is not None
        ok_b = agg_b is not None

        score_a = agg_a["score"] if ok_a else None
        score_b = agg_b["score"] if ok_b else None

        scores_a_list.append(score_a)
        scores_b_list.append(score_b)
        n_yes_a_list.append(agg_a["n_yes"] if ok_a else None)
        n_yes_b_list.append(agg_b["n_yes"] if ok_b else None)
        n_answered_a_list.append(agg_a["n_answered"] if ok_a else None)
        n_answered_b_list.append(agg_b["n_answered"] if ok_b else None)
        n_na_a_list.append(parsed_a.get("n_na", 0))
        n_na_b_list.append(parsed_b.get("n_na", 0))
        na_qnums_a_list.append(parsed_a.get("na_qnums", []))
        na_qnums_b_list.append(parsed_b.get("na_qnums", []))
        stop_reason_a_list.append(parsed_a.get("stop_reason"))
        stop_reason_b_list.append(parsed_b.get("stop_reason"))
        expected_n_list.append(expected_n)

        # ── pairwise margin (authoritative winner) ──
        pw = compare_checklists_pairwise(parsed_a, parsed_b, expected_n, tie_delta)
        if pw is not None:
            winner = pw["winner"]
            margin = pw["margin"]
            n_aligned = pw["n_aligned"]
        else:
            winner = None
            margin = None
            n_aligned = None

        parse_successes.append(pw is not None)
        pairwise_margins.append(margin)
        pairwise_n_aligned_list.append(n_aligned)
        predicted_winners.append(winner)

        # diagnostic: pointwise score gap (no longer drives winner)
        gap = (score_a - score_b) if (score_a is not None and score_b is not None) else None
        score_gaps.append(gap)

    # ── store results ──
    results["raw_output_a"] = raw_a
    results["raw_output_b"] = raw_b
    results["score_a"] = scores_a_list
    results["score_b"] = scores_b_list
    results["n_yes_a"] = n_yes_a_list
    results["n_yes_b"] = n_yes_b_list
    results["n_answered_a"] = n_answered_a_list
    results["n_answered_b"] = n_answered_b_list
    results["n_na_a"] = n_na_a_list
    results["n_na_b"] = n_na_b_list
    results["na_qnums_a"] = na_qnums_a_list
    results["na_qnums_b"] = na_qnums_b_list
    results["stop_reason_a"] = stop_reason_a_list
    results["stop_reason_b"] = stop_reason_b_list
    results["score_gap"] = score_gaps
    results["pairwise_margin"] = pairwise_margins
    results["pairwise_n_aligned"] = pairwise_n_aligned_list
    results["predicted_winner"] = predicted_winners
    results["checklist_parsed"] = parse_successes
    results["expected_n_questions"] = expected_n_list

    # ── logging: parse rate ──
    n_parsed = sum(parse_successes)
    log.info("Checklist parse rate: %s/%s (%.1f%%)  [na_policy=%s]",
             n_parsed, n, 100 * n_parsed / n, na_policy)

    # ── logging: predictions ──
    n_a = sum(1 for w in predicted_winners if w == "A")
    n_b = sum(1 for w in predicted_winners if w == "B")
    n_tie = sum(1 for w in predicted_winners if w == "Tie")
    n_none = sum(1 for w in predicted_winners if w is None)
    log.info("Predictions: A=%d, B=%d, Tie=%d, Unparsed=%d", n_a, n_b, n_tie, n_none)

    # ── logging: scores ──
    valid_sa = [s for s in scores_a_list if s is not None]
    valid_sb = [s for s in scores_b_list if s is not None]
    if valid_sa:
        log.info("Avg score A: %.4f  |  Avg score B: %.4f",
                 sum(valid_sa) / len(valid_sa), sum(valid_sb) / len(valid_sb))
    valid_answered_a = [x for x in n_answered_a_list if x is not None]
    valid_answered_b = [x for x in n_answered_b_list if x is not None]
    if valid_answered_a:
        log.info("Avg questions answered  A: %.1f  |  B: %.1f",
                 sum(valid_answered_a) / len(valid_answered_a),
                 sum(valid_answered_b) / len(valid_answered_b))
    valid_gaps = [g for g in score_gaps if g is not None]
    if valid_gaps:
        log.info("Score gap  mean=%.4f  |abs| mean=%.4f",
                 sum(valid_gaps) / len(valid_gaps),
                 sum(abs(g) for g in valid_gaps) / len(valid_gaps))

    valid_margins = [m for m in pairwise_margins if m is not None]
    if valid_margins:
        log.info("Pairwise margin  mean=%.4f  |abs| mean=%.4f",
                 sum(valid_margins) / len(valid_margins),
                 sum(abs(m) for m in valid_margins) / len(valid_margins))
    valid_aligned = [a for a in pairwise_n_aligned_list if a is not None]
    if valid_aligned:
        log.info("Avg n_aligned: %.1f / expected_n", sum(valid_aligned) / len(valid_aligned))

    # ── logging: N/A summary ──
    n_with_na_a = sum(1 for x in n_na_a_list if x > 0)
    n_with_na_b = sum(1 for x in n_na_b_list if x > 0)
    n_with_na_either = sum(1 for a, b in zip(n_na_a_list, n_na_b_list) if a > 0 or b > 0)
    log.info("N/A:  A=%d/%d (%.1f%%)  B=%d/%d (%.1f%%)  either=%d/%d (%.1f%%)",
             n_with_na_a, n, 100 * n_with_na_a / n,
             n_with_na_b, n, 100 * n_with_na_b / n,
             n_with_na_either, n, 100 * n_with_na_either / n)
    if n_with_na_either > 0:
        avg_na_a = sum(n_na_a_list) / n
        avg_na_b = sum(n_na_b_list) / n
        log.info("Avg N/A per sample  A: %.2f  B: %.2f", avg_na_a, avg_na_b)

        # top N/A question IDs
        na_counter = Counter()
        for qnums in na_qnums_a_list + na_qnums_b_list:
            na_counter.update(qnums)
        top_na = na_counter.most_common(10)
        log.info("Top N/A question IDs: %s",
                 "  ".join(f"Q{q}={c}" for q, c in top_na))

    # ── logging: stop reasons ──
    sr_a = Counter(stop_reason_a_list)
    sr_b = Counter(stop_reason_b_list)
    log.info("Stop reasons A: %s", dict(sr_a))
    log.info("Stop reasons B: %s", dict(sr_b))

    # ── logging: error categorization ──
    error_cats = []
    for w_pred, w_true, ok in zip(predicted_winners, df["winner"].tolist(), parse_successes):
        if not ok or w_pred is None:
            error_cats.append("parse_failure")
        elif w_pred == "Tie":
            error_cats.append("tie")
        elif w_pred == w_true:
            error_cats.append("correct")
        else:
            error_cats.append("wrong_winner")
    results["error_category"] = error_cats
    cat_counts = Counter(error_cats)
    log.info("Error categories: %s", dict(cat_counts))

    return results, all_parsed_a, all_parsed_b


def main():
    parser = argparse.ArgumentParser(description="Pairwise CheckEval Qwen Judge (NA-aware)")
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Use a training subset (e.g. debug_5k)",
    )
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
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Auto shutdown the system 60s after task completes",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--tie-delta",
        type=float,
        default=TIE_DELTA,
        help="Pairwise margin threshold for Tie (default: 0.05)",
    )
    parser.add_argument(
        "--na-policy",
        type=str,
        default="skip",
        choices=["strict", "as_no", "skip", "partial"],
        help="How to handle N/A answers (default: strict)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Min n_answered/expected_n for 'partial' policy (default: 0.8)",
    )
    parser.add_argument(
        "--checklists-dir",
        type=Path,
        default=None,
        help="Override checklist bank directory (e.g. checklists/v2). "
             "Defaults to cfg.CHECKLISTS_DIR.",
    )
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the experiment_name, so runs on "
             "different banks don't overwrite each other (e.g. '_v2').",
    )
    args = parser.parse_args()

    if args.checklists_dir is not None:
        checklists, definitions = load_checklists(args.checklists_dir)
    else:
        checklists, definitions = load_checklists()
    total_q = sum(len(questions) for questions in checklists.values())
    log.info("Loaded %s dimensions, %s total questions", len(checklists), total_q)
    for dim_name, questions in checklists.items():
        log.info("  %s: %s questions", dim_name, len(questions))

    df = load_eval_data(args.eval_split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples)
        log.info("Capped to %s samples", len(df))

    model = load_judge_model(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

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
        elapsed,
        elapsed / len(df),
    )

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
    metrics["avg_pairwise_margin"] = results["pairwise_margin"].dropna().mean()
    metrics["avg_abs_pairwise_margin"] = results["pairwise_margin"].dropna().abs().mean()
    metrics["avg_n_aligned"] = results["pairwise_n_aligned"].dropna().mean()

    # ── N/A stats for metrics JSON ──
    n_total = len(results)
    metrics["pct_a_with_na"] = (results["n_na_a"] > 0).mean()
    metrics["pct_b_with_na"] = (results["n_na_b"] > 0).mean()
    metrics["pct_either_with_na"] = ((results["n_na_a"] > 0) | (results["n_na_b"] > 0)).mean()
    metrics["avg_n_na_a"] = results["n_na_a"].mean()
    metrics["avg_n_na_b"] = results["n_na_b"].mean()

    # top N/A qids by domain
    top_na_by_domain = {}
    for domain in results["domain"].unique():
        mask = results["domain"] == domain
        na_counter = Counter()
        for qnums in results.loc[mask, "na_qnums_a"]:
            na_counter.update(qnums)
        for qnums in results.loc[mask, "na_qnums_b"]:
            na_counter.update(qnums)
        top_na_by_domain[domain] = dict(na_counter.most_common(15))
    metrics["top_na_qids_by_domain"] = top_na_by_domain

    split_tag = args.subset or args.eval_split
    experiment_name = f"checkeval_pairwise_naaware_{split_tag}{args.experiment_suffix}"
    save_results(results, metrics, experiment_name)

    # ── per-question diagnostics ──
    q_diag = compute_question_diagnostics(
        parsed_a_list, parsed_b_list,
        results["domain"].tolist(), checklists,
    )
    if q_diag:
        import csv
        q_diag_path = cfg.RESULTS_DIR / f"{experiment_name}_question_diagnostics.csv"
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
        dim_path = cfg.RESULTS_DIR / f"{experiment_name}_dimension_diagnostics.json"
        with open(dim_path, "w", encoding="utf-8") as f:
            json.dump(dim_acc, f, indent=2, ensure_ascii=False)
        log.info("Saved dimension diagnostics -> %s", dim_path)
        for dim_name, d in dim_acc.items():
            log.info("  [%s] acc=%.4f  avg_A=%.4f  avg_B=%.4f  na_rate=%.4f  n=%d",
                     dim_name, d["dimension_accuracy"],
                     d["avg_score_a"], d["avg_score_b"],
                     d["avg_na_rate"], d["n_samples"])

    log.info("Done.")

    if args.shutdown:
        import os
        log.info("System will shutdown in 60 seconds. Run 'shutdown /a' (Windows) or 'shutdown -c' (Linux) to cancel.")
        if platform.system() == "Windows":
            os.system("shutdown /s /t 60")
        else:
            os.system("sudo shutdown -h +1")

if __name__ == "__main__":
    main()
