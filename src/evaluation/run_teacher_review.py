#!/usr/bin/env python3
"""
Teacher-model review: run CheckEval judge on ≤100 samples, then select
50 review examples (25 wrong + 25 correct) for the Streamlit review app.

Usage:
    python src/evaluation/run_teacher_review.py [--eval-split dev] [--max-samples 100]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import random
import time
from pathlib import Path

import pandas as pd

import config as cfg
from utils import (
    aggregate_checklist_score,
    build_checkeval_prompt,
    compare_checklists_pairwise,
    expected_question_count,
    generate_batch,
    load_checklists,
    load_eval_data,
    load_judge_model,
    parse_checkeval_output,
    save_results,
    compute_metrics,
)
from run_checkeval_judge import run_checkeval_judge, TIE_DELTA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def select_review_samples(
    results: pd.DataFrame,
    n_wrong: int = 25,
    n_correct: int = 25,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """Pick 25 wrong + 25 correct predictions (or as many as available)."""
    rng = random.Random(seed)

    wrong_mask = (
        results["predicted_winner"].isin(["A", "B"])
        & (results["predicted_winner"] != results["winner"])
    )
    correct_mask = (
        results["predicted_winner"].isin(["A", "B"])
        & (results["predicted_winner"] == results["winner"])
    )

    wrong_idx = results.index[wrong_mask].tolist()
    correct_idx = results.index[correct_mask].tolist()
    rng.shuffle(wrong_idx)
    rng.shuffle(correct_idx)

    chosen_wrong = wrong_idx[:n_wrong]
    chosen_correct = correct_idx[:n_correct]

    log.info(
        "Review set: %d wrong (wanted %d), %d correct (wanted %d)",
        len(chosen_wrong), n_wrong, len(chosen_correct), n_correct,
    )

    review = pd.concat([
        results.loc[chosen_wrong].assign(_review_split="wrong"),
        results.loc[chosen_correct].assign(_review_split="correct"),
    ]).reset_index(drop=True)
    return review


def main():
    parser = argparse.ArgumentParser(description="Teacher-model review runner")
    parser.add_argument("--eval-split", type=str, default="dev")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--tie-delta", type=float, default=TIE_DELTA)
    parser.add_argument(
        "--na-policy",
        type=str,
        default="skip",
        choices=["strict", "as_no", "skip", "partial"],
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument(
        "--checklists-dir",
        type=Path,
        default=None,
        help="Override checklist directory. Defaults to cfg.CHECKLISTS_DIR.",
    )
    parser.add_argument("--n-wrong", type=int, default=25)
    parser.add_argument("--n-correct", type=int, default=25)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    checklists_dir = args.checklists_dir or cfg.CHECKLISTS_DIR
    checklists, definitions = load_checklists(checklists_dir)
    total_q = sum(len(q) for q in checklists.values())
    log.info("Loaded %d dimensions, %d total questions", len(checklists), total_q)

    df = load_eval_data(args.eval_split, args.subset)
    if len(df) > args.max_samples:
        # Stratified sample by domain so all domains are represented
        df = (
            df.groupby("domain", group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, round(args.max_samples * len(g) / len(df))),
                random_state=args.seed,
            ))
            .sample(frac=1, random_state=args.seed)  # shuffle
            .head(args.max_samples)
            .reset_index(drop=True)
        )
    log.info("Using %d samples (domains: %s)", len(df), df["domain"].value_counts().to_dict())

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
    log.info("Inference done in %.1fs", elapsed)

    # ── attach prompts so the review app can display them ──
    prompts_a, prompts_b = [], []
    for _, row in df.iterrows():
        prompts_a.append(build_checkeval_prompt(row, checklists, definitions, side="A"))
        prompts_b.append(build_checkeval_prompt(row, checklists, definitions, side="B"))
    results["prompt_a"] = prompts_a
    results["prompt_b"] = prompts_b

    # ── attach serialised parsed dicts for the review app ──
    results["parsed_a_json"] = [json.dumps(p, ensure_ascii=False) for p in parsed_a_list]
    results["parsed_b_json"] = [json.dumps(p, ensure_ascii=False) for p in parsed_b_list]

    split_tag = args.subset or args.eval_split
    model_tag = Path(str(args.model_id)).name.replace(" ", "_")
    experiment_name = (
        f"teacher_review_{split_tag}_n{len(df)}"
        f"_{model_tag}"
        f"_na{args.na_policy}"
        f"_td{args.tie_delta}"
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
    metrics["model_id"] = str(args.model_id)
    metrics["na_policy"] = args.na_policy
    metrics["tie_delta"] = args.tie_delta
    save_results(results, metrics, experiment_name)

    # ── select review subset ──
    review = select_review_samples(
        results,
        n_wrong=args.n_wrong,
        n_correct=args.n_correct,
        seed=args.seed,
    )
    review_path = cfg.RESULTS_DIR / f"{experiment_name}_review_samples.parquet"
    review.to_parquet(review_path, index=False)
    log.info("Saved review samples (%d rows) -> %s", len(review), review_path)
    log.info(
        "Launch review app:  streamlit run src/evaluation/review_app.py -- --results %s",
        review_path,
    )


if __name__ == "__main__":
    main()
