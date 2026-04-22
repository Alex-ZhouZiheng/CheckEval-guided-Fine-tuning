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
import math
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


def stratified_sample_by_domain(
    df: pd.DataFrame,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exactly ``max_samples`` rows while preserving domain mix.

    This avoids the brittle ``groupby.apply(...).reset_index(...)`` pattern,
    which can drop grouping columns on some pandas versions.
    """
    if len(df) <= max_samples:
        return df.copy().reset_index(drop=True)

    rng = random.Random(seed)
    domain_counts = df["domain"].value_counts()
    total = len(df)

    base_alloc: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for domain_name, count in domain_counts.items():
        exact = max_samples * count / total
        take = min(count, max(1, math.floor(exact)))
        base_alloc[domain_name] = take
        remainders.append((exact - math.floor(exact), domain_name))

    allocated = sum(base_alloc.values())
    if allocated > max_samples:
        for _, domain_name in sorted(remainders):
            if allocated <= max_samples:
                break
            if base_alloc[domain_name] > 1:
                base_alloc[domain_name] -= 1
                allocated -= 1
    elif allocated < max_samples:
        for _, domain_name in sorted(remainders, reverse=True):
            if allocated >= max_samples:
                break
            available = int(domain_counts[domain_name]) - base_alloc[domain_name]
            if available > 0:
                base_alloc[domain_name] += 1
                allocated += 1

    sampled_parts: list[pd.DataFrame] = []
    for domain_name, take in base_alloc.items():
        domain_df = df[df["domain"] == domain_name]
        sampled_parts.append(domain_df.sample(n=take, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    shuffle_seed = rng.randint(0, 10**9)
    sampled = sampled.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    return sampled


def _attach_individual_preference_from_raw(results: pd.DataFrame) -> tuple[pd.DataFrame, int, str | None]:
    """Attach raw HelpSteer3 ``individual_preference`` if raw parquets exist."""
    import hashlib as _hashlib

    def _ctx_to_text(ctx) -> str:
        return "\n\n".join(f"[{t['role']}]\n{t['content']}" for t in ctx)

    def _pid(text: str) -> str:
        return _hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    needed = set(results["prompt_id"].dropna())
    raw_dir = cfg.DATA_DIR / "raw"
    best_source = None
    best_filled = 0
    out = results.copy()

    for fname in ("helpsteer3_train.parquet", "helpsteer3_test.parquet"):
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_parquet(fpath, columns=["context", "individual_preference"])
        if "individual_preference" not in raw.columns:
            continue
        raw_pids = raw["context"].apply(lambda c: _pid(_ctx_to_text(c)))
        mask = raw_pids.isin(needed)
        lookup = dict(zip(raw_pids[mask], raw.loc[mask, "individual_preference"]))
        candidate = out["prompt_id"].map(lookup)
        filled = int(candidate.notna().sum())
        if filled > best_filled:
            out["individual_preference"] = candidate
            best_filled = filled
            best_source = fname
        if filled == len(out):
            break

    return out, best_filled, best_source


def _attach_individual_preference_from_reasoning_slice(
    results: pd.DataFrame,
    split_tag: str,
) -> tuple[pd.DataFrame, int, str | None]:
    """Fallback: synthesize a single reviewer record from with_reason parquet."""
    candidates = [
        cfg.WITH_REASON_DIR / f"{split_tag}_reasoning.parquet",
    ]
    if split_tag != "dev":
        candidates.append(cfg.WITH_REASON_DIR / "dev_reasoning.parquet")

    out = results.copy()
    join_cols = ["domain", "context", "response_a", "response_b", "winner"]

    for path in candidates:
        if not path.exists():
            continue
        reason_df = pd.read_parquet(path)
        if "swap_flag" in reason_df.columns:
            reason_df = reason_df[reason_df["swap_flag"] == False].copy()
        required = set(join_cols) | {"reasoning_text", "feedback_a_text", "feedback_b_text"}
        if not required.issubset(reason_df.columns):
            continue

        synth = reason_df[list(join_cols) + ["reasoning_text", "feedback_a_text", "feedback_b_text"]].copy()
        synth["individual_preference"] = synth.apply(
            lambda row: json.dumps(
                [{
                    "score": None,
                    "reasoning": row.get("reasoning_text", ""),
                    "feedback1": row.get("feedback_a_text", ""),
                    "feedback2": row.get("feedback_b_text", ""),
                }],
                ensure_ascii=False,
            ),
            axis=1,
        )
        synth = synth[join_cols + ["individual_preference"]].drop_duplicates(subset=join_cols, keep="first")
        merged = out.merge(synth, on=join_cols, how="left", suffixes=("", "_reason"))
        filled = int(merged["individual_preference"].notna().sum())
        if filled:
            return merged, filled, path.name

    return out, 0, None


def attach_human_reasoning(results: pd.DataFrame, split_tag: str) -> pd.DataFrame:
    """Attach ``individual_preference`` from the best available local source."""
    if "individual_preference" in results.columns:
        return results

    enriched, filled, source = _attach_individual_preference_from_raw(results)
    if filled:
        log.info("Joined individual_preference: %d/%d rows filled from %s", filled, len(results), source)
        return enriched

    enriched, filled, source = _attach_individual_preference_from_reasoning_slice(results, split_tag)
    if filled:
        log.info(
            "Joined synthetic human reasoning: %d/%d rows filled from %s",
            filled, len(results), source,
        )
        return enriched

    log.warning(
        "No individual_preference data found. Looked for raw HelpSteer3 parquets in %s "
        "and reasoning slices in %s.",
        cfg.RAW_DIR,
        cfg.WITH_REASON_DIR,
    )
    return results


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
    parser.add_argument("--reasoning-parser", type=str, default=None,
                        help="vLLM reasoning parser (e.g. 'qwen3' for Qwen3 thinking models)")
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
        df = stratified_sample_by_domain(df, args.max_samples, args.seed)
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
        reasoning_parser=args.reasoning_parser,
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
    results = attach_human_reasoning(results, split_tag)
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
