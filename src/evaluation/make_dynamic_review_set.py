#!/usr/bin/env python3
"""Build a review_app-compatible parquet from dynamic-eval predictions.

Use this after running run_dynamic_eval.py with --save-raw-outputs. The output
can be opened directly by src/evaluation/review_app.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from utils import (  # noqa: E402
    aggregate_checklist_score,
    build_pointwise_prompt_from_qids,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _parse_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _parse_list(ast.literal_eval(text))
    raise TypeError(f"Unsupported qid list value: {type(value).__name__}")


def _load_qmeta(bank_dir: Path) -> dict[int, dict[str, str]]:
    path = bank_dir / "bank_index.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    bank_df = pd.read_parquet(path)
    needed = {"qid", "dimension", "question_text"}
    missing = needed - set(bank_df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "definition" not in bank_df.columns:
        bank_df["definition"] = ""
    return {
        int(r["qid"]): {
            "dimension": str(r["dimension"]),
            "question_text": str(r["question_text"]),
            "definition": str(r.get("definition", "") or ""),
        }
        for _, r in bank_df.iterrows()
    }


def _attach_source_fields(
    df: pd.DataFrame,
    split: str,
    subset: str | None,
    input_path: Path | None,
    max_samples: int | None,
) -> pd.DataFrame:
    required = {"winner", "predicted_winner", "context", "response_a", "response_b", "domain"}
    missing = required - set(df.columns)
    source_needed = {"context", "response_a", "response_b"} & missing
    if not source_needed:
        return df

    if "sample_id" not in df.columns:
        raise ValueError(
            "predictions parquet is missing context/response columns and has no sample_id "
            "for joining back to the source split."
        )

    from evaluation.selector_infer import load_eval_pairs

    source = load_eval_pairs(
        split=split,
        subset=subset,
        input_path=input_path,
        max_samples=max_samples,
    )
    source_cols = ["sample_id"] + [
        c for c in ["prompt_id", "domain", "context", "response_a", "response_b", "winner"]
        if c not in df.columns
    ]
    source = source[source_cols].drop_duplicates(subset=["sample_id"], keep="first")
    out = df.merge(source, on="sample_id", how="left", validate="many_to_one")

    still_missing = required - set(out.columns)
    if still_missing:
        raise ValueError(f"could not attach required columns: {sorted(still_missing)}")

    empty_source = out[["context", "response_a", "response_b"]].isna().any(axis=1)
    if bool(empty_source.any()):
        sample_ids = out.loc[empty_source, "sample_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"source split did not match {int(empty_source.sum())} prediction rows "
            f"(first 5 sample_ids: {sample_ids}). Pass the same --split/--subset/--input-path "
            "used for run_dynamic_eval.py."
        )

    log.info(
        "Attached context/response columns from split=%s subset=%s input_path=%s",
        split,
        subset,
        input_path,
    )
    return out


def _attach_individual_preference_from_raw(results: pd.DataFrame) -> tuple[pd.DataFrame, int, str | None]:
    """Attach raw HelpSteer3 individual_preference if raw parquets exist."""
    import hashlib

    def _ctx_to_text(ctx) -> str:
        return "\n\n".join(f"[{t['role']}]\n{t['content']}" for t in ctx)

    def _pid(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    if "prompt_id" not in results.columns:
        return results, 0, None

    needed = set(results["prompt_id"].dropna().astype(str))
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
        raw_pids = raw["context"].apply(lambda c: _pid(_ctx_to_text(c))).astype(str)
        mask = raw_pids.isin(needed)
        lookup = dict(zip(raw_pids[mask], raw.loc[mask, "individual_preference"]))
        candidate = out["prompt_id"].astype(str).map(lookup)
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
    """Fallback: synthesize review-app human reasoning from with_reason parquet."""
    candidates = [cfg.WITH_REASON_DIR / f"{split_tag}_reasoning.parquet"]
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
                [
                    {
                        "score": None,
                        "reasoning": row.get("reasoning_text", ""),
                        "feedback1": row.get("feedback_a_text", ""),
                        "feedback2": row.get("feedback_b_text", ""),
                    }
                ],
                ensure_ascii=False,
            ),
            axis=1,
        )
        synth = synth[join_cols + ["individual_preference"]].drop_duplicates(
            subset=join_cols,
            keep="first",
        )
        merged = out.merge(synth, on=join_cols, how="left", suffixes=("", "_reason"))
        filled = int(merged["individual_preference"].notna().sum())
        if filled:
            return merged, filled, path.name

    return out, 0, None


def _attach_human_reasoning(results: pd.DataFrame, split_tag: str) -> tuple[pd.DataFrame, int, str | None]:
    if "individual_preference" in results.columns and results["individual_preference"].notna().any():
        return results, int(results["individual_preference"].notna().sum()), "existing"

    enriched, filled, source = _attach_individual_preference_from_raw(results)
    if filled:
        return enriched, filled, source

    return _attach_individual_preference_from_reasoning_slice(results, split_tag)


def _first_text(row: pd.Series, names: list[str]) -> str | None:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None:
            continue
        if pd.isna(value):
            continue
        text = str(value)
        if text:
            return text
    return None


def _select_rows(
    df: pd.DataFrame,
    n_wrong: int,
    n_tie: int,
    n_correct: int,
    seed: int,
    include_all_wrong: bool,
) -> pd.DataFrame:
    rng = random.Random(seed)

    wrong = df[
        df["predicted_winner"].isin(["A", "B"])
        & (df["predicted_winner"].astype(str) != df["winner"].astype(str))
    ].copy()
    tie = df[df["predicted_winner"].astype(str) == "Tie"].copy()
    correct = df[
        df["predicted_winner"].isin(["A", "B"])
        & (df["predicted_winner"].astype(str) == df["winner"].astype(str))
    ].copy()

    def take(part: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or part.empty:
            return part.head(0)
        if len(part) <= n:
            return part
        return part.sample(n=n, random_state=rng.randint(0, 10**9))

    wrong_take = wrong if include_all_wrong else take(wrong, n_wrong)
    parts = [
        wrong_take.assign(_review_split="wrong"),
        take(tie, n_tie).assign(_review_split="wrong"),
        take(correct, n_correct).assign(_review_split="correct"),
    ]
    out = pd.concat(parts, ignore_index=True)
    if len(out):
        out = out.sample(frac=1, random_state=rng.randint(0, 10**9)).reset_index(drop=True)

    log.info(
        "Selected review rows: wrong=%d/%d tie=%d/%d correct=%d/%d total=%d",
        len(wrong_take),
        len(wrong),
        min(len(tie), n_tie),
        len(tie),
        min(len(correct), n_correct),
        len(correct),
        len(out),
    )
    return out


def _make_review_row(
    row: pd.Series,
    qmeta: dict[int, dict[str, str]],
    na_policy: str,
    coverage_threshold: float,
) -> dict[str, Any]:
    qids = _parse_list(row.get("asked_qids") if "asked_qids" in row else row.get("selected_qids"))
    if not qids:
        qids = _parse_list(row.get("selected_qids"))
    if not qids:
        raise ValueError(f"sample {row.get('sample_id')} has no asked_qids/selected_qids")

    missing_qids = [q for q in qids if q not in qmeta]
    if missing_qids:
        raise ValueError(f"sample {row.get('sample_id')} has unknown qids: {missing_qids[:5]}")

    raw_a = _first_text(row, ["raw_output_a", "stage1_a"])
    raw_b = _first_text(row, ["raw_output_b", "stage1_b"])
    if raw_a is None or raw_b is None:
        raise ValueError(
            "predictions parquet lacks raw judge outputs. Rerun run_dynamic_eval.py "
            "with --save-raw-outputs before building review data."
        )

    parsed_a = parse_checkeval_output(raw_a, expected_n=len(qids))
    parsed_b = parse_checkeval_output(raw_b, expected_n=len(qids))
    agg_a = aggregate_checklist_score(
        parsed_a,
        na_policy=na_policy,
        coverage_threshold=coverage_threshold,
        expected_n=len(qids),
    )
    agg_b = aggregate_checklist_score(
        parsed_b,
        na_policy=na_policy,
        coverage_threshold=coverage_threshold,
        expected_n=len(qids),
    )

    winner = str(row.get("winner"))
    predicted = str(row.get("predicted_winner"))
    if predicted == "Tie":
        error_category = "tie"
    elif predicted == winner:
        error_category = "correct"
    else:
        error_category = "wrong_winner"

    prompt_a = build_pointwise_prompt_from_qids(row=row, qids=qids, qmeta=qmeta, side="A")
    prompt_b = build_pointwise_prompt_from_qids(row=row, qids=qids, qmeta=qmeta, side="B")

    return {
        "sample_id": row.get("sample_id"),
        "prompt_id": row.get("prompt_id"),
        "domain": row.get("domain"),
        "context": row.get("context"),
        "response_a": row.get("response_a"),
        "response_b": row.get("response_b"),
        "winner": winner,
        "predicted_winner": predicted,
        "score_a": agg_a["score"] if agg_a else None,
        "score_b": agg_b["score"] if agg_b else None,
        "pairwise_margin": row.get("margin_final", row.get("pairwise_margin")),
        "margin_initial": row.get("margin_initial"),
        "margin_final": row.get("margin_final"),
        "n_yes_a": int(parsed_a.get("n_yes", 0)),
        "n_yes_b": int(parsed_b.get("n_yes", 0)),
        "n_no_a": int(parsed_a.get("n_no", 0)),
        "n_no_b": int(parsed_b.get("n_no", 0)),
        "n_na_a": int(parsed_a.get("n_na", 0)),
        "n_na_b": int(parsed_b.get("n_na", 0)),
        "n_answered_a": int(parsed_a.get("n_questions_parsed", 0)),
        "n_answered_b": int(parsed_b.get("n_questions_parsed", 0)),
        "expected_n_questions": len(qids),
        "selected_qids": _parse_list(row.get("selected_qids")) if "selected_qids" in row else qids,
        "asked_qids": qids,
        "k_selected": row.get("k_selected"),
        "k_after_escalation": row.get("k_after_escalation"),
        "escalated": row.get("escalated"),
        "fallback": row.get("fallback"),
        "parse_ok": row.get("parse_ok"),
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "raw_output_a": raw_a,
        "raw_output_b": raw_b,
        "parsed_a_json": json.dumps(parsed_a, ensure_ascii=False, default=str),
        "parsed_b_json": json.dumps(parsed_b, ensure_ascii=False, default=str),
        "error_category": error_category,
        "_review_split": row.get("_review_split", "wrong"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-wrong", type=int, default=60)
    parser.add_argument("--n-tie", type=int, default=20)
    parser.add_argument("--n-correct", type=int, default=20)
    parser.add_argument("--include-all-wrong", action="store_true")
    parser.add_argument(
        "--no-human-reasoning",
        action="store_true",
        help="Do not attempt to attach individual_preference for review_app.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--na-policy",
        type=str,
        default="skip",
        choices=["strict", "as_no", "skip", "partial"],
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    args = parser.parse_args()

    pred_path = args.predictions.resolve()
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_parquet(pred_path)
    df = _attach_source_fields(
        df,
        split=args.split,
        subset=args.subset,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    required = {"winner", "predicted_winner", "context", "response_a", "response_b", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{pred_path} missing columns: {sorted(missing)}. Pass the same "
            "--split/--subset/--input-path used for run_dynamic_eval.py so the script can "
            "join source text by sample_id."
        )

    if not {"raw_output_a", "stage1_a"} & set(df.columns):
        raise ValueError(
            f"{pred_path} does not contain raw outputs. Rerun dynamic eval with --save-raw-outputs."
        )

    qmeta = _load_qmeta(args.bank.resolve())
    selected = _select_rows(
        df=df,
        n_wrong=args.n_wrong,
        n_tie=args.n_tie,
        n_correct=args.n_correct,
        seed=args.seed,
        include_all_wrong=args.include_all_wrong,
    )

    rows = [
        _make_review_row(
            row,
            qmeta=qmeta,
            na_policy=args.na_policy,
            coverage_threshold=args.coverage_threshold,
        )
        for _, row in selected.iterrows()
    ]
    review = pd.DataFrame(rows)

    human_filled = 0
    human_source = None
    if not args.no_human_reasoning and len(review):
        split_tag = args.subset or args.split
        review, human_filled, human_source = _attach_human_reasoning(review, split_tag)
        if human_filled:
            log.info(
                "Joined individual_preference: %d/%d rows from %s",
                human_filled,
                len(review),
                human_source,
            )
        else:
            log.warning(
                "individual_preference not attached. Looked in %s and %s.",
                cfg.DATA_DIR / "raw",
                cfg.WITH_REASON_DIR,
            )

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_parquet(out_path, index=False)

    meta = {
        "predictions": str(pred_path),
        "bank": str(args.bank.resolve()),
        "out": str(out_path),
        "split": args.split,
        "subset": args.subset,
        "input_path": str(args.input_path) if args.input_path else None,
        "n_rows": len(review),
        "n_wrong": int((review["error_category"] == "wrong_winner").sum()) if len(review) else 0,
        "n_tie": int((review["error_category"] == "tie").sum()) if len(review) else 0,
        "n_correct": int((review["error_category"] == "correct").sum()) if len(review) else 0,
        "human_reasoning_rows": human_filled,
        "human_reasoning_source": human_source,
        "na_policy": args.na_policy,
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    log.info("Saved dynamic review parquet (%d rows) -> %s", len(review), out_path)
    log.info("Launch: streamlit run src/evaluation/review_app.py -- --results %s", out_path)


if __name__ == "__main__":
    main()
