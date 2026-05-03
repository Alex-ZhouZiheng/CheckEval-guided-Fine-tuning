#!/usr/bin/env python3
"""Analyze v5 validation oracle results on 595 tie/error review samples.

Usage:
    python scripts/analyze_v5_validation.py \
        --oracle data/oracle/v5_validation_595xN_9b.parquet \
        --review results/review/hroracle_weighted_tie_error_review_with_weights.parquet \
        --new-qids data/oracle/v5_new_qids.parquet \
        --out-dir results/v5_validation/
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# ── helpers ──────────────────────────────────────────────────────────


def _gold_support(row: pd.Series, gold_side: str) -> bool:
    """Does this (sample, Q) pair yield a verdict supporting the gold winner?"""
    if gold_side == "A":
        return row["ans_a"] == "yes" and row["ans_b"] == "no"
    elif gold_side == "B":
        return row["ans_a"] == "no" and row["ans_b"] == "yes"
    return False


def _wrong_support(row: pd.Series, gold_side: str) -> bool:
    """Does this (sample, Q) pair yield a verdict supporting the WRONG (non-gold) side?"""
    if gold_side == "A":
        return row["ans_a"] == "no" and row["ans_b"] == "yes"
    elif gold_side == "B":
        return row["ans_a"] == "yes" and row["ans_b"] == "no"
    return False


def _nonzero(row: pd.Series) -> bool:
    """Does this Q discriminate at all between A and B?"""
    return (row["ans_a"] == "yes" and row["ans_b"] == "no") or (
        row["ans_a"] == "no" and row["ans_b"] == "yes"
    )


def _flip_side(gold: str) -> str:
    return "B" if gold == "A" else "A"


def _answered(row: pd.Series) -> bool:
    return row["ans_a"] not in ("parse_fail",) and row["ans_b"] not in ("parse_fail",)


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True, help="Per-(sample,qid) oracle output parquet")
    parser.add_argument("--review", type=Path, required=True, help="Review parquet with error_category")
    parser.add_argument("--new-qids", type=Path, required=True, help="Parquet listing new qids")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────
    oracle = pd.read_parquet(args.oracle)
    review = pd.read_parquet(args.review)
    new_qids = set(pd.read_parquet(args.new_qids)["qid"].astype(int).tolist())

    # Keep only new-Q rows.
    oracle = oracle[oracle["qid"].isin(new_qids)].copy()
    log_msg = f"Oracle: {len(oracle)} (sample,qid) rows after filtering to {len(new_qids)} new qids"
    if len(oracle) == 0:
        raise SystemExit("No oracle rows after qid filter — check --new-qids vs oracle qid values.")
    print(log_msg)

    # Merge error_category from review.
    review_lookup = review[["sample_id", "winner", "error_category", "_review_split"]].drop_duplicates("sample_id")
    merged = oracle.merge(review_lookup, on="sample_id", how="inner", suffixes=("", "_review"))
    before = len(oracle)
    after = len(merged)
    if after < before:
        print(f"Warning: {before - after} oracle rows lost during review merge")
    oracle = merged

    # Flag answered (not parse_fail on either side).
    oracle["answered"] = oracle.apply(_answered, axis=1)
    oracle["nonzero"] = oracle.apply(_nonzero, axis=1)
    oracle["gold_support_flag"] = oracle.apply(lambda r: _gold_support(r, r["winner"]), axis=1)
    oracle["wrong_support_flag"] = oracle.apply(lambda r: _wrong_support(r, r["winner"]), axis=1)

    # ── split ─────────────────────────────────────────────────────
    wrong = oracle[oracle["error_category"] == "wrong_winner"].copy()
    tie = oracle[oracle["error_category"] == "tie"].copy()
    correct = oracle[oracle["error_category"] == "correct"].copy()

    print(f"  wrong_winner={wrong['sample_id'].nunique()}  tie={tie['sample_id'].nunique()}  correct={correct['sample_id'].nunique()}")

    # ────────────────────────────────────────────────────────────────
    # Per-question stats (all error categories combined)
    # ────────────────────────────────────────────────────────────────
    pq = (
        oracle.groupby("qid")
        .agg(
            dimension=("dim", "first"),
            sub_aspect=("sub_aspect", "first"),
            question_text=("question_text", "first"),
            selected_count=("qid", "count"),
            answered_count=("answered", "sum"),
            na_count=(
                "ans_a",
                lambda s: ((s == "na") & (oracle.loc[s.index, "ans_b"] == "na")).sum(),
            ),
            a_yes_b_no=("nonzero", lambda s: ((oracle.loc[s.index, "ans_a"] == "yes") & (oracle.loc[s.index, "ans_b"] == "no")).sum()),
            a_no_b_yes=("nonzero", lambda s: ((oracle.loc[s.index, "ans_a"] == "no") & (oracle.loc[s.index, "ans_b"] == "yes")).sum()),
            a_yes_b_yes=("ans_a", lambda s: ((s == "yes") & (oracle.loc[s.index, "ans_b"] == "yes")).sum()),
            a_no_b_no=("ans_a", lambda s: ((s == "no") & (oracle.loc[s.index, "ans_b"] == "no")).sum()),
        )
        .reset_index()
    )
    pq["nonzero_rate"] = (pq["a_yes_b_no"] + pq["a_no_b_yes"]) / pq["answered_count"].clip(lower=1)
    pq["nonzero_rate"] = pq["nonzero_rate"].fillna(0.0)

    # Gold / wrong support (wrong_winner subset only).
    if len(wrong) > 0:
        wq = (
            wrong.groupby("qid")
            .agg(
                gold_support_count=("gold_support_flag", "sum"),
                wrong_support_count=("wrong_support_flag", "sum"),
            )
            .reset_index()
        )
        total_gold = wq["gold_support_count"].sum()
        total_wrong = wq["wrong_support_count"].sum()
        wq["gold_support_rate"] = wq["gold_support_count"] / total_gold if total_gold > 0 else 0.0
        wq["wrong_support_rate"] = wq["wrong_support_count"] / total_wrong if total_wrong > 0 else 0.0
        wq["gold_wrong_ratio"] = (
            wq["gold_support_count"] / wq["wrong_support_count"].clip(upper=1e-9)
        ).replace([float("inf")], float("nan"))
        pq = pq.merge(wq, on="qid", how="left")
    else:
        pq["gold_support_count"] = 0
        pq["wrong_support_count"] = 0
        pq["gold_support_rate"] = 0.0
        pq["wrong_support_rate"] = 0.0
        pq["gold_wrong_ratio"] = float("nan")

    pq.to_csv(out_dir / "v5_new_q_per_question.csv", index=False)
    print(f"Wrote {len(pq)} per-question rows -> {out_dir / 'v5_new_q_per_question.csv'}")

    # ────────────────────────────────────────────────────────────────
    # Per-sample aggregates (wrong_winner)
    # ────────────────────────────────────────────────────────────────
    path_w = out_dir / "per_sample_wrong_winner.csv"
    if len(wrong) > 0:
        ps_w = (
            wrong.groupby("sample_id")
            .agg(
                domain=("domain", "first"),
                winner=("winner", "first"),
                n_new_q_answered=("answered", "sum"),
                n_gold_supporting=("gold_support_flag", "sum"),
                n_wrong_supporting=("wrong_support_flag", "sum"),
                n_nonzero=("nonzero", "sum"),
            )
            .reset_index()
        )
        ps_w["has_ge1_gold"] = ps_w["n_gold_supporting"] >= 1
        ps_w["has_ge2_gold"] = ps_w["n_gold_supporting"] >= 2
        ps_w["rescuable"] = (ps_w["n_gold_supporting"] >= 1) & (
            ps_w["n_gold_supporting"] > ps_w["n_wrong_supporting"]
        )
        ps_w.to_csv(path_w, index=False)
        n_ge1 = ps_w["has_ge1_gold"].sum()
        n_ge2 = ps_w["has_ge2_gold"].sum()
        n_rescuable = ps_w["rescuable"].sum()
        n_w = len(ps_w)
        print(f"wrong_winner (n={n_w}): ≥1 gold={n_ge1} ({n_ge1/n_w:.1%})  ≥2 gold={n_ge2} ({n_ge2/n_w:.1%})  rescuable={n_rescuable} ({n_rescuable/n_w:.1%})")
    else:
        n_w = 0; n_ge1 = 0; n_ge2 = 0; n_rescuable = 0
        pd.DataFrame(columns=["sample_id", "domain", "winner", "n_new_q_answered",
                              "n_gold_supporting", "n_wrong_supporting", "n_nonzero",
                              "has_ge1_gold", "has_ge2_gold", "rescuable"]).to_csv(path_w, index=False)

    # ────────────────────────────────────────────────────────────────
    # Per-sample aggregates (tie)
    # ────────────────────────────────────────────────────────────────
    path_t = out_dir / "per_sample_tie.csv"
    if len(tie) > 0:
        ps_t = (
            tie.groupby("sample_id")
            .agg(
                domain=("domain", "first"),
                n_new_q_answered=("answered", "sum"),
                n_differentiator=("nonzero", "sum"),
            )
            .reset_index()
        )
        ps_t["differentiator_rate"] = ps_t["n_differentiator"] / ps_t["n_new_q_answered"].clip(lower=1)
        ps_t["tie_breakable"] = ps_t["n_differentiator"] >= 1
        ps_t.to_csv(path_t, index=False)
        n_tie_breakable = ps_t["tie_breakable"].sum()
        n_t = len(ps_t)
        print(f"tie (n={n_t}): breakable={n_tie_breakable} ({n_tie_breakable/n_t:.1%})")
    else:
        n_t = 0; n_tie_breakable = 0
        pd.DataFrame(columns=["sample_id", "domain", "n_new_q_answered",
                              "n_differentiator", "differentiator_rate",
                              "tie_breakable"]).to_csv(path_t, index=False)

    # ────────────────────────────────────────────────────────────────
    # Per-sample aggregates (correct — control)
    # ────────────────────────────────────────────────────────────────
    path_c = out_dir / "per_sample_correct.csv"
    if len(correct) > 0:
        ps_c = (
            correct.groupby("sample_id")
            .agg(
                domain=("domain", "first"),
                winner=("winner", "first"),
                n_new_q_answered=("answered", "sum"),
                n_wrong_supporting=("wrong_support_flag", "sum"),
                n_nonzero=("nonzero", "sum"),
            )
            .reset_index()
        )
        ps_c["regression_risk"] = ps_c["n_wrong_supporting"] > ps_c["n_nonzero"] / 2
        ps_c.to_csv(path_c, index=False)
        n_regression = ps_c["regression_risk"].sum()
        n_c = len(ps_c)
        print(f"correct (n={n_c}): regression_risk={n_regression} ({n_regression/n_c:.1%})")
    else:
        n_c = 0; n_regression = 0
        pd.DataFrame(columns=["sample_id", "domain", "winner", "n_new_q_answered",
                              "n_wrong_supporting", "n_nonzero", "regression_risk"]).to_csv(path_c, index=False)

    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────
    nonzero_rate_all = pq["nonzero_rate"].mean() if len(pq) > 0 else 0.0

    # Overall gold/wrong support rates from raw counts (wrong_winner subset only).
    total_gold_support = int(pq["gold_support_count"].sum()) if "gold_support_count" in pq.columns else 0
    total_wrong_support = int(pq["wrong_support_count"].sum()) if "wrong_support_count" in pq.columns else 0
    total_answered_ww = int(wrong["answered"].sum()) if len(wrong) > 0 else 0
    gold_support_rate = total_gold_support / total_answered_ww if total_answered_ww > 0 else 0.0
    wrong_support_rate = total_wrong_support / total_answered_ww if total_answered_ww > 0 else 0.0
    all_zero_wrong = wrong.groupby("sample_id").filter(
        lambda g: g["nonzero"].sum() == 0
    )["sample_id"].nunique() if len(wrong) > 0 else 0

    denom_rescue = n_w  # wrong_winner count
    rescuable_pct = n_rescuable / denom_rescue if denom_rescue > 0 else 0.0

    summary = {
        "n_new_questions": len(pq),
        "n_wrong_winner_samples": n_w,
        "n_tie_samples": n_t,
        "n_correct_control_samples": n_c,
        "new_question_nonzero_rate": round(float(nonzero_rate_all), 4),
        "new_question_gold_support_rate": round(float(gold_support_rate), 4),
        "new_question_wrong_support_rate": round(float(wrong_support_rate), 4),
        "samples_with_at_least_1_gold_supporting_new_question": int(n_ge1),
        "samples_with_at_least_1_gold_supporting_new_question_pct": round(float(n_ge1 / n_w), 4) if n_w > 0 else 0.0,
        "samples_with_at_least_2_gold_supporting_new_questions": int(n_ge2),
        "samples_with_at_least_2_gold_supporting_new_questions_pct": round(float(n_ge2 / n_w), 4) if n_w > 0 else 0.0,
        "samples_rescuable_by_new_bank": int(n_rescuable),
        "samples_rescuable_by_new_bank_pct": round(float(rescuable_pct), 4),
        "wrong_winner_all_zero_samples": int(all_zero_wrong),
        "tie_differentiator_pct": round(float(n_tie_breakable / n_t), 4) if n_t > 0 else 0.0,
        "control_regression_count": int(n_regression),
        "verdict": "pass" if (nonzero_rate_all > 0.25 and gold_support_rate > wrong_support_rate * 1.5 and rescuable_pct >= 0.30) else "fail",
    }

    summary_path = out_dir / "v5_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")

    # ────────────────────────────────────────────────────────────────
    # Markdown report (brief)
    # ────────────────────────────────────────────────────────────────
    lines = [
        "# v5 Validation Report on Tie/Error Samples",
        "",
        f"**Verdict: {summary['verdict'].upper()}**",
        "",
        f"- New questions evaluated: {summary['n_new_questions']}",
        f"- Wrong-winner samples: {summary['n_wrong_winner_samples']}",
        f"- Tie samples: {summary['n_tie_samples']}",
        f"- Correct-prediction controls: {summary['n_correct_control_samples']}",
        "",
        "## Core Metrics",
        "",
        f"| Metric | Value | Threshold |",
        f"|---|---|---|",
        f"| New-Q nonzero rate | {summary['new_question_nonzero_rate']:.2%} | >25% |",
        f"| Gold-support rate (wrong_winner) | {summary['new_question_gold_support_rate']:.2%} | > wrong-support ×1.5 |",
        f"| Wrong-support rate | {summary['new_question_wrong_support_rate']:.2%} | — |",
        f"| Samples with ≥1 gold-supporting Q | {summary['samples_with_at_least_1_gold_supporting_new_question']} / {n_w} ({summary['samples_with_at_least_1_gold_supporting_new_question_pct']:.1%}) | — |",
        f"| Samples with ≥2 gold-supporting Qs | {summary['samples_with_at_least_2_gold_supporting_new_questions']} / {n_w} ({summary['samples_with_at_least_2_gold_supporting_new_questions_pct']:.1%}) | — |",
        f"| Rescuable samples (gold > wrong, ≥1) | {summary['samples_rescuable_by_new_bank']} / {n_w} ({summary['samples_rescuable_by_new_bank_pct']:.1%}) | ≥30% |",
        f"| Tie breakable rate | {summary['tie_differentiator_pct']:.1%} | — |",
        f"| Control regression count | {summary['control_regression_count']} / {n_c} | min |",
        "",
    ]

    if summary["verdict"] == "pass":
        lines.append("**Conclusion**: v5 new questions fire on hard samples. Proceed with full v5 oracle relabel + selector retrain.")
    elif summary["verdict"] == "fail":
        lines.append("**Conclusion**: v5 new questions do not provide sufficient signal. Consider revising bank before committing.")
    lines.append("")

    report_path = out_dir / "v5_validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
