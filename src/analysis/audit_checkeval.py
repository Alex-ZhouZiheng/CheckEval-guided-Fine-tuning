#!/usr/bin/env python3
"""
One-shot audit script for CheckEval judge prediction files.

Usage:
    python src/audit_checkeval_run.py \
        --pred results/checkeval_judge_dev_600_predictions.parquet

Optional:
    python src/audit_checkeval_run.py \
        --pred results/checkeval_judge_dev_600_predictions.parquet \
        --export-raw 20
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from utils import load_checklists, _select_dimensions


LINE_RE = re.compile(r"^\s*Q(\d+):\s*(yes|no)\s*$", re.IGNORECASE)
NA_RE = re.compile(
    r"^\s*Q(\d+):\s*(?:N\s*/\s*A|NA|not\s*applicable)\s*[.。]?\s*$",
    re.IGNORECASE,
)


def expected_questions_for_domain(domain: str, checklists: dict[str, list[str]]) -> int:
    allowed = _select_dimensions(domain)
    allowed_lower = {d.lower() for d in allowed}
    total = 0
    for dim_name, questions in checklists.items():
        if dim_name in allowed or dim_name.lower() in allowed_lower:
            total += len(questions)
    return total


def parse_initial_answer_block(raw: str, expected_n: int | None = None) -> dict:
    """
    Parse the initial block of lines matching Qk: yes/no or Qk: N/A.
    N/A lines are recorded separately but do NOT stop parsing.
    Stop at the first non-empty line that matches neither pattern.
    """
    answers: list[dict[str, object]] = []
    na_answers: list[dict[str, object]] = []
    qnums: list[int] = []
    na_qnums: list[int] = []
    started = False

    lines = raw.splitlines()
    stop_idx = len(lines)

    for i, line in enumerate(lines):
        s = line.strip()

        if not s:
            continue

        m = LINE_RE.match(s)
        if m:
            started = True
            q = int(m.group(1))
            ans = m.group(2).lower()

            if expected_n is not None and not (1 <= q <= expected_n):
                continue

            qnums.append(q)
            answers.append({"q": q, "answer": ans})
            continue

        m_na = NA_RE.match(s)
        if m_na:
            started = True
            q = int(m_na.group(1))

            if expected_n is not None and not (1 <= q <= expected_n):
                continue

            na_qnums.append(q)
            na_answers.append({"q": q})
            continue

        if started:
            stop_idx = i
            break

    n_yes = sum(1 for a in answers if a["answer"] == "yes")
    n_no = sum(1 for a in answers if a["answer"] == "no")
    n_na = len(na_answers)

    unique_qnums = sorted(set(qnums))
    has_duplicates = len(unique_qnums) != len(qnums)

    exact_contiguous = False
    if expected_n is not None:
        exact_contiguous = unique_qnums == list(range(1, expected_n + 1)) and not has_duplicates

    # complete_with_na: answered ∪ na covers {1..expected_n}
    exact_contiguous_with_na = False
    if expected_n is not None:
        all_ids = set(qnums) | set(na_qnums)
        exact_contiguous_with_na = all_ids == set(range(1, expected_n + 1))

    remainder = "\n".join(lines[stop_idx:]).strip() if stop_idx < len(lines) else ""
    has_rationale = bool(re.search(r"rationale|reasoning|summary|explanation", remainder, re.I))
    has_more_q_after_block = bool(re.search(r"^\s*Q\d+:", remainder, re.I | re.M))

    return {
        "answers": answers,
        "na_answers": na_answers,
        "qnums": qnums,
        "na_qnums": sorted(set(na_qnums)),
        "unique_qnums": unique_qnums,
        "strict_n_answered": len(answers),
        "strict_n_yes": n_yes,
        "strict_n_no": n_no,
        "n_na": n_na,
        "has_duplicates": has_duplicates,
        "last_q": qnums[-1] if qnums else None,
        "exact_contiguous": exact_contiguous,
        "exact_contiguous_with_na": exact_contiguous_with_na,
        "remainder_text": remainder,
        "has_rationale_after_block": has_rationale,
        "has_more_q_after_block": has_more_q_after_block,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit a CheckEval prediction parquet.")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to prediction parquet, e.g. results/checkeval_judge_dev_600_predictions.parquet",
    )
    parser.add_argument(
        "--export-raw",
        type=int,
        default=10,
        help="How many anomalous samples to export as txt files",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_parquet(pred_path)

    required_cols = {
        "prompt_id",
        "domain",
        "raw_output_a",
        "raw_output_b",
        "n_answered_a",
        "n_answered_b",
        "n_yes_a",
        "n_yes_b",
        "score_a",
        "score_b",
        "predicted_winner",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    checklists, _ = load_checklists()

    audit_rows = []

    for _, row in df.iterrows():
        expected_n = expected_questions_for_domain(row["domain"], checklists)

        pa = parse_initial_answer_block(str(row["raw_output_a"]), expected_n=expected_n)
        pb = parse_initial_answer_block(str(row["raw_output_b"]), expected_n=expected_n)

        strict_complete_a = pa["exact_contiguous"]
        strict_complete_b = pb["exact_contiguous"]

        stored_na = row["n_answered_a"]
        stored_nb = row["n_answered_b"]

        stored_ny_a = row["n_yes_a"]
        stored_ny_b = row["n_yes_b"]

        stored_count_mismatch_a = pd.notna(stored_na) and int(stored_na) != pa["strict_n_answered"]
        stored_count_mismatch_b = pd.notna(stored_nb) and int(stored_nb) != pb["strict_n_answered"]

        stored_yes_mismatch_a = pd.notna(stored_ny_a) and int(stored_ny_a) != pa["strict_n_yes"]
        stored_yes_mismatch_b = pd.notna(stored_ny_b) and int(stored_ny_b) != pb["strict_n_yes"]

        predicted_winner = row["predicted_winner"]

        score_equal = (row["score_a"] == row["score_b"]) if pd.notna(row["score_a"]) and pd.notna(row["score_b"]) else False
        yes_equal = (row["n_yes_a"] == row["n_yes_b"]) if pd.notna(row["n_yes_a"]) and pd.notna(row["n_yes_b"]) else False
        answered_equal = (row["n_answered_a"] == row["n_answered_b"]) if pd.notna(row["n_answered_a"]) and pd.notna(row["n_answered_b"]) else False

        suspicious_tie = (
            predicted_winner == "Tie"
            and (
                (not yes_equal)
                or (not answered_equal)
                or (not strict_complete_a)
                or (not strict_complete_b)
                or stored_count_mismatch_a
                or stored_count_mismatch_b
                or stored_yes_mismatch_a
                or stored_yes_mismatch_b
            )
        )

        audit_rows.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "expected_n_questions": expected_n,
            "predicted_winner": predicted_winner,

            "stored_n_answered_a": row["n_answered_a"],
            "stored_n_answered_b": row["n_answered_b"],
            "strict_n_answered_a": pa["strict_n_answered"],
            "strict_n_answered_b": pb["strict_n_answered"],

            "stored_n_yes_a": row["n_yes_a"],
            "stored_n_yes_b": row["n_yes_b"],
            "strict_n_yes_a": pa["strict_n_yes"],
            "strict_n_yes_b": pb["strict_n_yes"],

            "n_na_a": pa["n_na"],
            "n_na_b": pb["n_na"],
            "na_qnums_a": pa["na_qnums"],
            "na_qnums_b": pb["na_qnums"],

            "stored_score_a": row["score_a"],
            "stored_score_b": row["score_b"],

            "last_q_a": pa["last_q"],
            "last_q_b": pb["last_q"],

            "strict_complete_a": strict_complete_a,
            "strict_complete_b": strict_complete_b,
            "both_strict_complete": strict_complete_a and strict_complete_b,
            "complete_with_na_a": pa["exact_contiguous_with_na"],
            "complete_with_na_b": pb["exact_contiguous_with_na"],

            "has_duplicates_a": pa["has_duplicates"],
            "has_duplicates_b": pb["has_duplicates"],
            "has_rationale_after_block_a": pa["has_rationale_after_block"],
            "has_rationale_after_block_b": pb["has_rationale_after_block"],
            "has_more_q_after_block_a": pa["has_more_q_after_block"],
            "has_more_q_after_block_b": pb["has_more_q_after_block"],

            "stored_count_mismatch_a": stored_count_mismatch_a,
            "stored_count_mismatch_b": stored_count_mismatch_b,
            "stored_yes_mismatch_a": stored_yes_mismatch_a,
            "stored_yes_mismatch_b": stored_yes_mismatch_b,

            "score_equal": score_equal,
            "yes_equal": yes_equal,
            "answered_equal": answered_equal,
            "suspicious_tie": suspicious_tie,
        })

    audit_df = pd.DataFrame(audit_rows)

    out_dir = pred_path.with_suffix("")
    out_dir = out_dir.parent / f"{pred_path.stem}_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full audit table
    audit_path = out_dir / "audit_table.parquet"
    audit_df.to_parquet(audit_path, index=False)

    # Summary
    summary = {
        "n_total": int(len(audit_df)),
        "strict_complete_a_rate": float(audit_df["strict_complete_a"].mean()),
        "strict_complete_b_rate": float(audit_df["strict_complete_b"].mean()),
        "both_strict_complete_rate": float(audit_df["both_strict_complete"].mean()),
        "complete_with_na_a_rate": float(audit_df["complete_with_na_a"].mean()),
        "complete_with_na_b_rate": float(audit_df["complete_with_na_b"].mean()),
        "pct_a_with_na": float((audit_df["n_na_a"] > 0).mean()),
        "pct_b_with_na": float((audit_df["n_na_b"] > 0).mean()),
        "avg_n_na_a": float(audit_df["n_na_a"].mean()),
        "avg_n_na_b": float(audit_df["n_na_b"].mean()),
        "stored_count_mismatch_a_rate": float(audit_df["stored_count_mismatch_a"].mean()),
        "stored_count_mismatch_b_rate": float(audit_df["stored_count_mismatch_b"].mean()),
        "stored_yes_mismatch_a_rate": float(audit_df["stored_yes_mismatch_a"].mean()),
        "stored_yes_mismatch_b_rate": float(audit_df["stored_yes_mismatch_b"].mean()),
        "has_rationale_after_block_a_rate": float(audit_df["has_rationale_after_block_a"].mean()),
        "has_rationale_after_block_b_rate": float(audit_df["has_rationale_after_block_b"].mean()),
        "has_more_q_after_block_a_rate": float(audit_df["has_more_q_after_block_a"].mean()),
        "has_more_q_after_block_b_rate": float(audit_df["has_more_q_after_block_b"].mean()),
        "n_ties": int((audit_df["predicted_winner"] == "Tie").sum()),
        "n_suspicious_ties": int(audit_df["suspicious_tie"].sum()),
    }

    by_domain = (
        audit_df.groupby("domain")[[
            "strict_complete_a",
            "strict_complete_b",
            "both_strict_complete",
            "stored_count_mismatch_a",
            "stored_count_mismatch_b",
            "suspicious_tie",
        ]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    summary["by_domain"] = by_domain

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable prints
    print("=== OVERALL SUMMARY ===")
    for k, v in summary.items():
        if k == "by_domain":
            continue
        print(f"{k}: {v}")

    print("\n=== BY DOMAIN ===")
    print(
        audit_df.groupby("domain")[[
            "strict_complete_a",
            "strict_complete_b",
            "both_strict_complete",
            "stored_count_mismatch_a",
            "stored_count_mismatch_b",
            "suspicious_tie",
        ]].mean()
    )

    print("\n=== FIRST ANOMALIES ===")
    anomalies = audit_df[
        (~audit_df["both_strict_complete"])
        | (audit_df["stored_count_mismatch_a"])
        | (audit_df["stored_count_mismatch_b"])
        | (audit_df["stored_yes_mismatch_a"])
        | (audit_df["stored_yes_mismatch_b"])
        | (audit_df["suspicious_tie"])
    ].copy()

    anomaly_cols = [
        "prompt_id",
        "domain",
        "expected_n_questions",
        "stored_n_answered_a",
        "strict_n_answered_a",
        "stored_n_answered_b",
        "strict_n_answered_b",
        "last_q_a",
        "last_q_b",
        "both_strict_complete",
        "stored_count_mismatch_a",
        "stored_count_mismatch_b",
        "suspicious_tie",
    ]
    print(anomalies[anomaly_cols].head(30).to_string(index=False))

    anomalies.to_csv(out_dir / "anomalies.csv", index=False)

    # Export raw outputs for first N anomalous samples
    export_n = min(args.export_raw, len(anomalies))
    raw_dir = out_dir / "raw_examples"
    raw_dir.mkdir(exist_ok=True)

    raw_lookup = df.set_index("prompt_id")

    for _, row in anomalies.head(export_n).iterrows():
        pid = row["prompt_id"]
        src = raw_lookup.loc[pid]

        meta_txt = (
            f"prompt_id: {pid}\n"
            f"domain: {row['domain']}\n"
            f"expected_n_questions: {row['expected_n_questions']}\n"
            f"stored_n_answered_a: {row['stored_n_answered_a']}\n"
            f"strict_n_answered_a: {row['strict_n_answered_a']}\n"
            f"stored_n_answered_b: {row['stored_n_answered_b']}\n"
            f"strict_n_answered_b: {row['strict_n_answered_b']}\n"
            f"last_q_a: {row['last_q_a']}\n"
            f"last_q_b: {row['last_q_b']}\n"
            f"both_strict_complete: {row['both_strict_complete']}\n"
            f"suspicious_tie: {row['suspicious_tie']}\n"
        )

        (raw_dir / f"{pid}_meta.txt").write_text(meta_txt, encoding="utf-8")
        (raw_dir / f"{pid}_A.txt").write_text(str(src["raw_output_a"]), encoding="utf-8")
        (raw_dir / f"{pid}_B.txt").write_text(str(src["raw_output_b"]), encoding="utf-8")

    print(f"\nSaved audit artifacts to: {out_dir}")


if __name__ == "__main__":
    main()