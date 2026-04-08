#!/usr/bin/env python3
"""
Build a debug dataset containing only samples whose model outputs contain N/A.

Usage:
    python src/make_na_debug_set.py \
        --pred results/checkeval_judge_dev_600_predictions.parquet

Optional:
    python src/make_na_debug_set.py \
        --pred results/checkeval_judge_dev_600_predictions.parquet \
        --export-raw 30
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from utils import build_question_index, load_checklists


NA_PAT = re.compile(
    r"^\s*Q(\d+):\s*(?:N/A|NA|not applicable)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_na_qnums(text: str) -> list[int]:
    return [int(x) for x in NA_PAT.findall(str(text))]


def build_question_map(checklists: dict[str, list[str]]) -> dict[str, dict[int, dict[str, str]]]:
    """Domain-specific qid mapping. Delegates to utils.build_question_index."""
    return {
        domain: build_question_index(checklists, domain)
        for domain in ["general", "code", "stem"]
    }


def map_qnums_to_questions(
    domain: str,
    qnums: list[int],
    qmap_by_domain: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, str | int | None]]:
    qmap = qmap_by_domain.get(domain, {})
    rows = []
    for qid in qnums:
        info = qmap.get(qid, {})
        rows.append({
            "qid": qid,
            "dimension": info.get("dimension"),
            "question": info.get("question"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create a debug subset for outputs containing N/A")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Prediction parquet path, e.g. results/checkeval_judge_dev_600_predictions.parquet",
    )
    parser.add_argument(
        "--export-raw",
        type=int,
        default=20,
        help="Export first N anomalous raw outputs as txt files",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_parquet(pred_path)

    required_cols = {"prompt_id", "domain", "raw_output_a", "raw_output_b"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    checklists, _ = load_checklists()
    qmap_by_domain = build_question_map(checklists)

    df = df.copy()
    df["na_qnums_a"] = df["raw_output_a"].apply(extract_na_qnums)
    df["na_qnums_b"] = df["raw_output_b"].apply(extract_na_qnums)
    df["n_na_a"] = df["na_qnums_a"].apply(len)
    df["n_na_b"] = df["na_qnums_b"].apply(len)
    df["has_na_a"] = df["n_na_a"] > 0
    df["has_na_b"] = df["n_na_b"] > 0
    df["has_na_any"] = df["has_na_a"] | df["has_na_b"]

    debug_df = df[df["has_na_any"]].copy()

    # add human-readable mapped questions
    debug_df["na_questions_a"] = debug_df.apply(
        lambda r: map_qnums_to_questions(r["domain"], r["na_qnums_a"], qmap_by_domain),
        axis=1,
    )
    debug_df["na_questions_b"] = debug_df.apply(
        lambda r: map_qnums_to_questions(r["domain"], r["na_qnums_b"], qmap_by_domain),
        axis=1,
    )

    # flatten a row-wise summary for easier viewing
    debug_df["first_na_q_a"] = debug_df["na_qnums_a"].apply(lambda xs: xs[0] if xs else None)
    debug_df["first_na_q_b"] = debug_df["na_qnums_b"].apply(lambda xs: xs[0] if xs else None)

    out_dir = pred_path.parent / f"{pred_path.stem}_na_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # main debug dataset
    parquet_path = "/root/autodl-tmp/Thesis/data/splits/na_debug.parquet"
    csv_path = out_dir / "na_debug.csv"
    debug_df.to_parquet(parquet_path, index=False)

    csv_cols = [
        c for c in [
            "prompt_id",
            "domain",
            "winner",
            "predicted_winner",
            "score_a",
            "score_b",
            "n_yes_a",
            "n_yes_b",
            "n_answered_a",
            "n_answered_b",
            "has_na_a",
            "has_na_b",
            "n_na_a",
            "n_na_b",
            "first_na_q_a",
            "first_na_q_b",
            "na_qnums_a",
            "na_qnums_b",
        ] if c in debug_df.columns
    ]
    debug_df[csv_cols].to_csv(csv_path, index=False)

    # flat table: one row per (sample, side, qid)
    flat_rows = []
    for _, row in debug_df.iterrows():
        for side, qnums_col, mapped_col in [
            ("A", "na_qnums_a", "na_questions_a"),
            ("B", "na_qnums_b", "na_questions_b"),
        ]:
            qnums = row[qnums_col]
            mapped = row[mapped_col]
            for item in mapped:
                flat_rows.append({
                    "prompt_id": row["prompt_id"],
                    "domain": row["domain"],
                    "side": side,
                    "qid": item["qid"],
                    "dimension": item["dimension"],
                    "question": item["question"],
                })

    flat_df = pd.DataFrame(flat_rows)
    flat_parquet = out_dir / "na_debug_flat.parquet"
    flat_csv = out_dir / "na_debug_flat.csv"
    flat_df.to_parquet(flat_parquet, index=False)
    flat_df.to_csv(flat_csv, index=False)

    # summary/meta
    meta = {
        "source_prediction_file": str(pred_path),
        "n_total_rows": int(len(df)),
        "n_rows_with_na_any": int(len(debug_df)),
        "n_rows_with_na_a": int(debug_df["has_na_a"].sum()),
        "n_rows_with_na_b": int(debug_df["has_na_b"].sum()),
        "domain_counts": debug_df["domain"].value_counts().to_dict(),
        "top_na_qids_a": pd.Series([q for xs in debug_df["na_qnums_a"] for q in xs]).value_counts().head(30).to_dict(),
        "top_na_qids_b": pd.Series([q for xs in debug_df["na_qnums_b"] for q in xs]).value_counts().head(30).to_dict(),
    }

    with open(out_dir / "na_debug_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # export raw outputs for first N rows
    raw_dir = out_dir / "raw_examples"
    raw_dir.mkdir(exist_ok=True)
    n_export = min(args.export_raw, len(debug_df))

    for _, row in debug_df.head(n_export).iterrows():
        pid = row["prompt_id"]

        meta_txt = (
            f"prompt_id: {pid}\n"
            f"domain: {row['domain']}\n"
            f"has_na_a: {row['has_na_a']}\n"
            f"has_na_b: {row['has_na_b']}\n"
            f"na_qnums_a: {row['na_qnums_a']}\n"
            f"na_qnums_b: {row['na_qnums_b']}\n"
            f"na_questions_a: {json.dumps(row['na_questions_a'], ensure_ascii=False, indent=2)}\n"
            f"na_questions_b: {json.dumps(row['na_questions_b'], ensure_ascii=False, indent=2)}\n"
        )
        (raw_dir / f"{pid}_meta.txt").write_text(meta_txt, encoding="utf-8")
        (raw_dir / f"{pid}_A.txt").write_text(str(row["raw_output_a"]), encoding="utf-8")
        (raw_dir / f"{pid}_B.txt").write_text(str(row["raw_output_b"]), encoding="utf-8")

    print("=== done ===")
    print("source:", pred_path)
    print("rows with N/A:", len(debug_df))
    print("saved parquet:", parquet_path)
    print("saved csv:", csv_path)
    print("saved flat parquet:", flat_parquet)
    print("saved flat csv:", flat_csv)
    print("saved meta:", out_dir / "na_debug_meta.json")
    print("saved raw examples dir:", raw_dir)


if __name__ == "__main__":
    main()