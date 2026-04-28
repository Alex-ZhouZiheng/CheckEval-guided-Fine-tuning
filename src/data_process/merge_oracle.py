#!/usr/bin/env python3
"""Merge a stronger-model oracle rerun back into the original oracle.

Overwrite strategy: for every sample_id present in the strong oracle, the
matching question rows and sample row in the original are dropped and replaced
by the strong-model rows. Samples not in the strong file are left untouched.

Usage:
    python src/data_process/merge_oracle.py \
        --orig data/oracle/dev_600_oracle_v4_9b.parquet \
        --strong data/oracle/dev_600_oracle_v4_9b_rerun_27b.parquet \
        --out data/oracle/dev_600_oracle_v4_merged.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _sample_path(question_path: Path) -> Path:
    return question_path.with_name(f"{question_path.stem}_sample.parquet")


def _meta_path(question_path: Path) -> Path:
    return question_path.with_suffix(".meta.json")


def _agreement_rate(s_df: pd.DataFrame) -> tuple[float | None, float | None, int, int, int]:
    valid_mask = s_df["winner_pred_full"].isin(["A", "B"])
    n_valid = int(valid_mask.sum())
    n_tie = int((s_df["winner_pred_full"] == "Tie").sum())
    n_unparseable = int(s_df["winner_pred_full"].isna().sum())
    agree_valid = (
        float((s_df.loc[valid_mask, "winner_pred_full"] == s_df.loc[valid_mask, "winner_gt"]).mean())
        if n_valid else None
    )
    agree_total = (
        float(s_df["oracle_agrees_gt"].dropna().mean())
        if "oracle_agrees_gt" in s_df else None
    )
    return agree_valid, agree_total, n_valid, n_tie, n_unparseable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True, help="Original oracle question parquet")
    parser.add_argument("--strong", type=Path, required=True, help="Strong-model oracle question parquet (rerun output)")
    parser.add_argument("--out", type=Path, required=True, help="Merged oracle question parquet output")
    parser.add_argument(
        "--orig-sample",
        type=Path,
        default=None,
        help="Original sample parquet (default: <orig_stem>_sample.parquet). "
             "Use when the producer was run with build_oracle_labels.py --sample-out.",
    )
    parser.add_argument(
        "--strong-sample",
        type=Path,
        default=None,
        help="Strong-model sample parquet (default: <strong_stem>_sample.parquet).",
    )
    parser.add_argument(
        "--out-sample",
        type=Path,
        default=None,
        help="Output sample parquet (default: <out_stem>_sample.parquet).",
    )
    parser.add_argument(
        "--allow-extra-strong-ids",
        action="store_true",
        help="Allow strong oracle to contain sample_ids not in original (default: error).",
    )
    args = parser.parse_args()

    orig_q_path = args.orig.resolve()
    strong_q_path = args.strong.resolve()
    out_q_path = args.out.resolve()

    orig_s_path = (args.orig_sample.resolve() if args.orig_sample else _sample_path(orig_q_path))
    strong_s_path = (args.strong_sample.resolve() if args.strong_sample else _sample_path(strong_q_path))
    out_s_path = (args.out_sample.resolve() if args.out_sample else _sample_path(out_q_path))

    for p in (orig_q_path, strong_q_path, orig_s_path, strong_s_path):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    orig_q = pd.read_parquet(orig_q_path)
    strong_q = pd.read_parquet(strong_q_path)
    orig_s = pd.read_parquet(orig_s_path)
    strong_s = pd.read_parquet(strong_s_path)

    orig_ids = set(orig_s["sample_id"].unique())
    strong_ids = set(strong_s["sample_id"].unique())
    extra = strong_ids - orig_ids
    if extra:
        msg = f"Strong oracle contains {len(extra)} sample_id(s) not in original (e.g. {next(iter(extra))})."
        if not args.allow_extra_strong_ids:
            raise SystemExit(msg + " Pass --allow-extra-strong-ids to merge anyway.")
        log.warning(msg + " Merging anyway per --allow-extra-strong-ids.")

    overlap_ids = orig_ids & strong_ids

    if not overlap_ids and not extra:
        raise SystemExit("Strong oracle has no sample_ids in common with original — nothing to merge.")

    # Schema sanity: strong must carry the same columns we splice in.
    for col in ("sample_id",):
        if col not in orig_q.columns or col not in strong_q.columns:
            raise SystemExit(f"Both question parquets must have column '{col}'")
        if col not in orig_s.columns or col not in strong_s.columns:
            raise SystemExit(f"Both sample parquets must have column '{col}'")

    overwrite_ids = strong_ids  # rows to drop from original
    merged_q = pd.concat(
        [orig_q[~orig_q["sample_id"].isin(overwrite_ids)], strong_q],
        ignore_index=True,
    )
    merged_s = pd.concat(
        [orig_s[~orig_s["sample_id"].isin(overwrite_ids)], strong_s],
        ignore_index=True,
    )

    # Align column unions so nothing silently disappears on concat.
    for df_name, df in (("merged_q", merged_q), ("merged_s", merged_s)):
        if df.isna().all(axis=0).any():
            empty_cols = df.columns[df.isna().all(axis=0)].tolist()
            log.info("%s has all-NaN columns after merge: %s", df_name, empty_cols)

    out_q_path.parent.mkdir(parents=True, exist_ok=True)
    merged_q.to_parquet(out_q_path, index=False)
    merged_s.to_parquet(out_s_path, index=False)

    orig_valid, orig_total, _, _, _ = _agreement_rate(orig_s)
    merged_valid, merged_total, n_valid, n_tie, n_unparseable = _agreement_rate(merged_s)

    metrics = {
        "n_samples_total": int(len(merged_s)),
        "n_samples_overwritten": int(len(overlap_ids)),
        "n_samples_added_from_strong": int(len(extra)),
        "n_question_rows": int(len(merged_q)),
        "orig_oracle_agreement_valid": orig_valid,
        "orig_oracle_agreement_total": orig_total,
        "merged_oracle_agreement_valid": merged_valid,
        "merged_oracle_agreement_total": merged_total,
        "n_valid": n_valid,
        "n_tie": n_tie,
        "n_unparseable": n_unparseable,
        "orig_path": str(orig_q_path),
        "strong_path": str(strong_q_path),
    }

    with _meta_path(out_q_path).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("Merged question rows -> %s (%d rows)", out_q_path, len(merged_q))
    log.info("Merged sample rows   -> %s (%d rows, %d overwritten)", out_s_path, len(merged_s), len(overlap_ids))
    log.info("Saved meta           -> %s", _meta_path(out_q_path))
    log.info(
        "Agreement (valid): orig=%.4f -> merged=%.4f  (delta=%+.4f)",
        orig_valid if orig_valid is not None else float("nan"),
        merged_valid if merged_valid is not None else float("nan"),
        (merged_valid - orig_valid) if (orig_valid is not None and merged_valid is not None) else float("nan"),
    )


if __name__ == "__main__":
    main()
