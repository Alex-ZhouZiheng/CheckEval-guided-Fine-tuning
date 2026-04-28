#!/usr/bin/env python3
"""Merge one or more stronger-model oracle reruns back into the original oracle.

Overwrite strategy: layers are applied in order. For every sample_id present in
a layer, the matching question rows and sample row from earlier layers (or the
original) are dropped and replaced. Later layers win over earlier layers, so
pass strong files in increasing order of trust (e.g. 27B before DeepSeek).
Samples not in any layer are left untouched.

Usage (single layer, backwards compatible):
    python src/data_process/merge_oracle.py \
        --orig data/oracle/dev_600_oracle_v4_9b.parquet \
        --strong data/oracle/dev_600_oracle_v4_9b_rerun_27b.parquet \
        --out data/oracle/dev_600_oracle_v4_merged.parquet

Usage (multiple layers, last wins):
    python src/data_process/merge_oracle.py \
        --orig data/oracle/train_10k_oracle_v4_9b.parquet \
        --strong data/oracle/train_10k_oracle_v4_9b_rerun_27b.parquet \
                 data/oracle/train_10k_oracle_v4_27b_rerun_dsv4.parquet \
        --out data/oracle/train_10k_oracle_v4_final.parquet
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
    parser.add_argument(
        "--strong",
        type=Path,
        nargs="+",
        required=True,
        help="One or more strong-model oracle question parquets. Applied in order; "
             "later layers overwrite earlier layers for any shared sample_id.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Merged oracle question parquet output")
    parser.add_argument(
        "--orig-sample",
        type=Path,
        default=None,
        help="Original sample parquet (default: <orig_stem>_sample.parquet).",
    )
    parser.add_argument(
        "--strong-sample",
        type=Path,
        nargs="+",
        default=None,
        help="Sample parquets for each --strong (same length and order). "
             "Default: <strong_stem>_sample.parquet for each.",
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
        help="Allow strong oracles to contain sample_ids not in original (default: error).",
    )
    args = parser.parse_args()

    orig_q_path = args.orig.resolve()
    out_q_path = args.out.resolve()
    orig_s_path = (args.orig_sample.resolve() if args.orig_sample else _sample_path(orig_q_path))
    out_s_path = (args.out_sample.resolve() if args.out_sample else _sample_path(out_q_path))

    strong_q_paths = [p.resolve() for p in args.strong]
    if args.strong_sample is not None:
        if len(args.strong_sample) != len(args.strong):
            raise SystemExit(
                f"--strong-sample count ({len(args.strong_sample)}) must match "
                f"--strong count ({len(args.strong)})"
            )
        strong_s_paths = [p.resolve() for p in args.strong_sample]
    else:
        strong_s_paths = [_sample_path(p) for p in strong_q_paths]

    for p in [orig_q_path, orig_s_path, *strong_q_paths, *strong_s_paths]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    orig_q = pd.read_parquet(orig_q_path)
    orig_s = pd.read_parquet(orig_s_path)

    if "sample_id" not in orig_q.columns or "sample_id" not in orig_s.columns:
        raise SystemExit("Original parquets must have column 'sample_id'")

    orig_valid, orig_total, _, _, _ = _agreement_rate(orig_s)

    merged_q = orig_q.copy()
    merged_s = orig_s.copy()
    orig_ids = set(orig_s["sample_id"].unique())

    layer_metrics: list[dict] = []
    cumulative_overwritten: set[str] = set()

    for layer_idx, (sq_path, ss_path) in enumerate(zip(strong_q_paths, strong_s_paths)):
        s_q = pd.read_parquet(sq_path)
        s_s = pd.read_parquet(ss_path)

        if "sample_id" not in s_q.columns or "sample_id" not in s_s.columns:
            raise SystemExit(f"Strong parquets must have column 'sample_id' (layer {layer_idx}: {sq_path})")

        s_ids = set(s_s["sample_id"].unique())
        extra = s_ids - orig_ids
        if extra and not args.allow_extra_strong_ids:
            raise SystemExit(
                f"Layer {layer_idx} ({sq_path}) contains {len(extra)} sample_id(s) "
                f"not in original (e.g. {next(iter(extra))}). "
                "Pass --allow-extra-strong-ids to merge anyway."
            )
        if extra:
            log.warning("Layer %d adds %d new sample_ids not in original.", layer_idx, len(extra))

        before_valid, before_total, _, _, _ = _agreement_rate(merged_s)

        merged_q = pd.concat(
            [merged_q[~merged_q["sample_id"].isin(s_ids)], s_q],
            ignore_index=True,
        )
        merged_s = pd.concat(
            [merged_s[~merged_s["sample_id"].isin(s_ids)], s_s],
            ignore_index=True,
        )

        after_valid, after_total, _, _, _ = _agreement_rate(merged_s)

        layer_metrics.append({
            "layer_idx": layer_idx,
            "path": str(sq_path),
            "n_overwritten": int(len(s_ids & (orig_ids | cumulative_overwritten))),
            "n_added_new": int(len(extra)),
            "agreement_valid_before": before_valid,
            "agreement_valid_after": after_valid,
            "delta_valid": (after_valid - before_valid) if (before_valid is not None and after_valid is not None) else None,
            "agreement_total_before": before_total,
            "agreement_total_after": after_total,
        })
        cumulative_overwritten |= s_ids
        log.info(
            "Layer %d (%s): overwrote %d, added %d new. valid %.4f -> %.4f (Δ %+.4f)",
            layer_idx, sq_path.name, len(s_ids & orig_ids), len(extra),
            before_valid if before_valid is not None else float("nan"),
            after_valid if after_valid is not None else float("nan"),
            (after_valid - before_valid) if (before_valid is not None and after_valid is not None) else float("nan"),
        )

    for df_name, df in (("merged_q", merged_q), ("merged_s", merged_s)):
        if df.isna().all(axis=0).any():
            empty_cols = df.columns[df.isna().all(axis=0)].tolist()
            log.info("%s has all-NaN columns after merge: %s", df_name, empty_cols)

    out_q_path.parent.mkdir(parents=True, exist_ok=True)
    merged_q.to_parquet(out_q_path, index=False)
    merged_s.to_parquet(out_s_path, index=False)

    final_valid, final_total, n_valid, n_tie, n_unparseable = _agreement_rate(merged_s)

    metrics = {
        "n_samples_total": int(len(merged_s)),
        "n_question_rows": int(len(merged_q)),
        "n_layers": len(strong_q_paths),
        "n_samples_touched": int(len(cumulative_overwritten & orig_ids)),
        "orig_path": str(orig_q_path),
        "orig_oracle_agreement_valid": orig_valid,
        "orig_oracle_agreement_total": orig_total,
        "merged_oracle_agreement_valid": final_valid,
        "merged_oracle_agreement_total": final_total,
        "n_valid": n_valid,
        "n_tie": n_tie,
        "n_unparseable": n_unparseable,
        "layers": layer_metrics,
    }

    with _meta_path(out_q_path).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("Merged question rows -> %s (%d rows)", out_q_path, len(merged_q))
    log.info("Merged sample rows   -> %s (%d rows)", out_s_path, len(merged_s))
    log.info("Saved meta           -> %s", _meta_path(out_q_path))
    log.info(
        "Final agreement (valid): orig=%.4f -> merged=%.4f  (Δ=%+.4f)",
        orig_valid if orig_valid is not None else float("nan"),
        final_valid if final_valid is not None else float("nan"),
        (final_valid - orig_valid) if (orig_valid is not None and final_valid is not None) else float("nan"),
    )


if __name__ == "__main__":
    main()
