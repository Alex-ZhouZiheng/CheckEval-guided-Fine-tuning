"""Extract tie rows from HR-oracle+weights predictions into eval-ready subset + picks parquet."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--split-parquet", type=Path, required=True)
    ap.add_argument("--out-subset", type=Path, required=True)
    ap.add_argument("--out-picks", type=Path, required=True)
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions)
    ties = preds[preds["predicted_winner"] == "Tie"].copy()
    print(f"tie rows: {len(ties)}")

    split = pd.read_parquet(args.split_parquet)
    print(f"split rows: {len(split)}, cols: {list(split.columns)[:10]}...")

    if "sample_id" not in split.columns:
        from src.evaluation.selector_infer import make_sample_id  # type: ignore
        split["sample_id"] = split.apply(
            lambda r: make_sample_id(
                prompt_id=r["prompt_id"],
                response_a=r["response_a"],
                response_b=r["response_b"],
                winner=r["winner"],
            ),
            axis=1,
        )

    tie_sids = set(ties["sample_id"].astype(str).tolist())
    subset = split[split["sample_id"].astype(str).isin(tie_sids)].reset_index(drop=True)
    print(f"subset rows after join: {len(subset)}")
    if len(subset) != len(ties):
        missing = tie_sids - set(subset["sample_id"].astype(str))
        print(f"WARNING: {len(missing)} tie sample_ids not in split. Sample: {list(missing)[:3]}")

    args.out_subset.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args.out_subset, index=False)
    print(f"wrote subset: {args.out_subset}")

    picks = ties[["sample_id", "selected_qids"]].rename(columns={"selected_qids": "qids"}).copy()
    picks["sample_id"] = picks["sample_id"].astype(str)
    args.out_picks.parent.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(args.out_picks, index=False)
    print(f"wrote picks: {args.out_picks} ({len(picks)} rows)")

if __name__ == "__main__":
    main()
