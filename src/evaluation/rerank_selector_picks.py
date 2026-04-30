"""Rerank selector candidates with an empirical discriminativeness prior.

This is an experiment helper: it keeps the selector's per-sample ranked list but
nudges candidates whose historical checklist contribution is both nonzero and
directionally aligned with the gold winner.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_qid_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _parse_qid_list(ast.literal_eval(text))
    raise TypeError(f"unsupported qid list value: {type(value).__name__}")


def _load_bank(path: Path) -> pd.DataFrame:
    index_path = path / "bank_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    bank = pd.read_parquet(index_path)
    if "qid" not in bank.columns:
        raise ValueError(f"{index_path} missing qid")
    if "dimension" not in bank.columns and "dim" not in bank.columns:
        raise ValueError(f"{index_path} missing dimension/dim")
    return bank


def _select_topk_with_quota(
    ranked_active_qids: list[int],
    bank_df: pd.DataFrame,
    domain: str,
    k: int,
    enforce_quota: bool,
) -> list[int]:
    k_eff = min(k, len(ranked_active_qids))
    if k_eff <= 0:
        return []
    if not enforce_quota:
        return ranked_active_qids[:k_eff]

    dim_col = "dimension" if "dimension" in bank_df.columns else "dim"
    q_to_dim = {int(r["qid"]): str(r[dim_col]) for _, r in bank_df[["qid", dim_col]].iterrows()}

    allowed_dims: list[str] = []
    seen: set[str] = set()
    for qid in ranked_active_qids:
        dim = q_to_dim.get(int(qid))
        if dim is None or dim in seen:
            continue
        seen.add(dim)
        allowed_dims.append(dim)

    if not allowed_dims:
        return ranked_active_qids[:k_eff]

    min_per_dim = k_eff // len(allowed_dims)
    selected: list[int] = []
    used: set[int] = set()

    if min_per_dim > 0:
        for dim in allowed_dims:
            dim_qids = [q for q in ranked_active_qids if q_to_dim.get(int(q)) == dim]
            for qid in dim_qids[:min_per_dim]:
                if qid not in used:
                    selected.append(qid)
                    used.add(qid)

    for qid in ranked_active_qids:
        if len(selected) >= k_eff:
            break
        if qid in used:
            continue
        selected.append(qid)
        used.add(qid)

    return selected[:k_eff]


def _build_prior(contrib_path: Path, min_support: int) -> pd.DataFrame:
    contrib = pd.read_parquet(contrib_path)
    required = {"qid", "winner"}
    missing = required - set(contrib.columns)
    if missing:
        raise ValueError(f"{contrib_path} missing columns: {sorted(missing)}")

    contrib_col = "raw_contribution"
    if contrib_col not in contrib.columns:
        contrib_col = "weighted_contribution" if "weighted_contribution" in contrib.columns else "contribution"
    if contrib_col not in contrib.columns:
        raise ValueError(f"{contrib_path} missing contribution column")

    work = contrib.copy()
    work["qid"] = work["qid"].astype(int)
    work["contribution"] = work[contrib_col].astype(float)
    work["nonzero"] = work["contribution"].abs() > 1e-12

    def direction_ok(row: pd.Series) -> bool:
        if not row["nonzero"]:
            return False
        winner = str(row["winner"])
        contribution = float(row["contribution"])
        return (winner == "A" and contribution > 0) or (winner == "B" and contribution < 0)

    work["direction_ok"] = work.apply(direction_ok, axis=1)

    rows: list[dict[str, Any]] = []
    for qid, group in work.groupby("qid"):
        n = int(len(group))
        n_nonzero = int(group["nonzero"].sum())
        nonzero_rate = n_nonzero / n if n else 0.0
        if n_nonzero:
            direction_acc = float(group.loc[group["nonzero"], "direction_ok"].mean())
        else:
            direction_acc = 0.0
        if n < min_support:
            direction_acc_for_score = 0.5
        else:
            direction_acc_for_score = direction_acc
        decisiveness = nonzero_rate * max(direction_acc_for_score - 0.5, 0.0)
        rows.append(
            {
                "qid": int(qid),
                "n": n,
                "n_nonzero": n_nonzero,
                "nonzero_rate": nonzero_rate,
                "direction_acc_when_nonzero": direction_acc,
                "decisiveness": decisiveness,
            }
        )
    return pd.DataFrame(rows).sort_values("qid")


def _rerank_qids(
    qids: list[int],
    prior: dict[int, float],
    *,
    rank_alpha: float,
    prior_beta: float,
    epsilon: float,
) -> list[int]:
    scored: list[tuple[float, int, int]] = []
    for rank, qid in enumerate(qids):
        rank_score = 1.0 / ((rank + 1) ** rank_alpha)
        prior_score = (epsilon + float(prior.get(int(qid), 0.0))) ** prior_beta
        scored.append((rank_score * prior_score, -rank, int(qid)))
    scored.sort(reverse=True)
    return [qid for _, _, qid in scored]


def _apply_subaspect_cap(
    ranked_qids: list[int],
    bank_df: pd.DataFrame,
    k: int,
    cap: int | None,
) -> list[int]:
    if cap is None or cap <= 0:
        return ranked_qids
    if "sub_aspect" not in bank_df.columns:
        return ranked_qids

    q_to_sub = {
        int(row["qid"]): str(row["sub_aspect"])
        for _, row in bank_df[["qid", "sub_aspect"]].iterrows()
    }
    selected: list[int] = []
    delayed: list[int] = []
    counts: dict[str, int] = {}

    for qid in ranked_qids:
        sub = q_to_sub.get(int(qid), "")
        if len(selected) < k and counts.get(sub, 0) < cap:
            selected.append(qid)
            counts[sub] = counts.get(sub, 0) + 1
        else:
            delayed.append(qid)

    return selected + delayed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--selector-picks", type=Path, required=True)
    parser.add_argument("--question-contributions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--rank-alpha", type=float, default=0.5)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--subaspect-cap", type=int, default=2)
    parser.add_argument("--no-dim-quota", action="store_true")
    parser.add_argument("--prior-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.rank_alpha < 0:
        raise ValueError("--rank-alpha must be >= 0")
    if args.prior_beta < 0:
        raise ValueError("--prior-beta must be >= 0")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be > 0")

    bank_df = _load_bank(args.bank)
    picks = pd.read_parquet(args.selector_picks)
    if "sample_id" not in picks.columns:
        raise ValueError(f"{args.selector_picks} missing sample_id")
    qid_col = "ranked_qids" if "ranked_qids" in picks.columns else "selected_qids"
    if qid_col not in picks.columns:
        raise ValueError(f"{args.selector_picks} must contain ranked_qids or selected_qids")

    prior_df = _build_prior(args.question_contributions, args.min_support)
    prior = dict(zip(prior_df["qid"].astype(int), prior_df["decisiveness"].astype(float)))

    if args.prior_out is not None:
        args.prior_out.parent.mkdir(parents=True, exist_ok=True)
        prior_df.to_parquet(args.prior_out, index=False)

    rows: list[dict[str, Any]] = []
    overlaps: list[int] = []
    changed = 0
    before_prior: list[float] = []
    after_prior: list[float] = []

    for _, row in picks.iterrows():
        original_ranked = _parse_qid_list(row[qid_col])
        old_selected = _parse_qid_list(row["selected_qids"])[: args.k] if "selected_qids" in picks.columns else original_ranked[: args.k]

        reranked = _rerank_qids(
            original_ranked,
            prior,
            rank_alpha=args.rank_alpha,
            prior_beta=args.prior_beta,
            epsilon=args.epsilon,
        )
        reranked = _apply_subaspect_cap(reranked, bank_df, args.k, args.subaspect_cap)
        selected = _select_topk_with_quota(
            reranked,
            bank_df,
            str(row.get("domain", "")),
            args.k,
            enforce_quota=not args.no_dim_quota,
        )
        selected_set = set(selected)
        final_ranked = selected + [qid for qid in reranked if qid not in selected_set]

        overlap = len(set(old_selected) & selected_set)
        overlaps.append(overlap)
        changed += int(overlap < min(len(old_selected), len(selected)))
        before_prior.extend(float(prior.get(qid, 0.0)) for qid in old_selected)
        after_prior.extend(float(prior.get(qid, 0.0)) for qid in selected)

        out_row = dict(row)
        out_row["k"] = int(args.k)
        out_row["selected_qids"] = selected
        out_row["ranked_qids"] = final_ranked
        out_row["reranker_overlap_at_k"] = int(overlap)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    summary = {
        "n_samples": int(len(out_df)),
        "k": int(args.k),
        "qid_source_column": qid_col,
        "rank_alpha": args.rank_alpha,
        "prior_beta": args.prior_beta,
        "epsilon": args.epsilon,
        "min_support": args.min_support,
        "subaspect_cap": args.subaspect_cap,
        "changed_at_k": int(changed),
        "mean_overlap_at_k": float(pd.Series(overlaps).mean()) if overlaps else 0.0,
        "mean_prior_before": float(pd.Series(before_prior).mean()) if before_prior else 0.0,
        "mean_prior_after": float(pd.Series(after_prior).mean()) if after_prior else 0.0,
        "out": str(args.out),
    }
    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
