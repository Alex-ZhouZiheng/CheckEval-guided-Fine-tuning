#!/usr/bin/env python3
"""
Phase 0 — Checklist Bank Cleanup.

Diagnoses and prunes the checklist bank used by CheckEval, producing a
``checklists/v2/`` directory plus audit artefacts under
``results/bank_cleanup/``.

Core principle: NA means "not applicable" — it is NOT a defect signal.
Pruning decisions are made only from per-item behaviour on the *non-NA*
subset (yes-rate saturation, label-correlation signal). A high NA rate
is recorded as an advisory tag only.

Prerequisite (run manually first):

    python src/run_checkeval_judge.py --eval-split dev_600 --na-policy strict
    python src/run_checkeval_judge.py --eval-split test    --na-policy strict

which creates::

    results/checkeval_pairwise_naaware_{dev_600,test}_predictions.parquet
    results/checkeval_pairwise_naaware_{dev_600,test}_question_diagnostics.csv

Then::

    python src/diagnose_and_clean_bank.py all --dev-split dev_600 --test-split test
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

import config as cfg
from utils import (
    build_question_index,
    expected_question_count,
    parse_checkeval_output,
)

log = logging.getLogger("bank_cleanup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ─────────────────────── paths ───────────────────────
BANK_IN_DIR = cfg.PROJECT_ROOT / "checklists" / "filtered"
BANK_OUT_DIR = cfg.PROJECT_ROOT / "checklists" / "v2"
CLEANUP_DIR = cfg.RESULTS_DIR / "bank_cleanup"

EXP_PREFIX = "checkeval_pairwise_naaware"


def pred_path(split: str) -> Path:
    return cfg.RESULTS_DIR / f"{EXP_PREFIX}_{split}_predictions.parquet"


def qdiag_path(split: str) -> Path:
    return cfg.RESULTS_DIR / f"{EXP_PREFIX}_{split}_question_diagnostics.csv"


def dimdiag_path(split: str) -> Path:
    return cfg.RESULTS_DIR / f"{EXP_PREFIX}_{split}_dimension_diagnostics.json"


# ─────────────── structured bank loader ────────────

def load_structured_bank(bank_dir: Path) -> dict[str, dict]:
    """Load YAML bank *preserving* sub_aspects structure.

    Returns ``{dimension: {definition, dimension, sub_aspects: {...}}}``.
    Unlike ``utils.load_checklists`` this does not flatten sub_aspects.
    """
    if not bank_dir.exists():
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")

    structured: dict[str, dict] = {}
    for yaml_path in sorted(bank_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dim = data.get("dimension", yaml_path.stem)
        structured[dim] = data
    if not structured:
        raise RuntimeError(f"No *_filtered.yaml files in {bank_dir}")
    return structured


def flat_from_structured(
    structured: dict[str, dict],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Project the structured bank into the flat shape used by ``utils``.

    Mirrors ``utils.load_checklists`` but without disk I/O, so that the
    ephemeral qids produced by ``build_question_index`` match what the
    judge saw at inference time.
    """
    checklists: dict[str, list[str]] = {}
    definitions: dict[str, str] = {}
    for dim, data in structured.items():
        definitions[dim] = data.get("definition", "")
        seen: set[str] = set()
        questions: list[str] = []
        for sub_data in data.get("sub_aspects", {}).values():
            for q in sub_data.get("filtered_questions", []):
                q_norm = q.strip()
                if q_norm not in seen:
                    seen.add(q_norm)
                    questions.append(q_norm)
        checklists[dim] = questions
    return checklists, definitions


def build_question_to_subaspect(
    structured: dict[str, dict],
) -> dict[tuple[str, str], str]:
    """Map (dimension, question_text) → sub_aspect name for joining stats."""
    q_to_sub: dict[tuple[str, str], str] = {}
    for dim, data in structured.items():
        for sub_name, sub_data in data.get("sub_aspects", {}).items():
            for q in sub_data.get("filtered_questions", []):
                key = (dim, q.strip())
                if key in q_to_sub and q_to_sub[key] != sub_name:
                    log.warning(
                        "Duplicate question text in dim=%s across sub_aspects %s and %s",
                        dim, q_to_sub[key], sub_name,
                    )
                q_to_sub[key] = sub_name
    return q_to_sub


# ─────────────── prerequisite checks ────────────

def require_prereqs(splits: list[str]) -> None:
    missing: list[str] = []
    for split in splits:
        if not pred_path(split).exists():
            missing.append(str(pred_path(split)))
        if not qdiag_path(split).exists():
            missing.append(str(qdiag_path(split)))
    if missing:
        log.error("Missing prerequisite files:")
        for m in missing:
            log.error("  - %s", m)
        log.error("")
        log.error("Please run the judge first. For each missing split:")
        for split in splits:
            log.error(
                "  python src/run_checkeval_judge.py --eval-split %s --na-policy strict",
                split,
            )
        sys.exit(2)


# ─────────────── re-parse raw predictions ────────────

def reparse_predictions(
    pred_df: pd.DataFrame,
    flat: dict[str, list[str]],
) -> tuple[list[dict], list[dict]]:
    """Rebuild parsed_a_list / parsed_b_list by running ``parse_checkeval_output``
    on the raw judge text stored in the predictions parquet."""
    parsed_a: list[dict] = []
    parsed_b: list[dict] = []
    for _, row in pred_df.iterrows():
        exp_n = expected_question_count(row["domain"], flat)
        parsed_a.append(parse_checkeval_output(row["raw_output_a"], expected_n=exp_n))
        parsed_b.append(parse_checkeval_output(row["raw_output_b"], expected_n=exp_n))
    return parsed_a, parsed_b


# ─────────────── label correlation ────────────

def compute_label_correlation(
    pred_df: pd.DataFrame,
    parsed_a: list[dict],
    parsed_b: list[dict],
    flat: dict[str, list[str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per-item agreement rate with gold winner.

    For every example where an item is answered (non-NA) on BOTH sides and
    the two sides disagree, cast a vote: +1 → A, -1 → B. Compare with the
    gold winner. Return per ``(dimension, question_text)``:

        {"agree_rate": n_correct / n_effective, "n_effective": n_effective}

    Ties (both sides agree) are excluded from n_effective — they provide
    no pairwise signal.
    """
    acc: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n_correct": 0, "n_effective": 0}
    )

    domains = pred_df["domain"].tolist()
    winners = pred_df["winner"].tolist()

    # cache qindex per domain to avoid rebuilding 600 times
    qindex_cache: dict[str, dict[int, dict[str, str]]] = {}
    for d in set(domains):
        qindex_cache[d] = build_question_index(flat, d)

    for i, (pa, pb, domain, winner) in enumerate(zip(parsed_a, parsed_b, domains, winners)):
        if pa.get("_raw_fallback") or pb.get("_raw_fallback"):
            continue
        if winner not in ("A", "B"):
            continue
        qindex = qindex_cache[domain]

        a_ans = {a["q"]: a["answer"] for a in pa.get("answers", [])}
        b_ans = {a["q"]: a["answer"] for a in pb.get("answers", [])}
        a_na = {a["q"] for a in pa.get("na_answers", [])}
        b_na = {a["q"] for a in pb.get("na_answers", [])}

        # items answered (non-NA) on both sides
        common = (set(a_ans) & set(b_ans)) - a_na - b_na

        for q in common:
            av = a_ans[q]
            bv = b_ans[q]
            if av == bv:
                continue  # no pairwise signal
            info = qindex.get(q)
            if info is None:
                continue
            key = (info["dimension"], info["question"].strip())
            vote_a = av == "yes"  # A gets yes, B gets no → vote for A
            correct = (vote_a and winner == "A") or ((not vote_a) and winner == "B")
            acc[key]["n_effective"] += 1
            if correct:
                acc[key]["n_correct"] += 1

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, v in acc.items():
        n_eff = v["n_effective"]
        out[key] = {
            "agree_rate": (v["n_correct"] / n_eff) if n_eff > 0 else None,
            "n_effective": n_eff,
        }
    return out


# ─────────────── aggregate by question text ────────────

def aggregate_to_question_text(
    diag_df: pd.DataFrame,
    structured: dict[str, dict],
    corr_map: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Collapse per-(domain, qid) rows to per-(dimension, sub_aspect, question_text).

    - Sums n_yes / n_no / n_na across domains (same question can get different
      qids on general vs code when coding_communication_conditional shifts
      the numbering).
    - Computes ``yes_rate_nonNA = n_yes / (n_yes + n_no)`` — the authoritative
      saturation metric used by drop rules.
    - Looks up sub_aspect from the structured bank.
    - Attaches label-correlation signal.
    """
    # normalize text before joining
    diag_df = diag_df.copy()
    diag_df["question_text"] = diag_df["question_text"].astype(str).str.strip()

    grp = diag_df.groupby(["dimension", "question_text"], as_index=False).agg(
        n_yes=("n_yes", "sum"),
        n_no=("n_no", "sum"),
        n_na=("n_na", "sum"),
        n_total=("n_total", "sum"),
        domains_seen=("domain", lambda s: ",".join(sorted(set(s)))),
    )

    nonna = (grp["n_yes"] + grp["n_no"]).astype("float64")
    grp["n_effective_nonna"] = nonna.astype("int64")
    grp["yes_rate_nonNA"] = grp["n_yes"] / nonna.where(nonna > 0)
    total_f = grp["n_total"].astype("float64")
    grp["na_rate"] = grp["n_na"] / total_f.where(total_f > 0)

    # attach sub_aspect
    q_to_sub = build_question_to_subaspect(structured)
    grp["sub_aspect"] = [
        q_to_sub.get((d, q), "?UNKNOWN")
        for d, q in zip(grp["dimension"], grp["question_text"])
    ]

    unknown = grp[grp["sub_aspect"] == "?UNKNOWN"]
    if not unknown.empty:
        log.warning(
            "%d items could not be matched to a sub_aspect (text drift?):",
            len(unknown),
        )
        for _, r in unknown.iterrows():
            log.warning("  [%s] %s", r["dimension"], r["question_text"][:80])

    # attach correlation
    grp["agree_rate"] = [
        corr_map.get((d, q), {}).get("agree_rate")
        for d, q in zip(grp["dimension"], grp["question_text"])
    ]
    grp["n_effective_corr"] = [
        int(corr_map.get((d, q), {}).get("n_effective", 0))
        for d, q in zip(grp["dimension"], grp["question_text"])
    ]

    cols = [
        "dimension", "sub_aspect", "question_text",
        "n_yes", "n_no", "n_na", "n_total",
        "n_effective_nonna", "yes_rate_nonNA", "na_rate",
        "agree_rate", "n_effective_corr",
        "domains_seen",
    ]
    return grp[cols].sort_values(["dimension", "sub_aspect", "question_text"]).reset_index(drop=True)


# ─────────────── drop rules ────────────

DROP_TRIGGER_REASONS = frozenset({"saturated_yes", "saturated_no", "low_signal"})


def apply_drop_rules(
    df: pd.DataFrame,
    *,
    yes_high: float,
    yes_low: float,
    signal_threshold: float,
    signal_min_n: int,
    rare_na_threshold: float,
    sat_min_n: int = 30,
) -> pd.DataFrame:
    """Annotate each row with ``drop_reasons`` and ``drop``.

    Rules (see design doc for rationale):
      - saturated_yes : yes_rate_nonNA > yes_high  AND n_effective_nonna >= sat_min_n
      - saturated_no  : yes_rate_nonNA < yes_low   AND n_effective_nonna >= sat_min_n
      - low_signal    : |agree_rate - 0.5| < signal_threshold AND n_effective_corr >= signal_min_n
      - rarely_applicable  (advisory, never drops)
      - tiny_effective_n   (advisory, never drops)
    """
    df = df.copy()
    reasons_col: list[list[str]] = []

    for row in df.itertuples(index=False):
        rs: list[str] = []

        nonna = int(getattr(row, "n_effective_nonna") or 0)
        yes_rate = getattr(row, "yes_rate_nonNA")
        na_rate = getattr(row, "na_rate")
        agree = getattr(row, "agree_rate")
        n_eff_corr = int(getattr(row, "n_effective_corr") or 0)

        if nonna >= sat_min_n:
            if yes_rate is not None and not pd.isna(yes_rate):
                if yes_rate > yes_high:
                    rs.append("saturated_yes")
                if yes_rate < yes_low:
                    rs.append("saturated_no")
        else:
            rs.append("tiny_effective_n")

        if (
            n_eff_corr >= signal_min_n
            and agree is not None
            and not pd.isna(agree)
            and abs(agree - 0.5) < signal_threshold
        ):
            rs.append("low_signal")

        if na_rate is not None and not pd.isna(na_rate) and na_rate > rare_na_threshold:
            rs.append("rarely_applicable")

        reasons_col.append(rs)

    df["drop_reasons"] = reasons_col
    df["drop"] = df["drop_reasons"].apply(
        lambda rs: any(r in DROP_TRIGGER_REASONS for r in rs)
    )
    return df


# ─────────────── Bank v2 writer ────────────

def write_bank_v2(
    structured: dict[str, dict],
    drop_set: set[tuple[str, str]],
    out_dir: Path,
    *,
    prune_empty_subaspects: bool = False,
) -> dict[str, dict[str, int]]:
    """Write cleaned YAML files preserving the original schema.

    Returns per-dimension ``{kept, dropped, orig}`` counts for reporting.
    Raises if a whole dimension would become empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dim_counts: dict[str, dict[str, int]] = {}

    for dim, data in structured.items():
        new_sub_aspects: dict[str, Any] = {}
        kept_total = 0
        dropped_total = 0
        orig_total = 0

        for sub_name, sub_data in data.get("sub_aspects", {}).items():
            old_qs_raw = list(sub_data.get("filtered_questions", []))
            old_qs = [q.strip() for q in old_qs_raw]
            kept = [q for q in old_qs if (dim, q) not in drop_set]
            dropped_here = len(old_qs) - len(kept)
            orig_count_field = int(sub_data.get("original_count", len(old_qs)))
            orig_total += orig_count_field
            kept_total += len(kept)
            dropped_total += dropped_here

            if not kept:
                log.warning(
                    "  [%s / %s] sub_aspect empty after cleanup (was %d items)",
                    dim, sub_name, len(old_qs),
                )
                if prune_empty_subaspects:
                    continue

            # cumulative removed: original_count - current kept
            new_sub_aspects[sub_name] = {
                "filtered_questions": kept,
                "original_count": orig_count_field,
                "removed_count": orig_count_field - len(kept),
                "seed_question": sub_data.get("seed_question", ""),
            }

        if kept_total == 0:
            raise RuntimeError(
                f"Dimension {dim!r} has zero kept items after cleanup. "
                "Thresholds are too aggressive — aborting."
            )

        new_data = {
            "definition": data.get("definition", ""),
            "dimension": dim,
            "sub_aspects": new_sub_aspects,
        }
        out_path = out_dir / f"{dim}_filtered.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                new_data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                width=1000,
            )
        log.info(
            "Wrote %s — kept %d / orig %d (dropped %d)",
            out_path.name, kept_total, orig_total, dropped_total,
        )

        per_dim_counts[dim] = {
            "kept": kept_total,
            "dropped": dropped_total,
            "orig": orig_total,
        }

    return per_dim_counts


# ─────────────── reports ────────────

def write_drop_report(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["drop_reasons"] = df_out["drop_reasons"].apply(lambda rs: "|".join(rs))
    df_out.to_csv(path, index=False)
    n_drop = int(df_out["drop"].sum())
    log.info("Wrote drop_report -> %s  (%d items flagged, %d drop)", path, len(df_out), n_drop)


def write_item_stats(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote item_stats -> %s  (%d rows)", path, len(df))


def write_before_after_summary(
    per_dim_counts: dict[str, dict[str, int]],
    stats_df: pd.DataFrame,
    dim_acc_before: dict[str, dict[str, Any]] | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # compute avg NA rate per dimension BEFORE from the item stats
    avg_na_before: dict[str, float] = (
        stats_df.groupby("dimension")["na_rate"].mean().to_dict()
    )
    # avg NA rate AFTER: only consider kept items
    kept_mask = ~stats_df["drop"]
    avg_na_after: dict[str, float] = (
        stats_df[kept_mask].groupby("dimension")["na_rate"].mean().to_dict()
    )

    lines = [
        "# Checklist Bank Cleanup — Before / After",
        "",
        "| dimension | items_before | items_after | dropped | avg_na_before | avg_na_after | dim_acc_before |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dim in sorted(per_dim_counts.keys()):
        counts = per_dim_counts[dim]
        dim_acc = "-"
        if dim_acc_before and dim in dim_acc_before:
            dim_acc = f"{dim_acc_before[dim].get('dimension_accuracy', float('nan')):.4f}"
        na_b = avg_na_before.get(dim, float("nan"))
        na_a = avg_na_after.get(dim, float("nan"))
        lines.append(
            f"| {dim} | {counts['orig']} | {counts['kept']} | {counts['dropped']} | "
            f"{na_b:.4f} | {na_a:.4f} | {dim_acc} |"
        )

    lines += [
        "",
        "Note: `dim_acc_before` is read from the existing",
        "`*_dimension_diagnostics.json` if present. Re-run `run_checkeval_judge.py`",
        "with `--checklists-dir checklists/v2` to measure the *after* dim_accuracy.",
        "",
        "Drop reason legend:",
        "- **saturated_yes / saturated_no** (hard drop): `yes_rate_nonNA` above `--yes-high` or below `--yes-low` with ≥ `sat_min_n` non-NA answers.",
        "- **low_signal** (advisory drop): `|agree_rate - 0.5| < --signal-threshold` with ≥ `--signal-min-n` effective comparisons.",
        "- **rarely_applicable** (tag only): `na_rate > --rare-na-threshold`. Does NOT trigger drop — NA means 'not applicable', which is legitimate.",
        "- **tiny_effective_n** (tag only): too few non-NA answers to judge. Does NOT trigger drop.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote before_after summary -> %s", path)


def load_dim_acc_before(split: str) -> dict[str, dict[str, Any]] | None:
    p = dimdiag_path(split)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to parse %s: %s", p, e)
        return None


# ─────────────── teacher audit sampler ────────────

def sample_teacher_audit(
    pred_df: pd.DataFrame,
    parsed_a: list[dict],
    parsed_b: list[dict],
    flat: dict[str, list[str]],
    n: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stratified (by dimension) sample of teacher labels for manual review."""
    rng = random.Random(seed)
    pred_df = pred_df.reset_index(drop=True)

    # cache qindex per domain
    qindex_cache: dict[str, dict[int, dict[str, str]]] = {}
    for d in set(pred_df["domain"].tolist()):
        qindex_cache[d] = build_question_index(flat, d)

    pool: list[dict[str, Any]] = []
    for idx, row in pred_df.iterrows():
        pa = parsed_a[idx]
        pb = parsed_b[idx]
        if pa.get("_raw_fallback") or pb.get("_raw_fallback"):
            continue
        domain = row["domain"]
        qindex = qindex_cache[domain]
        for side, parsed in (("A", pa), ("B", pb)):
            for a in parsed.get("answers", []):
                info = qindex.get(a["q"])
                if info is None:
                    continue
                pool.append({
                    "row_idx": int(idx),
                    "side": side,
                    "qid": int(a["q"]),
                    "dimension": info["dimension"],
                    "question_text": info["question"],
                    "teacher_answer": a["answer"],
                })
            for a in parsed.get("na_answers", []):
                info = qindex.get(a["q"])
                if info is None:
                    continue
                pool.append({
                    "row_idx": int(idx),
                    "side": side,
                    "qid": int(a["q"]),
                    "dimension": info["dimension"],
                    "question_text": info["question"],
                    "teacher_answer": "NA",
                })

    if not pool:
        log.warning("Teacher audit pool is empty — no parseable items found.")
        return []

    # stratified: evenly split across dimensions present in the pool
    dims = sorted({p["dimension"] for p in pool})
    per_dim = max(1, n // max(1, len(dims)))
    sampled: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    for dim in dims:
        dim_pool = [p for p in pool if p["dimension"] == dim]
        rng.shuffle(dim_pool)
        for item in dim_pool[:per_dim]:
            sampled.append(item)
            used_ids.add(id(item))

    # fill any remainder
    remaining = n - len(sampled)
    if remaining > 0:
        extra = [p for p in pool if id(p) not in used_ids]
        rng.shuffle(extra)
        sampled.extend(extra[:remaining])

    sampled = sampled[:n]

    # attach context / response snippets
    for s in sampled:
        row = pred_df.iloc[s["row_idx"]]
        s["prompt_id"] = row.get("prompt_id", f"row_{s['row_idx']}")
        s["domain"] = row["domain"]
        s["gold_winner"] = row["winner"]
        resp_col = "response_a" if s["side"] == "A" else "response_b"
        s["context_excerpt"] = str(row.get("context", ""))[:500]
        s["response_excerpt"] = str(row.get(resp_col, ""))[:500]

    return sampled


def write_teacher_audit(sampled: list[dict[str, Any]], md_path: Path, csv_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV (flat, one row per item)
    fieldnames = [
        "prompt_id", "domain", "gold_winner", "side", "dimension", "qid",
        "question_text", "teacher_answer", "human_label", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sampled:
            writer.writerow({
                "prompt_id": s["prompt_id"],
                "domain": s["domain"],
                "gold_winner": s["gold_winner"],
                "side": s["side"],
                "dimension": s["dimension"],
                "qid": s["qid"],
                "question_text": s["question_text"],
                "teacher_answer": s["teacher_answer"],
                "human_label": "",  # to be filled manually
                "notes": "",
            })

    # Markdown (human-readable)
    lines: list[str] = [
        "# Teacher Label Audit (manual review)",
        "",
        f"Sample size: {len(sampled)}.  Fill in `human_label` with yes/no/NA and optionally `notes`.",
        "",
        f"Paired CSV (for tallying): `{csv_path.name}`.",
        "",
    ]
    by_dim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sampled:
        by_dim[s["dimension"]].append(s)

    for dim in sorted(by_dim.keys()):
        lines.append(f"## {dim}")
        lines.append("")
        for s in by_dim[dim]:
            lines.append(f"### [{s['prompt_id']}] side={s['side']}  domain={s['domain']}  gold_winner={s['gold_winner']}")
            lines.append("")
            lines.append(f"**Question (Q{s['qid']}):** {s['question_text']}")
            lines.append("")
            lines.append(f"**Teacher answer:** `{s['teacher_answer']}`")
            lines.append("")
            lines.append("**Context excerpt:**")
            lines.append("")
            lines.append("> " + s["context_excerpt"].replace("\n", "\n> "))
            lines.append("")
            lines.append(f"**Response {s['side']} excerpt:**")
            lines.append("")
            lines.append("> " + s["response_excerpt"].replace("\n", "\n> "))
            lines.append("")
            lines.append("**Human label:** ___    **Notes:**")
            lines.append("")
            lines.append("---")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote teacher_audit -> %s (+ %s)", md_path, csv_path.name)


# ─────────────── subcommand: stats ────────────

def run_stats(split: str, structured: dict[str, dict]) -> pd.DataFrame:
    """Compute per-question aggregated stats for ``split`` and write CSV.

    Returns the aggregated DataFrame for in-memory reuse by ``run_clean``.
    """
    log.info("── stats [%s] ──", split)
    diag_df = pd.read_csv(qdiag_path(split))
    pred_df = pd.read_parquet(pred_path(split))
    log.info("Loaded %d diagnostic rows, %d prediction rows", len(diag_df), len(pred_df))

    flat, _defs = flat_from_structured(structured)
    parsed_a, parsed_b = reparse_predictions(pred_df, flat)
    corr_map = compute_label_correlation(pred_df, parsed_a, parsed_b, flat)

    stats_df = aggregate_to_question_text(diag_df, structured, corr_map)

    out_path = CLEANUP_DIR / f"item_stats_{split}.csv"
    write_item_stats(stats_df, out_path)
    return stats_df


# ─────────────── subcommand: clean ────────────

def run_clean(
    dev_stats_df: pd.DataFrame,
    structured: dict[str, dict],
    dev_split: str,
    *,
    yes_high: float,
    yes_low: float,
    signal_threshold: float,
    signal_min_n: int,
    rare_na_threshold: float,
    sat_min_n: int,
    prune_empty_subaspects: bool,
    meta_extra: dict[str, Any],
) -> None:
    log.info("── clean (decisions from %s) ──", dev_split)

    flagged = apply_drop_rules(
        dev_stats_df,
        yes_high=yes_high,
        yes_low=yes_low,
        signal_threshold=signal_threshold,
        signal_min_n=signal_min_n,
        rare_na_threshold=rare_na_threshold,
        sat_min_n=sat_min_n,
    )

    drop_set: set[tuple[str, str]] = {
        (row.dimension, row.question_text)
        for row in flagged.itertuples(index=False)
        if row.drop
    }
    log.info("Dropping %d / %d items", len(drop_set), len(flagged))

    per_dim_counts = write_bank_v2(
        structured, drop_set, BANK_OUT_DIR,
        prune_empty_subaspects=prune_empty_subaspects,
    )

    drop_report_path = CLEANUP_DIR / "drop_report.csv"
    write_drop_report(flagged, drop_report_path)

    summary_path = CLEANUP_DIR / "before_after.md"
    dim_acc_before = load_dim_acc_before(dev_split)
    write_before_after_summary(per_dim_counts, flagged, dim_acc_before, summary_path)

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "dev_split": dev_split,
        "bank_in_dir": str(BANK_IN_DIR),
        "bank_out_dir": str(BANK_OUT_DIR),
        "thresholds": {
            "yes_high": yes_high,
            "yes_low": yes_low,
            "signal_threshold": signal_threshold,
            "signal_min_n": signal_min_n,
            "rare_na_threshold": rare_na_threshold,
            "sat_min_n": sat_min_n,
            "prune_empty_subaspects": prune_empty_subaspects,
        },
        "per_dim_counts": per_dim_counts,
        "n_items_flagged": int(flagged["drop"].sum()),
        "n_items_total": int(len(flagged)),
        **meta_extra,
    }
    (CLEANUP_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote meta.json")


# ─────────────── subcommand: audit ────────────

def run_audit(
    split: str,
    structured: dict[str, dict],
    n: int,
    seed: int,
) -> None:
    log.info("── audit [%s] n=%d seed=%d ──", split, n, seed)
    pred_df = pd.read_parquet(pred_path(split))
    flat, _defs = flat_from_structured(structured)
    parsed_a, parsed_b = reparse_predictions(pred_df, flat)
    sampled = sample_teacher_audit(pred_df, parsed_a, parsed_b, flat, n=n, seed=seed)
    md_path = CLEANUP_DIR / "teacher_audit.md"
    csv_path = CLEANUP_DIR / "teacher_audit.csv"
    write_teacher_audit(sampled, md_path, csv_path)


# ─────────────── CLI ────────────

def _common_cleanup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dev-split", default="dev_600",
                        help="Split used for drop decisions (default: dev_600)")
    parser.add_argument("--test-split", default="test",
                        help="Extra split to compute stats on for audit (default: test)")
    parser.add_argument("--yes-high", type=float, default=0.95)
    parser.add_argument("--yes-low", type=float, default=0.05)
    parser.add_argument("--sat-min-n", type=int, default=30,
                        help="Min non-NA answers required before saturation rules apply")
    parser.add_argument("--signal-threshold", type=float, default=0.02,
                        help="|agree_rate - 0.5| strictly below this → low_signal")
    parser.add_argument("--signal-min-n", type=int, default=30)
    parser.add_argument("--rare-na-threshold", type=float, default=0.80,
                        help="NA rate above this → advisory 'rarely_applicable' tag only")
    parser.add_argument("--audit-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-empty-subaspects", action="store_true")
    parser.add_argument("--bank-in", type=Path, default=BANK_IN_DIR,
                        help="Input bank directory (default: checklists/filtered)")
    parser.add_argument("--bank-out", type=Path, default=BANK_OUT_DIR,
                        help="Output bank directory (default: checklists/v2)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checklist Bank Cleanup (Phase 0)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_stats = sub.add_parser("stats", help="Compute per-item stats CSVs")
    _common_cleanup_args(sp_stats)

    sp_clean = sub.add_parser("clean", help="Apply drop rules and write Bank v2")
    _common_cleanup_args(sp_clean)

    sp_audit = sub.add_parser("audit", help="Sample 50 teacher labels for manual review")
    _common_cleanup_args(sp_audit)

    sp_all = sub.add_parser("all", help="Run stats → clean → audit")
    _common_cleanup_args(sp_all)

    args = parser.parse_args()

    # resolve bank paths (let user override via flags)
    global BANK_IN_DIR, BANK_OUT_DIR
    BANK_IN_DIR = args.bank_in
    BANK_OUT_DIR = args.bank_out

    CLEANUP_DIR.mkdir(parents=True, exist_ok=True)
    structured = load_structured_bank(BANK_IN_DIR)
    log.info(
        "Loaded bank from %s — %d dimensions, %d total items",
        BANK_IN_DIR, len(structured),
        sum(
            sum(len(sa.get("filtered_questions", [])) for sa in d.get("sub_aspects", {}).values())
            for d in structured.values()
        ),
    )

    if args.cmd == "stats":
        require_prereqs([args.dev_split, args.test_split])
        run_stats(args.dev_split, structured)
        run_stats(args.test_split, structured)

    elif args.cmd == "clean":
        require_prereqs([args.dev_split])
        dev_stats = run_stats(args.dev_split, structured)
        run_clean(
            dev_stats, structured, args.dev_split,
            yes_high=args.yes_high, yes_low=args.yes_low,
            signal_threshold=args.signal_threshold,
            signal_min_n=args.signal_min_n,
            rare_na_threshold=args.rare_na_threshold,
            sat_min_n=args.sat_min_n,
            prune_empty_subaspects=args.prune_empty_subaspects,
            meta_extra={},
        )

    elif args.cmd == "audit":
        require_prereqs([args.dev_split])
        run_audit(args.dev_split, structured, args.audit_n, args.seed)

    elif args.cmd == "all":
        require_prereqs([args.dev_split, args.test_split])
        dev_stats = run_stats(args.dev_split, structured)
        run_stats(args.test_split, structured)
        run_clean(
            dev_stats, structured, args.dev_split,
            yes_high=args.yes_high, yes_low=args.yes_low,
            signal_threshold=args.signal_threshold,
            signal_min_n=args.signal_min_n,
            rare_na_threshold=args.rare_na_threshold,
            sat_min_n=args.sat_min_n,
            prune_empty_subaspects=args.prune_empty_subaspects,
            meta_extra={"test_split": args.test_split},
        )
        run_audit(args.dev_split, structured, args.audit_n, args.seed)

    log.info("Done.")


if __name__ == "__main__":
    main()
