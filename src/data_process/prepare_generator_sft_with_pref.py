#!/usr/bin/env python3
"""
Build generator SFT parquet that mirrors `prepare_generator_sft.py` but
additionally carries the raw ``individual_preference`` list from HelpSteer3
so that generator training can consume per-annotator signals.

Key properties:
- The underlying pair rows are taken verbatim from an existing split parquet
  (e.g. ``splits/train_tier_10k.parquet`` or ``splits/dev_600.parquet``).
  Rows are never resampled or reordered, preserving experimental consistency.
- ``individual_preference`` is looked up from the HelpSteer3 HF dataset by
  rebuilding ``prompt_id`` with the exact same hash as ``prepare_data.py``.
- Checklist target construction is identical to ``prepare_generator_sft.py``.

Usage:
    # Train tier
    python prepare_generator_sft_with_pref.py --split train_tier_10k

    # Dev subset (already materialised by make_dev600.py — not resampled)
    python prepare_generator_sft_with_pref.py --split dev_600

    # Arbitrary dev subset created ad hoc (stratified, reproducible)
    python prepare_generator_sft_with_pref.py --split dev --dev-size 200 \
        --dev-out-name dev_200.parquet
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from data_process.prepare_generator_sft import (
    DOMAIN_ORDER,
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_TEMPLATE,
    aggregate_questions,
    build_generator_messages,
    format_checklist_target,
    load_questions,
)
from data_process.make_dev600 import exact_stratified_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── prompt_id helpers ──────────────────────────
# Must match prepare_data.py exactly so the hashes line up.

def _context_to_text(context: list[dict]) -> str:
    parts = []
    for turn in context:
        parts.append(f"[{turn['role']}]\n{turn['content']}")
    return "\n\n".join(parts)


def _prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ────────────────────────── HelpSteer3 lookup ──────────────────────────

def build_individual_pref_map(
    hf_split: str,
    cache_dir: str | None,
) -> dict[tuple[str, str, str], list[int]]:
    """
    Return {(prompt_id, response_a, response_b): individual_preference}.

    Keying on the triple avoids ambiguity when the same prompt has several
    (response_a, response_b) pairs, which happens in HelpSteer3.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset  # local import so --help stays fast

    log.info("Loading HelpSteer3 %s split from HF ...", hf_split)
    ds = load_dataset(cfg.HF_DATASET_ID, split=hf_split, cache_dir=cache_dir)
    df = ds.to_pandas()

    df["domain_lower"] = df["domain"].str.strip().str.lower()
    df = df[df["domain_lower"].isin(cfg.KEEP_DOMAINS)]
    df = df[df["overall_preference"].isin(cfg.KEEP_PREFERENCES)]

    if "individual_preference" not in df.columns:
        raise RuntimeError(
            "HelpSteer3 rows don't carry an 'individual_preference' column; "
            "the dataset schema may have changed."
        )

    log.info("  usable rows after filtering: %d", len(df))

    mapping: dict[tuple[str, str, str], list[int]] = {}
    for _, row in df.iterrows():
        ctx_text = _context_to_text(row["context"])
        pid = _prompt_id(ctx_text)
        key = (pid, str(row["response1"]), str(row["response2"]))
        prefs = row["individual_preference"]
        # datasets may give np.ndarray — normalise to plain list[int]
        mapping[key] = [int(x) for x in list(prefs)]
    return mapping


# ────────────────────────── split loading / dev split ──────────────────────

def load_or_make_split(
    split: str,
    dev_size: int | None,
    dev_source: str,
    dev_out_name: str | None,
    seed: int,
) -> tuple[pd.DataFrame, Path]:
    """
    Resolve the pair parquet for this run.

    If ``dev_size`` is given and the target parquet doesn't already exist,
    create one by stratified sampling from ``dev_source`` (mirroring
    ``make_dev600.py``). Existing files are returned as-is to preserve
    experimental consistency.
    """
    out_name = dev_out_name or f"{split}.parquet"
    path = cfg.SPLITS_DIR / out_name

    if dev_size is not None and not path.exists():
        src_path = cfg.SPLITS_DIR / dev_source
        if not src_path.exists():
            raise FileNotFoundError(f"dev source {src_path} not found")
        src_df = pd.read_parquet(src_path)
        log.info("Sampling new dev subset n=%d from %s", dev_size, src_path)
        subset = exact_stratified_sample(
            df=src_df,
            n=dev_size,
            strat_cols=["domain", "winner"],
            seed=seed,
        )
        subset.to_parquet(path, index=False)
        log.info("Wrote new dev subset → %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide --dev-size to create it, or run "
            f"prepare_data.py / make_dev600.py first."
        )

    df = pd.read_parquet(path)
    required = {"prompt_id", "context", "response_a", "response_b", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df, path


# ────────────────────────── SFT build ──────────────────────────

def build_sft_rows_with_pref(
    pairs: pd.DataFrame,
    questions_by_sample: dict[str, dict[str, list[str]]],
    pref_map: dict[tuple[str, str, str], list[int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict] = []
    n_no_questions = 0
    n_no_pref = 0
    for _, r in pairs.iterrows():
        sid = r.get("sample_id", r.get("prompt_id"))
        per_domain = questions_by_sample.get(sid) if sid is not None else None
        if not per_domain:
            n_no_questions += 1
            continue
        target = format_checklist_target(per_domain)
        if not target:
            n_no_questions += 1
            continue

        pid = r["prompt_id"]
        key = (pid, str(r["response_a"]), str(r["response_b"]))
        prefs = pref_map.get(key)
        if prefs is None:
            n_no_pref += 1
            continue

        messages = build_generator_messages(r)
        rows.append(
            {
                "sample_id": sid,
                "prompt_id": pid,
                "domain": r["domain"],
                "messages": json.dumps(messages, ensure_ascii=False),
                "target_output": target,
                "individual_preference": prefs,
                "n_annotators": len(prefs),
                "pref_mean": float(np.mean(prefs)) if prefs else float("nan"),
                "pref_unanimous": bool(len(set(np.sign(prefs))) == 1) if prefs else False,
                "n_questions": sum(len(v) for v in per_domain.values()),
                "n_domains": len(per_domain),
            }
        )

    stats = {
        "n_rows": len(rows),
        "skipped_no_questions": n_no_questions,
        "skipped_no_individual_pref": n_no_pref,
    }
    log.info(
        "Built %d rows (dropped %d w/o questions, %d w/o individual_preference)",
        len(rows), n_no_questions, n_no_pref,
    )
    return pd.DataFrame(rows), stats


def print_summary(df: pd.DataFrame, out_path: Path, stats: dict[str, int]) -> None:
    table = Table(title="Generator SFT (+ individual_preference)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{len(df):,}")
    for k, v in stats.items():
        if k == "n_rows":
            continue
        table.add_row(k, f"{v:,}")
    if len(df):
        table.add_row("Avg questions / sample", f"{df['n_questions'].mean():.1f}")
        table.add_row("Avg annotators / sample", f"{df['n_annotators'].mean():.2f}")
        table.add_row("Unanimous share", f"{df['pref_unanimous'].mean():.1%}")
    table.add_row("Output", str(out_path))
    console.print(table)


# ────────────────────────── CLI ──────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        required=True,
        help="Split basename under data/splits/ (e.g. train_tier_10k, dev, dev_600).",
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help="Override reasoning questions parquet "
             "(default: data/<split>_reasoning_questions.parquet for train_* splits, "
             "else data/<split>_reasoning_questions.parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output (default: data/generator_sft/<split>.parquet)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="HelpSteer3 split to pull individual_preference from. "
             "Use 'train' for train_* and dev_* splits (dev is carved out of train).",
    )
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF cache dir (forwarded to datasets.load_dataset).")

    # Dev subset creation (optional; never overwrites existing files).
    parser.add_argument("--dev-size", type=int, default=None,
                        help="If set and the target split file doesn't exist, "
                             "create it by stratified sampling.")
    parser.add_argument("--dev-source", type=str, default="dev.parquet",
                        help="Source parquet used when sampling a new dev subset.")
    parser.add_argument("--dev-out-name", type=str, default=None,
                        help="Override output name for the sampled dev subset "
                             "(default: <split>.parquet).")
    parser.add_argument("--seed", type=int, default=cfg.SEED)

    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N pairs (debugging).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats + one example without writing.")
    args = parser.parse_args()

    # 1) Resolve pair parquet (optionally materialise a new dev subset).
    pairs, pair_path = load_or_make_split(
        split=args.split,
        dev_size=args.dev_size,
        dev_source=args.dev_source,
        dev_out_name=args.dev_out_name,
        seed=args.seed,
    )
    log.info("Pairs: %d rows from %s", len(pairs), pair_path)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)

    # 2) Reasoning questions → target checklists.
    questions_path = (
        Path(args.questions_path)
        if args.questions_path
        else cfg.DATA_DIR / f"{args.split}_reasoning_questions.parquet"
    )
    df_q = load_questions(questions_path)
    questions_by_sample = aggregate_questions(df_q)

    # 3) Pull individual_preference from HelpSteer3 and index by prompt_id + responses.
    pref_map = build_individual_pref_map(args.hf_split, args.cache_dir)

    # 4) Build SFT rows.
    sft_df, stats = build_sft_rows_with_pref(pairs, questions_by_sample, pref_map)

    out_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.GENERATOR_SFT_DIR / f"{args.split}.parquet"
    )
    print_summary(sft_df, out_path, stats)

    if args.dry_run:
        if not sft_df.empty:
            r = sft_df.iloc[0]
            console.print(f"\n[cyan]sample_id[/cyan]: {r['sample_id']}")
            console.print(f"[cyan]individual_preference[/cyan]: {r['individual_preference']}")
            console.print(f"[cyan]target_output[/cyan]:\n{r['target_output']}")
        return

    if sft_df.empty:
        raise SystemExit("No SFT rows produced — check inputs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(out_path, index=False)

    meta = {
        "split": args.split,
        "pair_source": str(pair_path),
        "questions_source": str(questions_path),
        "hf_split": args.hf_split,
        "seed": args.seed,
        "n_rows": int(len(sft_df)),
        **stats,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved %d rows → %s", len(sft_df), out_path)
    log.info("Saved metadata → %s", meta_path)
    console.print(f"\n[bold green]Done. {out_path}[/bold green]")


if __name__ == "__main__":
    main()
