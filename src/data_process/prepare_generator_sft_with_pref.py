#!/usr/bin/env python3
"""
Build generator SFT parquet that mirrors `prepare_generator_sft.py` but
additionally carries the raw ``individual_preference`` list from HelpSteer3
so that generator training can consume per-annotator signals.

Key properties:
- The underlying pair rows are taken verbatim from an existing split parquet
  (e.g. ``splits/train_tier_10k.parquet`` or ``splits/dev_600.parquet``).
  Rows are never resampled or reordered, preserving experimental consistency.
- ``individual_preference`` (list of dicts with score/reasoning/feedback1/
  feedback2) is looked up from the HelpSteer3 HF dataset, keyed by
  (prompt_id, response_a, response_b).
- ``sample_id`` is computed with the exact same function as
  ``prepare_data_reasoning.py`` so that the join with the questions parquet
  always aligns.
- Checklist target construction is identical to ``prepare_generator_sft.py``.

Usage:
    # Train tier (questions produced by extract_reasoning_checklist_labels.py)
    python prepare_generator_sft_with_pref.py --split train_tier_10k

    # Dev subset (not resampled — preserves experimental consistency)
    python prepare_generator_sft_with_pref.py --split dev_600

    # Create a new dev subset on the fly (stratified), then build SFT data
    python prepare_generator_sft_with_pref.py --split dev --dev-size 200 \\
        --dev-out-name dev_200.parquet
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from prepare_generator_sft import (
    aggregate_questions,
    build_generator_messages,
    format_checklist_target,
    load_questions,
)
from prepare_data_reasoning import make_sample_id
from make_dev600 import exact_stratified_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── HelpSteer3 lookup ──────────────────────────

def build_pref_map(
    hf_split: str,
    cache_dir: str | None,
) -> dict[tuple[str, str, str], list[dict]]:
    """
    Return {(prompt_id, response_a, response_b): individual_preference list}.

    Each value is the raw list of dicts from HelpSteer3, e.g.:
        [{"score": -2, "reasoning": "...", "feedback1": "...", "feedback2": "..."},
         ...]

    Keying on the (prompt_id, response_a, response_b) triple avoids ambiguity
    when the same prompt appears with multiple response pairs.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset  # deferred so --help stays fast

    log.info("Loading HelpSteer3 '%s' split from HF ...", hf_split)
    ds = load_dataset(cfg.HF_DATASET_ID, split=hf_split, cache_dir=cache_dir)
    df = ds.to_pandas()

    df["domain_lower"] = df["domain"].str.strip().str.lower()
    df = df[df["domain_lower"].isin(cfg.KEEP_DOMAINS)]
    df = df[df["overall_preference"].isin(cfg.KEEP_PREFERENCES)]

    if "individual_preference" not in df.columns:
        raise RuntimeError(
            "HelpSteer3 rows don't have an 'individual_preference' column. "
            "The dataset schema may have changed."
        )

    log.info("  Usable rows after domain/preference filter: %d", len(df))

    # Re-use the same context_to_text + make_prompt_id as prepare_data.py
    # to guarantee hash alignment.
    from prepare_data import context_to_text, make_prompt_id

    mapping: dict[tuple[str, str, str], list[dict]] = {}
    for _, row in df.iterrows():
        ctx_text = context_to_text(row["context"])
        pid = make_prompt_id(ctx_text)
        key = (pid, str(row["response1"]), str(row["response2"]))
        raw_prefs = row["individual_preference"]
        # Normalise: datasets may return a list of dicts or a list of
        # numpy-dict-like objects; convert each element to a plain dict.
        if isinstance(raw_prefs, (list, tuple)):
            prefs = [
                {k: (v.item() if hasattr(v, "item") else v) for k, v in item.items()}
                if isinstance(item, dict) else item
                for item in raw_prefs
            ]
        else:
            prefs = []
        mapping[key] = prefs

    log.info("  Built pref map with %d entries.", len(mapping))
    return mapping


# ────────────────────────── split loading / on-the-fly dev subset ──────────

def load_or_make_split(
    split: str,
    dev_size: int | None,
    dev_source: str,
    dev_out_name: str | None,
    seed: int,
) -> tuple[pd.DataFrame, Path]:
    """
    Resolve the pair parquet for this run.

    If ``dev_size`` is given AND the target file does not yet exist, create it
    by stratified sampling from ``dev_source``. Existing files are never
    overwritten — this preserves experimental consistency.
    """
    out_name = dev_out_name or f"{split}.parquet"
    path = cfg.SPLITS_DIR / out_name

    if dev_size is not None and not path.exists():
        src_path = cfg.SPLITS_DIR / dev_source
        if not src_path.exists():
            raise FileNotFoundError(f"dev source not found: {src_path}")
        src_df = pd.read_parquet(src_path)
        log.info("Sampling new dev subset (n=%d) from %s ...", dev_size, src_path)
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
            f"{path} not found. Run prepare_data.py / make_dev600.py first, "
            f"or pass --dev-size to create it automatically."
        )

    df = pd.read_parquet(path)
    required = {"prompt_id", "context", "response_a", "response_b", "domain", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df, path


# ────────────────────────── SFT row construction ──────────────────────────

def _scores_from_prefs(prefs: list[dict]) -> list[int]:
    """Extract the integer score from each annotator dict."""
    out = []
    for item in prefs:
        if isinstance(item, dict) and "score" in item:
            out.append(int(item["score"]))
    return out


def build_sft_rows(
    pairs: pd.DataFrame,
    questions_by_sample: dict[str, dict[str, list[str]]],
    pref_map: dict[tuple[str, str, str], list[dict]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict] = []
    n_no_questions = 0
    n_no_pref = 0

    for _, r in pairs.iterrows():
        # Compute sample_id with the same function used by
        # prepare_data_reasoning.py so the join with questions aligns.
        pid = r["prompt_id"]
        sid = make_sample_id(
            prompt_id=pid,
            response_a=str(r["response_a"]),
            response_b=str(r["response_b"]),
            winner=str(r["winner"]),
        )

        per_domain = questions_by_sample.get(sid)
        if not per_domain:
            n_no_questions += 1
            continue
        target = format_checklist_target(per_domain)
        if not target:
            n_no_questions += 1
            continue

        key = (pid, str(r["response_a"]), str(r["response_b"]))
        prefs = pref_map.get(key)
        if prefs is None:
            n_no_pref += 1
            continue

        scores = _scores_from_prefs(prefs)

        messages = build_generator_messages(r)
        rows.append(
            {
                "sample_id": sid,
                "prompt_id": pid,
                "domain": r["domain"],
                "winner": r["winner"],
                "messages": json.dumps(messages, ensure_ascii=False),
                "target_output": target,
                # Full raw individual_preference list (list of dicts)
                "individual_preference": json.dumps(prefs, ensure_ascii=False),
                # Derived numeric signals for training use
                "pref_scores": scores,
                "n_annotators": len(scores),
                "pref_mean": float(np.mean(scores)) if scores else float("nan"),
                "pref_unanimous": (
                    bool(len({1 if s > 0 else -1 if s < 0 else 0 for s in scores}) == 1)
                    if scores else False
                ),
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
        "Built %d rows (skipped %d without questions, %d without individual_preference)",
        len(rows), n_no_questions, n_no_pref,
    )
    return pd.DataFrame(rows), stats


# ────────────────────────── summary ──────────────────────────

def print_summary(df: pd.DataFrame, out_path: Path, stats: dict[str, int]) -> None:
    table = Table(title="Generator SFT + individual_preference Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Output rows", f"{len(df):,}")
    table.add_row("Skipped (no questions)", f"{stats['skipped_no_questions']:,}")
    table.add_row("Skipped (no pref)", f"{stats['skipped_no_individual_pref']:,}")
    if len(df):
        table.add_row("Avg questions / sample", f"{df['n_questions'].mean():.1f}")
        table.add_row("Avg annotators / sample", f"{df['n_annotators'].mean():.2f}")
        table.add_row("Unanimous share", f"{df['pref_unanimous'].mean():.1%}")
    table.add_row("Output path", str(out_path))
    console.print(table)


# ────────────────────────── CLI ──────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Split basename under data/splits/ without extension, e.g. "
            "train_tier_10k, dev, dev_600."
        ),
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help=(
            "Path to the reasoning questions parquet produced by "
            "extract_reasoning_checklist_labels.py. "
            "Default: data/<split>_reasoning_questions.parquet"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output parquet path. Default: data/generator_sft/<split>.parquet",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help=(
            "HelpSteer3 HF split to pull individual_preference from. "
            "Use 'train' for train_* and dev_* splits (dev is carved from train). "
            "Use 'validation' for test splits."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Hugging Face dataset cache directory.",
    )
    # On-the-fly dev subset creation.
    parser.add_argument(
        "--dev-size",
        type=int,
        default=None,
        help=(
            "If the target split file does not exist, create it by stratified "
            "sampling this many rows from --dev-source. Never overwrites existing files."
        ),
    )
    parser.add_argument(
        "--dev-source",
        type=str,
        default="dev.parquet",
        help="Source parquet (under data/splits/) used when sampling a new dev subset.",
    )
    parser.add_argument(
        "--dev-out-name",
        type=str,
        default=None,
        help=(
            "Filename for the sampled dev subset under data/splits/. "
            "Default: <split>.parquet"
        ),
    )
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pairs (for debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and one example row without writing any files.",
    )
    args = parser.parse_args()

    # 1) Resolve (and optionally create) the split parquet.
    pairs, pair_path = load_or_make_split(
        split=args.split,
        dev_size=args.dev_size,
        dev_source=args.dev_source,
        dev_out_name=args.dev_out_name,
        seed=args.seed,
    )
    log.info("Loaded %d pairs from %s", len(pairs), pair_path)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)

    # 2) Load checklist questions (produced by extract_reasoning_checklist_labels.py).
    questions_path = (
        Path(args.questions_path)
        if args.questions_path
        else cfg.DATA_DIR / f"{args.split}_reasoning_questions.parquet"
    )
    df_q = load_questions(questions_path)
    questions_by_sample = aggregate_questions(df_q)

    # 3) Build individual_preference lookup from HelpSteer3.
    pref_map = build_pref_map(args.hf_split, args.cache_dir)

    # 4) Build SFT rows.
    sft_df, stats = build_sft_rows(pairs, questions_by_sample, pref_map)

    out_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.GENERATOR_SFT_DIR / f"{args.split}.parquet"
    )
    print_summary(sft_df, out_path, stats)

    if args.dry_run:
        if not sft_df.empty:
            r = sft_df.iloc[0]
            console.print(f"\n[cyan]sample_id[/cyan]:          {r['sample_id']}")
            console.print(f"[cyan]winner[/cyan]:              {r['winner']}")
            console.print(f"[cyan]pref_scores[/cyan]:         {r['pref_scores']}")
            console.print(f"[cyan]pref_mean[/cyan]:           {r['pref_mean']:.2f}")
            console.print(f"[cyan]pref_unanimous[/cyan]:      {r['pref_unanimous']}")
            console.print(f"[cyan]n_questions[/cyan]:         {r['n_questions']}")
            console.print(f"[cyan]target_output[/cyan]:\n{r['target_output']}")
        return

    if sft_df.empty:
        raise SystemExit("No SFT rows produced — check questions and pref inputs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(out_path, index=False)

    meta = {
        "split": args.split,
        "pair_source": str(pair_path),
        "questions_source": str(questions_path),
        "hf_split": args.hf_split,
        "seed": args.seed,
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
