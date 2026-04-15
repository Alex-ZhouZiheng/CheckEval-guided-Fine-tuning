from __future__ import annotations
"""
Prepare a reasoning-augmented pairwise slice for strict A/B vs B/A comparisons.

The script:
1. Loads an existing pairwise split (default: data/splits/dev_600.parquet).
2. Loads the matching raw HelpSteer3 preference data.
3. Aligns context / response_a / response_b / winner with
   individual_preference[*].reasoning, feedback1, feedback2.
4. Cleans reasoning / feedbacks into usable text fields.
5. Emits both the original A/B order and a swapped B/A order.

Output columns:
    sample_id
    domain
    context
    response_a
    response_b
    winner
    gold_label
    reasoning_text
    feedback_a_text
    feedback_b_text
    swap_flag

Usage:
    python src/prepare_data_reasoning.py
    python src/prepare_data_reasoning.py --split dev
    python src/prepare_data_reasoning.py --input-path data/splits/dev_600.parquet
    python src/prepare_data_reasoning.py --dry-run
"""



import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # src/
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                      # src/data_process/

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from prepare_data import context_to_text, make_prompt_id, preference_to_winner

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

KEY_COLS = ["prompt_id", "domain", "context", "response_a", "response_b", "winner"]
OUTPUT_COLS = [
    "sample_id",
    "domain",
    "context",
    "response_a",
    "response_b",
    "winner",
    "gold_label",
    "reasoning_text",
    "feedback_a_text",
    "feedback_b_text",
    "swap_flag",
]


def _normalize_domain(value: Any) -> str:
    return str(value).strip().lower()


def make_sample_id(prompt_id: str, response_a: str, response_b: str, winner: str) -> str:
    payload = json.dumps(
        {
            "prompt_id": prompt_id,
            "response_a": response_a,
            "response_b": response_b,
            "winner": winner,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prompt_id}_{digest}"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_reasoning_fragments(value: Any) -> list[str]:
    """Extract reasoning strings from nested list/dict/json structures."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [_normalize_whitespace(text)]
            return _extract_reasoning_fragments(parsed)

        return [_normalize_whitespace(text)]

    if isinstance(value, dict):
        for key in ("reasoning", "rationale", "explanation", "summary", "text", "content"):
            if key in value:
                return _extract_reasoning_fragments(value[key])
        return []

    if isinstance(value, (list, tuple)):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_extract_reasoning_fragments(item))
        return fragments

    return [_normalize_whitespace(str(value))]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _standardize_response_refs(text: str) -> str:
    """Map Response 1/2 references to Response A/B for downstream consistency."""
    substitutions = [
        (r"(?i)@?\bresponse\s*1\b", "Response A"),
        (r"(?i)@?\bresponse\s*2\b", "Response B"),
        (r"(?i)@?\bresponse\s*a\b", "Response A"),
        (r"(?i)@?\bresponse\s*b\b", "Response B"),
    ]

    out = text
    for pattern, repl in substitutions:
        out = re.sub(pattern, repl, out)
    return _normalize_whitespace(out)


def clean_reasoning_text(raw_reasoning: Any) -> str:
    fragments = _extract_reasoning_fragments(raw_reasoning)
    fragments = [_standardize_response_refs(fragment) for fragment in fragments]
    fragments = _dedupe_preserve_order([fragment for fragment in fragments if fragment])
    return "\n\n".join(fragments).strip()


def _extract_feedback_fragments(value: Any, feedback_key: str) -> list[str]:
    """Collect strings under `feedback_key` (e.g. 'feedback1') across annotator entries.

    The raw individual_preference column may be a JSON string or a list/tuple of
    dicts — each dict carries 'reasoning', 'feedback1', 'feedback2'.
    """
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            return _extract_feedback_fragments(parsed, feedback_key)
        return []

    if isinstance(value, dict):
        if feedback_key in value:
            inner = value[feedback_key]
            if isinstance(inner, str):
                text = inner.strip()
                return [_normalize_whitespace(text)] if text else []
            return _extract_feedback_fragments(inner, feedback_key)
        return []

    if isinstance(value, (list, tuple)):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_extract_feedback_fragments(item, feedback_key))
        return fragments

    return []


def clean_feedback_text(raw_individual_preference: Any, feedback_key: str) -> str:
    fragments = _extract_feedback_fragments(raw_individual_preference, feedback_key)
    fragments = [_standardize_response_refs(fragment) for fragment in fragments]
    fragments = _dedupe_preserve_order([fragment for fragment in fragments if fragment])
    return "\n\n".join(fragments).strip()


def _swap_response_refs(text: str) -> str:
    if not text:
        return ""

    out = text.replace("Response A", "__RESPONSE_A__")
    out = out.replace("Response B", "Response A")
    out = out.replace("__RESPONSE_A__", "Response B")
    return out


def _swap_label(label: Any) -> str:
    label = str(label)
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return label


def infer_raw_source(split_name: str) -> str:
    split = split_name.lower()
    if split in {"test", "validation", "valid"}:
        return "validation"
    return "train"


def _filter_raw_preference(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "domain_lower" not in work.columns and "domain" in work.columns:
        work["domain_lower"] = work["domain"].map(_normalize_domain)

    if "domain_lower" in work.columns:
        work = work[work["domain_lower"].isin(cfg.KEEP_DOMAINS)].copy()

    if "overall_preference" in work.columns:
        work = work[work["overall_preference"].isin(cfg.KEEP_PREFERENCES)].copy()

    return work


def load_raw_preference(raw_source: str, cache_dir: str | None = None) -> pd.DataFrame:
    local_name = "helpsteer3_train.parquet" if raw_source == "train" else "helpsteer3_test.parquet"
    local_path = cfg.RAW_DIR / local_name

    if local_path.exists():
        log.info("Loading raw preference parquet: %s", local_path)
        return _filter_raw_preference(pd.read_parquet(local_path))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FileNotFoundError(
            f"{local_path} not found, and `datasets` is not installed for HF fallback. "
            "Run download_data.py first or install datasets."
        ) from exc

    hf_split = "train" if raw_source == "train" else "validation"
    log.info("Loading HelpSteer3 %s split from Hugging Face", hf_split)
    ds = load_dataset(cfg.HF_DATASET_ID, cache_dir=cache_dir)
    return _filter_raw_preference(ds[hf_split].to_pandas())


def build_reasoning_lookup(raw_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "context",
        "response1",
        "response2",
        "overall_preference",
        "individual_preference",
    }
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw preference data is missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        context_text = context_to_text(row["context"])
        indiv_pref = row.get("individual_preference")
        reasoning_text = clean_reasoning_text(indiv_pref)
        feedback_a_text = clean_feedback_text(indiv_pref, "feedback1")
        feedback_b_text = clean_feedback_text(indiv_pref, "feedback2")
        records.append(
            {
                "prompt_id": make_prompt_id(context_text),
                "domain": _normalize_domain(row.get("domain_lower", row.get("domain", ""))),
                "context": context_text,
                "response_a": row["response1"],
                "response_b": row["response2"],
                "winner": preference_to_winner(int(row["overall_preference"])),
                "reasoning_text": reasoning_text,
                "feedback_a_text": feedback_a_text,
                "feedback_b_text": feedback_b_text,
            }
        )

    lookup = pd.DataFrame(records)
    before = len(lookup)
    lookup = lookup.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    dropped = before - len(lookup)
    if dropped:
        log.warning("Dropped %s duplicate raw rows while building reasoning lookup", dropped)
    return lookup


def load_pairwise_split(split_name: str | None, input_path: str | None) -> tuple[pd.DataFrame, str, Path]:
    if input_path:
        path = Path(input_path)
        resolved_split = path.stem
    else:
        resolved_split = split_name or "dev_600"
        path = cfg.SPLITS_DIR / f"{resolved_split}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Pairwise split not found: {path}")

    df = pd.read_parquet(path)
    missing = set(KEY_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Pairwise split is missing columns: {sorted(missing)}")

    return df.copy(), resolved_split, path


def align_reasoning(split_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    aux_cols = ["reasoning_text", "feedback_a_text", "feedback_b_text"]
    merged = split_df.merge(
        lookup_df[KEY_COLS + aux_cols],
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )

    for col in aux_cols:
        merged[col] = merged[col].fillna("")
    merged["sample_id"] = merged.apply(
        lambda row: make_sample_id(
            prompt_id=row["prompt_id"],
            response_a=row["response_a"],
            response_b=row["response_b"],
            winner=row["winner"],
        ),
        axis=1,
    )
    merged["gold_label"] = merged["winner"]
    return merged


def make_original_and_swapped(df: pd.DataFrame) -> pd.DataFrame:
    original = df.copy()
    original["swap_flag"] = False

    swapped = df.copy()
    swapped["response_a"] = df["response_b"]
    swapped["response_b"] = df["response_a"]
    swapped["winner"] = df["winner"].map(_swap_label)
    swapped["gold_label"] = swapped["winner"]
    swapped["reasoning_text"] = df["reasoning_text"].map(_swap_response_refs)
    # feedback_a describes response_a; after swap it should describe the NEW
    # response_a, which is the OLD response_b → swap the two feedback columns
    # and also swap any internal "Response A"/"Response B" references.
    swapped["feedback_a_text"] = df["feedback_b_text"].map(_swap_response_refs)
    swapped["feedback_b_text"] = df["feedback_a_text"].map(_swap_response_refs)
    swapped["swap_flag"] = True

    combined = pd.concat([original, swapped], ignore_index=True)
    combined = combined.sort_values(["sample_id", "swap_flag"], kind="stable").reset_index(drop=True)
    return combined[OUTPUT_COLS]


def print_summary(base_df: pd.DataFrame, final_df: pd.DataFrame, split_name: str, output_path: Path) -> None:
    matched = int(base_df["reasoning_text"].ne("").sum())
    missing = int(base_df["reasoning_text"].eq("").sum())
    fa_matched = int(base_df["feedback_a_text"].ne("").sum())
    fb_matched = int(base_df["feedback_b_text"].ne("").sum())

    table = Table(title=f"Reasoning Slice Summary — {split_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Base rows", f"{len(base_df):,}")
    table.add_row("Rows with reasoning", f"{matched:,}")
    table.add_row("Rows missing reasoning", f"{missing:,}")
    table.add_row("Rows with feedback_a", f"{fa_matched:,}")
    table.add_row("Rows with feedback_b", f"{fb_matched:,}")
    table.add_row("Final rows (with swap)", f"{len(final_df):,}")
    table.add_row("Output", str(output_path))
    console.print(table)

    dom_table = Table(title="Per-domain counts")
    dom_table.add_column("Domain", style="bold")
    dom_table.add_column("Base", justify="right")
    dom_table.add_column("Final", justify="right")

    base_counts = base_df["domain"].value_counts()
    final_counts = final_df["domain"].value_counts()
    for domain in sorted(set(base_counts.index) | set(final_counts.index)):
        dom_table.add_row(
            domain,
            f"{int(base_counts.get(domain, 0)):,}",
            f"{int(final_counts.get(domain, 0)):,}",
        )
    console.print(dom_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reasoning-augmented pairwise data")
    parser.add_argument(
        "--split",
        type=str,
        default="dev_600",
        help="Split name under data/splits (default: dev_600)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional explicit path to a pairwise parquet file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output parquet path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir for raw-data fallback",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    split_df, split_name, split_path = load_pairwise_split(args.split, args.input_path)
    raw_source = infer_raw_source(split_name)
    raw_df = load_raw_preference(raw_source=raw_source, cache_dir=args.cache_dir)
    lookup_df = build_reasoning_lookup(raw_df)
    base_df = align_reasoning(split_df, lookup_df)
    final_df = make_original_and_swapped(base_df)

    output_path = Path(args.output_path) if args.output_path else (cfg.DATA_DIR / f"{split_name}_reasoning.parquet")
    print_summary(base_df, final_df, split_name=split_name, output_path=output_path)

    if args.dry_run:
        log.info("Dry run -- not writing files. Source split: %s", split_path)
        if len(final_df) > 0:
            sample = final_df.iloc[0].to_dict()
            console.print("\n[bold]Sample record:[/bold]")
            for key in OUTPUT_COLS:
                value = sample[key]
                if isinstance(value, str) and len(value) > 220:
                    value = value[:220] + "..."
                console.print(f"  [cyan]{key}[/cyan]: {value}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    log.info("Saved %s rows -> %s", len(final_df), output_path)
    console.print(f"\n[bold green]Done. Reasoning slice saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
