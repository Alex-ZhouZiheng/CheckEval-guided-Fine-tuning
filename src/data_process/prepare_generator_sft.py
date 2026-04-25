#!/usr/bin/env python3
"""
Build SFT parquet for the checklist *generator* model.

Given:
- Pairwise split  (data/splits/train_<tier>.parquet)
- Reasoning-extracted questions
  (data/train_<tier>_reasoning_questions.parquet from
   prepare_data_reasoning.py + extract_reasoning_checklist_labels.py)

For each sample (only ``swap_flag == False``), aggregate all Yes/No checklist
questions by checklist dimension and produce a single structured target string:

    ### correctness_and_completeness
    - Does the response ...
    - Are all facts ...

    ### clarity_and_communication
    - ...

The generator is trained to map (instruction + response_a + response_b) to this
structured checklist. Samples without any extracted question are skipped.

Usage:
    python prepare_generator_sft.py --tier tier_10k
    python prepare_generator_sft.py --tier debug_5k --dry-run --limit 3
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # src/data_process/

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from prepare_data_reasoning import make_sample_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()



DOMAINS = cfg.DOMAINS
DOMAIN_DESCRIPTIONS = cfg.DOMAIN_DESCRIPTIONS

GENERATOR_SYSTEM_PROMPT = (
    "You are a checklist writer for LLM response evaluation. Given a user "
    "request, produce a list of Yes/No evaluation questions, grouped by "
    "quality dimension, that could be used to judge any candidate response to "
    "this request. Each question must be phrased so that 'Yes' means the "
    "response meets the criterion. Output ONLY the checklist in the required "
    "format."
)

GENERATOR_USER_TEMPLATE = """\
Produce a Yes/No evaluation checklist for judging responses to the following request.

Group questions under these section headers (skip a section if it does not apply):
{domain}

Output format:
### <domain>
- <question>
- <question>

### <domain>
- ...

Rules:
- Phrase each question so that "Yes" means the response is good on that criterion.
- Do not reference any specific response; questions must apply to any candidate.
- Keep each question under ~40 words.
- Output only the checklist, no commentary.

# Conversation Context
{context}

# Checklist"""

def _domain_block() -> str:
    lines = []
    for name in DOMAINS:
        lines.append(f"- {name}: {DOMAIN_DESCRIPTIONS[name]}")
    return "\n".join(lines)

def build_generator_messages(row: dict | pd.Series) -> list[dict[str, str]]:
    """Chat-template messages consumed by the generator model at both train and eval."""
    user = GENERATOR_USER_TEMPLATE.format(
        context=row["context"],
        domain=_domain_block(),
    )
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_checklist_target(questions_by_domain: dict[str, list[str]]) -> str:
    """Render aggregated questions as the canonical `### <domain>\\n- ...` string."""
    sections: list[str] = []
    for domain in DOMAINS:
        qs = questions_by_domain.get(domain, [])
        if not qs:
            continue
        body = "\n".join(f"- {q}" for q in qs)
        sections.append(f"### {domain}\n{body}")
    return "\n\n".join(sections).strip()


def load_questions(questions_path: Path) -> pd.DataFrame:
    if not questions_path.exists():
        raise FileNotFoundError(
            f"{questions_path} not found. "
            "Run prepare_data_reasoning.py then extract_reasoning_checklist_labels.py first."
        )
    df = pd.read_parquet(questions_path)
    required = {"sample_id", "swap_flag", "domain", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Questions parquet missing columns: {sorted(missing)}")
    # Only use the original A/B ordering as canonical training target.
    df = df[~df["swap_flag"].astype(bool)].copy()
    return df


def load_pairwise(tier: str) -> pd.DataFrame:
    path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    df = pd.read_parquet(path)
    required = {"prompt_id", "context", "response_a", "response_b", "domain", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairwise parquet missing columns: {sorted(missing)}")
    # Compute sample_id with the same function as prepare_data_reasoning.py so
    # the key aligns with the reasoning_questions parquet.
    if "sample_id" not in df.columns:
        df["sample_id"] = df.apply(
            lambda r: make_sample_id(
                prompt_id=r["prompt_id"],
                response_a=str(r["response_a"]),
                response_b=str(r["response_b"]),
                winner=str(r["winner"]),
            ),
            axis=1,
        )
    return df


def aggregate_questions(df_q: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """{sample_id: {domain: [question, ...]}} preserving source_idx order."""
    if "source_idx" in df_q.columns:
        df_q = df_q.sort_values(["sample_id", "source_idx"], kind="stable")
    out: dict[str, dict[str, list[str]]] = {}
    for sid, group in df_q.groupby("sample_id", sort=False):
        per_domain: dict[str, list[str]] = {}
        for _, r in group.iterrows():
            dom = r["domain"]
            q = str(r["question"]).strip()
            if not q:
                continue
            per_domain.setdefault(dom, []).append(q)
        # Deduplicate within domain, preserve order.
        for dom, qs in per_domain.items():
            seen: set[str] = set()
            dedup: list[str] = []
            for q in qs:
                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(q)
            per_domain[dom] = dedup
        out[sid] = per_domain
    return out


def build_sft_rows(
    pairs: pd.DataFrame,
    questions_by_sample: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    rows: list[dict] = []
    n_skipped = 0
    for _, r in pairs.iterrows():
        sid = r["sample_id"]
        per_domain = questions_by_sample.get(sid)
        if not per_domain:
            n_skipped += 1
            continue
        target = format_checklist_target(per_domain)
        if not target:
            n_skipped += 1
            continue
        messages = build_generator_messages(r)
        rows.append(
            {
                "sample_id": sid,
                "domain": r["domain"],
                "messages": json.dumps(messages, ensure_ascii=False),
                "target_output": target,
                "n_questions": sum(len(v) for v in per_domain.values()),
                "n_domains": len(per_domain),
            }
        )
    log.info(
        "Built %d SFT rows (skipped %d samples without any extracted questions)",
        len(rows), n_skipped,
    )
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    table = Table(title="Generator SFT Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{len(df):,}")
    if len(df):
        table.add_row("Avg questions / sample", f"{df['n_questions'].mean():.1f}")
        table.add_row("Avg domains / sample", f"{df['n_domains'].mean():.2f}")
    table.add_row("Output", str(output_path))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_10k",
        choices=["debug_5k", "tier_10k", "tier_20k"],
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help="Override path to *_reasoning_questions.parquet "
             "(default: data/train_<tier>_reasoning_questions.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output path (default: data/generator_sft/train_<tier>.parquet)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N pairs (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print a few examples without writing")
    args = parser.parse_args()

    questions_path = (
        Path(args.questions_path)
        if args.questions_path
        else cfg.WITH_REASON_DIR / f"train_{args.tier}_reasoning_questions.parquet"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.GENERATOR_SFT_DIR / f"train_{args.tier}.parquet"
    )

    pairs = load_pairwise(args.tier)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)

    df_q = load_questions(questions_path)
    questions_by_sample = aggregate_questions(df_q)

    sft_df = build_sft_rows(pairs, questions_by_sample)

    print_summary(sft_df, output_path)

    if args.dry_run:
        console.print("\n[bold]Sample row(s):[/bold]")
        for i, r in sft_df.head(2).iterrows():
            console.print(f"\n[cyan]sample_id[/cyan]: {r['sample_id']}")
            console.print(f"[cyan]target_output[/cyan]:\n{r['target_output']}")
        return

    if sft_df.empty:
        raise SystemExit("No SFT rows produced — check reasoning questions input.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(output_path, index=False)
    log.info("Saved %d rows -> %s", len(sft_df), output_path)
    console.print(f"\n[bold green]Done. Saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
