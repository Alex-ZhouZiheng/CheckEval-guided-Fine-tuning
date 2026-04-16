#!/usr/bin/env python3
"""
Build SFT parquet for the checklist-conditioned *judge* model.

For every training pair, expand into two pointwise rows (side A and side B) that
feed the CheckEval pointwise prompt with the *generator-produced* checklist as
the question set.  Target output is ``Q1: yes / Q2: no / ...`` where each label
is derived heuristically from the preference winner:

    winner=A   →  side A = all yes,  side B = all no
    winner=B   →  side A = all no,   side B = all yes
    winner=Tie →  both sides = all yes

This matches the synthetic-label strategy used elsewhere in the repo
(``prepare_checklist_sft.py``'s synthetic mode) and is the simplest coherent
training signal when no per-question gold labels exist for the generator's
(sample-specific) questions.

Usage:
    python prepare_judge_sft.py --tier tier_10k \\
        --generated data/generated_checklists/tier_10k.parquet
    python prepare_judge_sft.py --tier debug_5k --dry-run --limit 3
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from run_generator_infer import parse_generated_checklist
from utils import CHECKEVAL_POINTWISE_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


DOMAIN_DEFS = {
    "correctness_completeness": "Factual accuracy, logical consistency, coverage of the user's request.",
    "clarity_communication": "Clarity, structure, readability, conciseness of the response.",
    "helpfulness_usefulness": "Practical value, relevance, actionability for the user's real need.",
    "coding_communication_conditional": "Code-specific quality (syntax, runnability, comments); apply only when code is present.",
}


def flatten_checklist(per_domain: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """Return (dimension_block_lines, flat_questions, flat_domains_per_q)."""
    from prepare_generator_sft import DOMAIN_ORDER

    dim_lines: list[str] = []
    flat_q: list[str] = []
    flat_dom: list[str] = []
    for domain in DOMAIN_ORDER:
        qs = per_domain.get(domain, [])
        if not qs:
            continue
        dim_lines.append(f"{domain} - {DOMAIN_DEFS[domain]}")
        for q in qs:
            flat_q.append(q)
            flat_dom.append(domain)
    return dim_lines, flat_q, flat_dom


def build_pointwise_prompt(
    row: dict | pd.Series,
    per_domain: dict[str, list[str]],
    side: str,
) -> tuple[str, int]:
    dim_lines, flat_q, _ = flatten_checklist(per_domain)
    dimension_block = "\n".join(dim_lines)
    checklist_text = "\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(flat_q))
    response_key = "response_a" if side == "A" else "response_b"
    prompt = CHECKEVAL_POINTWISE_PROMPT.format(
        dimension_block=dimension_block,
        context=row["context"],
        response=row[response_key],
        checklist_text=checklist_text,
    )
    return prompt, len(flat_q)


def build_target(side: str, winner: str, n_q: int) -> str:
    if winner == "Tie":
        label = "yes"
    elif winner == side:
        label = "yes"
    else:
        label = "no"
    return "\n".join(f"Q{i + 1}: {label}" for i in range(n_q))


def load_pairwise(tier: str) -> pd.DataFrame:
    path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    df = pd.read_parquet(path)
    required = {"sample_id", "context", "response_a", "response_b", "domain", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairwise parquet missing columns: {sorted(missing)}")
    return df


def load_generated(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_generator_infer.py on the same tier first."
        )
    df = pd.read_parquet(path)
    out: dict[str, dict[str, list[str]]] = {}
    for _, r in df.iterrows():
        sid = r["sample_id"]
        text = r.get("generated_checklist", "") or r.get("raw_output", "")
        if not isinstance(text, str) or not text.strip():
            continue
        per_domain = parse_generated_checklist(text)
        if any(per_domain.values()):
            out[sid] = per_domain
    return out


def build_rows(
    pairs: pd.DataFrame,
    gen_by_sample: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    rows: list[dict] = []
    n_skipped = 0
    for _, r in pairs.iterrows():
        sid = r["sample_id"]
        per_domain = gen_by_sample.get(sid)
        if not per_domain:
            n_skipped += 1
            continue
        winner = r["winner"]
        for side in ("A", "B"):
            prompt, n_q = build_pointwise_prompt(r, per_domain, side)
            if n_q == 0:
                continue
            target = build_target(side, winner, n_q)
            messages = [{"role": "user", "content": prompt}]
            rows.append({
                "sample_id": sid,
                "side": side,
                "domain": r["domain"],
                "winner": winner,
                "n_questions": n_q,
                "messages": json.dumps(messages, ensure_ascii=False),
                "target_output": target,
            })
    log.info("Built %d judge SFT rows (skipped %d pairs with no generated checklist)",
             len(rows), n_skipped)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    table = Table(title="Judge SFT Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{len(df):,}")
    if len(df):
        table.add_row("Avg Q per row", f"{df['n_questions'].mean():.1f}")
        table.add_row("Rows per side A", f"{int((df['side']=='A').sum()):,}")
        table.add_row("Rows per side B", f"{int((df['side']=='B').sum()):,}")
    table.add_row("Output", str(output_path))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="tier_10k",
                        choices=["debug_5k", "tier_10k", "tier_20k"])
    parser.add_argument("--generated", type=str, default=None,
                        help="Path to data/generated_checklists/<tier>.parquet "
                             "(default: inferred from --tier)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Override output (default: data/judge_sft/train_<tier>.parquet)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generated_path = (
        Path(args.generated)
        if args.generated
        else cfg.GENERATED_CHECKLIST_DIR / f"{args.tier}.parquet"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.JUDGE_SFT_DIR / f"train_{args.tier}.parquet"
    )

    pairs = load_pairwise(args.tier)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)

    gen_by_sample = load_generated(generated_path)
    log.info("Loaded generated checklists for %d samples", len(gen_by_sample))

    sft_df = build_rows(pairs, gen_by_sample)
    print_summary(sft_df, output_path)

    if args.dry_run:
        if len(sft_df):
            r = sft_df.iloc[0]
            console.print(f"\n[cyan]sample_id[/cyan]: {r['sample_id']}  side={r['side']}  winner={r['winner']}")
            msgs = json.loads(r["messages"])
            console.print(f"[cyan]prompt (first 500c)[/cyan]:\n{msgs[0]['content'][:500]}...")
            console.print(f"[cyan]target_output[/cyan]:\n{r['target_output']}")
        return

    if sft_df.empty:
        raise SystemExit("No judge SFT rows produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(output_path, index=False)
    log.info("Saved %d rows -> %s", len(sft_df), output_path)
    console.print(f"\n[bold green]Done. Saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
