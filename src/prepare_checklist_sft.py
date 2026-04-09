#!/usr/bin/env python3
"""
Generate checklist evaluation SFT data for joint DPO + checklist training.

For each pairwise training sample, creates checklist evaluation prompts for
both responses (A and B), paired with target checklist answers.

Two modes:
- **synthetic**: heuristic answers derived from preference labels (fast, no GPU)
- **teacher**:  base model generates checklist answers via vLLM (accurate, GPU)

Usage:
    # Quick synthetic data for pipeline testing
    python prepare_checklist_sft.py --tier debug_5k --mode synthetic

    # Teacher-generated data (requires GPU)
    python prepare_checklist_sft.py --tier debug_5k --mode teacher

    # Different checklist source
    python prepare_checklist_sft.py --tier tier_10k --mode teacher \
        --checklist-dir ../checklists/v2

    # Custom output directory (for control experiments)
    python prepare_checklist_sft.py --tier tier_10k --mode teacher \
        --output-dir ../data/checklist_sft_v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config as cfg
from utils import (
    build_checkeval_prompt,
    expected_question_count,
    load_checklists,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ────────────────────────── synthetic mode ──────────────────────


def generate_synthetic_answers(
    n_questions: int,
    is_chosen: bool,
    preference_strength: int,
    seed: int,
) -> str:
    """Generate synthetic checklist answers based on preference signal.

    Chosen responses get more "yes" answers; rejected get more "no".
    The proportion scales with preference_strength (2 or 3).
    """
    rng = np.random.RandomState(seed)
    strength = abs(preference_strength)

    if is_chosen:
        p_yes = 0.65 + 0.05 * strength   # strength 2 → 0.75, strength 3 → 0.80
    else:
        p_yes = 0.35 - 0.05 * strength   # strength 2 → 0.25, strength 3 → 0.20

    lines = []
    for i in range(1, n_questions + 1):
        ans = "yes" if rng.random() < p_yes else "no"
        lines.append(f"Q{i}: {ans}")
    return "\n".join(lines)


def build_synthetic_sft(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
) -> pd.DataFrame:
    """Build checklist SFT data using synthetic (heuristic) answers."""
    records = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                         desc="Synthetic SFT")):
        domain = row["domain"]
        winner = row["winner"]
        pref = row.get("preference_strength", 2)
        n_q = expected_question_count(domain, checklists)

        for side in ("A", "B"):
            prompt_text = build_checkeval_prompt(
                row, checklists, definitions, domain=domain, side=side,
            )
            is_chosen = (side == winner)
            completion = generate_synthetic_answers(
                n_q, is_chosen, pref, seed=idx * 2 + (0 if side == "A" else 1),
            )
            records.append({
                "prompt_text": prompt_text,
                "completion_text": completion,
                "domain": domain,
                "side": side,
                "is_chosen": is_chosen,
                "parse_valid": True,
                "n_questions": n_q,
            })

    return pd.DataFrame(records)


# ────────────────────────── teacher mode ───────────────────────


def build_teacher_sft(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    model_id: str = cfg.JUDGE_MODEL_ID,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Build checklist SFT data using base model (teacher) via vLLM."""
    from utils import generate_batch, load_judge_model

    log.info("Loading teacher model: %s", model_id)
    llm = load_judge_model(model_id)

    # Build all prompts (2 per sample: side A and B)
    all_prompts: list[str] = []
    meta: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
        domain = row["domain"]
        n_q = expected_question_count(domain, checklists)
        winner = row["winner"]

        for side in ("A", "B"):
            prompt_text = build_checkeval_prompt(
                row, checklists, definitions, domain=domain, side=side,
            )
            all_prompts.append(prompt_text)
            meta.append({
                "domain": domain,
                "side": side,
                "is_chosen": (side == winner),
                "n_questions": n_q,
            })

    # Batch inference
    log.info("Running teacher inference on %d prompts ...", len(all_prompts))
    messages_list = [[{"role": "user", "content": p}] for p in all_prompts]
    raw_outputs = generate_batch(
        llm, messages_list, batch_size=batch_size, max_new_tokens=2048,
    )

    # Parse and validate
    records = []
    n_valid = 0
    for prompt_text, raw, m in zip(all_prompts, raw_outputs, meta):
        parsed = parse_checkeval_output(raw, expected_n=m["n_questions"])
        valid = not parsed.get("_raw_fallback", False)
        if valid:
            n_valid += 1

        records.append({
            "prompt_text": prompt_text,
            "completion_text": raw.strip(),
            "domain": m["domain"],
            "side": m["side"],
            "is_chosen": m["is_chosen"],
            "parse_valid": valid,
            "n_questions": m["n_questions"],
        })

    log.info("Teacher parse rate: %d/%d (%.1f%%)",
             n_valid, len(records), 100 * n_valid / len(records))

    # Delete vLLM model to free GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return pd.DataFrame(records)


# ────────────────────────── summary ────────────────────────────


def print_summary(sft_df: pd.DataFrame) -> None:
    table = Table(title="Checklist SFT Data Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total samples", f"{len(sft_df):,}")
    table.add_row("Chosen side", f"{sft_df['is_chosen'].sum():,}")
    table.add_row("Rejected side", f"{(~sft_df['is_chosen']).sum():,}")
    table.add_row("Parse valid", f"{sft_df['parse_valid'].sum():,}")
    table.add_row("Parse rate",
                  f"{100 * sft_df['parse_valid'].mean():.1f}%")

    for domain in sorted(sft_df["domain"].unique()):
        n = (sft_df["domain"] == domain).sum()
        table.add_row(f"  {domain}", f"{n:,}")

    console.print(table)


# ────────────────────────── main ───────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate checklist SFT data for joint training"
    )
    parser.add_argument(
        "--tier", type=str, default="debug_5k",
        choices=["debug_5k", "tier_10k", "tier_20k", "full"],
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "teacher"],
        help="synthetic = fast heuristic; teacher = base model via vLLM",
    )
    parser.add_argument(
        "--checklist-dir", type=str, default=None,
        help="Override checklist directory (default: checklists/filtered)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: data/checklist_sft)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-id", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument(
        "--filter-valid", action="store_true",
        help="Only keep samples that parsed successfully (teacher mode)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve paths
    checklist_dir = Path(args.checklist_dir) if args.checklist_dir else cfg.CHECKLISTS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else cfg.CHECKLIST_SFT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checklists
    log.info("Loading checklists from %s", checklist_dir)
    checklists, definitions = load_checklists(checklist_dir)
    total_q = sum(len(qs) for qs in checklists.values())
    log.info("  %d dimensions, %d total questions", len(checklists), total_q)

    # Load pairwise data
    if args.tier == "full":
        src_path = cfg.SPLITS_DIR / "train.parquet"
    else:
        src_path = cfg.SPLITS_DIR / f"train_{args.tier}.parquet"

    if not src_path.exists():
        log.error("Source file not found: %s. Run prepare_data.py first.", src_path)
        return

    df = pd.read_parquet(src_path)
    # Filter out ties
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    log.info("Loaded %d pairwise samples from %s (ties excluded)", len(df), src_path.name)

    # Generate SFT data
    if args.mode == "synthetic":
        sft_df = build_synthetic_sft(df, checklists, definitions)
    else:
        sft_df = build_teacher_sft(
            df, checklists, definitions,
            model_id=args.model_id, batch_size=args.batch_size,
        )

    # Optionally filter invalid parses
    if args.filter_valid:
        before = len(sft_df)
        sft_df = sft_df[sft_df["parse_valid"]].reset_index(drop=True)
        log.info("Filtered: %d → %d valid samples", before, len(sft_df))

    print_summary(sft_df)

    if args.dry_run:
        log.info("Dry run — not saving.")
        sample = sft_df.iloc[0]
        console.print("\n[bold]Sample:[/bold]")
        console.print(f"  [cyan]domain[/cyan]: {sample['domain']}")
        console.print(f"  [cyan]side[/cyan]: {sample['side']}")
        console.print(f"  [cyan]is_chosen[/cyan]: {sample['is_chosen']}")
        console.print(f"  [cyan]prompt (first 200 chars)[/cyan]:\n{sample['prompt_text'][:200]}...")
        console.print(f"  [cyan]completion[/cyan]:\n{sample['completion_text'][:300]}")
        return

    # Save
    tag = f"_{args.mode}" if args.mode != "teacher" else ""
    out_name = f"train_{args.tier}{tag}.parquet" if args.tier != "full" else f"train{tag}.parquet"
    out_path = output_dir / out_name
    sft_df.to_parquet(out_path, index=False)
    log.info("Saved %d SFT samples to %s", len(sft_df), out_path)

    # Save metadata
    meta = {
        "tier": args.tier,
        "mode": args.mode,
        "checklist_dir": str(checklist_dir),
        "n_samples": len(sft_df),
        "n_valid": int(sft_df["parse_valid"].sum()),
        "n_dimensions": len(checklists),
        "n_total_questions": total_q,
        "filter_valid": args.filter_valid,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", meta_path)

    console.print(f"\n[bold green]Done. Checklist SFT data → {out_path}[/bold green]")


if __name__ == "__main__":
    main()
