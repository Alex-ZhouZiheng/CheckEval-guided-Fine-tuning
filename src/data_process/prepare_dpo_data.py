#!/usr/bin/env python3
"""
Convert HelpSteer3 pairwise splits into DPO training format.

DPO format requires each sample to contain:
  - prompt:   user input (conversation context)
  - chosen:   the preferred response
  - rejected: the dispreferred response

Usage:
    python prepare_dpo_data.py                          # convert all splits
    python prepare_dpo_data.py --splits train dev       # specific splits only
    python prepare_dpo_data.py --tier train_debug_5k    # a specific tier subset
    python prepare_dpo_data.py --chat-template           # chat message list format
    python prepare_dpo_data.py --dry-run                 # print stats only
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from utils import build_vanilla_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

DPO_DIR = cfg.DATA_DIR / "dpo"
DPO_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────── conversion ──────────────────────────
def pairwise_to_dpo(df: pd.DataFrame, chat_template: bool = False) -> pd.DataFrame:
    """Convert pairwise judge data to DPO triplets for training a *judge* model.

    The judge is trained to emit a verdict ("A" or "B") given the full
    VANILLA_JUDGE_PROMPT (context + response_a + response_b). So:
        prompt   = formatted judge prompt
        chosen   = correct verdict letter
        rejected = incorrect verdict letter

    Input columns:  context, response_a, response_b, winner, domain,
                    prompt_id, preference_strength
    Output columns: prompt, chosen, rejected, domain, prompt_id,
                    preference_strength

    Parameters
    ----------
    df : pd.DataFrame
        Pairwise data produced by prepare_data.py.
    chat_template : bool
        If True, wrap prompt/chosen/rejected as chat message lists
        compatible with trl DPOTrainer's chat template mode.
    """
    records = []
    for _, row in df.iterrows():
        winner = row["winner"]
        if winner == "A":
            chosen_letter, rejected_letter = "A", "B"
        elif winner == "B":
            chosen_letter, rejected_letter = "B", "A"
        else:
            continue

        judge_prompt = build_vanilla_prompt(row)

        if chat_template:
            prompt_val = [{"role": "user", "content": judge_prompt}]
            chosen_val = [{"role": "assistant", "content": chosen_letter}]
            rejected_val = [{"role": "assistant", "content": rejected_letter}]
        else:
            prompt_val = judge_prompt
            chosen_val = chosen_letter
            rejected_val = rejected_letter

        records.append(
            {
                "prompt": prompt_val,
                "chosen": chosen_val,
                "rejected": rejected_val,
                "domain": row["domain"],
                "prompt_id": row["prompt_id"],
                "preference_strength": row["preference_strength"],
            }
        )

    return pd.DataFrame(records)


def print_summary(dpo_splits: dict[str, pd.DataFrame]) -> None:
    table = Table(title="DPO Data Summary")
    table.add_column("Split", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("General", justify="right")
    table.add_column("STEM", justify="right")
    table.add_column("Code", justify="right")
    table.add_column("Avg Pref Strength", justify="right")

    for name, sdf in dpo_splits.items():
        dom = sdf["domain"].value_counts()
        avg_str = f"{sdf['preference_strength'].mean():.2f}" if len(sdf) > 0 else "N/A"
        table.add_row(
            name,
            f"{len(sdf):,}",
            f"{dom.get('general', 0):,}",
            f"{dom.get('stem', 0):,}",
            f"{dom.get('code', 0):,}",
            avg_str,
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Convert pairwise data to DPO format")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Which splits to convert (default: train dev test)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Convert a specific training tier (e.g. train_debug_5k, train_tier_10k)",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        default=True,
        help="Output in chat message list format for trl DPOTrainer (default: True)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_false",
        dest="chat_template",
        help="Output as plain text instead of chat message lists",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "jsonl", "both"],
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    dpo_splits: dict[str, pd.DataFrame] = {}

    if args.tier:
        src = cfg.SPLITS_DIR / f"{args.tier}.parquet"
        if not src.exists():
            log.error(f"Tier file not found: {src}")
            return
        log.info(f"Loading tier: {src}")
        df = pd.read_parquet(src)
        dpo_splits[args.tier] = pairwise_to_dpo(df, chat_template=args.chat_template)
    else:
        for split_name in args.splits:
            src = cfg.SPLITS_DIR / f"{split_name}.parquet"
            if not src.exists():
                log.warning(f"Split file not found, skipping: {src}")
                continue
            log.info(f"Loading split: {src}")
            df = pd.read_parquet(src)
            dpo_splits[split_name] = pairwise_to_dpo(df, chat_template=args.chat_template)

    if not dpo_splits:
        log.error("No data loaded. Run prepare_data.py first.")
        return

    print_summary(dpo_splits)

    if args.dry_run:
        log.info("Dry run -- not writing files.")
        first_split = next(iter(dpo_splits.values()))
        if len(first_split) > 0:
            sample = first_split.iloc[0]
            console.print("\n[bold]Sample record:[/bold]")
            for col in ["prompt", "chosen", "rejected"]:
                val = sample[col]
                if isinstance(val, str):
                    display = val[:200] + "..." if len(val) > 200 else val
                else:
                    display = json.dumps(val, ensure_ascii=False)[:200] + "..."
                console.print(f"  [cyan]{col}[/cyan]: {display}")
        return

    suffix = "" if args.chat_template else "_raw"
    for name, dpo_df in dpo_splits.items():
        if args.output_format in ("parquet", "both"):
            out_path = DPO_DIR / f"{name}{suffix}.parquet"
            dpo_df.to_parquet(out_path, index=False)
            log.info(f"Saved {out_path}  ({len(dpo_df):,} samples)")

        if args.output_format in ("jsonl", "both"):
            out_path = DPO_DIR / f"{name}{suffix}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for _, row in dpo_df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            log.info(f"Saved {out_path}  ({len(dpo_df):,} samples)")

    console.print(f"\n[bold green]Done. DPO data saved to {DPO_DIR}[/bold green]")


if __name__ == "__main__":
    main()
