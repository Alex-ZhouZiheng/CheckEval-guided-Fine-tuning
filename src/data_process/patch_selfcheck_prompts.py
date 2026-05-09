#!/usr/bin/env python3
"""
Rewrite the cached student_prompt / messages in existing self-checklist SFT
and GRPO parquets/jsonls after the prompt template was edited (e.g. to drop
"Tie" from the final Winner instructions).

Keeps existing teacher traces (target output) intact — only patches the
prompt that the student sees.

Usage:
    # Patch SFT parquet
    python -m src.data_process.patch_selfcheck_prompts \
        --sft-parquet data/checklist_sft/self_checklist_sft_tier_10k.parquet \
        --thinking

    # Patch GRPO jsonl
    python -m src.data_process.patch_selfcheck_prompts \
        --grpo-jsonl data/judge_sft/grpo_train_tier_10k_selfcheck.jsonl \
        --thinking

    # Both at once, write to new files instead of in place
    python -m src.data_process.patch_selfcheck_prompts \
        --sft-parquet IN.parquet --sft-out OUT.parquet \
        --grpo-jsonl IN.jsonl --grpo-out OUT.jsonl \
        --thinking
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data_process.prepare_self_checklist_sft import (
    SELF_CHECKLIST_STUDENT_PROMPT_THINKING,
    SELF_CHECKLIST_STUDENT_PROMPT_LEGACY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _rebuild_prompt(row: dict, thinking: bool) -> str:
    template = (
        SELF_CHECKLIST_STUDENT_PROMPT_THINKING if thinking
        else SELF_CHECKLIST_STUDENT_PROMPT_LEGACY
    )
    return template.format(
        context=row["context"],
        response_a=row["response_a"],
        response_b=row["response_b"],
    )


def patch_sft_parquet(in_path: Path, out_path: Path, thinking: bool) -> None:
    df = pd.read_parquet(in_path)
    log.info("Loaded SFT parquet: %d rows from %s", len(df), in_path)

    needed = {"context", "response_a", "response_b"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"SFT parquet missing required columns {missing}. "
            f"Cannot rebuild prompt without context/response_a/response_b."
        )

    new_messages = []
    for _, row in df.iterrows():
        prompt = _rebuild_prompt(row, thinking=thinking)
        new_messages.append([{"role": "user", "content": prompt}])
    df["messages"] = new_messages

    if "student_prompt" in df.columns:
        df["student_prompt"] = [m[0]["content"] for m in new_messages]

    df.to_parquet(out_path, index=False)
    log.info("Wrote patched SFT parquet -> %s", out_path)


def patch_grpo_jsonl(in_path: Path, out_path: Path, thinking: bool) -> None:
    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ctx = d.get("context")
            ra = d.get("response_a")
            rb = d.get("response_b")
            if ctx is None or ra is None or rb is None:
                # Fall back: GRPO jsonl may store prompt under "messages"
                msgs = d.get("messages")
                if msgs and isinstance(msgs, list):
                    # Cannot rebuild without raw fields — skip
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                    continue
                raise RuntimeError(
                    f"GRPO row missing context/response_a/response_b: keys={list(d.keys())}"
                )
            new_prompt = _rebuild_prompt(
                {"context": ctx, "response_a": ra, "response_b": rb},
                thinking=thinking,
            )
            d["messages"] = [{"role": "user", "content": new_prompt}]
            if "prompt" in d:
                d["prompt"] = new_prompt
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d patched GRPO rows -> %s", n, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-parquet", type=Path, default=None)
    ap.add_argument("--sft-out", type=Path, default=None,
                    help="Default: in-place overwrite of --sft-parquet")
    ap.add_argument("--grpo-jsonl", type=Path, default=None)
    ap.add_argument("--grpo-out", type=Path, default=None,
                    help="Default: in-place overwrite of --grpo-jsonl")
    ap.add_argument("--thinking", action="store_true",
                    help="Use the THINKING student prompt variant")
    args = ap.parse_args()

    if args.sft_parquet is not None:
        out = args.sft_out or args.sft_parquet
        patch_sft_parquet(args.sft_parquet, out, thinking=args.thinking)

    if args.grpo_jsonl is not None:
        out = args.grpo_out or args.grpo_jsonl
        patch_grpo_jsonl(args.grpo_jsonl, out, thinking=args.thinking)

    if args.sft_parquet is None and args.grpo_jsonl is None:
        ap.error("Provide at least one of --sft-parquet or --grpo-jsonl")


if __name__ == "__main__":
    main()
