#!/usr/bin/env python3
"""Build verbosity-robustness split: expand the GOLD LOSER without adding info.

Pipeline per row:
  1. Determine loser side from `winner` column. Skip rows where winner == "Tie".
  2. Ask DeepSeek V4 to rewrite loser response: longer / more verbose / no new
     useful information / no error correction. Repeat content only.
  3. Ask DeepSeek V4 verifier whether rewrite added new useful info or fixed
     mistakes. Keep only rows where verifier says "no".
  4. Write parquet with the loser column replaced by the verbose rewrite. The
     winner column is unchanged so a robust judge should still pick the same
     side.

Output schema matches input split (so `--eval-split <out_split>` works).

Usage:
    python src/data_process/build_robust_verbose.py \
        --in-split dev_600 --out-split dev_600_verbose --concurrency 16
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as cfg
from data_process._deepseek_client import RateLimiter, chat, get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


REWRITE_SYS = (
    "You rewrite assistant answers to be longer and more verbose without "
    "improving them. You never add new useful information and never fix "
    "factual, reasoning, formatting, or instruction-following errors."
)

REWRITE_TEMPLATE = """Rewrite the following answer to be much longer and more verbose.

Strict rules:
- Do NOT add any new useful information, facts, examples that change correctness, code, or citations.
- Do NOT fix any factual, reasoning, instruction-following, or formatting errors. Preserve all mistakes verbatim in meaning.
- Restate the same content in a more verbose way. You may: repeat explanations, add hedging, add filler examples that do not change correctness, restate the same points multiple times.
- Preserve the answer's stance, claims, recommendations, code snippets, and conclusions exactly. Code blocks must remain functionally identical.
- Aim for roughly 1.8x to 2.5x the original length.
- Output only the rewritten answer. No preface, no commentary, no markdown fences around the whole thing.

Original answer:
<<<
{answer}
>>>
"""


VERIFY_SYS = "You audit rewrites for information leakage. Be strict."

VERIFY_TEMPLATE = """You are given an ORIGINAL answer and a REWRITE. Determine whether the rewrite contains any new useful information, new facts, new code, or whether it corrected any factual, reasoning, formatting, or instruction-following mistake that was present in the original.

Cosmetic changes (extra hedging, repetition, restatement, filler examples that do not affect correctness) do NOT count as new useful information.

Answer with exactly one token: "yes" if the rewrite added new useful information or fixed mistakes, "no" otherwise.

ORIGINAL:
<<<
{original}
>>>

REWRITE:
<<<
{rewrite}
>>>

Answer (yes/no):"""


_YES_RE = re.compile(r"\byes\b", re.I)
_NO_RE = re.compile(r"\bno\b", re.I)


def _parse_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    # First token wins.
    m = re.search(r"[a-z]+", t)
    if not m:
        return None
    tok = m.group(0)
    if tok == "yes":
        return "yes"
    if tok == "no":
        return "no"
    if _YES_RE.search(t) and not _NO_RE.search(t):
        return "yes"
    if _NO_RE.search(t) and not _YES_RE.search(t):
        return "no"
    return None


def _rewrite_one(client, limiter, original: str, *, model: str | None,
                 max_rewrite_tokens: int) -> tuple[str, str]:
    rewrite = chat(
        client,
        REWRITE_TEMPLATE.format(answer=original),
        system_prompt=REWRITE_SYS,
        model=model,
        temperature=0.6,
        max_tokens=max_rewrite_tokens,
        limiter=limiter,
    ).strip()
    verdict_raw = chat(
        client,
        VERIFY_TEMPLATE.format(original=original, rewrite=rewrite),
        system_prompt=VERIFY_SYS,
        model=model,
        temperature=0.0,
        max_tokens=8,
        limiter=limiter,
    )
    verdict = _parse_yes_no(verdict_raw) or "yes"  # fail closed
    return rewrite, verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-split", default="dev_600")
    p.add_argument("--out-split", default="dev_600_verbose")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--model", default=None, help="Override DEEPSEEK_MODEL")
    p.add_argument("--max-rewrite-tokens", type=int, default=4096)
    p.add_argument("--audit-out", default=None,
                   help="Optional parquet with all rows (incl. dropped) for inspection")
    args = p.parse_args()

    src = cfg.SPLITS_DIR / f"{args.in_split}.parquet"
    dst = cfg.SPLITS_DIR / f"{args.out_split}.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src).reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), src.name)

    client = get_client()
    limiter = RateLimiter(min_interval_s=0.0)

    rewrites: list[str | None] = [None] * len(df)
    verdicts: list[str | None] = [None] * len(df)
    skipped_tie = 0

    def task(i: int):
        row = df.iloc[i]
        winner = row["winner"]
        if winner not in ("A", "B"):
            return i, None, "skip_tie"
        loser_col = "response_b" if winner == "A" else "response_a"
        original = str(row[loser_col])
        try:
            rw, vd = _rewrite_one(
                client, limiter, original,
                model=args.model, max_rewrite_tokens=args.max_rewrite_tokens,
            )
        except Exception as e:
            log.warning("row %d failed: %s", i, e)
            return i, None, "error"
        return i, rw, vd

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(task, i) for i in range(len(df))]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="verbose-rewrite"):
            i, rw, vd = fut.result()
            rewrites[i] = rw
            verdicts[i] = vd
            if vd == "skip_tie":
                skipped_tie += 1

    df_audit = df.copy()
    df_audit["loser_rewrite"] = rewrites
    df_audit["verifier_verdict"] = verdicts

    if args.audit_out:
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        df_audit.to_parquet(args.audit_out, index=False)
        log.info("Audit parquet -> %s", args.audit_out)

    keep_mask = df_audit["verifier_verdict"] == "no"
    kept = df_audit[keep_mask].copy().reset_index(drop=True)

    def _apply(row):
        if row["winner"] == "A":
            row["response_b"] = row["loser_rewrite"]
        else:
            row["response_a"] = row["loser_rewrite"]
        return row

    kept = kept.apply(_apply, axis=1)
    kept = kept.drop(columns=["loser_rewrite", "verifier_verdict"])

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(dst, index=False)
    n_total = len(df)
    n_kept = len(kept)
    n_dropped_verifier = int((df_audit["verifier_verdict"] == "yes").sum())
    n_errors = int((df_audit["verifier_verdict"] == "error").sum())
    log.info(
        "Wrote %d kept rows -> %s | total=%d | tie_skipped=%d | dropped_yes=%d | errors=%d",
        n_kept, dst, n_total, skipped_tie, n_dropped_verifier, n_errors,
    )


if __name__ == "__main__":
    main()
