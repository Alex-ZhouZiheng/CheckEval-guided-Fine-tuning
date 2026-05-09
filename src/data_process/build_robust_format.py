#!/usr/bin/env python3
"""Build format-robustness split: format-only perturbation of BOTH responses.

For each row, applies the same format template to response_a AND response_b
(keeps semantics, only reshapes presentation). DeepSeek V4 rewrites; a verifier
asks whether semantics changed; rows where either side is judged
semantics-changed are dropped.

Templates (4):
  v1 markdown    : sectioned headings (**Answer:** / **Reasoning:**)
  v2 bullets     : bullet list of points
  v3 reorder     : same content, reordered paragraphs / sentence grouping
  v4 plain       : strip markdown, merge into plain prose paragraphs

Default produces ONE output file with a random template per row.
Use --all-templates to produce 4 separate parquets (dev_600_format_v1..v4).

Usage:
    python src/data_process/build_robust_format.py \
        --in-split dev_600 --out-split dev_600_format \
        --concurrency 24

    python src/data_process/build_robust_format.py \
        --in-split dev_600 --out-prefix dev_600_format \
        --all-templates --concurrency 24
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as cfg
from data_process._deepseek_client import RateLimiter, chat, get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


TEMPLATES = {
    "v1_markdown": (
        "Reformat the following answer using markdown sections. Use bold "
        "headings such as **Answer:** and **Reasoning:** (or **Steps:**, "
        "**Notes:** as appropriate). Preserve every claim, fact, code block, "
        "number, and conclusion exactly. Do not add new content. Do not "
        "remove content. Code blocks must remain functionally identical."
    ),
    "v2_bullets": (
        "Reformat the following answer as a bullet list of concise points. "
        "Each substantive point becomes one bullet. Preserve every claim, "
        "fact, code block, number, and conclusion exactly. Do not add new "
        "content. Do not remove content. Keep code blocks intact."
    ),
    "v3_reorder": (
        "Reorder the paragraphs and sentence groupings of the following "
        "answer for stylistic variation, while preserving all content "
        "exactly. Do not change wording of factual claims, numbers, code, "
        "or conclusions. Do not add or remove content. Keep code blocks "
        "intact and in their original form."
    ),
    "v4_plain": (
        "Reformat the following answer as plain prose. Remove markdown "
        "formatting (headings, bold, bullets) and merge points into "
        "flowing paragraphs. Preserve every claim, fact, code block, "
        "number, and conclusion exactly. Do not add new content. Do not "
        "remove content. Keep code blocks intact (still wrapped in triple "
        "backticks)."
    ),
}

TEMPLATE_KEYS = list(TEMPLATES.keys())


REWRITE_SYS = (
    "You reformat assistant answers without changing their meaning. You "
    "preserve all factual content, code, numbers, and conclusions exactly."
)

REWRITE_TEMPLATE = """{instruction}

Output only the reformatted answer. No preface, no commentary.

Original answer:
<<<
{answer}
>>>
"""


VERIFY_SYS = "You audit reformats for semantic drift. Be strict."

VERIFY_TEMPLATE = """You are given an ORIGINAL answer and a REFORMATTED version. Determine whether the reformat changed semantics: added, removed, or altered any claim, fact, number, code, or conclusion. Pure presentation changes (markdown, bullets, paragraph order, prose vs list) do NOT count.

Answer with exactly one token: "yes" if semantics changed, "no" if only formatting changed.

ORIGINAL:
<<<
{original}
>>>

REFORMATTED:
<<<
{rewrite}
>>>

Answer (yes/no):"""


def _parse_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    m = re.search(r"[a-z]+", t)
    if not m:
        return None
    tok = m.group(0)
    if tok in ("yes", "no"):
        return tok
    return None


def _do_pair(client, limiter, response: str, instruction: str, *,
             model: str | None, max_tokens: int) -> tuple[str, str]:
    rewrite = chat(
        client,
        REWRITE_TEMPLATE.format(instruction=instruction, answer=response),
        system_prompt=REWRITE_SYS,
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
        limiter=limiter,
    ).strip()
    verdict_raw = chat(
        client,
        VERIFY_TEMPLATE.format(original=response, rewrite=rewrite),
        system_prompt=VERIFY_SYS,
        model=model,
        temperature=0.0,
        max_tokens=8,
        limiter=limiter,
    )
    verdict = _parse_yes_no(verdict_raw) or "yes"  # fail closed
    return rewrite, verdict


def build_one_template(df: pd.DataFrame, template_key: str, *,
                       client, limiter, concurrency: int, model: str | None,
                       max_tokens: int) -> pd.DataFrame:
    instruction = TEMPLATES[template_key]
    rewrites_a: list[str | None] = [None] * len(df)
    rewrites_b: list[str | None] = [None] * len(df)
    verdicts_a: list[str | None] = [None] * len(df)
    verdicts_b: list[str | None] = [None] * len(df)

    def task(i: int):
        row = df.iloc[i]
        try:
            ra, va = _do_pair(client, limiter, str(row["response_a"]), instruction,
                              model=model, max_tokens=max_tokens)
            rb, vb = _do_pair(client, limiter, str(row["response_b"]), instruction,
                              model=model, max_tokens=max_tokens)
        except Exception as e:
            log.warning("row %d template %s failed: %s", i, template_key, e)
            return i, None, None, "error", "error"
        return i, ra, rb, va, vb

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(task, i) for i in range(len(df))]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"format[{template_key}]"):
            i, ra, rb, va, vb = fut.result()
            rewrites_a[i], rewrites_b[i] = ra, rb
            verdicts_a[i], verdicts_b[i] = va, vb

    out = df.copy()
    out["response_a_rewrite"] = rewrites_a
    out["response_b_rewrite"] = rewrites_b
    out["verifier_a"] = verdicts_a
    out["verifier_b"] = verdicts_b
    out["format_template"] = template_key
    return out


def finalize(df_audit: pd.DataFrame) -> pd.DataFrame:
    keep = (df_audit["verifier_a"] == "no") & (df_audit["verifier_b"] == "no")
    kept = df_audit[keep].copy().reset_index(drop=True)
    kept["response_a"] = kept["response_a_rewrite"]
    kept["response_b"] = kept["response_b_rewrite"]
    kept = kept.drop(columns=[
        "response_a_rewrite", "response_b_rewrite",
        "verifier_a", "verifier_b",
    ])
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-split", default="dev_600")
    p.add_argument("--out-split", default="dev_600_format",
                   help="Used in single-template mode (random per row).")
    p.add_argument("--out-prefix", default="dev_600_format",
                   help="Used with --all-templates: writes <prefix>_<vN>.parquet")
    p.add_argument("--all-templates", action="store_true",
                   help="Produce 4 parquets, one per template.")
    p.add_argument("--templates", nargs="+", default=None,
                   help="Subset of template keys (default: all 4).")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--audit-dir", default=None,
                   help="If set, dumps audit parquet(s) with verifier columns.")
    args = p.parse_args()

    src = cfg.SPLITS_DIR / f"{args.in_split}.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src).reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), src.name)

    template_keys = args.templates or TEMPLATE_KEYS
    for k in template_keys:
        if k not in TEMPLATES:
            raise SystemExit(f"unknown template key: {k}")

    client = get_client()
    limiter = RateLimiter(min_interval_s=0.0)
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)

    if args.all_templates:
        for tk in template_keys:
            audit = build_one_template(
                df, tk, client=client, limiter=limiter,
                concurrency=args.concurrency, model=args.model,
                max_tokens=args.max_tokens,
            )
            if audit_dir:
                audit.to_parquet(audit_dir / f"{args.out_prefix}_{tk}_audit.parquet",
                                 index=False)
            kept = finalize(audit)
            dst = cfg.SPLITS_DIR / f"{args.out_prefix}_{tk}.parquet"
            kept.to_parquet(dst, index=False)
            log.info("[%s] kept %d/%d -> %s", tk, len(kept), len(df), dst)
        return

    # Single-output mode: random template per row.
    rng = random.Random(args.seed)
    df = df.copy()
    df["format_template"] = [rng.choice(template_keys) for _ in range(len(df))]

    parts: list[pd.DataFrame] = []
    for tk in template_keys:
        sub = df[df["format_template"] == tk].reset_index(drop=True)
        if len(sub) == 0:
            continue
        audit = build_one_template(
            sub, tk, client=client, limiter=limiter,
            concurrency=args.concurrency, model=args.model,
            max_tokens=args.max_tokens,
        )
        parts.append(audit)

    audit_all = pd.concat(parts, ignore_index=True)
    if audit_dir:
        audit_all.to_parquet(audit_dir / f"{args.out_split}_audit.parquet", index=False)

    kept = finalize(audit_all)
    dst = cfg.SPLITS_DIR / f"{args.out_split}.parquet"
    kept.to_parquet(dst, index=False)
    log.info("kept %d/%d -> %s", len(kept), len(df), dst)


if __name__ == "__main__":
    main()
