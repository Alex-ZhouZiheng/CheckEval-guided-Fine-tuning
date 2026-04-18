#!/usr/bin/env python3
"""
Verify the synthetic-label assumption used by prepare_judge_sft.py.

The judge SFT builder assumes:
    winner == side   -> all questions = yes
    winner != side   -> all questions = no
    winner == "Tie"  -> both sides = all yes

This script actually runs a zero-shot CheckEval judge on each
(sample, side) pair using the *same* generator-produced checklist,
parses the per-question yes/no/N/A labels, and compares them against
the synthetic labels. It reports:

    - per-question agreement rate
    - per-row "fully agrees" rate  (all n_q questions match synthetic)
    - per-side yes-rate vs. synthetic yes-rate
    - breakdowns by winner (A/B/Tie) and by domain

Usage:
    python -m src.analysis.verify_judge_sft_labels --tier debug_5k --limit 200
    python -m src.analysis.verify_judge_sft_labels --tier tier_10k --limit 1000
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from utils import (
    CHECKEVAL_POINTWISE_PROMPT,
    generate_batch,
    load_judge_model,
    parse_checkeval_output,
)
from eval.run_generator_infer import parse_generated_checklist
from data_process.prepare_judge_sft import (
    flatten_checklist,
    load_pairwise,
    load_generated,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


def build_prompt(row, per_domain, side):
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
    return prompt, flat_q


def synthetic_label(side: str, winner: str) -> str:
    if winner == "Tie":
        return "yes"
    return "yes" if winner == side else "no"


def parsed_to_map(parsed: dict, n_q: int) -> dict[int, str]:
    """Return {q_num: 'yes'|'no'|'na'|'missing'} for q = 1..n_q."""
    out: dict[int, str] = {q: "missing" for q in range(1, n_q + 1)}
    if parsed.get("_raw_fallback"):
        return out
    for a in parsed.get("answers", []):
        q = a.get("q")
        if isinstance(q, int) and 1 <= q <= n_q:
            out[q] = a.get("answer", "missing")
    for a in parsed.get("na_answers", []):
        q = a.get("q")
        if isinstance(q, int) and 1 <= q <= n_q:
            out[q] = "na"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="debug_5k",
                    choices=["debug_5k", "tier_10k", "tier_20k"])
    ap.add_argument("--generated", type=str, default=None,
                    help="Path to generated checklists parquet (default inferred from tier)")
    ap.add_argument("--limit", type=int, default=200,
                    help="Number of pairs to verify (not rows — each pair yields 2 rows)")
    ap.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int,
                    default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    ap.add_argument("--max-num-seqs", type=int,
                    default=cfg.VLLM_ENGINE_KWARGS["max_num_seqs"])
    ap.add_argument("--tensor-parallel-size", type=int,
                    default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    ap.add_argument("--output-dir", type=Path,
                    default=cfg.RESULTS_DIR / "judge_sft_label_audit")
    ap.add_argument("--na-as", default="no", choices=["no", "skip"],
                    help="How to treat judge N/A when comparing to synthetic yes/no")
    args = ap.parse_args()

    gen_path = Path(args.generated) if args.generated else (
        cfg.GENERATED_CHECKLIST_DIR / f"{args.tier}.parquet"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data (mirror prepare_judge_sft) ──
    pairs = load_pairwise(args.tier)
    gen_by_sample = load_generated(gen_path)
    log.info("Loaded %d pairs and %d generated checklists", len(pairs), len(gen_by_sample))

    # keep only pairs that have a generated checklist
    pairs = pairs[pairs["sample_id"].isin(gen_by_sample)].reset_index(drop=True)
    if args.limit:
        pairs = pairs.head(args.limit).reset_index(drop=True)
    log.info("Verifying %d pairs (%d rows)", len(pairs), 2 * len(pairs))

    # ── build prompts: A then B, same order as pairs ──
    prompts: list[list[dict]] = []
    meta: list[dict] = []
    for _, r in pairs.iterrows():
        per_domain = gen_by_sample[r["sample_id"]]
        for side in ("A", "B"):
            prompt, flat_q = build_prompt(r, per_domain, side)
            if not flat_q:
                continue
            prompts.append([{"role": "user", "content": prompt}])
            meta.append({
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "winner": r["winner"],
                "side": side,
                "n_q": len(flat_q),
                "questions": flat_q,
            })
    log.info("Built %d prompts", len(prompts))

    # ── run judge ──
    model = load_judge_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    t0 = time.time()
    raws = generate_batch(model, prompts, batch_size=args.batch_size,
                          max_new_tokens=args.max_new_tokens)
    elapsed = time.time() - t0
    log.info("Judge inference: %.1fs (%.2fs / row)", elapsed, elapsed / max(len(prompts), 1))

    # ── compare to synthetic labels ──
    per_row = []
    per_q_records = []
    for m, raw in zip(meta, raws):
        parsed = parse_checkeval_output(raw, expected_n=m["n_q"])
        qmap = parsed_to_map(parsed, m["n_q"])
        synth = synthetic_label(m["side"], m["winner"])

        n_yes = sum(1 for v in qmap.values() if v == "yes")
        n_no = sum(1 for v in qmap.values() if v == "no")
        n_na = sum(1 for v in qmap.values() if v == "na")
        n_missing = sum(1 for v in qmap.values() if v == "missing")

        # per-question agreement (na policy)
        n_agree = 0
        n_scored = 0
        for q, v in qmap.items():
            if v == "missing":
                continue
            if v == "na":
                effective = "no" if args.na_as == "no" else None
                if effective is None:
                    continue
            else:
                effective = v
            n_scored += 1
            if effective == synth:
                n_agree += 1
            per_q_records.append({
                "sample_id": m["sample_id"],
                "domain": m["domain"],
                "winner": m["winner"],
                "side": m["side"],
                "q_num": q,
                "question": m["questions"][q - 1],
                "judge_label": v,
                "synthetic_label": synth,
                "agree": int(effective == synth),
            })

        per_row.append({
            "sample_id": m["sample_id"],
            "domain": m["domain"],
            "winner": m["winner"],
            "side": m["side"],
            "n_q": m["n_q"],
            "n_yes": n_yes,
            "n_no": n_no,
            "n_na": n_na,
            "n_missing": n_missing,
            "synthetic_label": synth,
            "n_scored": n_scored,
            "n_agree": n_agree,
            "row_agreement": (n_agree / n_scored) if n_scored else None,
            "fully_agrees": int(n_scored > 0 and n_agree == n_scored),
            "yes_rate": (n_yes / m["n_q"]) if m["n_q"] else None,
            "raw_output": raw,
        })

    row_df = pd.DataFrame(per_row)
    q_df = pd.DataFrame(per_q_records)

    # ── summary ──
    summary: dict = {
        "tier": args.tier,
        "n_pairs": int(len(pairs)),
        "n_rows": int(len(row_df)),
        "n_questions_total": int(q_df.shape[0]) if len(q_df) else 0,
        "na_policy": args.na_as,
    }

    def _agg(mask: pd.Series) -> dict:
        sub = row_df[mask]
        if len(sub) == 0:
            return {"n_rows": 0}
        keys = set(zip(sub["sample_id"], sub["side"]))
        if len(q_df):
            sub_q = q_df[q_df.apply(lambda r: (r["sample_id"], r["side"]) in keys, axis=1)]
        else:
            sub_q = q_df
        return {
            "n_rows": int(len(sub)),
            "avg_row_agreement": float(sub["row_agreement"].dropna().mean()),
            "fully_agrees_rate": float(sub["fully_agrees"].mean()),
            "avg_yes_rate": float(sub["yes_rate"].dropna().mean()),
            "per_question_agreement": (
                float(sub_q["agree"].mean()) if len(sub_q) else None
            ),
        }

    summary["overall"] = _agg(pd.Series([True] * len(row_df)))

    # winner-side rows (synthetic = yes) vs loser-side rows (synthetic = no)
    is_winner_side = (
        ((row_df["winner"] == "A") & (row_df["side"] == "A"))
        | ((row_df["winner"] == "B") & (row_df["side"] == "B"))
    )
    is_loser_side = (
        ((row_df["winner"] == "A") & (row_df["side"] == "B"))
        | ((row_df["winner"] == "B") & (row_df["side"] == "A"))
    )
    is_tie = row_df["winner"] == "Tie"
    summary["by_role"] = {
        "winner_side (synth=yes)": _agg(is_winner_side),
        "loser_side  (synth=no)":  _agg(is_loser_side),
        "tie        (synth=yes)":  _agg(is_tie),
    }

    summary["by_winner"] = {
        w: _agg(row_df["winner"] == w) for w in sorted(row_df["winner"].unique())
    }
    summary["by_domain"] = {
        d: _agg(row_df["domain"] == d) for d in sorted(row_df["domain"].unique())
    }

    # ── print ──
    t = Table(title=f"Judge SFT Label Audit — {args.tier}")
    t.add_column("Slice", style="bold")
    t.add_column("rows", justify="right")
    t.add_column("per-Q agree", justify="right")
    t.add_column("fully-agree rate", justify="right")
    t.add_column("avg yes-rate", justify="right")

    def _row(label, d):
        if d.get("n_rows", 0) == 0:
            t.add_row(label, "0", "-", "-", "-")
            return
        t.add_row(
            label,
            f"{d['n_rows']}",
            f"{d['per_question_agreement']:.3f}" if d.get('per_question_agreement') is not None else "-",
            f"{d['fully_agrees_rate']:.3f}",
            f"{d['avg_yes_rate']:.3f}" if d.get('avg_yes_rate') is not None else "-",
        )

    _row("OVERALL", summary["overall"])
    for k, d in summary["by_role"].items():
        _row(k, d)
    for k, d in summary["by_winner"].items():
        _row(f"winner={k}", d)
    for k, d in summary["by_domain"].items():
        _row(f"domain={k}", d)
    console.print(t)

    # ── save ──
    tag = f"{args.tier}_n{len(pairs)}"
    row_out = args.output_dir / f"audit_rows_{tag}.parquet"
    q_out = args.output_dir / f"audit_questions_{tag}.parquet"
    sum_out = args.output_dir / f"audit_summary_{tag}.json"

    row_df.to_parquet(row_out, index=False)
    if len(q_df):
        q_df.to_parquet(q_out, index=False)
    with open(sum_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]Saved:[/bold green]")
    console.print(f"  rows     -> {row_out}")
    console.print(f"  per-Q    -> {q_out}")
    console.print(f"  summary  -> {sum_out}")


if __name__ == "__main__":
    main()
