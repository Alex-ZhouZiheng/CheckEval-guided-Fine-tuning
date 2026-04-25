#!/usr/bin/env python3
"""
Swap-consistent evaluation for the pairwise warm-up adapter.

For each pair, runs the judge twice:
  (A,B) → winner_1      (B,A) → winner_2 (mapped back to original sides)

Agreement rate (winner_1 == winner_2) measures consistency (inverse of
position bias). When both agree, we take the agreed side; when they
disagree we mark 'tie' and exclude from accuracy denominator (or count
as wrong, reported both ways).

Usage:
    python run_warmup_swap_eval.py \\
        --adapter-path results/checkpoints/judge_warmup_tier_5k_r16_lr5e-06/final_adapter \\
        --split test --batch-size 32
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import config as cfg
from utils import (
    build_vanilla_prompt,
    generate_batch,
    load_eval_data,
    load_judge_model,
    make_lora_handle,
    parse_winner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _messages(row_or_swapped):
    p = build_vanilla_prompt(row_or_swapped)
    return [
        {"role": "system", "content": "You are an impartial judge."},
        {"role": "user", "content": p},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--split", default="test", choices=["test", "dev", "dev_600"])
    p.add_argument("--subset", default=None)
    p.add_argument("--model-id", default=cfg.JUDGE_MODEL_ID)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-lora-rank", type=int, default=cfg.LORA_RANK)
    p.add_argument("--backend", type=str, default=None,
                   choices=["llamacpp", "vllm"],
                   help="Inference backend; defaults to cfg.INFERENCE_BACKEND.")
    args = p.parse_args()

    df = load_eval_data(args.split, args.subset)
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    log.info("Evaluating %d pairs", len(df))

    model = load_judge_model(
        model_id=args.model_id,
        backend=args.backend,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        llamacpp_adapter_path=str(args.adapter_path),
    )
    lora_req = make_lora_handle(
        adapter_path=str(args.adapter_path),
        backend=args.backend,
        name="warmup",
        lora_int_id=1,
    )

    # normal order
    msgs_norm = [_messages(r) for _, r in df.iterrows()]
    # swapped order
    swapped = df.copy()
    swapped["response_a"], swapped["response_b"] = df["response_b"], df["response_a"]
    msgs_swap = [_messages(r) for _, r in swapped.iterrows()]

    log.info("Pass 1: normal order")
    raw_norm = generate_batch(model, msgs_norm, batch_size=args.batch_size, lora_request=lora_req)
    log.info("Pass 2: swapped order")
    raw_swap = generate_batch(model, msgs_swap, batch_size=args.batch_size, lora_request=lora_req)

    pred_norm = [parse_winner(r) for r in raw_norm]        # already in original frame
    pred_swap_raw = [parse_winner(r) for r in raw_swap]    # but A/B here mean swapped
    # map swapped pred back to original: A→B, B→A
    pred_swap = [("B" if w == "A" else "A" if w == "B" else w) for w in pred_swap_raw]

    gold = df["winner"].tolist()

    n = len(df)
    agree = sum(1 for a, b in zip(pred_norm, pred_swap) if a == b and a in ("A", "B"))
    # metrics
    def _acc(preds):
        correct = sum(1 for p, g in zip(preds, gold) if p == g)
        frac_A = sum(1 for p in preds if p == "A") / n
        return correct / n, frac_A

    acc_norm, fa_norm = _acc(pred_norm)
    acc_swap, fa_swap = _acc(pred_swap)

    # consistent (both agree) subset
    consistent_idx = [i for i in range(n) if pred_norm[i] == pred_swap[i] and pred_norm[i] in ("A", "B")]
    acc_consistent = sum(1 for i in consistent_idx if pred_norm[i] == gold[i]) / max(1, len(consistent_idx))

    # strict: disagreement = wrong
    strict_correct = sum(
        1 for i in range(n)
        if pred_norm[i] == pred_swap[i] == gold[i]
    )
    acc_strict = strict_correct / n

    log.info("─" * 60)
    log.info("Normal-order acc   : %.4f   frac_pred_A = %.3f", acc_norm, fa_norm)
    log.info("Swapped-order acc  : %.4f   frac_pred_A = %.3f", acc_swap, fa_swap)
    log.info("Agreement rate     : %.4f   (%d/%d consistent)", agree / n, agree, n)
    log.info("Acc on consistent  : %.4f   (n=%d)", acc_consistent, len(consistent_idx))
    log.info("Strict acc (agree+correct): %.4f", acc_strict)
    log.info("Position bias      : %.3f   (|A_norm + A_swap - 1|; 0 = unbiased)",
             abs(fa_norm + fa_swap - 1.0))

    # save
    out_dir = cfg.RESULTS_DIR / f"warmup_swap_{Path(args.adapter_path).parent.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = df.copy()
    results["pred_norm"] = pred_norm
    results["pred_swap"] = pred_swap
    results["raw_norm"] = raw_norm
    results["raw_swap"] = raw_swap
    results.to_parquet(out_dir / f"{args.split}_preds.parquet", index=False)
    metrics = {
        "split": args.split,
        "n": n,
        "acc_norm": acc_norm,
        "acc_swap": acc_swap,
        "acc_consistent": acc_consistent,
        "acc_strict": acc_strict,
        "agreement_rate": agree / n,
        "frac_pred_A_norm": fa_norm,
        "frac_pred_A_swap": fa_swap,
        "position_bias": abs(fa_norm + fa_swap - 1.0),
        "adapter": str(args.adapter_path),
        "date": str(date.today()),
    }
    with open(out_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
