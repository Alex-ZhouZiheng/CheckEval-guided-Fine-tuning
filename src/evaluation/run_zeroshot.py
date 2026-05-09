#!/usr/bin/env python3
"""
Baseline 1: Zero-shot Vanilla Qwen Judge.

Loads the test split (or debug subset) and asks the Qwen model a simple
pairwise question: "Which response is better? Answer A or B."

No training and no checklist: pure model capability.
"""

from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as cfg
from utils import (
    build_vanilla_prompt,
    compute_metrics,
    generate_batch,
    generate_single,
    load_eval_data,
    load_judge_model,
    parse_winner,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def run_vanilla_judge(
    df: pd.DataFrame,
    model,
    batch_size: int = 1,
    enable_thinking: bool = False,
    max_new_tokens: int | None = None,
) -> pd.DataFrame:
    """Run zero-shot vanilla pairwise judge on each row."""
    results = df.copy()
    raw_outputs = []
    predicted_winners = []

    chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}
    gen_kwargs = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    if batch_size == 1:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Vanilla Judge"):
            prompt_text = build_vanilla_prompt(row)
            messages = [
                {"role": "system", "content": "You are an impartial judge."},
                {"role": "user", "content": prompt_text},
            ]
            raw = generate_single(
                model, messages,
                chat_template_kwargs=chat_template_kwargs,
                **gen_kwargs,
            )
            raw_outputs.append(raw)
            predicted_winners.append(parse_winner(raw))
    else:
        all_messages = []
        for _, row in df.iterrows():
            prompt_text = build_vanilla_prompt(row)
            all_messages.append(
                [
                    {"role": "system", "content": "You are an impartial judge."},
                    {"role": "user", "content": prompt_text},
                ]
            )

        raw_outputs = generate_batch(
            model, all_messages, batch_size=batch_size,
            chat_template_kwargs=chat_template_kwargs,
            **gen_kwargs,
        )
        predicted_winners = [parse_winner(raw) for raw in raw_outputs]

    results["raw_output"] = raw_outputs
    results["predicted_winner"] = predicted_winners
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Vanilla Qwen Judge")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name under data/splits/ (without .parquet).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Use a training subset instead (e.g. debug_5k)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"],
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"],
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"],
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_num_seqs"],
    )
    parser.add_argument("--backend", type=str, default=None,
                        choices=["llamacpp", "vllm"],
                        help="Inference backend; defaults to cfg.INFERENCE_BACKEND.")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Set chat_template_kwargs={'enable_thinking': True} so Qwen3 "
                             "emits a thinking trace before its final answer. Bump "
                             "--max-new-tokens accordingly.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override cfg.GENERATION_KWARGS['max_new_tokens'] (default 512). "
                             "Use ~4096+ when --enable-thinking.")
    parser.add_argument("--enable-mtp", action="store_true",
                        help="Enable vLLM MTP speculative decoding (vllm backend only).")
    parser.add_argument("--mtp-method", type=str, default="mtp",
                        help="vLLM speculative_config method (default: mtp).")
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=10,
                        help="MTP speculative depth (default: 1).")
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    if args.suffix is None:
        args.suffix = Path(args.model_id).name

    speculative_config = (
        {"method": args.mtp_method,
         "num_speculative_tokens": args.mtp_num_speculative_tokens}
        if args.enable_mtp else None
    )

    df = load_eval_data(args.split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples)
        log.info("Capped to %s samples", len(df))

    model = load_judge_model(
        model_id=args.model_id,
        backend=args.backend,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        speculative_config=speculative_config,
    )

    t0 = time.time()
    results = run_vanilla_judge(
        df, model,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0
    log.info(
        "Inference complete in %.1fs  (%.2fs/sample)",
        elapsed,
        elapsed / len(df),
    )

    metrics = compute_metrics(
        y_true=results["winner"].tolist(),
        y_pred=results["predicted_winner"].tolist(),
        domains=results["domain"].tolist(),
    )
    
    metrics["inference_time_s"] = elapsed
    metrics["samples_per_second"] = len(df) / elapsed
    metrics["model_id"] = args.model_id
    metrics["backend"] = "vllm"
    metrics["tensor_parallel_size"] = args.tensor_parallel_size
    metrics["max_model_len"] = args.max_model_len
    metrics["enable_thinking"] = bool(args.enable_thinking)

    split_tag = args.subset or args.split
    experiment_name = f"vanilla_judge_{split_tag}_{args.suffix}"
    save_results(results, metrics, experiment_name)

    log.info("Done.")


if __name__ == "__main__":
    main()
