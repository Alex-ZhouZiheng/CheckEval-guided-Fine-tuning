#!/usr/bin/env python3
"""
Evaluate a DPO fine-tuned LoRA adapter as a pairwise judge.

Loads the base model via vLLM with native LoRA support and serves the
adapter through ``LoRARequest`` — avoiding the ``merge_and_unload`` →
``save_pretrained`` roundtrip that mangles VL weight key names (the
``language_model.language_model.*`` vs ``language_model.model.*``
mismatch between transformers and vLLM).

Usage:
    python run_eval_finetuned.py \\
        --adapter-path results/checkpoints/dpo_debug_5k_.../final_adapter \\
        --eval-mode vanilla

    python run_eval_finetuned.py \\
        --adapter-path results/checkpoints/dpo_debug_5k_.../final_adapter \\
        --eval-mode both --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import date

import pandas as pd
from tqdm import tqdm
from vllm.lora.request import LoRARequest

import config as cfg
from utils import (
    build_checkeval_prompt,
    build_vanilla_prompt,
    compare_checklists_pairwise,
    compute_metrics,
    generate_batch,
    load_checklists,
    load_eval_data,
    load_judge_model,
    parse_checkeval_output,
    parse_winner,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ────────────────────────── adapter config ───────────────────


def read_adapter_rank(adapter_path: Path) -> int:
    """Read the LoRA rank from ``adapter_config.json``."""
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found — is {adapter_path} a PEFT adapter dir?"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    r = int(adapter_cfg.get("r", 16))
    return r


# ────────────────────────── evaluation modes ─────────────────


def run_vanilla_eval(
    df: pd.DataFrame,
    model,
    batch_size: int,
    lora_request: LoRARequest | None = None,
) -> pd.DataFrame:
    """Run vanilla pairwise judge evaluation."""
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
        model, all_messages, batch_size=batch_size, lora_request=lora_request
    )
    predicted_winners = [parse_winner(raw) for raw in raw_outputs]

    results = df.copy()
    results["raw_output"] = raw_outputs
    results["predicted_winner"] = predicted_winners
    return results


TIE_DELTA = 0.0 


def run_checkeval_eval(
    df: pd.DataFrame,
    model,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    batch_size: int,
    expected_n: int,
    tie_delta: float = TIE_DELTA,
    lora_request: LoRARequest | None = None,
) -> pd.DataFrame:
    """Run pointwise CheckEval judge evaluation with per-question pairwise aggregation.

    Uses :func:`compare_checklists_pairwise` — N/A is a first-class label in the
    margin table, so we no longer need to drop whole samples via a strict na_policy.
    """
    messages_a: list[list[dict]] = []
    messages_b: list[list[dict]] = []
    for _, row in df.iterrows():
        prompt_a = build_checkeval_prompt(row, checklists, definitions, side="A")
        prompt_b = build_checkeval_prompt(row, checklists, definitions, side="B")
        messages_a.append([{"role": "user", "content": prompt_a}])
        messages_b.append([{"role": "user", "content": prompt_b}])

    all_messages = messages_a + messages_b
    all_raw = generate_batch(
        model,
        all_messages,
        batch_size=batch_size,
        max_new_tokens=2048,
        lora_request=lora_request,
    )
    n = len(df)
    raw_a = all_raw[:n]
    raw_b = all_raw[n:]

    predicted_winners: list[str | None] = []
    margins: list[float | None] = []
    n_aligned_list: list[int | None] = []
    parse_successes: list[bool] = []

    for ra, rb in tqdm(zip(raw_a, raw_b), total=n, desc="CheckEval Parse"):
        parsed_a = parse_checkeval_output(ra, expected_n=expected_n)
        parsed_b = parse_checkeval_output(rb, expected_n=expected_n)

        cmp = compare_checklists_pairwise(
            parsed_a, parsed_b,
            expected_n=expected_n,
            tie_delta=tie_delta,
        )

        if cmp is None:
            predicted_winners.append(None)
            margins.append(None)
            n_aligned_list.append(None)
            parse_successes.append(False)
        else:
            predicted_winners.append(cmp["winner"])
            margins.append(cmp["margin"])
            n_aligned_list.append(cmp["n_aligned"])
            parse_successes.append(True)

    results = df.copy()
    results["raw_output_a"] = raw_a
    results["raw_output_b"] = raw_b
    results["margin"] = margins
    results["n_aligned"] = n_aligned_list
    results["predicted_winner"] = predicted_winners
    results["checklist_parsed"] = parse_successes

    n_parsed = sum(parse_successes)
    log.info(
        "Parse rate: %s/%s (%.1f%%)  [pairwise, tie_delta=%.3f]",
        n_parsed, n, 100 * n_parsed / n, tie_delta,
    )
    return results


# ────────────────────────── main ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LoRA judge"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to PEFT LoRA adapter directory (must contain adapter_config.json)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["test", "dev"],
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="vanilla",
        choices=["vanilla", "checkeval", "both"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(cfg.JUDGE_MODEL_ID),
        help="Base model to load with vLLM (LoRA adapter attached via LoRARequest)",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="Override max LoRA rank (auto-detected from adapter_config.json if omitted)",
    )
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
        "--tie-delta",
        type=float,
        default=TIE_DELTA,
        help="Margin threshold for Tie in pairwise checklist scoring (|margin| <= tie_delta → Tie).",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()

    # 1. Detect LoRA rank from the adapter and load base model with vLLM
    #    native LoRA support. We avoid merge+save entirely to sidestep the
    #    `language_model.language_model.*` vs `language_model.model.*` key
    #    rename that transformers' `save_pretrained` introduces for VL models.
    detected_rank = read_adapter_rank(adapter_path)
    max_lora_rank = args.max_lora_rank or max(detected_rank, 16)
    log.info(
        "Adapter rank=%d  →  loading base %s with enable_lora=True (max_lora_rank=%d)",
        detected_rank, args.base_model, max_lora_rank,
    )

    model = load_judge_model(
        model_id=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=1,
    )

    lora_request = LoRARequest(
        lora_name=adapter_path.name,
        lora_int_id=1,
        lora_path=str(adapter_path),
    )

    # 3. Load eval data
    df = load_eval_data(args.eval_split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples)
        log.info("Capped to %s samples", len(df))

    adapter_name = adapter_path.name
    split_tag = args.subset or args.eval_split

    # 4. Vanilla evaluation
    if args.eval_mode in ("vanilla", "both"):
        log.info("Running vanilla evaluation...")
        t0 = time.time()
        results = run_vanilla_eval(
            df, model, args.batch_size, lora_request=lora_request
        )
        elapsed = time.time() - t0
        log.info("Vanilla inference: %.1fs (%.2fs/sample)", elapsed, elapsed / len(df))

        metrics = compute_metrics(
            y_true=results["winner"].tolist(),
            y_pred=results["predicted_winner"].tolist(),
            domains=results["domain"].tolist(),
        )
        metrics["inference_time_s"] = elapsed
        metrics["samples_per_second"] = len(df) / elapsed
        metrics["adapter_path"] = str(adapter_path)
        metrics["eval_mode"] = "vanilla"
        metrics["model_id"] = args.base_model
        
        time_now = date.today()

        experiment_name = f"finetuned_vanilla_{adapter_name}_{split_tag}_{time_now}"
        save_results(results, metrics, experiment_name)

    # 5. CheckEval evaluation
    if args.eval_mode in ("checkeval", "both"):
        checklists, definitions = load_checklists()
        total_q = sum(len(q) for q in checklists.values())
        log.info("Running CheckEval evaluation (%d questions)...", total_q)

        t0 = time.time()
        results = run_checkeval_eval(
            df,
            model,
            checklists,
            definitions,
            args.batch_size,
            expected_n=total_q,
            tie_delta=args.tie_delta,
            lora_request=lora_request,
        )
        elapsed = time.time() - t0
        log.info("CheckEval inference: %.1fs (%.2fs/sample)", elapsed, elapsed / len(df))

        metrics = compute_metrics(
            y_true=results["winner"].tolist(),
            y_pred=results["predicted_winner"].tolist(),
            domains=results["domain"].tolist(),
        )
        metrics["inference_time_s"] = elapsed
        metrics["samples_per_second"] = len(df) / elapsed
        metrics["adapter_path"] = str(adapter_path)
        metrics["eval_mode"] = "checkeval"
        metrics["model_id"] = args.base_model
        metrics["parse_rate"] = results["checklist_parsed"].mean()
        metrics["n_checklist_questions"] = total_q
        metrics["tie_delta"] = args.tie_delta
        metrics["scoring"] = "pairwise_per_question"

        time_now = date.today()

        experiment_name = f"finetuned_checkeval_{adapter_name}_{split_tag}_{time_now}"
        save_results(results, metrics, experiment_name)

    log.info("Done.")


if __name__ == "__main__":
    main()
