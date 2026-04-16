#!/usr/bin/env python3
"""
Evaluate the checklist-conditioned judge (LoRA adapter) on a pairwise split,
using pre-generated checklists from ``run_generator_infer.py``.

Pipeline per sample:
  1. Look up the generated per-domain checklist.
  2. Build two pointwise CheckEval prompts (side A, side B) over the same
     flattened question list.
  3. Batch-generate with the judge adapter.
  4. Parse Yes/No/N/A, pairwise-aggregate with ``compare_checklists_pairwise``.
  5. Compute accuracy / macro-F1 / per-domain metrics, persist results.

Usage:
    python run_judge_eval.py \\
        --judge-adapter results/checkpoints/judge_sft_tier_10k_.../final_adapter \\
        --generated data/generated_checklists/dev_600.parquet \\
        --eval-split dev --subset dev_600
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm.lora.request import LoRARequest

import config as cfg
from data_process.prepare_judge_sft import build_pointwise_prompt
from run_generator_infer import parse_generated_checklist
from utils import (
    compare_checklists_pairwise,
    compute_metrics,
    generate_batch,
    load_judge_model,
    parse_checkeval_output,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


TIE_DELTA = 0.0


def load_eval_data(eval_split: str = "dev", subset: str | None = None) -> pd.DataFrame:
    """Load reasoning-augmented eval data from ``data/with_reason/``.

    The judge eval needs ``sample_id`` to look up generated checklists, so it
    must read the same reasoning parquet that ``run_generator_infer.py``
    consumed. We also filter to ``swap_flag == False`` to match the generator
    (one row per sample_id).
    """
    split_tag = subset if (subset and subset != "full") else eval_split
    path = cfg.WITH_REASON_DIR / f"{split_tag}_reasoning.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_data_reasoning.py first, e.g. "
            f"`python src/data_process/prepare_data_reasoning.py --split {split_tag}`."
        )

    df = pd.read_parquet(path)
    if "swap_flag" in df.columns:
        df = df[df["swap_flag"] == False].reset_index(drop=True)  # noqa: E712
    log.info("Loaded %s pairs from %s", f"{len(df):,}", path.name)
    return df


def load_generated_map(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_generator_infer.py first on the same split."
        )
    df = pd.read_parquet(path)
    out: dict[str, dict[str, list[str]]] = {}
    for _, r in df.iterrows():
        text = r.get("generated_checklist", "") or r.get("raw_output", "")
        if isinstance(text, str) and text.strip():
            per_domain = parse_generated_checklist(text)
            if any(per_domain.values()):
                out[r["sample_id"]] = per_domain
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-adapter", type=str, default=None,
                        help="Path to judge final_adapter directory. "
                             "Omit to run the base model (pre-FT baseline).")
    parser.add_argument("--base-model", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--generated", type=str, required=True,
                        help="data/generated_checklists/<split>.parquet")
    parser.add_argument("--eval-split", type=str, default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tie-delta", type=float, default=TIE_DELTA)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    args = parser.parse_args()

    if args.judge_adapter:
        adapter_path = Path(args.judge_adapter).resolve()
        with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
            lora_rank = int(json.load(f).get("r", 16))
    else:
        adapter_path = None
        lora_rank = 16
        log.info("No judge adapter — running base %s as pre-FT baseline",
                 args.base_model)

    df = load_eval_data(args.eval_split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)

    gen_map = load_generated_map(Path(args.generated))
    log.info("Loaded generated checklists for %d / %d samples", len(gen_map), len(df))

    # Build per-sample prompts and remember expected_n per row.
    messages_a: list[list[dict]] = []
    messages_b: list[list[dict]] = []
    expected_ns: list[int] = []
    keep_idx: list[int] = []
    for i, (_, row) in enumerate(df.iterrows()):
        per_domain = gen_map.get(row["sample_id"])
        if not per_domain:
            expected_ns.append(0)
            continue
        prompt_a, n_q = build_pointwise_prompt(row, per_domain, "A")
        prompt_b, _ = build_pointwise_prompt(row, per_domain, "B")
        if n_q == 0:
            expected_ns.append(0)
            continue
        messages_a.append([{"role": "user", "content": prompt_a}])
        messages_b.append([{"role": "user", "content": prompt_b}])
        expected_ns.append(n_q)
        keep_idx.append(i)

    log.info("Evaluable samples: %d / %d (rest skipped: empty checklist)",
             len(keep_idx), len(df))

    if not keep_idx:
        raise SystemExit("No evaluable samples — regenerate checklists first.")

    # vLLM with optional judge LoRA.
    enable_lora = adapter_path is not None
    model = load_judge_model(
        model_id=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=enable_lora,
        max_lora_rank=max(lora_rank, 16) if enable_lora else None,
        max_loras=1 if enable_lora else None,
    )
    lora_request = (
        LoRARequest(
            lora_name=adapter_path.name,
            lora_int_id=1,
            lora_path=str(adapter_path),
        )
        if enable_lora
        else None
    )

    t0 = time.time()
    all_msgs = messages_a + messages_b
    raw = generate_batch(
        model, all_msgs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        lora_request=lora_request,
    )
    elapsed = time.time() - t0
    n_eval = len(messages_a)
    raw_a = raw[:n_eval]
    raw_b = raw[n_eval:]
    log.info("Judge inference: %.1fs (%.2fs/sample pair)", elapsed, elapsed / n_eval)

    # Parse and score.
    df_eval = df.iloc[keep_idx].reset_index(drop=True).copy()
    kept_n = [expected_ns[i] for i in keep_idx]

    predicted_winners: list[str | None] = []
    margins: list[float | None] = []
    parse_ok: list[bool] = []
    raw_a_keep: list[str] = []
    raw_b_keep: list[str] = []

    for ra, rb, n_q in tqdm(zip(raw_a, raw_b, kept_n), total=n_eval, desc="Judge parse"):
        parsed_a = parse_checkeval_output(ra, expected_n=n_q)
        parsed_b = parse_checkeval_output(rb, expected_n=n_q)
        cmp = compare_checklists_pairwise(parsed_a, parsed_b,
                                          expected_n=n_q,
                                          tie_delta=args.tie_delta)
        raw_a_keep.append(ra)
        raw_b_keep.append(rb)
        if cmp is None:
            predicted_winners.append(None)
            margins.append(None)
            parse_ok.append(False)
        else:
            predicted_winners.append(cmp["winner"])
            margins.append(cmp["margin"])
            parse_ok.append(True)

    df_eval["raw_output_a"] = raw_a_keep
    df_eval["raw_output_b"] = raw_b_keep
    df_eval["expected_n"] = kept_n
    df_eval["margin"] = margins
    df_eval["predicted_winner"] = predicted_winners
    df_eval["checklist_parsed"] = parse_ok

    metrics = compute_metrics(
        y_true=df_eval["winner"].tolist(),
        y_pred=df_eval["predicted_winner"].tolist(),
        domains=df_eval["domain"].tolist(),
    )
    metrics["inference_time_s"] = elapsed
    metrics["n_samples_total"] = len(df)
    metrics["n_samples_evaluable"] = n_eval
    metrics["coverage_rate"] = n_eval / len(df) if len(df) else 0.0
    metrics["parse_rate"] = float(df_eval["checklist_parsed"].mean())
    metrics["tie_delta"] = args.tie_delta
    metrics["judge_adapter"] = str(adapter_path) if adapter_path else "base"
    metrics["base_model"] = args.base_model
    metrics["generated_source"] = str(Path(args.generated))

    split_tag = args.subset or args.eval_split
    adapter_tag = adapter_path.name if adapter_path else "base"
    exp_name = f"pipeline_judge_{adapter_tag}_{split_tag}_{date.today()}"
    save_results(df_eval, metrics, exp_name)
    log.info("Done.")


if __name__ == "__main__":
    main()
