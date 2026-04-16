#!/usr/bin/env python3
"""
Run the trained checklist-generator over a pairwise split and persist one
checklist per sample_id.  Output becomes the input to ``prepare_judge_sft.py``
and to ``run_judge_eval.py``.

Usage:
    python run_generator_infer.py \\
        --adapter-path results/checkpoints/generator_sft_tier_10k_.../final_adapter \\
        --split dev_600
    python run_generator_infer.py \\
        --adapter-path results/checkpoints/generator_sft_tier_10k_.../final_adapter \\
        --subset tier_10k
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))           # src/
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data_process"))  # src/data_process/

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from vllm.lora.request import LoRARequest

import config as cfg
from prepare_generator_sft import (
    DOMAIN_ORDER,
    build_generator_messages,
    format_checklist_target,
)
from utils import generate_batch, load_judge_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


_SECTION_RE = re.compile(r"^\s*#{2,}\s*(\S+)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_VALID_DOMAINS = set(DOMAIN_ORDER)


def parse_generated_checklist(raw: str) -> dict[str, list[str]]:
    """Parse `### <domain>\\n- ...` blocks into {domain: [questions]}."""
    if not raw:
        return {}
    per_domain: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _SECTION_RE.match(stripped)
        if m:
            dom = m.group(1).lower()
            current = dom if dom in _VALID_DOMAINS else None
            if current is not None and current not in per_domain:
                per_domain[current] = []
            continue
        m_b = _BULLET_RE.match(stripped)
        if m_b and current is not None:
            q = m_b.group(1).strip()
            if q:
                per_domain[current].append(q)
    # Dedup within domain.
    for dom, qs in per_domain.items():
        seen: set[str] = set()
        dedup: list[str] = []
        for q in qs:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(q)
        per_domain[dom] = dedup
    return per_domain


def load_eval_data(eval_split: str = "dev", subset: str | None = None) -> pd.DataFrame:
    """Load reasoning-augmented eval data from data/with_reason/.

    Expected filename: ``{split_tag}_reasoning.parquet`` — produced by
    ``src/data_process/prepare_data_reasoning.py``. This file carries the
    ``sample_id`` column required by the generator → judge pipeline.

    The reasoning parquet contains both original and swapped A/B orderings
    (``swap_flag == False`` / ``True``). For checklist generation we only
    want one row per sample, so we filter to ``swap_flag == False``.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to the generator's final_adapter directory. "
                             "Omit to run the base model (pre-FT baseline).")
    parser.add_argument("--base-model", type=str,
                        default=str(cfg.GENERATOR_MODEL_ID))
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--subset", type=str, default=None,
                        help="Training tier (e.g. tier_10k). Overrides --split.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output parquet (default: data/generated_checklists/<split>.parquet)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    args = parser.parse_args()

    if args.adapter_path:
        adapter_path = Path(args.adapter_path).resolve()
        with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        lora_rank = int(adapter_cfg.get("r", 16))
    else:
        adapter_path = None
        lora_rank = 16
        log.info("No adapter provided — running base %s as pre-FT baseline",
                 args.base_model)

    df = load_eval_data(args.split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)

    split_tag = args.subset or args.split
    default_fname = f"{split_tag}.parquet" if adapter_path else f"{split_tag}_base.parquet"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.GENERATED_CHECKLIST_DIR / default_fname
    )

    enable_lora = adapter_path is not None
    log.info("Loading %s (enable_lora=%s, rank=%d)",
             args.base_model, enable_lora, lora_rank)
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

    all_messages = [build_generator_messages(r) for _, r in df.iterrows()]

    t0 = time.time()
    raw_outputs = generate_batch(
        model,
        all_messages,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        lora_request=lora_request,
    )
    elapsed = time.time() - t0
    log.info("Generator inference: %.1fs (%.2fs/sample)", elapsed, elapsed / len(df))

    records: list[dict] = []
    n_empty = 0
    n_nonempty_parsed = 0
    total_questions = 0
    for (_, row), raw in zip(df.iterrows(), raw_outputs):
        per_domain = parse_generated_checklist(raw)
        canonical = format_checklist_target(per_domain)
        n_q = sum(len(v) for v in per_domain.values())
        if n_q == 0:
            n_empty += 1
        else:
            n_nonempty_parsed += 1
            total_questions += n_q
        records.append({
            "sample_id": row["sample_id"],
            "domain": row.get("domain", ""),
            "generated_checklist": canonical,
            "raw_output": raw,
            "n_questions": n_q,
            "n_domains": len(per_domain),
        })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    meta = {
        "adapter_path": str(adapter_path) if adapter_path else "base",
        "base_model": args.base_model,
        "split": split_tag,
        "n_samples": len(df),
        "n_empty": n_empty,
        "empty_rate": n_empty / len(df) if len(df) else 0.0,
        "avg_questions_nonempty": (
            total_questions / n_nonempty_parsed if n_nonempty_parsed else 0.0
        ),
        "inference_time_s": elapsed,
    }
    with (output_path.with_suffix(".meta.json")).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info(
        "Saved %d rows -> %s  (empty: %d / %d = %.1f%%)",
        len(out_df), output_path, n_empty, len(df), 100 * meta["empty_rate"],
    )


if __name__ == "__main__":
    main()
