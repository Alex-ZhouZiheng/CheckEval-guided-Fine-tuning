#!/usr/bin/env python3
"""Generate per-question importance weights for selected qids.

For each eval sample, takes the top-k qids from a selector picks parquet,
asks a base judge model (e.g. Qwen3.5-9B) to assign an importance score
0..5 to each question conditional on the *context only* (no responses).
Writes raw + softmax-normalized weights per sample.

Output schema (parquet):
    sample_id              str
    prompt_id              str
    domain                 str
    selected_qids          list[int]   # global qids, top-k of picks
    importance_raw         list[float] # 0..5 (filler 1.0 for missing)
    importance_softmax     list[float] # softmax of importance_raw (sum≈1)
    raw_output             str
    parse_ok               bool
    n_missing_qids         int

Usage:
    python src/evaluation/build_question_importance_weights.py \
        --bank checklists/v4_frozen \
        --split test \
        --selector-picks results/.../picks.parquet \
        --k 15 \
        --base-model models/Qwen3.5-9B \
        --out results/.../weights.parquet
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

import config as cfg
from evaluation.selector_infer import load_eval_pairs
from utils import generate_batch, load_judge_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


IMPORTANCE_SYSTEM = (
    "You assign importance weights to evaluation questions for a given user "
    "context. You do NOT see candidate responses. Score each question by how "
    "decisive it is for judging response quality on this specific context."
)

IMPORTANCE_PROMPT = """You are assigning importance weights to evaluation questions for this user context.

[Context]
{context}

[Candidate Questions]
{questions_block}

For each question above, give an integer importance from 0 to 5:
- 5 = critical for this context (a wrong answer should sink the response)
- 3 = relevant
- 1 = marginally useful
- 0 = irrelevant or trivially satisfied for this context

Return strict JSON, no prose, no markdown fence:
{{
  "weights": [
    {{"qid": <global qid>, "importance": <int 0-5>}},
    ...
  ]
}}
Include exactly one entry per question above. Use the same qid numbers shown."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_bank(bank_dir: Path) -> dict[int, dict[str, str]]:
    path = bank_dir / "bank_index.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path).sort_values("qid", kind="stable").reset_index(drop=True)
    needed = {"qid", "dimension", "question_text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"bank_index missing columns: {sorted(missing)}")
    return {
        int(r["qid"]): {
            "dimension": str(r["dimension"]),
            "question_text": str(r["question_text"]),
        }
        for _, r in df.iterrows()
    }


def _parse_qid_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = ast.literal_eval(text)
        return _parse_qid_list(parsed)
    raise TypeError(f"Unsupported qid list value: {type(value).__name__}")


def _load_picks(path: Path, df: pd.DataFrame, k: int) -> dict[str, list[int]]:
    picks = pd.read_parquet(path)
    if "sample_id" not in picks.columns:
        raise ValueError(f"{path} missing sample_id")
    qid_col = "ranked_qids" if "ranked_qids" in picks.columns else "selected_qids"
    if qid_col not in picks.columns:
        raise ValueError(f"{path} must contain ranked_qids or selected_qids")
    pick_map = {
        str(row["sample_id"]): _parse_qid_list(row[qid_col])[:k]
        for _, row in picks.drop_duplicates(subset=["sample_id"], keep="first").iterrows()
    }
    missing = [str(sid) for sid in df["sample_id"].astype(str).tolist() if str(sid) not in pick_map]
    if missing:
        raise ValueError(
            f"{path} missing picks for {len(missing)} samples (first 5: {missing[:5]})"
        )
    log.info("Loaded picks from %s using %s, k=%d", path, qid_col, k)
    return pick_map


def _build_prompt(context: str, qids: list[int], qmeta: dict[int, dict[str, str]]) -> str:
    questions_block = "\n".join(
        f"Q{q}: {qmeta[q]['question_text']}" for q in qids
    )
    return IMPORTANCE_PROMPT.format(context=context, questions_block=questions_block)


def _parse_weights(raw: str, qids: list[int]) -> tuple[dict[int, float], bool]:
    """Return (qid → importance, parse_ok). Missing qids absent from dict."""
    text = raw.strip()
    if not text:
        return {}, False
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}, False
    weights = obj.get("weights")
    if not isinstance(weights, list):
        return {}, False
    qid_set = set(qids)
    out: dict[int, float] = {}
    for entry in weights:
        if not isinstance(entry, dict):
            continue
        try:
            qid = int(entry.get("qid"))
            imp = float(entry.get("importance"))
        except (TypeError, ValueError):
            continue
        if qid in qid_set:
            out[qid] = max(0.0, min(5.0, imp))
    return out, len(out) > 0


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(values)] * len(values)
    return [e / s for e in exps]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--selector-picks", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--base-model", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["llamacpp", "vllm"])
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    parser.add_argument("--max-num-seqs", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-chunk-size", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-fill", type=float, default=1.0,
                        help="Importance assigned to qids the model did not return.")
    args = parser.parse_args()

    bank_dir = args.bank.resolve()
    qmeta = _load_bank(bank_dir)

    df = load_eval_pairs(
        split=args.split,
        subset=args.subset,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    df_by_sid = {str(r["sample_id"]): r for _, r in df.iterrows()}

    picks_map = _load_picks(args.selector_picks.resolve(), df, k=args.k)

    metas: list[dict[str, Any]] = []
    messages_list: list[list[dict[str, str]]] = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        qids = picks_map[sid]
        if not qids:
            metas.append({"sid": sid, "qids": []})
            messages_list.append([
                {"role": "system", "content": IMPORTANCE_SYSTEM},
                {"role": "user", "content": ""},
            ])
            continue
        prompt = _build_prompt(str(row["context"]), qids, qmeta)
        messages_list.append([
            {"role": "system", "content": IMPORTANCE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        metas.append({"sid": sid, "qids": list(qids)})

    log.info("Loading judge model: %s (backend=%s)", args.base_model, args.backend)
    model = load_judge_model(
        model_id=args.base_model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )

    raw_all: list[str] = []
    for start in range(0, len(messages_list), args.prompt_chunk_size):
        chunk = messages_list[start : start + args.prompt_chunk_size]
        raw_all.extend(generate_batch(
            model, chunk,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        ))

    rows: list[dict[str, Any]] = []
    n_parse_ok = 0
    for meta, raw in zip(metas, raw_all):
        sid = meta["sid"]
        qids = meta["qids"]
        row = df_by_sid[sid]

        if not qids:
            rows.append({
                "sample_id": sid,
                "prompt_id": row.get("prompt_id"),
                "domain": row.get("domain"),
                "selected_qids": [],
                "importance_raw": [],
                "importance_softmax": [],
                "raw_output": raw,
                "parse_ok": False,
                "n_missing_qids": 0,
            })
            continue

        parsed, ok = _parse_weights(raw, qids)
        if ok:
            n_parse_ok += 1
        n_missing = sum(1 for q in qids if q not in parsed)
        raw_vals = [float(parsed.get(q, args.missing_fill)) for q in qids]
        sm = _softmax(raw_vals)

        rows.append({
            "sample_id": sid,
            "prompt_id": row.get("prompt_id"),
            "domain": row.get("domain"),
            "selected_qids": qids,
            "importance_raw": raw_vals,
            "importance_softmax": sm,
            "raw_output": raw,
            "parse_ok": ok,
            "n_missing_qids": int(n_missing),
        })

    out_df = pd.DataFrame(rows)
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    parse_rate = n_parse_ok / len(rows) if rows else 0.0
    avg_missing = float(out_df["n_missing_qids"].mean()) if len(out_df) else 0.0
    log.info("Saved %d weight rows -> %s", len(out_df), out_path)
    log.info("parse_ok_rate=%.4f  avg_missing_qids=%.3f", parse_rate, avg_missing)


if __name__ == "__main__":
    main()
