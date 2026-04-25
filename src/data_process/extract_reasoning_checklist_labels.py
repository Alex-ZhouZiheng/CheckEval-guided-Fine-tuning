#!/usr/bin/env python3
"""
Extract checklist-dimension-tagged questions from HelpSteer3 reasoning text.

For each row in a reasoning-augmented pairwise parquet (produced by
`prepare_data_reasoning.py`), prompt an LLM to convert the free-form
individual-preference `reasoning_text` into a list of Yes/No checklist
questions, each assigned to exactly one checklist dimension:

    - clarity_and_communication
    - helpfulness_and_usefulness
    - correctness_and_completeness
    - coding_communication_conditional
    - relevance_instruction_following

Output columns (one row per extracted question):
    sample_id
    swap_flag
    domain
    gold_label
    question
    source_idx                # index into the original reasoning parquet

Usage:
    # Local vLLM (default judge model)
    python src/extract_reasoning_checklist_labels.py \
        --input data/dev_600_reasoning.parquet

    # OpenAI teacher
    python src/extract_reasoning_checklist_labels.py \
        --input data/dev_600_reasoning.parquet \
        --backend openai --model-id gpt-4o-mini

    # Dry run (just print a few samples)
    python src/extract_reasoning_checklist_labels.py \
        --input data/dev_600_reasoning.parquet --dry-run --limit 5
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

DOMAINS = cfg.DOMAINS
DOMAIN_DESCRIPTIONS = cfg.DOMAIN_DESCRIPTIONS

EXTRACTION_SYSTEM_PROMPT = (
    "You convert free-form evaluation reasoning about two LLM responses into a "
    "structured checklist of Yes/No evaluation questions. Each question must "
    "be phrased so that 'Yes' means the response meets the criterion. Return "
    "ONLY valid JSON, no commentary."
)

EXTRACTION_USER_TEMPLATE = """\
You will be given an annotator's reasoning that explains why one response is \
better than another. Your job is to turn the key evaluation criteria in that \
reasoning into a list of self-contained Yes/No checklist questions that could \
be asked about ANY response to this kind of user request. The questions must \
NOT reference "Response A", "Response B", or the specific content of either \
response; they must be reusable evaluation questions.

Each question must be tagged with exactly one of these checklist dimensions:

{domain_block}

Rules:
- Only produce questions that are actually implied by the reasoning. Do not \
invent criteria that the reasoning does not discuss.
- Phrase every question so that "Yes" is the desirable answer.
- Use the `coding_communication_conditional` dimension ONLY when the reasoning \
explicitly discusses code, syntax, or implementation details.
- If the reasoning is empty or produces no usable criteria, return `[]`.
- Output ONLY a JSON array of objects, each with keys `domain` and `question`.
- Keep each question under ~40 words.

### Reasoning
{reasoning_text}

### Output
JSON array:"""


# ────────────────────────── prompt building ──────────────────


def _domain_block() -> str:
    lines = []
    for name in DOMAINS:
        lines.append(f"- {name}: {DOMAIN_DESCRIPTIONS[name]}")
    return "\n".join(lines)


def build_extraction_prompt(reasoning_text: str) -> list[dict[str, str]]:
    user = EXTRACTION_USER_TEMPLATE.format(
        domain_block=_domain_block(),
        reasoning_text=reasoning_text.strip() or "(empty)",
    )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ────────────────────────── output parsing ───────────────────


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_extraction_output(raw: str) -> list[dict[str, str]]:
    """Parse LLM output into a list of {domain, question} dicts."""
    if not raw:
        return []

    text = raw.strip()
    # Strip Markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    out: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "")).strip().lower()
        question = str(item.get("question", "")).strip()
        if domain not in DOMAINS or not question:
            continue
        # Drop questions that still leak A/B references.
        if re.search(r"\bresponse\s*[ab12]\b", question, re.IGNORECASE):
            continue
        out.append({"domain": domain, "question": question})
    return out


# ────────────────────────── backends ─────────────────────────


def generate_vllm(
    prompts: list[list[dict[str, str]]],
    model_id: str,
    max_new_tokens: int,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    quantization: str | None = None,
    load_format: str | None = None,
) -> list[str]:
    from utils import generate_batch, load_judge_model

    model = load_judge_model(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        quantization=quantization,
        load_format=load_format,
    )
    return generate_batch(model, prompts, max_new_tokens=max_new_tokens)


def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI backend requires `pip install openai`.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in env/.env.")

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def generate_llama_cpp(
    prompts: list[list[dict[str, str]]],
    model_path: str,
    max_new_tokens: int,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_concurrent: int = 1,
) -> list[str]:
    """Generate completions using llama-cpp-python (supports GGUF models).

    Parameters
    ----------
    model_path:
        Absolute path to a .gguf model file.
    n_ctx:
        Context window size (tokens).
    n_gpu_layers:
        Number of layers to offload to GPU; -1 means all layers.
    max_concurrent:
        Llama.cpp is not thread-safe for a single model instance, so this
        defaults to 1. Increase only if you load separate instances per thread.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp backend requires `pip install llama-cpp-python`."
        ) from exc

    log.info("Loading GGUF model via llama.cpp: %s", model_path)
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    log.info("  Model loaded.")

    results: list[str] = [""] * len(prompts)

    def _call(idx: int, messages: list[dict[str, str]]) -> tuple[int, str]:
        try:
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
            text = (resp["choices"][0]["message"]["content"] or "").strip()
            return idx, text
        except Exception as exc:  # noqa: BLE001
            log.warning("llama.cpp call failed (idx=%d): %s", idx, exc)
            return idx, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futs = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(prompts), desc="llama.cpp extract"):
            idx, text = fut.result()
            results[idx] = text

    return results


def generate_openai(
    prompts: list[list[dict[str, str]]],
    model_id: str,
    max_new_tokens: int,
    max_concurrent: int = 4,
) -> list[str]:
    client = _get_openai_client()
    results = [""] * len(prompts)

    def _call(idx: int, messages: list[dict[str, str]]) -> tuple[int, str]:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    log.warning("OpenAI call failed (idx=%d): %s", idx, exc)
                    return idx, ""
                time.sleep(2 ** attempt)
        return idx, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futs = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(prompts), desc="OpenAI extract"):
            idx, text = fut.result()
            results[idx] = text
    return results


# ────────────────────────── main ─────────────────────────────


def load_reasoning_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reasoning parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"sample_id", "reasoning_text", "gold_label", "swap_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")
    return df


def print_summary(df_questions: pd.DataFrame, n_source: int, output_path: Path) -> None:
    table = Table(title="Reasoning → Checklist Extraction Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Source rows", f"{n_source:,}")
    table.add_row("Rows with ≥1 question", f"{df_questions['sample_id'].nunique():,}")
    table.add_row("Total questions", f"{len(df_questions):,}")
    table.add_row("Output", str(output_path))
    console.print(table)

    if not df_questions.empty:
        dom_table = Table(title="Questions per domain")
        dom_table.add_column("Domain", style="bold")
        dom_table.add_column("Count", justify="right")
        for domain, count in df_questions["domain"].value_counts().items():
            dom_table.add_row(domain, f"{int(count):,}")
        console.print(dom_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True,
                        help="Path to *_reasoning.parquet from prepare_data_reasoning.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: <input>_questions.parquet)")
    parser.add_argument("--backend", choices=["vllm", "openai", "llama-cpp"], default="vllm")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model id (default: cfg.JUDGE_MODEL_ID for vllm)")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (for debugging)")
    parser.add_argument("--dedup-within-sample", action="store_true",
                        help="Drop duplicate questions within the same sample")
    parser.add_argument("--only-original", action="store_true",
                        help="Only extract from swap_flag=False rows (A/B order)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print a few extractions without saving")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--quantization", type=str, default=None,
                        help="vLLM quantization backend, e.g. 'bitsandbytes'")
    parser.add_argument("--load-format", type=str, default=None,
                        help="vLLM load format, e.g. 'bitsandbytes' for on-the-fly int4")
    parser.add_argument("--openai-concurrency", type=int, default=4)
    # llama.cpp-specific args
    parser.add_argument("--llama-cpp-model-path", type=str, default=None,
                        help="Path to a .gguf model file (required for --backend llama-cpp)")
    parser.add_argument("--llama-cpp-n-ctx", type=int, default=8192,
                        help="Context window size for llama.cpp (default: 8192)")
    parser.add_argument("--llama-cpp-n-gpu-layers", type=int, default=-1,
                        help="GPU layers to offload; -1 = all (default: -1)")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = load_reasoning_parquet(input_path)

    if args.only_original and "swap_flag" in df.columns:
        df = df[~df["swap_flag"].astype(bool)].reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    # Only rows with non-empty reasoning contribute prompts.
    work = df[df["reasoning_text"].fillna("").str.strip().ne("")].copy()
    log.info("Extracting checklist questions from %d / %d rows with reasoning",
             len(work), len(df))

    prompts = [build_extraction_prompt(r) for r in work["reasoning_text"].tolist()]

    if args.backend == "vllm":
        model_id = args.model_id or str(cfg.JUDGE_MODEL_ID)
        raw_outputs = generate_vllm(
            prompts,
            model_id=model_id,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            quantization=args.quantization,
            load_format=args.load_format,
        )
    elif args.backend == "llama-cpp":
        if not args.llama_cpp_model_path:
            parser.error("--llama-cpp-model-path is required when --backend llama-cpp")
        raw_outputs = generate_llama_cpp(
            prompts,
            model_path=args.llama_cpp_model_path,
            max_new_tokens=args.max_new_tokens,
            n_ctx=args.llama_cpp_n_ctx,
            n_gpu_layers=args.llama_cpp_n_gpu_layers,
        )
    else:
        model_id = args.model_id or "gpt-4o-mini"
        raw_outputs = generate_openai(
            prompts,
            model_id=model_id,
            max_new_tokens=args.max_new_tokens,
            max_concurrent=args.openai_concurrency,
        )

    records: list[dict[str, Any]] = []
    for (src_idx, row), raw in zip(work.iterrows(), raw_outputs):
        questions = parse_extraction_output(raw)
        if args.dedup_within_sample:
            seen: set[str] = set()
            deduped = []
            for q in questions:
                key = q["question"].lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(q)
            questions = deduped

        for q in questions:
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "swap_flag": bool(row.get("swap_flag", False)),
                    "domain": q["domain"],
                    "gold_label": row.get("gold_label", ""),
                    "question": q["question"],
                    "source_idx": int(src_idx),
                }
            )

    df_questions = pd.DataFrame.from_records(
        records,
        columns=["sample_id", "swap_flag", "domain", "gold_label", "question", "source_idx"],
    )

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_questions.parquet"
    )
    print_summary(df_questions, n_source=len(work), output_path=output_path)

    if args.dry_run:
        console.print("\n[bold]Sample extractions:[/bold]")
        for sid, group in df_questions.head(15).groupby("sample_id"):
            console.print(f"\n[cyan]{sid}[/cyan]")
            for _, r in group.iterrows():
                console.print(f"  [{r['domain']}] {r['question']}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_questions.to_parquet(output_path, index=False)
    log.info("Saved %d extracted questions -> %s", len(df_questions), output_path)
    console.print(f"\n[bold green]Done. Saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
