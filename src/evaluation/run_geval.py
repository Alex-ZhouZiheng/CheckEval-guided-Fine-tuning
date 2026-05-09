#!/usr/bin/env python3
"""
Baseline: G-Eval (Liu et al., 2023) on HelpSteer3 pairwise data.

G-Eval uses an LLM with auto-generated chain-of-thought evaluation steps
to score responses on pre-defined criteria (1-5 Likert), then aggregates
scores to determine the winner.

Backends:
  api      OpenAI-compatible HTTP API (DeepSeek, GPT-4). Requires API key.
  vllm     Local vLLM in-process engine. Supports Qwen3.5-4B/9B.
  llamacpp Local llama-server HTTP API.

Reference: https://arxiv.org/abs/2303.16634
"""

from __future__ import annotations
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from tqdm import tqdm

import config as cfg
from utils import (
    compute_metrics,
    generate_batch,
    load_eval_data,
    load_judge_model,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ────────────────────────── G-Eval prompts ──────────────────────

# Auto-CoT evaluation steps per dimension. Generated once by GPT-4 and
# held fixed for reproducibility (core G-Eval methodology).
G_EVAL_STEPS: dict[str, list[str]] = {
    "clarity_and_communication": [
        "Read the conversation history to understand what the user is asking.",
        "Read the response and check if the main answer or information is "
        "immediately identifiable without re-reading.",
        "Check if the logical flow is easy to follow — does each sentence "
        "build naturally on the previous one?",
        "Check for unnecessary jargon, run-on sentences, or wall-of-text "
        "formatting that hurts readability.",
        "If the response includes structure (headings, lists, sections), "
        "verify it aids comprehension rather than being cosmetic.",
        "Assess whether a typical user could extract the answer and act on "
        "it within 30 seconds of reading.",
        "Assign a score 1-5 based on overall clarity and communication quality.",
    ],
    "coding_communication_conditional": [
        "Determine whether the response contains code, commands, configuration, "
        "APIs, or implementation details. If none are present, skip this "
        "dimension and mark it N/A.",
        "Check if the code is syntactically correct for the claimed language "
        "or framework.",
        "Check if the code is runnable as-is — are imports, dependencies, "
        "and setup steps included?",
        "Verify API calls, function signatures, and library usage are valid "
        "(not hallucinated).",
        "Check if the code is adequately explained — does the surrounding "
        "text clarify the logic?",
        "Assess readability: consistent naming, reasonable comments, not "
        "excessively clever.",
        "Assign a score 1-5 based on overall code quality and communication.",
    ],
    "correctness_and_completeness": [
        "Identify the core factual claims or answers in the response.",
        "Cross-check each claim against what a domain expert would verify "
        "as correct (use your knowledge; flag anything that contradicts "
        "well-established facts).",
        "Check if the response addresses ALL parts of the user's question. "
        "A partial answer to a multi-part question is incomplete.",
        "If the user's question contains a false premise or ambiguous term, "
        "check whether the response identifies and handles it appropriately.",
        "For time-sensitive queries, check whether the response acknowledges "
        "temporal context (e.g., current year, version-specific info).",
        "Check for overconfident statements about uncertain topics — does "
        "the response hedge appropriately?",
        "Assign a score 1-5 based on factual correctness and completeness.",
    ],
    "helpfulness_and_usefulness": [
        "Identify what the user actually needs (not just what they asked — "
        "consider the underlying goal).",
        "Check if the response goes beyond minimal correctness by providing "
        "actionable specifics, concrete examples, or tailored guidance.",
        "Assess whether the depth matches the complexity of the question. "
        "A 'yes/no' to a nuanced question is unhelpful.",
        "Check if relevant trade-offs, alternatives, or next steps are "
        "mentioned when appropriate.",
        "Verify the response does not drift into generic advice or "
        "tangential information that wastes the user's time.",
        "Assign a score 1-5 based on overall helpfulness and usefulness.",
    ],
    "relevance_instruction_following": [
        "Extract every explicit instruction from the user's message: "
        "requested format, length, style, structure, constraints.",
        "Check if the response follows each instruction. Flag any "
        "deviation (e.g., asked for bullet points but got paragraphs).",
        "If the user provided source material or context to transform, "
        "check whether the response actually uses that material.",
        "Check for numeric constraints: word count, number of items, "
        "specific quantities requested.",
        "Verify the response stays on-topic and does not introduce "
        "unrequested content that dilutes the answer.",
        "Assign a score 1-5 based on instruction following and relevance.",
    ],
}

G_EVAL_SCORE_PROMPT = """<Task Overview>
You will be given a conversation between a user and an assistant, followed by
a candidate response for the next turn. Your task is to evaluate the quality
of the response on a specific evaluation dimension using a 1-5 rating scale.

<Evaluation Dimension>
{dimension_name}

<Evaluation Steps>
{evaluation_steps}

<Scoring Rubric>
1 = {score1_desc}
2 = {score2_desc}
3 = {score3_desc}
4 = {score4_desc}
5 = {score5_desc}

<Conversation History>
{context}

<Response to Evaluate>
{response}

<Output Format>
First, work through each evaluation step, writing your observations.
Then, on a new line, output your final score as a single integer:
Score: X
where X is 1, 2, 3, 4, or 5. If the dimension is not applicable,
output: Score: N/A
"""

DIMENSION_SCORE_DESCRIPTIONS = {
    "clarity_and_communication": {
        1: "Incomprehensible — cannot extract meaning even with effort.",
        2: "Hard to follow — major clarity issues, needs re-reading multiple times.",
        3: "Adequate — main point is clear but structure or wording is suboptimal.",
        4: "Clear and well-structured — easy to read and understand on first pass.",
        5: "Exceptionally clear — elegant communication, perfectly structured, effortless to parse.",
    },
    "coding_communication_conditional": {
        1: "Code is incorrect, harmful, or completely broken.",
        2: "Code has significant errors or missing critical components to run.",
        3: "Code is mostly correct but has minor issues or poor readability.",
        4: "Code is correct, runnable, and well-explained.",
        5: "Code is exemplary — correct, efficient, well-documented, production-ready.",
    },
    "correctness_and_completeness": {
        1: "Mostly incorrect or entirely misses the question.",
        2: "Several factual errors or addresses only a small part of the question.",
        3: "Mostly correct but has minor inaccuracies or omissions.",
        4: "Factually accurate and addresses all parts of the question.",
        5: "Perfectly accurate, comprehensive, and handles edge cases or implicit needs.",
    },
    "helpfulness_and_usefulness": {
        1: "Not helpful — generic, irrelevant, or wastes the user's time.",
        2: "Marginally helpful — provides minimal value beyond basic correctness.",
        3: "Moderately helpful — some useful specifics but lacks depth or tailoring.",
        4: "Helpful — tailored, actionable, with good depth and practical guidance.",
        5: "Exceptionally helpful — anticipates needs, offers insights beyond what was asked.",
    },
    "relevance_instruction_following": {
        1: "Ignores most instructions; largely irrelevant.",
        2: "Follows few instructions; significant deviations from what was requested.",
        3: "Follows main instructions but misses some details or constraints.",
        4: "Follows all instructions accurately and stays on-topic.",
        5: "Follows every instruction precisely, including implicit constraints, perfectly on-task.",
    },
}

# ────────────────────────── prompt builders ─────────────────────


def build_geval_prompt(context: str, response: str, dimension: str) -> str:
    """Build a G-Eval scoring prompt for one response on one dimension."""
    steps = G_EVAL_STEPS[dimension]
    descs = DIMENSION_SCORE_DESCRIPTIONS[dimension]
    steps_text = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    dim_name = dimension.replace("_", " ").title()
    return G_EVAL_SCORE_PROMPT.format(
        dimension_name=dim_name,
        evaluation_steps=steps_text,
        score1_desc=descs[1],
        score2_desc=descs[2],
        score3_desc=descs[3],
        score4_desc=descs[4],
        score5_desc=descs[5],
        context=context,
        response=response,
    )


def build_all_prompts(
    df: pd.DataFrame, dimensions: list[str]
) -> list[dict[str, Any]]:
    """Build flat list of G-Eval prompts: one per (sample, side, dimension).

    Returns list of dicts with keys: sample_idx, side, dimension, prompt_text.
    """
    prompts: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        for side in ("A", "B"):
            response = row["response_a"] if side == "A" else row["response_b"]
            for dim in dimensions:
                prompts.append({
                    "sample_idx": i,
                    "side": side,
                    "dimension": dim,
                    "prompt_text": build_geval_prompt(row["context"], response, dim),
                })
    return prompts


# ────────────────────────── parsing ─────────────────────────────

_SCORE_RE = re.compile(r"Score:\s*(N/?A|na|\d)", re.IGNORECASE)


def parse_geval_score(raw_output: str) -> int | None:
    """Extract the numeric score (1-5) from G-Eval output. Returns None if N/A."""
    match = _SCORE_RE.search(raw_output)
    if not match:
        return None
    token = match.group(1).upper()
    if token in ("N/A", "NA"):
        return None
    try:
        score = int(token)
        if 1 <= score <= 5:
            return score
    except ValueError:
        pass
    return None


def aggregate_scores(scores: dict[str, int | None]) -> float:
    """Sum valid (non-None) dimension scores. None treated as 0 (N/A)."""
    return sum(v for v in scores.values() if v is not None)


# ────────────────────────── API backend ─────────────────────────


def _get_api_client():
    from openai import OpenAI

    api_key = _os.getenv("OPENAI_API_KEY")
    base_url = _os.getenv("OPENAI_BASE_URL")
    if not api_key:
        env_path = cfg.PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                    elif line.startswith("OPENAI_BASE_URL="):
                        base_url = line.split("=", 1)[1].strip().strip("'\"")

    if not api_key:
        api_key = _os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            env_path = cfg.PROJECT_ROOT / ".env"
            if env_path.exists():
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("DEEPSEEK_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip("'\"")
                            break
        if api_key and not base_url:
            base_url = "https://api.deepseek.com/v1"

    if not api_key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env or environment."
        )

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _run_api_backend(
    df: pd.DataFrame,
    dimensions: list[str],
    model: str,
    concurrency: int,
    temperature: float,
    max_tokens: int,
) -> pd.DataFrame:
    """G-Eval via OpenAI-compatible HTTP API with concurrent requests."""
    client = _get_api_client()
    results = df.copy()
    scores_a: list[dict] = [{} for _ in range(len(df))]
    scores_b: list[dict] = [{} for _ in range(len(df))]
    predicted_winners: list[str] = []
    start_lock = Lock()
    next_allowed_start = 0.0

    def _score_one(idx: int, side: str, dim: str, prompt_text: str) -> tuple:
        nonlocal next_allowed_start
        if concurrency > 1:
            with start_lock:
                now = time.monotonic()
                wait_s = max(0.0, next_allowed_start - now)
                if wait_s > 0:
                    time.sleep(wait_s)
                next_allowed_start = time.monotonic() + 0.5
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as e:
            log.warning("API call failed sample=%d side=%s dim=%s: %s", idx, side, dim, e)
            raw = ""
        return idx, side, dim, parse_geval_score(raw)

    all_prompts = build_all_prompts(df, dimensions)
    tasks = [(p["sample_idx"], p["side"], p["dimension"], p["prompt_text"])
             for p in all_prompts]

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_score_one, *t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="G-Eval API"):
            idx, side, dim, score = future.result()
            target = scores_a if side == "A" else scores_b
            target[idx][dim] = score

    for i in range(len(df)):
        total_a = aggregate_scores(scores_a[i])
        total_b = aggregate_scores(scores_b[i])
        if total_a > total_b:
            predicted_winners.append("A")
        elif total_b > total_a:
            predicted_winners.append("B")
        else:
            predicted_winners.append("Tie")

    results["geval_scores_a"] = [json.dumps(s) for s in scores_a]
    results["geval_scores_b"] = [json.dumps(s) for s in scores_b]
    results["geval_total_a"] = [aggregate_scores(s) for s in scores_a]
    results["geval_total_b"] = [aggregate_scores(s) for s in scores_b]
    results["predicted_winner"] = predicted_winners
    return results


# ──────────────────────── local backend (vLLM / llamacpp) ───────


def _run_local_backend(
    df: pd.DataFrame,
    dimensions: list[str],
    model,
    backend: str,
    batch_size: int,
    enable_thinking: bool,
    max_new_tokens: int,
) -> pd.DataFrame:
    """G-Eval via local vLLM or llamacpp — batched inference."""
    all_prompts = build_all_prompts(df, dimensions)
    prompt_texts = [p["prompt_text"] for p in all_prompts]

    messages_list = [
        [{"role": "user", "content": pt}]
        for pt in prompt_texts
    ]

    chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}
    gen_kwargs = {"max_new_tokens": max_new_tokens}

    log.info("Running batched inference: %d prompts, batch_size=%d", len(messages_list), batch_size)
    t0 = time.time()

    if backend == "vllm":
        raw_outputs = generate_batch(
            model, messages_list, batch_size=batch_size,
            chat_template_kwargs=chat_template_kwargs,
            **gen_kwargs,
        )
    else:
        raw_outputs = generate_batch(
            model, messages_list, batch_size=batch_size,
            chat_template_kwargs=chat_template_kwargs,
            **gen_kwargs,
        )

    elapsed = time.time() - t0
    log.info("Inference done in %.1fs (%.2f s/prompt)", elapsed, elapsed / len(messages_list))

    # Parse and aggregate
    results = df.copy()
    scores_a: list[dict] = [{} for _ in range(len(df))]
    scores_b: list[dict] = [{} for _ in range(len(df))]
    predicted_winners: list[str] = []

    parse_failures = 0
    for p, raw in zip(all_prompts, raw_outputs):
        score = parse_geval_score(raw)
        if score is None and raw.strip():
            parse_failures += 1
        target = scores_a if p["side"] == "A" else scores_b
        target[p["sample_idx"]][p["dimension"]] = score

    if parse_failures:
        log.warning("Parse failures: %d/%d (%.1f%%)",
                    parse_failures, len(raw_outputs),
                    100 * parse_failures / len(raw_outputs))

    for i in range(len(df)):
        total_a = aggregate_scores(scores_a[i])
        total_b = aggregate_scores(scores_b[i])
        if total_a > total_b:
            predicted_winners.append("A")
        elif total_b > total_a:
            predicted_winners.append("B")
        else:
            predicted_winners.append("Tie")

    results["geval_scores_a"] = [json.dumps(s) for s in scores_a]
    results["geval_scores_b"] = [json.dumps(s) for s in scores_b]
    results["geval_total_a"] = [aggregate_scores(s) for s in scores_a]
    results["geval_total_b"] = [aggregate_scores(s) for s in scores_b]
    results["predicted_winner"] = predicted_winners
    return results


# ────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="G-Eval baseline on HelpSteer3 pairwise data"
    )
    parser.add_argument("--split", default="dev", choices=["test", "dev", "dev_600"])
    parser.add_argument("--subset", default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    # Backend selection
    parser.add_argument(
        "--backend", default="api",
        choices=["api", "vllm", "llamacpp"],
        help="Inference backend. 'api' for OpenAI-compatible HTTP, "
             "'vllm' for local vLLM, 'llamacpp' for local llama-server.",
    )

    # API backend args
    parser.add_argument(
        "--model", default=None,
        help="Model name. api backend: e.g. gpt-4o, deepseek-chat. "
             "vllm/llamacpp: path to local model dir. "
             "Default: DEEPSEEK_MODEL env, or gpt-4o-mini (api) / cfg.JUDGE_MODEL_ID (local).",
    )
    parser.add_argument("--concurrency", type=int, default=4,
                        help="API concurrency (api backend only).")

    # Local backend args
    parser.add_argument("--model-id", type=str, default=None,
                        help="Path to local model (vllm/llamacpp). Overrides --model for local backends.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="vLLM batch size for chat().")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 thinking traces (vllm/llamacpp).")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    parser.add_argument("--max-num-seqs", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_num_seqs"])

    # Generation
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per prompt (api backend).")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens per prompt (vllm/llamacpp backend).")

    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()

    dimensions = cfg.CHECKLIST_DIMENSIONS
    df = load_eval_data(args.split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples)
        log.info("Capped to %d samples", len(df))

    # Resolve model name per backend
    if args.backend == "api":
        model_name = args.model or _os.getenv("DEEPSEEK_MODEL") or "gpt-4o-mini"
        suffix = args.suffix or model_name.replace("/", "_").replace(":", "_")
        log.info("G-Eval backend=api model=%s samples=%d dims=%d concurrency=%d",
                 model_name, len(df), len(dimensions), args.concurrency)

        t0 = time.time()
        results = _run_api_backend(
            df, dimensions,
            model=model_name,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        elapsed = time.time() - t0
    else:
        model_path = args.model_id or args.model or str(cfg.JUDGE_MODEL_ID)
        model_name = Path(model_path).name
        suffix = args.suffix or model_name
        log.info("G-Eval backend=%s model=%s samples=%d dims=%d batch_size=%d",
                 args.backend, model_path, len(df), len(dimensions), args.batch_size)

        model = load_judge_model(
            model_id=model_path,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
        )

        t0 = time.time()
        results = _run_local_backend(
            df, dimensions, model,
            backend=args.backend,
            batch_size=args.batch_size,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0

    log.info("G-Eval complete in %.1fs (%.2fs/sample)", elapsed, elapsed / len(df))

    metrics = compute_metrics(
        y_true=results["winner"].tolist(),
        y_pred=results["predicted_winner"].tolist(),
        domains=results["domain"].tolist(),
    )
    metrics["inference_time_s"] = elapsed
    metrics["samples_per_second"] = len(df) / elapsed
    metrics["model"] = model_name
    metrics["method"] = "geval"
    metrics["backend"] = args.backend

    split_tag = args.subset or args.split
    experiment_name = f"geval_{split_tag}_{suffix}"
    save_results(results, metrics, experiment_name)

    acc = metrics.get("accuracy", 0)
    valid = metrics.get("n_valid", len(df))
    tie_count = (results["predicted_winner"] == "Tie").sum()
    log.info("G-Eval accuracy (excl Tie): %.4f  (n_valid=%d, ties=%d, total=%d)",
             acc, valid, tie_count, len(df))
    log.info("Done.")


if __name__ == "__main__":
    main()
