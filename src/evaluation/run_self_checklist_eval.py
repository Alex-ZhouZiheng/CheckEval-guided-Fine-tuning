#!/usr/bin/env python3
"""
Evaluate the self-checklist CoT judge model (base or LoRA adapter) on a pairwise
split. The model is expected to produce a structured ``<think>`` trace with
checklist + verdicts, followed by a ``### Final`` block with ``Winner: A/B/Tie``.

Only the ``Winner:`` line is consumed for the core pairwise metrics; the think
block is parsed for diagnostics (checklist length, item-tie rate).

Usage:
    # Base model (zero-shot self-checklist baseline)
    python src/evaluation/run_self_checklist_eval.py --eval-split dev

    # Fine-tuned adapter
    python src/evaluation/run_self_checklist_eval.py \\
        --judge-adapter results/checkpoints/selfcheck_.../final_adapter \\
        --eval-split dev --subset dev_600

    # Remote HTTP server (e.g. standalone vLLM instance)
    JUDGE_MODE=http JUDGE_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=qwen3.5-9b \\
        python src/evaluation/run_self_checklist_eval.py --eval-split dev_600
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config as cfg
from data_process.prepare_self_checklist_sft import parse_self_checklist_trace
from utils import (
    compute_metrics,
    generate_batch,
    load_eval_data,
    load_judge_model,
    make_lora_handle,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

# ── Eval prompt (no gold winner line) ───────────────────────────────────

SELF_CHECKLIST_EVAL_PROMPT = """\
<Task Overview>
You will evaluate two candidate responses to a user request. Your task is to:
1. Generate a checklist of specific quality criteria for comparing these two responses.
2. For each criterion, decide which response is better (A, B, or Tie).
3. Based on your evaluation, output the final winner.

<Instructions>
1. Read the conversation history and both responses carefully.
2. Generate 8-20 specific, targeted comparison questions about these two specific responses.
   - Questions should compare the responses on different quality dimensions.
   - Each question should be answerable with A, B, or Tie.
3. For each question, compare the two responses and answer A, B, or Tie.
4. Based on your checklist evaluation, decide the final winner.
5. Output in the required format.

<Answer Format>
<think>
### Checklist
Q1: [your comparison question here]
Q2: [your comparison question here]
...

### Item Verdicts
Q1: A
Q2: Tie
...
</think>

### Final
Winner: A

# Conversation History #
{context}

# Response A #
{response_a}

# Response B #
{response_b}

# Your Evaluation #
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def parse_winner(raw: str) -> str | None:
    """Extract ``Winner: A/B/Tie`` from the model output.

    Returns ``"A"``, ``"B"``, ``"Tie"``, or ``None`` if no match.
    Normalises ``TIE`` to ``Tie``.
    """
    if not isinstance(raw, str):
        return None
    m = re.search(r'Winner:\s*(A|B|Tie)\s*$', raw, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    winner = m.group(1).strip()
    if winner.upper() == "TIE":
        return "Tie"
    return winner.upper()


def _trace_diagnostics(raw_outputs: list[str]) -> tuple[
    list[int | None], list[float | None],
]:
    """Parse checklist traces for diagnostics: n_questions and item tie rate."""
    n_questions: list[int | None] = []
    item_tie_rates: list[float | None] = []
    for raw in raw_outputs:
        parsed = parse_self_checklist_trace(raw)
        nq = parsed.get("n_questions", 0)
        nv = parsed.get("n_verdicts", 0)
        if nq > 0:
            n_questions.append(nq)
        else:
            n_questions.append(None)

        if nv > 0:
            n_tie = sum(1 for v in parsed["verdicts"].values() if v == "Tie")
            item_tie_rates.append(n_tie / nv)
        else:
            item_tie_rates.append(None)
    return n_questions, item_tie_rates


def build_eval_prompts(df: pd.DataFrame) -> tuple[list[list[dict]], list[int]]:
    """Build self-checklist eval prompts for each row.

    Returns:
        messages_list: one ``[{"role": "user", "content": prompt}]`` per sample.
        keep_idx: row indices of df that were kept (swap_flag=False rows).
    """
    messages_list: list[list[dict]] = []
    keep_idx: list[int] = []
    n_swapped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if "swap_flag" in df.columns and row.get("swap_flag") is True:
            n_swapped += 1
            continue
        prompt = SELF_CHECKLIST_EVAL_PROMPT.format(
            context=row["context"],
            response_a=row["response_a"],
            response_b=row["response_b"],
        )
        messages_list.append([{"role": "user", "content": prompt}])
        keep_idx.append(i)
    if n_swapped:
        log.info("Skipped %d swap_flag rows", n_swapped)
    return messages_list, keep_idx


# ── HTTP judge (remote server) ──────────────────────────────────────────

def _http_judge_generate(
    messages_list: list[list[dict]],
    url: str,
    model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    concurrency: int = 32,
) -> list[str]:
    """Fan out chat.completions calls to a running OpenAI-compatible server."""
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key=api_key)

    def one(msgs: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(
            tqdm(ex.map(one, messages_list), total=len(messages_list), desc="Judge HTTP")
        )


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-adapter", type=str, default=None,
                        help="Path to judge final_adapter directory. "
                             "Omit to run the base model (pre-FT baseline).")
    parser.add_argument("--base-model", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--eval-split", type=str, default="dev")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tie-delta", type=float, default=0.0)
    parser.add_argument("--backend", type=str, default=None,
                        choices=["llamacpp", "vllm"],
                        help="Inference backend; defaults to cfg.INFERENCE_BACKEND.")
    parser.add_argument("--http-concurrency", type=int, default=32,
                        help="Concurrent HTTP requests for judge_mode=http (default: 32).")
    parser.add_argument("--experiment-suffix", type=str, default=str(date.today()))
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])
    args = parser.parse_args()

    # ── adapter setup ──
    if args.judge_adapter:
        adapter_path = Path(args.judge_adapter).resolve()
        with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
            lora_rank = int(json.load(f).get("r", 16))
    else:
        adapter_path = None
        lora_rank = 16
        log.info("No judge adapter -- running base %s as pre-FT baseline",
                 args.base_model)

    # ── load data ──
    df = load_eval_data(args.eval_split, args.subset)
    if args.max_samples:
        df = df.head(args.max_samples).reset_index(drop=True)

    messages_list, keep_idx = build_eval_prompts(df)
    log.info("Evaluable samples: %d / %d", len(keep_idx), len(df))

    if not keep_idx:
        raise SystemExit("No evaluable samples found.")

    # ── inference ──
    judge_mode = _os.environ.get("JUDGE_MODE", "").lower()

    if judge_mode == "http":
        url = _os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1")
        judge_model_name = _os.environ.get("JUDGE_MODEL")
        if not judge_model_name:
            raise SystemExit("JUDGE_MODE=http requires JUDGE_MODEL env var.")
        api_key = _os.environ.get("JUDGE_API_KEY", "EMPTY")
        log.info("HTTP judge -> %s model=%s  (skipping local model load)",
                 url, judge_model_name)
        if adapter_path is not None:
            log.warning("--judge-adapter is ignored in HTTP mode; the server's "
                        "registered adapter/model is used instead.")
        t0 = time.time()
        raw_outputs = _http_judge_generate(
            messages_list, url, judge_model_name, api_key,
            max_new_tokens=args.max_new_tokens,
            concurrency=args.http_concurrency,
        )
        elapsed = time.time() - t0
    else:
        enable_lora = adapter_path is not None
        model = load_judge_model(
            model_id=args.base_model,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=enable_lora,
            max_lora_rank=max(lora_rank, 16) if enable_lora else None,
            max_loras=1 if enable_lora else None,
            llamacpp_adapter_path=str(adapter_path) if enable_lora else None,
        )
        lora_request = make_lora_handle(
            adapter_path=str(adapter_path) if enable_lora else None,
            backend=args.backend,
            name=adapter_path.name if enable_lora else "adapter",
            lora_int_id=1,
        )
        t0 = time.time()
        raw_outputs = generate_batch(
            model, messages_list,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            lora_request=lora_request,
        )
        elapsed = time.time() - t0

    n_eval = len(messages_list)
    log.info("Inference: %.1fs (%.2fs/sample)", elapsed, elapsed / max(n_eval, 1))

    # ── parse & diagnostics ──
    predicted_winners: list[str | None] = []
    parse_ok_list: list[bool] = []
    for raw in raw_outputs:
        w = parse_winner(raw)
        predicted_winners.append(w)
        parse_ok_list.append(w is not None)

    n_questions, item_tie_rates = _trace_diagnostics(raw_outputs)

    # ── assemble results dataframe ──
    df_eval = df.iloc[keep_idx].reset_index(drop=True).copy()
    df_eval["raw_output"] = raw_outputs
    df_eval["predicted_winner"] = predicted_winners
    df_eval["parse_ok"] = parse_ok_list
    df_eval["n_checklist_questions"] = n_questions
    df_eval["item_tie_rate"] = item_tie_rates

    # ── metrics ──
    metrics = compute_metrics(
        y_true=df_eval["winner"].tolist(),
        y_pred=df_eval["predicted_winner"].tolist(),
        domains=df_eval["domain"].tolist(),
    )
    metrics["inference_time_s"] = elapsed
    metrics["n_samples_total"] = len(df)
    metrics["n_samples_evaluable"] = n_eval
    metrics["parse_rate"] = float(df_eval["parse_ok"].mean())

    # Tie rate: fraction of parsed outputs predicted as Tie
    n_tie = sum(1 for w in predicted_winners if w == "Tie")
    metrics["tie_rate"] = n_tie / n_eval if n_eval else 0.0
    metrics["tie_delta"] = args.tie_delta

    # Trace diagnostics
    valid_nq = [v for v in n_questions if v is not None]
    valid_tie = [v for v in item_tie_rates if v is not None]
    metrics["avg_checklist_length"] = sum(valid_nq) / len(valid_nq) if valid_nq else None
    metrics["item_tie_rate"] = sum(valid_tie) / len(valid_tie) if valid_tie else None
    metrics["trace_parse_rate"] = len(valid_nq) / n_eval if n_eval else 0.0

    if judge_mode == "http":
        metrics["judge_adapter"] = f"http:{_os.environ.get('JUDGE_MODEL')}"
        metrics["judge_mode"] = "http"
    else:
        metrics["judge_adapter"] = str(adapter_path) if adapter_path else "base"
        metrics["judge_mode"] = "local"
    metrics["base_model"] = args.base_model

    # ── save ──
    split_tag = args.subset or args.eval_split
    if judge_mode == "http":
        adapter_tag = f"httpjudge_{_os.environ.get('JUDGE_MODEL', 'unknown')}"
    else:
        adapter_tag = adapter_path.name if adapter_path else "base"
    exp_name = f"selfchecklist_{adapter_tag}_{split_tag}_{args.experiment_suffix}"
    save_results(df_eval, metrics, exp_name)

    # ── summary table ──
    table = Table(title="Self-Checklist Eval Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    table.add_row("Macro-F1", f"{metrics.get('macro_f1', 0):.4f}")
    table.add_row("Parse rate", f"{metrics['parse_rate']:.4f}")
    table.add_row("Tie rate", f"{metrics['tie_rate']:.4f}")
    table.add_row("Avg checklist length",
                  f"{metrics['avg_checklist_length']:.1f}"
                  if metrics["avg_checklist_length"] is not None else "N/A")
    table.add_row("Item tie rate",
                  f"{metrics['item_tie_rate']:.4f}"
                  if metrics["item_tie_rate"] is not None else "N/A")
    table.add_row("Trace parse rate", f"{metrics['trace_parse_rate']:.4f}")
    table.add_row("N total", str(metrics["n_samples_total"]))
    table.add_row("N evaluable", str(metrics["n_samples_evaluable"]))
    table.add_row("Inference time", f"{elapsed:.1f}s")

    console.print(table)

    # Per-domain breakdown
    if "per_domain" in metrics:
        dt = Table(title="Per-Domain Accuracy")
        dt.add_column("Domain", style="bold")
        dt.add_column("Accuracy", justify="right")
        dt.add_column("F1", justify="right")
        dt.add_column("N", justify="right")
        for domain_name, dm in metrics["per_domain"].items():
            dt.add_row(domain_name, f"{dm['accuracy']:.4f}", f"{dm['macro_f1']:.4f}", str(dm["n"]))
        console.print(dt)

    # Winner distribution
    winner_counts = Counter(w for w in predicted_winners if w is not None)
    log.info("Predicted winner distribution: %s",
             {k: winner_counts.get(k, 0) for k in ["A", "B", "Tie"]})

    log.info("Done.")


if __name__ == "__main__":
    main()
