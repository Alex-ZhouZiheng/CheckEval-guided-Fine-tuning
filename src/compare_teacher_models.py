#!/usr/bin/env python3
"""
Compare teacher models (ChatGPT vs Qwen 3.5 9B) on checklist evaluation.

Runs both models on the same subset of pairwise data, parses checklist
outputs, and compares:
  - Parse rate (valid checklist format)
  - N/A rate
  - Accuracy vs ground-truth preference labels
  - Score distribution (chosen vs rejected)
  - Pairwise agreement between models

Usage:
    # Compare on dev_600 (default)
    python compare_teacher_models.py --split dev_600

    # Compare on a small subset for quick test
    python compare_teacher_models.py --split dev_600 --max-samples 50

    # Only run ChatGPT (if Qwen results already exist)
    python compare_teacher_models.py --only chatgpt --max-samples 100

    # Specify ChatGPT model
    python compare_teacher_models.py --chatgpt-model gpt-4o-mini

    # Compare multiple ChatGPT teacher models in one run
    python compare_teacher_models.py --only chatgpt \
        --chatgpt-models gpt-4o-mini gpt-4.1-mini gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config as cfg
from utils import (
    build_checkeval_prompt,
    compare_checklists_pairwise,
    expected_question_count,
    load_checklists,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

RESULTS_DIR = cfg.RESULTS_DIR / "teacher_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify_model_name(model_name: str) -> str:
    """Make a model name safe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


# ────────────────────────── ChatGPT inference ──────────────────


def _get_openai_client(timeout: float = 90.0):
    """Get OpenAI client, reading key from env or .env."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = cfg.PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                        break
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in environment or .env file."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
        log.info("Using custom OpenAI base URL: %s", base_url)

    return OpenAI(**kwargs)


def run_chatgpt_checklist(
    prompts: list[str],
    model: str = "gpt-4o-mini",
    max_concurrent: int = 2,
    temperature: float = 0.0,
    min_start_interval: float = 2.5,
    timeout: float = 90.0,
    max_retries: int = 5,
) -> list[str]:
    """Run checklist evaluation prompts through ChatGPT API.

    Uses synchronous API with threading for parallelism, plus a global
    request-start gate so concurrency can be >1 without bursting too hard
    into rate limits.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = _get_openai_client(timeout=timeout)

    results = [""] * len(prompts)
    errors = 0

    start_lock = Lock()
    next_allowed_start = 0.0

    def _retry_after_seconds(exc: Exception) -> float | None:
        text = str(exc)
        match = re.search(r"try again in ([0-9.]+)s", text, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0
        return None

    def _acquire_start_slot() -> None:
        nonlocal next_allowed_start
        if min_start_interval <= 0:
            return
        with start_lock:
            now = time.monotonic()
            wait_s = max(0.0, next_allowed_start - now)
            if wait_s > 0:
                time.sleep(wait_s)
            next_allowed_start = time.monotonic() + min_start_interval

    def _call(idx: int, prompt: str) -> tuple[int, str]:
        for attempt in range(max_retries):
            try:
                _acquire_start_slot()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return idx, response.choices[0].message.content.strip()
            except Exception as e:
                text = str(e)

                # 400 类请求错误通常不是瞬时错误，直接失败
                if (
                    "invalid_request_error" in text
                    or "Unsupported parameter" in text
                    or "Error code: 400" in text
                ):
                    log.warning("Non-retriable request error (idx=%d): %s", idx, e)
                    return idx, ""

                if attempt == max_retries - 1:
                    log.warning("ChatGPT call failed (idx=%d): %s", idx, e)
                    return idx, ""

                wait_s = _retry_after_seconds(e)
                if wait_s is None:
                    wait_s = min(45.0, (2 ** attempt) + random.uniform(0.0, 1.0))

                log.info("Transient error on idx=%d, sleeping %.1fs", idx, wait_s)
                time.sleep(wait_s)

        return idx, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        try:
            for future in tqdm(as_completed(futures), total=len(prompts), desc="ChatGPT"):
                idx, text = future.result()
                results[idx] = text
                if not text:
                    errors += 1
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt received, cancelling pending futures...")
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    if errors:
        log.warning("ChatGPT: %d/%d calls failed", errors, len(prompts))
    return results


# ────────────────────────── Qwen inference ─────────────────────


def run_qwen_checklist(
    prompts: list[str],
    model_id: str = str(cfg.JUDGE_MODEL_ID),
    batch_size: int = 16,
) -> list[str]:
    """Run checklist evaluation prompts through Qwen via vLLM."""
    from utils import generate_batch, load_judge_model

    log.info("Loading Qwen model: %s", model_id)
    llm = load_judge_model(model_id)

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    outputs = generate_batch(
        llm, messages_list, batch_size=batch_size, max_new_tokens=2048,
    )

    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return outputs


# ────────────────────────── evaluation ─────────────────────────


def evaluate_teacher(
    raw_outputs_a: list[str],
    raw_outputs_b: list[str],
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    label: str,
) -> dict:
    """Evaluate teacher model outputs against ground truth."""
    n = len(df)
    assert len(raw_outputs_a) == n and len(raw_outputs_b) == n

    predicted_winners = []
    parse_ok = 0
    total_na_a, total_na_b = 0, 0
    chosen_scores, rejected_scores = [], []
    all_parsed_a, all_parsed_b = [], []

    for i, (_, row) in enumerate(df.iterrows()):
        domain = row["domain"]
        winner = row["winner"]
        n_q = expected_question_count(domain, checklists)

        parsed_a = parse_checkeval_output(raw_outputs_a[i], expected_n=n_q)
        parsed_b = parse_checkeval_output(raw_outputs_b[i], expected_n=n_q)
        all_parsed_a.append(parsed_a)
        all_parsed_b.append(parsed_b)

        total_na_a += parsed_a.get("n_na", 0)
        total_na_b += parsed_b.get("n_na", 0)

        pw = compare_checklists_pairwise(parsed_a, parsed_b, n_q, tie_delta=0.05)
        if pw is not None:
            predicted_winners.append(pw["winner"])
            parse_ok += 1

            # Track chosen/rejected scores
            score_a = parsed_a.get("score", 0)
            score_b = parsed_b.get("score", 0)
            if winner == "A":
                chosen_scores.append(score_a)
                rejected_scores.append(score_b)
            elif winner == "B":
                chosen_scores.append(score_b)
                rejected_scores.append(score_a)
        else:
            predicted_winners.append(None)

    # Accuracy (excluding unparseable and ties)
    valid_mask = [
        i for i, (pw, gt) in enumerate(zip(predicted_winners, df["winner"]))
        if pw is not None and pw != "Tie"
    ]
    gt_valid = [df.iloc[i]["winner"] for i in valid_mask]
    pred_valid = [predicted_winners[i] for i in valid_mask]
    accuracy = (
        sum(p == g for p, g in zip(pred_valid, gt_valid)) / len(gt_valid)
        if gt_valid else 0
    )

    # Full accuracy (including ties as wrong)
    full_valid_mask = [i for i, pw in enumerate(predicted_winners) if pw is not None]
    gt_full = [df.iloc[i]["winner"] for i in full_valid_mask]
    pred_full = [predicted_winners[i] for i in full_valid_mask]
    accuracy_full = (
        sum(p == g for p, g in zip(pred_full, gt_full)) / len(gt_full)
        if gt_full else 0
    )

    n_ties = sum(1 for pw in predicted_winners if pw == "Tie")
    n_unparseable = sum(1 for pw in predicted_winners if pw is None)

    # Parse rates per response
    parse_a_ok = sum(1 for p in all_parsed_a if not p.get("_raw_fallback", False))
    parse_b_ok = sum(1 for p in all_parsed_b if not p.get("_raw_fallback", False))

    # Per-domain accuracy
    per_domain = {}
    for domain in sorted(df["domain"].unique()):
        domain_idx = [i for i in valid_mask if df.iloc[i]["domain"] == domain]
        if domain_idx:
            dgt = [df.iloc[i]["winner"] for i in domain_idx]
            dpred = [predicted_winners[i] for i in domain_idx]
            per_domain[domain] = {
                "n": len(domain_idx),
                "accuracy": sum(p == g for p, g in zip(dpred, dgt)) / len(dgt),
            }

    return {
        "label": label,
        "n_total": n,
        "parse_rate_a": parse_a_ok / n,
        "parse_rate_b": parse_b_ok / n,
        "pairwise_parse_rate": parse_ok / n,
        "n_ties": n_ties,
        "n_unparseable": n_unparseable,
        "accuracy_excl_ties": accuracy,
        "accuracy_incl_ties": accuracy_full,
        "n_evaluated": len(gt_valid),
        "avg_na_a": total_na_a / n,
        "avg_na_b": total_na_b / n,
        "avg_chosen_score": float(np.mean(chosen_scores)) if chosen_scores else None,
        "avg_rejected_score": float(np.mean(rejected_scores)) if rejected_scores else None,
        "score_gap": (
            float(np.mean(chosen_scores) - np.mean(rejected_scores))
            if chosen_scores else None
        ),
        "per_domain": per_domain,
    }


# ────────────────────────── display ────────────────────────────


def print_comparison(results: list[dict]) -> None:
    table = Table(title="Teacher Model Comparison — Checklist Evaluation")
    table.add_column("Metric", style="bold")
    for r in results:
        table.add_column(r["label"], justify="right")

    metrics = [
        ("Samples", "n_total"),
        ("Parse rate (A)", "parse_rate_a"),
        ("Parse rate (B)", "parse_rate_b"),
        ("Pairwise parse rate", "pairwise_parse_rate"),
        ("Ties", "n_ties"),
        ("Unparseable", "n_unparseable"),
        ("Accuracy (excl ties)", "accuracy_excl_ties"),
        ("Accuracy (incl ties)", "accuracy_incl_ties"),
        ("# Evaluated", "n_evaluated"),
        ("Avg N/A (A)", "avg_na_a"),
        ("Avg N/A (B)", "avg_na_b"),
        ("Avg chosen score", "avg_chosen_score"),
        ("Avg rejected score", "avg_rejected_score"),
        ("Score gap (chosen-rej)", "score_gap"),
        ("Runtime (s)", "time_s"),
    ]

    for name, key in metrics:
        vals = []
        for r in results:
            v = r.get(key)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        table.add_row(name, *vals)

    # Per-domain accuracy
    all_domains = set()
    for r in results:
        all_domains.update(r.get("per_domain", {}).keys())
    for domain in sorted(all_domains):
        vals = []
        for r in results:
            d = r.get("per_domain", {}).get(domain)
            if d:
                vals.append(f"{d['accuracy']:.4f} (n={d['n']})")
            else:
                vals.append("—")
        table.add_row(f"  {domain}", *vals)

    console.print(table)




def main():
    parser = argparse.ArgumentParser(
        description="Compare ChatGPT vs Qwen as teacher models"
    )
    parser.add_argument("--split", type=str, default="dev_600",
                        help="Eval split (default: dev_600)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples for quick testing")
    parser.add_argument("--only", type=str, default=None,
                        choices=["chatgpt", "qwen"],
                        help="Run only one model")
    parser.add_argument("--chatgpt-model", type=str, default="gpt-4o-mini",
                        help="ChatGPT model name (default: gpt-4o-mini)")
    parser.add_argument("--chatgpt-models", type=str, nargs="+", default=None,
                        help="Optional list of ChatGPT models to compare in one run")
    parser.add_argument("--chatgpt-concurrent", type=int, default=2,
                        help="Max concurrent ChatGPT API calls")
    parser.add_argument("--chatgpt-max-tokens", type=int, default=None,
                        help="Max completion tokens per ChatGPT request (auto if omitted)")
    parser.add_argument("--chatgpt-min-start-interval", type=float, default=2.5,
                        help="Minimum seconds between starting ChatGPT requests")
    parser.add_argument("--chatgpt-timeout", type=float, default=90.0,
                        help="Per-request client timeout in seconds")
    parser.add_argument("--qwen-model", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--qwen-batch-size", type=int, default=16)
    parser.add_argument("--checklist-dir", type=str, default=None)
    parser.add_argument("--save-raw", action="store_true",
                        help="Save raw model outputs for analysis")
    args = parser.parse_args()

    # Load checklists
    checklist_dir = Path(args.checklist_dir) if args.checklist_dir else cfg.CHECKLISTS_DIR
    checklists, definitions = load_checklists(checklist_dir)
    log.info("Checklists: %d dims, %d total questions",
             len(checklists), sum(len(q) for q in checklists.values()))

    # Load eval data
    split_path = cfg.SPLITS_DIR / f"{args.split}.parquet"
    if not split_path.exists():
        # Try dev.parquet
        split_path = cfg.SPLITS_DIR / "dev.parquet"
    if not split_path.exists():
        log.error("No eval data found at %s. Run on server with data.", split_path)
        return

    df = pd.read_parquet(split_path)
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    log.info("Loaded %d samples from %s", len(df), split_path.name)


    # Build prompts (2 per sample)
    prompts_a, prompts_b = [], []
    domains = []
    for _, row in df.iterrows():
        domain = row["domain"]
        domains.append(domain)
        prompts_a.append(
            build_checkeval_prompt(row, checklists, definitions, side="A")
        )
        prompts_b.append(
            build_checkeval_prompt(row, checklists, definitions, side="B")
        )

    all_prompts = prompts_a + prompts_b
    prompt_domains = domains + domains
    n = len(df)
    all_results = []
    chatgpt_models = args.chatgpt_models or [args.chatgpt_model]

    # Reorder prompts by domain and then length to improve prompt-prefix reuse.
    ordered_indices = sorted(
        range(len(all_prompts)),
        key=lambda i: (str(prompt_domains[i]), len(all_prompts[i]), i),
    )
    ordered_prompts = [all_prompts[i] for i in ordered_indices]

    # ── ChatGPT ──
    if args.only != "qwen":
        for chatgpt_model in chatgpt_models:
            log.info(
                "Running ChatGPT (%s) on %d prompts (reordered for cache locality) ...",
                chatgpt_model, len(ordered_prompts)
            )
            t0 = time.time()
            ordered_outputs = run_chatgpt_checklist(
                ordered_prompts,
                model=chatgpt_model,
                max_concurrent=args.chatgpt_concurrent,
                min_start_interval=args.chatgpt_min_start_interval,
                timeout=args.chatgpt_timeout,
            )
            chatgpt_time = time.time() - t0

            # Restore original prompt order
            chatgpt_outputs = [""] * len(all_prompts)
            for orig_idx, text in zip(ordered_indices, ordered_outputs):
                chatgpt_outputs[orig_idx] = text

            log.info("ChatGPT (%s) done in %.1fs", chatgpt_model, chatgpt_time)

            chatgpt_eval = evaluate_teacher(
                chatgpt_outputs[:n], chatgpt_outputs[n:],
                df, checklists, label=f"ChatGPT ({chatgpt_model})",
            )
            chatgpt_eval["time_s"] = chatgpt_time
            chatgpt_eval["model_id"] = chatgpt_model
            all_results.append(chatgpt_eval)

            if args.save_raw:
                safe_model_name = _slugify_model_name(chatgpt_model)
                raw_path = RESULTS_DIR / f"chatgpt_{safe_model_name}_{args.split}_raw.json"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": chatgpt_model,
                            "outputs_a": chatgpt_outputs[:n],
                            "outputs_b": chatgpt_outputs[n:],
                        },
                        f,
                        ensure_ascii=False,
                    )
                log.info("Saved raw outputs to %s", raw_path)

    # ── Qwen ──
    if args.only != "chatgpt":
        log.info("Running Qwen (%s) on %d prompts ...", args.qwen_model, len(all_prompts))
        t0 = time.time()
        qwen_outputs = run_qwen_checklist(
            all_prompts,
            model_id=args.qwen_model,
            batch_size=args.qwen_batch_size,
        )
        qwen_time = time.time() - t0
        log.info("Qwen done in %.1fs", qwen_time)

        qwen_eval = evaluate_teacher(
            qwen_outputs[:n], qwen_outputs[n:],
            df, checklists, label="Qwen 3.5 9B",
        )
        qwen_eval["time_s"] = qwen_time
        all_results.append(qwen_eval)

        if args.save_raw:
            raw_path = RESULTS_DIR / f"qwen_{args.split}_raw.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump({"outputs_a": qwen_outputs[:n],
                           "outputs_b": qwen_outputs[n:]}, f, ensure_ascii=False)
            log.info("Saved raw outputs to %s", raw_path)

    # ── Compare ──
    if all_results:
        print_comparison(all_results)

        ranked_results = sorted(
            all_results,
            key=lambda r: (
                r.get("accuracy_excl_ties", -1),
                r.get("pairwise_parse_rate", -1),
                r.get("score_gap", float("-inf")) if r.get("score_gap") is not None else float("-inf"),
            ),
            reverse=True,
        )
        best_result = ranked_results[0]
        log.info(
            "Best teacher candidate: %s | acc_excl_ties=%.4f | parse_rate=%.4f | score_gap=%.4f",
            best_result["label"],
            best_result.get("accuracy_excl_ties", 0.0),
            best_result.get("pairwise_parse_rate", 0.0),
            best_result.get("score_gap", 0.0) or 0.0,
        )

        # Save metrics
        out_path = RESULTS_DIR / f"comparison_{args.split}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log.info("Saved comparison metrics to %s", out_path)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()