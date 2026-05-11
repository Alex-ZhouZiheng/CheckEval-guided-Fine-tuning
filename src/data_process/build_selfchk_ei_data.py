#!/usr/bin/env python3
"""
Expert-iteration / STaR-style data builder for the self-checklist judge —
**no gold labels used**.

Pipeline:
  1. Load pairs from ``WITH_REASON_DIR/<tier>_reasoning.parquet``.
  2. For each pair, sample ``K`` traces with the current self-checklist model
     on the canonical (A, B) order, and another ``K`` traces on the swapped
     (B, A) order — both at temperature > 0 for diversity.
  3. Aggregate the 2K winner verdicts (permuting swap-order outputs) into a
     pseudo winner Ŵ with swap-consistency + majority-confidence filters.
  4. Score every (A, B)-order trace against Ŵ via
     ``train/plugin/selfchk_trace_scorer.score_trace``.
  5. For each pair, keep the highest-reward orig-order trace whose final
     winner matches Ŵ AND whose reward exceeds ``--min-reward``.
  6. Emit an SFT parquet with the same schema as
     ``prepare_self_checklist_sft.py`` (``messages``, ``target_output``,
     ``sample_id``, ``domain``, plus diagnostics) — drops in directly to
     ``run_judge_sft.py`` / ``run_judge_sft_swift.sh``.

The gold ``winner`` column from the input parquet is read **only** for
diagnostics (audit pseudo-vs-gold agreement) and is never written to the
output. Inference-time prompts and SFT targets contain no gold reference.

Usage:
    # MVP on tier_5k_clean with current best self-checklist adapter:
    python -m src.data_process.build_selfchk_ei_data \\
        --tier tier_5k_clean \\
        --adapter-path results/checkpoints/selfcheck_.../final_adapter \\
        --k 8 --temperature 0.8 \\
        --output-path data/judge_sft/train_tier_5k_clean_selfcheck_ei_r1.parquet

    # Base model (no adapter), useful for cold-start exploration:
    python -m src.data_process.build_selfchk_ei_data \\
        --tier debug_5k --base-model models/Qwen3.5-4B --k 4 \\
        --output-path data/judge_sft/train_debug_5k_selfcheck_ei_base.parquet

    # Dry-run: print one prompt and exit.
    python -m src.data_process.build_selfchk_ei_data --tier debug_5k --dry-run
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from data_process.prepare_self_checklist_sft import (
    SELF_CHECKLIST_STUDENT_PROMPT_THINKING,
    SELF_CHECKLIST_STUDENT_PROMPT_LEGACY,
    build_self_checklist_student_prompt,
    load_pairs,
    stratified_sample,
)
from train.plugin.selfchk_trace_scorer import (
    DEFAULTS as SCORER_DEFAULTS,
    derive_pseudo_winner,
    load_diversity_encoder,
    permute_winner_for_swap,
    score_trace,
)
from utils import generate_batch, load_judge_model, make_lora_handle

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()


# ── Helpers ──

def _swap_row(row: pd.Series) -> dict:
    """Return a row-like dict with response_a / response_b swapped."""
    return {
        "context": row["context"],
        "response_a": row["response_b"],
        "response_b": row["response_a"],
    }


def build_prompts_for_pair(row: pd.Series, *, thinking: bool) -> tuple[str, str]:
    """Return (orig_prompt, swap_prompt) for one pair, both no-gold."""
    orig = build_self_checklist_student_prompt(row, thinking=thinking)
    swap = build_self_checklist_student_prompt(_swap_row(row), thinking=thinking)
    return orig, swap


def messages_from_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


# ── Sampling ──

def sample_k_traces(
    model,
    messages_list: list[list[dict]],
    *,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    enable_thinking: bool,
    lora_request,
) -> list[list[str]]:
    """For each prompt, return a list of K rollouts.

    Implemented by duplicating prompts K times within a single batched call so
    vLLM can pack them efficiently. Distinct seeds per replica encourage
    diverse completions even when ``temperature`` is moderate.
    """
    n = len(messages_list)
    if n == 0 or k == 0:
        return [[] for _ in range(n)]

    duped: list[list[dict]] = []
    seeds: list[int] = []
    for i, m in enumerate(messages_list):
        for r in range(k):
            duped.append(m)
            seeds.append(base_seed + i * 10_000 + r)

    chat_template_kwargs = {"enable_thinking": True} if enable_thinking else None

    # vLLM accepts a single ``seed`` per SamplingParams — to get distinct seeds
    # per replica we have to fall back to per-call seeding when seeds differ.
    # Fastest path: rely on temperature for diversity, single batch call.
    raws = generate_batch(
        model,
        duped,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        lora_request=lora_request,
        chat_template_kwargs=chat_template_kwargs,
        seed=base_seed,
    )

    grouped: list[list[str]] = []
    cursor = 0
    for _ in range(n):
        grouped.append(raws[cursor:cursor + k])
        cursor += k
    return grouped


# ── Main builder ──

def build(
    pairs: pd.DataFrame,
    model,
    *,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    lora_request,
    min_reward: float,
    weights: dict,
    encoder,
    seed: int,
    log_dir: Path | None,
) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)

    orig_msgs: list[list[dict]] = []
    swap_msgs: list[list[dict]] = []
    metas: list[dict] = []

    for _, row in pairs.iterrows():
        orig_prompt, swap_prompt = build_prompts_for_pair(row, thinking=enable_thinking)
        orig_msgs.append(messages_from_prompt(orig_prompt))
        swap_msgs.append(messages_from_prompt(swap_prompt))
        metas.append({
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "gold_winner": row["winner"],  # diagnostic ONLY; never written out
            "orig_prompt": orig_prompt,
        })

    log.info(
        "Sampling traces: pairs=%d  k=%d  temperature=%.2f  top_p=%.2f",
        len(pairs), k, temperature, top_p,
    )
    t0 = time.time()
    orig_groups = sample_k_traces(
        model, orig_msgs,
        k=k, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        base_seed=seed, enable_thinking=enable_thinking,
        lora_request=lora_request,
    )
    t_orig = time.time() - t0
    log.info("Orig-order sampling: %.1fs (%.2fs/sample)", t_orig, t_orig / max(len(pairs) * k, 1))

    t0 = time.time()
    swap_groups = sample_k_traces(
        model, swap_msgs,
        k=k, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        base_seed=seed + 1, enable_thinking=enable_thinking,
        lora_request=lora_request,
    )
    t_swap = time.time() - t0
    log.info("Swap-order sampling: %.1fs (%.2fs/sample)", t_swap, t_swap / max(len(pairs) * k, 1))

    # ── score + select ──
    rows: list[dict] = []
    sample_log: list[dict] = []

    n_pseudo_rejected = 0
    n_no_qualifying_trace = 0
    n_below_min_reward = 0
    n_kept = 0
    pseudo_vs_gold = Counter()    # {"agree": ..., "disagree": ..., "pseudo_none": ...}
    pseudo_winner_counts = Counter()
    reward_samples: list[float] = []

    for meta, orig_raws, swap_raws in zip(metas, orig_groups, swap_groups):
        from data_process.prepare_self_checklist_sft import parse_self_checklist_trace
        orig_parsed = [parse_self_checklist_trace(r or "") for r in orig_raws]
        swap_parsed = [parse_self_checklist_trace(r or "") for r in swap_raws]
        pseudo = derive_pseudo_winner(
            [p.get("winner") for p in orig_parsed],
            [p.get("winner") for p in swap_parsed],
            min_confidence=weights["min_pseudo_confidence"],
            min_swap_consistency=weights["min_swap_consistency"],
        )

        if pseudo["pseudo_winner"] is None:
            n_pseudo_rejected += 1
            pseudo_vs_gold["pseudo_none"] += 1
            sample_log.append({
                "sample_id": meta["sample_id"],
                "status": "rejected_pseudo",
                "pseudo": pseudo,
                "gold_winner": meta["gold_winner"],
            })
            continue

        pseudo_winner_counts[pseudo["pseudo_winner"]] += 1
        if meta["gold_winner"] in ("A", "B"):
            if pseudo["pseudo_winner"] == meta["gold_winner"]:
                pseudo_vs_gold["agree"] += 1
            else:
                pseudo_vs_gold["disagree"] += 1

        # Score every orig-order rollout; pick best parseable one whose final
        # winner matches pseudo and whose reward >= min_reward.
        candidates: list[tuple[int, dict]] = []
        for i, raw in enumerate(orig_raws):
            s = score_trace(
                raw,
                pseudo["pseudo_winner"],
                swap_consistency=pseudo["swap_consistency"],
                pseudo_confidence=pseudo["confidence"],
                encoder=encoder,
                weights=weights,
            )
            reward_samples.append(s["reward"])
            if not s["parse_ok"]:
                continue
            if s["pred_winner"] != pseudo["pseudo_winner"]:
                continue
            candidates.append((i, s))

        if not candidates:
            n_no_qualifying_trace += 1
            sample_log.append({
                "sample_id": meta["sample_id"],
                "status": "no_qualifying_trace",
                "pseudo": pseudo,
            })
            continue

        candidates.sort(key=lambda kv: kv[1]["reward"], reverse=True)
        best_i, best_s = candidates[0]
        if best_s["reward"] < min_reward:
            n_below_min_reward += 1
            sample_log.append({
                "sample_id": meta["sample_id"],
                "status": "below_min_reward",
                "best_reward": best_s["reward"],
                "pseudo": pseudo,
            })
            continue

        best_raw = orig_raws[best_i]

        # Build target_output. Mirror prepare_self_checklist_sft logic so the
        # trainer sees identical formatting. In thinking-mode prompts the chat
        # template appends "<think>\n" so the assistant generation starts
        # inside the think block — we keep raw verbatim.
        has_open = "<think>" in best_raw
        has_close = "</think>" in best_raw
        if enable_thinking:
            target = best_raw
        elif has_open and has_close:
            target = best_raw
        else:
            final_pos = best_raw.rfind("### Final")
            if final_pos != -1:
                target = (
                    "<think>\n"
                    + best_raw[:final_pos].rstrip()
                    + "\n</think>\n\n"
                    + best_raw[final_pos:].lstrip()
                )
            else:
                target = "<think>\n" + best_raw + "\n</think>"

        messages = messages_from_prompt(meta["orig_prompt"])
        parsed_best = best_s["parsed"]
        rows.append({
            "sample_id": meta["sample_id"],
            "domain": meta["domain"],
            "pseudo_winner": pseudo["pseudo_winner"],
            "pseudo_confidence": pseudo["confidence"],
            "swap_consistency": pseudo["swap_consistency"],
            "trace_reward": best_s["reward"],
            "n_questions": int(parsed_best.get("n_questions") or 0),
            "n_verdicts": int(parsed_best.get("n_verdicts") or 0),
            "item_margin": best_s["item_margin"],
            "diversity": best_s["diversity"],
            "discriminative": best_s["discriminative"],
            "messages": json.dumps(messages, ensure_ascii=False),
            "target_output": target,
        })
        n_kept += 1

        sample_log.append({
            "sample_id": meta["sample_id"],
            "status": "kept",
            "pseudo": pseudo,
            "best_reward": best_s["reward"],
            "best_rank_among_k": [c[0] for c in candidates].index(best_i),
        })

    stats: dict = {
        "n_pairs": int(len(pairs)),
        "n_kept": n_kept,
        "n_pseudo_rejected": n_pseudo_rejected,
        "n_no_qualifying_trace": n_no_qualifying_trace,
        "n_below_min_reward": n_below_min_reward,
        "k": k,
        "temperature": temperature,
        "top_p": top_p,
        "min_reward": min_reward,
        "orig_sampling_seconds": float(t_orig),
        "swap_sampling_seconds": float(t_swap),
        "pseudo_vs_gold": dict(pseudo_vs_gold),
        "pseudo_winner_counts": dict(pseudo_winner_counts),
    }
    if reward_samples:
        stats["reward_mean"] = float(sum(reward_samples) / len(reward_samples))
        stats["reward_max"] = float(max(reward_samples))
        stats["reward_min"] = float(min(reward_samples))

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "sample_log.jsonl").open("w", encoding="utf-8") as f:
            for entry in sample_log:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    return pd.DataFrame(rows), stats


# ── Reporting ──

def print_summary(df: pd.DataFrame, stats: dict, output_path: Path) -> None:
    table = Table(title="Self-Checklist Expert-Iteration Summary (no gold)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Pairs in", f"{stats.get('n_pairs', 0):,}")
    table.add_row("Rows kept", f"{stats.get('n_kept', 0):,}")
    table.add_row("Pseudo rejected (low conf/swap)", f"{stats.get('n_pseudo_rejected', 0):,}")
    table.add_row("No qualifying trace", f"{stats.get('n_no_qualifying_trace', 0):,}")
    table.add_row("Below min reward", f"{stats.get('n_below_min_reward', 0):,}")
    table.add_row("K", str(stats.get("k", "?")))
    table.add_row("Temperature", f"{stats.get('temperature', 0.0):.2f}")
    table.add_row("Min reward", f"{stats.get('min_reward', 0.0):.3f}")

    if "reward_mean" in stats:
        table.add_row("Reward mean (all traces)", f"{stats['reward_mean']:.3f}")
        table.add_row("Reward max", f"{stats['reward_max']:.3f}")
        table.add_row("Reward min", f"{stats['reward_min']:.3f}")

    pvg = stats.get("pseudo_vs_gold", {})
    if pvg:
        total_resolved = pvg.get("agree", 0) + pvg.get("disagree", 0)
        agree_rate = pvg.get("agree", 0) / total_resolved if total_resolved else 0.0
        table.add_row("Pseudo vs gold agree", f"{pvg.get('agree', 0):,}")
        table.add_row("Pseudo vs gold disagree", f"{pvg.get('disagree', 0):,}")
        table.add_row("Pseudo accuracy vs gold", f"{agree_rate:.4f}")

    pwc = stats.get("pseudo_winner_counts", {})
    for w in ("A", "B"):
        table.add_row(f"Pseudo winner = {w}", f"{pwc.get(w, 0):,}")

    if len(df):
        table.add_row("Avg n_questions", f"{df['n_questions'].mean():.1f}")
        table.add_row("Avg trace reward (kept)", f"{df['trace_reward'].mean():.3f}")
        table.add_row("Avg pseudo confidence", f"{df['pseudo_confidence'].mean():.3f}")
        table.add_row("Avg swap consistency", f"{df['swap_consistency'].mean():.3f}")
        table.add_row("Avg item margin", f"{df['item_margin'].mean():.3f}")
        table.add_row("Avg diversity", f"{df['diversity'].mean():.3f}")

    table.add_row("Output", str(output_path))
    console.print(table)


# ── CLI ──

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="tier_5k_clean",
                        help="Reasoning parquet name under data/with_reason/.")
    parser.add_argument("--input-parquet", type=str, default=None,
                        help="Override --tier: load pairs from this exact parquet path. "
                             "Required columns: sample_id, context, response_a, response_b, "
                             "domain, winner. swap_flag is optional.")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to dump per-sample decision log.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one prompt pair and exit without inference.")

    # Model
    parser.add_argument("--base-model", type=str, default=str(cfg.GENERATOR_MODEL_ID))
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Optional LoRA adapter (e.g. existing self-checklist SFT).")
    parser.add_argument("--backend", type=str, default=cfg.INFERENCE_BACKEND,
                        choices=["llamacpp", "vllm"])
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS.get("max_num_batched_tokens", 12288))
    parser.add_argument("--quantization", type=str, default=None)

    # Sampling
    parser.add_argument("--k", type=int, default=8,
                        help="Rollouts per (pair, order). Total = 2 * K per pair.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true")

    # Scoring / filtering
    parser.add_argument("--min-reward", type=float, default=0.6,
                        help="Drop pair if best trace's reward is below this.")
    parser.add_argument("--min-pseudo-confidence", type=float,
                        default=SCORER_DEFAULTS["min_pseudo_confidence"])
    parser.add_argument("--min-swap-consistency", type=float,
                        default=SCORER_DEFAULTS["min_swap_consistency"])
    parser.add_argument("--min-q", type=int, default=SCORER_DEFAULTS["min_q"])
    parser.add_argument("--max-q", type=int, default=SCORER_DEFAULTS["max_q"])
    parser.add_argument("--diversity-encoder", type=str, default=None)
    parser.add_argument("--diversity-device", type=str, default="cpu")
    parser.add_argument("--skip-diversity", action="store_true",
                        help="Skip diversity scoring (saves a model load).")

    args = parser.parse_args()

    weights = dict(SCORER_DEFAULTS)
    weights.update({
        "min_pseudo_confidence": args.min_pseudo_confidence,
        "min_swap_consistency": args.min_swap_consistency,
        "min_q": args.min_q,
        "max_q": args.max_q,
    })

    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.JUDGE_SFT_DIR / f"train_{args.tier}_selfcheck_ei.parquet"
    )

    # ── load + sample pairs ──
    if args.input_parquet:
        input_path = Path(args.input_parquet)
        pairs = pd.read_parquet(input_path)
        if "swap_flag" in pairs.columns:
            pairs = pairs[pairs["swap_flag"] == False].reset_index(drop=True)
        required = {"sample_id", "context", "response_a", "response_b", "domain", "winner"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"Input parquet missing columns: {sorted(missing)}")
        log.info("Loaded %d pairs from %s", len(pairs), input_path)
    else:
        pairs = load_pairs(args.tier)
        log.info("Loaded %d pairs from tier=%s", len(pairs), args.tier)
    pairs = stratified_sample(pairs, args.n_samples, seed=args.seed)
    log.info("Using %d pairs (seed=%d)", len(pairs), args.seed)

    if args.dry_run:
        row = pairs.iloc[0]
        orig, swap = build_prompts_for_pair(row, thinking=args.enable_thinking)
        console.print(f"[cyan]sample_id[/cyan]={row['sample_id']}  domain={row['domain']}  gold_winner={row['winner']}")
        console.print("[cyan]orig prompt (first 600c):[/cyan]\n" + orig[:600])
        console.print("[cyan]swap prompt (first 600c):[/cyan]\n" + swap[:600])
        console.print("[yellow]--dry-run set: skipping inference.[/yellow]")
        return

    # ── encoder ──
    encoder = None
    if not args.skip_diversity:
        encoder = load_diversity_encoder(args.diversity_encoder, args.diversity_device)

    # ── load model ──
    adapter_path = Path(args.adapter_path).resolve() if args.adapter_path else None
    enable_lora = adapter_path is not None
    if enable_lora:
        with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
            lora_rank = int(json.load(f).get("r", 16))
    else:
        lora_rank = 16

    log.info("Loading model: base=%s  adapter=%s  backend=%s",
             args.base_model, adapter_path, args.backend)
    model = load_judge_model(
        model_id=args.base_model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_lora=enable_lora,
        max_lora_rank=max(lora_rank, 16) if enable_lora else None,
        max_loras=1 if enable_lora else None,
        llamacpp_adapter_path=str(adapter_path) if enable_lora else None,
        quantization=args.quantization,
    )
    lora_request = make_lora_handle(
        adapter_path=str(adapter_path) if enable_lora else None,
        backend=args.backend,
        name=adapter_path.name if enable_lora else "adapter",
        lora_int_id=1,
    )

    # ── run ──
    df, stats = build(
        pairs,
        model,
        k=args.k,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        lora_request=lora_request,
        min_reward=args.min_reward,
        weights=weights,
        encoder=encoder,
        seed=args.seed,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )

    print_summary(df, stats, output_path)
    if df.empty:
        raise SystemExit("No EI rows produced; loosen --min-reward / --min-pseudo-confidence.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    stats_path = output_path.with_suffix(".stats.json")
    stats["adapter_path"] = str(adapter_path) if adapter_path else None
    stats["base_model"] = str(args.base_model)
    stats["enable_thinking"] = bool(args.enable_thinking)
    stats["tier"] = args.tier
    stats["weights"] = weights
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    log.info("Saved %d EI rows -> %s", len(df), output_path)
    console.print(f"\n[bold green]Done.[/bold green] parquet={output_path}  stats={stats_path}")


if __name__ == "__main__":
    main()
