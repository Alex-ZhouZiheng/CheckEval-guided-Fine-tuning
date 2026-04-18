#!/usr/bin/env python3
"""
Build SFT parquet for the checklist-conditioned *judge* model, using a
zero-shot teacher (Qwen3.5-27B-AWQ-4bit by default) to provide per-question
yes/no/N/A labels on the *generator-produced* checklist.

Why not the old winner-heuristic?
    winner=A → all yes / all no introduces a systematic shortcut — the model
    learns to read pair-level preference rather than judge each question. A
    zero-shot teacher gives noisy but distributionally-correct per-question
    labels, which is what we want if SFT is "learn the format + a weak prior"
    and the real judging ability comes from later RL/DPO.

Each training pair expands into two pointwise rows (side A / side B); each row
carries the *teacher's actual* ``Q1: yes / Q2: no / Q3: N/A`` output as the
target. Rows where the teacher fails to parse / cover all questions are
dropped.

Usage:
    python -m src.data_process.prepare_judge_sft \\
        --tier tier_10k --n-samples 1500

    python -m src.data_process.prepare_judge_sft \\
        --tier debug_5k --n-samples 50 --dry-run
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from eval.run_generator_infer import parse_generated_checklist
from utils import (
    CHECKEVAL_POINTWISE_PROMPT_BINARY,
    generate_batch,
    load_judge_model,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-27B-GPTQ-Int4"
DOMAIN_DEFS = cfg.DOMAIN_DESCRIPTIONS


# ────────────────────────── checklist helpers ─────────────────────
def flatten_checklist(per_domain: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    dim_lines: list[str] = []
    flat_q: list[str] = []
    flat_dom: list[str] = []
    for domain in cfg.DOMAINS:
        qs = per_domain.get(domain, [])
        if not qs:
            continue
        dim_lines.append(f"{domain} - {DOMAIN_DEFS[domain]}")
        for q in qs:
            flat_q.append(q)
            flat_dom.append(domain)
    return dim_lines, flat_q, flat_dom


def build_pointwise_prompt(
    row: dict | pd.Series,
    per_domain: dict[str, list[str]],
    side: str,
) -> tuple[str, int]:
    dim_lines, flat_q, _ = flatten_checklist(per_domain)
    dimension_block = "\n".join(dim_lines)
    checklist_text = "\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(flat_q))
    response_key = "response_a" if side == "A" else "response_b"
    prompt = CHECKEVAL_POINTWISE_PROMPT_BINARY.format(
        dimension_block=dimension_block,
        context=row["context"],
        response=row[response_key],
        checklist_text=checklist_text,
    )
    return prompt, len(flat_q)


# ────────────────────────── data loading ─────────────────────────
def load_pairwise(tier: str) -> pd.DataFrame:
    """Load pairs from the reasoning parquet (same source run_generator_infer uses).

    The basic splits parquet lacks ``sample_id`` — it only has ``prompt_id``.
    ``sample_id`` is materialized in ``{tier}_reasoning.parquet`` and is the
    key used to join with generated_checklists.
    """
    path = cfg.WITH_REASON_DIR / f"{tier}_reasoning.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_data_reasoning.py --split {tier} first."
        )
    df = pd.read_parquet(path)
    if "swap_flag" in df.columns:
        df = df[df["swap_flag"] == False].reset_index(drop=True)
    required = {"sample_id", "context", "response_a", "response_b", "domain", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairwise parquet missing columns: {sorted(missing)}")
    return df


def load_generated(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_generator_infer.py on the same tier first."
        )
    df = pd.read_parquet(path)
    out: dict[str, dict[str, list[str]]] = {}
    for _, r in df.iterrows():
        sid = r["sample_id"]
        text = r.get("generated_checklist", "") or r.get("raw_output", "")
        if not isinstance(text, str) or not text.strip():
            continue
        per_domain = parse_generated_checklist(text)
        if any(per_domain.values()):
            out[sid] = per_domain
    return out


def stratified_sample(
    pairs: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Sample ~n pairs stratified by (domain, winner)."""
    if n >= len(pairs):
        return pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = pairs.groupby(["domain", "winner"], group_keys=False)
    frac = n / len(pairs)
    sampled = groups.apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
    # top up / trim to n exactly
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(sampled) > n:
        sampled = sampled.head(n)
    elif len(sampled) < n:
        extra = pairs.drop(sampled.index, errors="ignore").sample(
            n - len(sampled), random_state=seed
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.reset_index(drop=True)


# ────────────────────────── teacher labeling ─────────────────────
def reconstruct_target(parsed: dict, n_q: int) -> tuple[str | None, int]:
    """Build canonical binary ``Q{i}: yes|no`` target from teacher output.

    Binary-only: if the teacher emits N/A despite the binary prompt, coerce it
    to ``no`` (matches the "when in doubt, no" rule). Returns (target, n_coerced).
    Target is None if any question is still unanswered.
    """
    labels: dict[int, str] = {}
    for a in parsed.get("answers", []):
        q = a.get("q")
        if isinstance(q, int) and 1 <= q <= n_q:
            labels[q] = a["answer"]  # 'yes' / 'no'
    n_coerced = 0
    for a in parsed.get("na_answers", []):
        q = a.get("q")
        if isinstance(q, int) and 1 <= q <= n_q and q not in labels:
            labels[q] = "no"
            n_coerced += 1
    if len(labels) != n_q:
        return None, n_coerced
    return "\n".join(f"Q{i}: {labels[i]}" for i in range(1, n_q + 1)), n_coerced


def build_rows(
    pairs: pd.DataFrame,
    gen_by_sample: dict[str, dict[str, list[str]]],
    teacher_model,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[pd.DataFrame, dict]:
    """Build (sample_id, side) prompts, run teacher, parse, return rows + stats."""
    prompts: list[list[dict]] = []
    metas: list[dict] = []
    n_skipped_no_checklist = 0

    for _, r in pairs.iterrows():
        sid = r["sample_id"]
        per_domain = gen_by_sample.get(sid)
        if not per_domain:
            n_skipped_no_checklist += 1
            continue
        for side in ("A", "B"):
            prompt, n_q = build_pointwise_prompt(r, per_domain, side)
            if n_q == 0:
                continue
            prompts.append([{"role": "user", "content": prompt}])
            metas.append({
                "sample_id": sid,
                "domain": r["domain"],
                "winner": r["winner"],
                "side": side,
                "n_q": n_q,
                "prompt": prompt,
            })

    log.info("Built %d teacher prompts (skipped %d pairs with no checklist)",
             len(prompts), n_skipped_no_checklist)

    t0 = time.time()
    raws = generate_batch(
        teacher_model,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    log.info("Teacher inference: %.1fs (%.2fs/row)", elapsed, elapsed / max(len(prompts), 1))

    rows: list[dict] = []
    n_parse_fail = 0
    n_incomplete = 0
    n_na_coerced_total = 0
    label_counter = Counter()
    for m, raw in zip(metas, raws):
        parsed = parse_checkeval_output(raw, expected_n=m["n_q"])
        if parsed.get("_raw_fallback"):
            n_parse_fail += 1
            continue
        target, n_coerced = reconstruct_target(parsed, m["n_q"])
        if target is None:
            n_incomplete += 1
            continue
        n_na_coerced_total += n_coerced

        # diagnostics
        n_yes = sum(1 for ln in target.splitlines() if ln.endswith(": yes"))
        n_no = sum(1 for ln in target.splitlines() if ln.endswith(": no"))
        label_counter["yes"] += n_yes
        label_counter["no"] += n_no

        messages = [{"role": "user", "content": m["prompt"]}]
        rows.append({
            "sample_id": m["sample_id"],
            "side": m["side"],
            "domain": m["domain"],
            "winner": m["winner"],
            "n_questions": m["n_q"],
            "n_yes": n_yes,
            "n_no": n_no,
            "n_na_coerced": n_coerced,
            "messages": json.dumps(messages, ensure_ascii=False),
            "target_output": target,
            "teacher_raw": raw,
        })

    stats = {
        "n_prompts": len(prompts),
        "n_parse_fail": n_parse_fail,
        "n_incomplete": n_incomplete,
        "n_rows_kept": len(rows),
        "n_na_coerced_to_no": n_na_coerced_total,
        "label_totals": dict(label_counter),
        "teacher_inference_seconds": elapsed,
    }
    log.info("Teacher parse results: kept=%d  parse_fail=%d  incomplete=%d",
             len(rows), n_parse_fail, n_incomplete)
    return pd.DataFrame(rows), stats


# ────────────────────────── reporting ─────────────────────────
def print_summary(df: pd.DataFrame, stats: dict, output_path: Path) -> None:
    table = Table(title="Judge SFT Summary (teacher-labelled)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{len(df):,}")
    if len(df):
        table.add_row("Avg Q per row", f"{df['n_questions'].mean():.1f}")
        table.add_row("Rows side A", f"{int((df['side']=='A').sum()):,}")
        table.add_row("Rows side B", f"{int((df['side']=='B').sum()):,}")
        n_total = df[["n_yes", "n_no"]].sum().sum()
        if n_total:
            table.add_row("yes %", f"{df['n_yes'].sum() / n_total:.1%}")
            table.add_row("no %",  f"{df['n_no'].sum()  / n_total:.1%}")
        table.add_row("N/A→no coerced", f"{int(df['n_na_coerced'].sum()):,}")
    table.add_row("Parse failures", f"{stats.get('n_parse_fail', 0):,}")
    table.add_row("Incomplete teacher outputs", f"{stats.get('n_incomplete', 0):,}")
    table.add_row("Output", str(output_path))
    console.print(table)


# ────────────────────────── main ─────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="tier_10k")
    parser.add_argument("--generated", type=str, default=None,
                        help="Path to data/generated_checklists/<tier>.parquet "
                             "(default: inferred from --tier)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Override output (default: data/judge_sft/train_<tier>_teacher.parquet)")
    parser.add_argument("--n-samples", type=int, default=1500,
                        help="Number of *pairs* to keep (each yields up to 2 rows). Default 1500.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one example and exit without saving parquet.")

    # ── teacher model args ──
    parser.add_argument("--teacher-model-id", type=str, default=DEFAULT_TEACHER_MODEL,
                        help=f"Teacher LLM id/path (default: {DEFAULT_TEACHER_MODEL})")
    parser.add_argument("--quantization", type=str, default=None,
                        help="vLLM quantization (auto-detected from model name; e.g. gptq_marlin, awq)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--cache-dir", type=str, default=None)

    args = parser.parse_args()

    generated_path = (
        Path(args.generated)
        if args.generated
        else cfg.GENERATED_CHECKLIST_DIR / f"{args.tier}.parquet"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.JUDGE_SFT_DIR / f"train_{args.tier}_teacher.parquet"
    )

    # ── load + sample pairs ──
    pairs = load_pairwise(args.tier)
    gen_by_sample = load_generated(generated_path)
    log.info("Loaded %d generated checklists", len(gen_by_sample))

    pairs = pairs[pairs["sample_id"].isin(gen_by_sample)].reset_index(drop=True)
    log.info("%d pairs have a matching generated checklist", len(pairs))

    pairs = stratified_sample(pairs, args.n_samples, seed=args.seed)
    log.info("Sampled %d pairs (seed=%d)", len(pairs), args.seed)
    console.print("  winner counts: ", dict(Counter(pairs["winner"])))
    console.print("  domain counts: ", dict(Counter(pairs["domain"])))

    # ── dry-run before loading the big model ──
    if args.dry_run:
        r = pairs.iloc[0]
        per_domain = gen_by_sample[r["sample_id"]]
        prompt, n_q = build_pointwise_prompt(r, per_domain, "A")
        console.print(f"\n[cyan]sample_id[/cyan]: {r['sample_id']}  winner={r['winner']}  n_q={n_q}")
        console.print(f"[cyan]prompt (first 600c)[/cyan]:\n{prompt[:600]}...")
        console.print("\n[yellow]--dry-run set: skipping teacher inference.[/yellow]")
        return

    # ── load teacher ──
    # Let vLLM auto-detect quantization from the model's config.json.
    # Only override when --quantization is explicitly passed.
    quantization = args.quantization
    log.info("Teacher: %s  (quantization=%s)",
             args.teacher_model_id, quantization or "auto")

    teacher = load_judge_model(
        model_id=args.teacher_model_id,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        quantization=quantization,
    )

    # ── run teacher and collect rows ──
    sft_df, stats = build_rows(
        pairs,
        gen_by_sample,
        teacher,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print_summary(sft_df, stats, output_path)

    if sft_df.empty:
        raise SystemExit("No judge SFT rows produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(output_path, index=False)

    # sidecar stats
    stats_path = output_path.with_suffix(".stats.json")
    stats["teacher_model_id"] = str(args.teacher_model_id)
    stats["tier"] = args.tier
    stats["n_pairs_sampled"] = int(len(pairs))
    stats["seed"] = args.seed
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info("Saved %d rows -> %s", len(sft_df), output_path)
    console.print(f"\n[bold green]Done.[/bold green] parquet={output_path}  stats={stats_path}")


if __name__ == "__main__":
    main()
