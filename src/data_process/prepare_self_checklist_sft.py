#!/usr/bin/env python3
"""
Build SFT parquet for a **self-checklist CoT judge** model that learns to
produce a structured intermediate trace (``<think>`` block with checklist +
verdicts) before the final ``Winner: A/B/Tie`` line. At inference, only the
``Winner:`` line is consumed; the think block serves as chain-of-thought.

Teacher (Qwen3.5-27B-AWQ-4bit by default) sees the gold winner so it can
generate a coherent trace. Rows where the final winner does not match gold are
dropped.

Usage:
    python -m src.data_process.prepare_self_checklist_sft \\
        --tier tier_10k --n-samples 1500

    python -m src.data_process.prepare_self_checklist_sft \\
        --tier debug_5k --n-samples 50 --dry-run
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
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from utils import generate_batch, load_judge_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

DEFAULT_TEACHER_MODEL = "/root/autodl-tmp/Thesis/models/Qwen3.5-27B-AWQ-4bit"


# ── Prompts ──
# Teacher prompt includes gold winner so the teacher can generate a coherent
# trace. Student prompt omits it to avoid distribution mismatch at inference.

SELF_CHECKLIST_TEACHER_PROMPT = """\
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

The correct winner is: {gold_winner}

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

SELF_CHECKLIST_STUDENT_PROMPT = """\
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


# ── Parsing ──

def parse_self_checklist_trace(raw: str) -> dict:
    """Parse the self-checklist CoT trace from raw teacher output.

    Returns a dict with keys:
        checklist: list[str]       -- question strings from ### Checklist
        verdicts: dict[int, str]   -- q_number -> "A"|"B"|"Tie"
        winner: str | None         -- parsed from ### Final block
        n_questions: int
        n_verdicts: int
        checklist_matched: bool    -- len(checklist) == len(verdicts)
        parse_error: str | None
        raw: str                   -- original raw string
    """
    result: dict = {
        "checklist": [],
        "verdicts": {},
        "winner": None,
        "n_questions": 0,
        "n_verdicts": 0,
        "checklist_matched": False,
        "parse_error": None,
        "raw": raw,
    }

    if not isinstance(raw, str) or not raw.strip():
        result["parse_error"] = "empty output"
        return result

    # Split into sections using ### markers
    # Find ### Checklist block
    checklist_match = re.search(
        r'###\s*Checklist\s*\n(.*?)(?=###\s*Item\s*Verdicts|###\s*Final|\Z)',
        raw, re.DOTALL | re.IGNORECASE
    )
    if not checklist_match:
        result["parse_error"] = "missing ### Checklist section"
        return result

    checklist_block = checklist_match.group(1)
    for m in re.finditer(r'^Q(\d+):\s*(.+)$', checklist_block, re.MULTILINE):
        q_num = int(m.group(1))
        question = m.group(2).strip()
        result["checklist"].append(question)

    # Find ### Item Verdicts block
    verdicts_match = re.search(
        r'###\s*Item\s*Verdicts\s*\n(.*?)(?=###\s*Final|\Z)',
        raw, re.DOTALL | re.IGNORECASE
    )
    if not verdicts_match:
        result["parse_error"] = "missing ### Item Verdicts section"
        return result

    verdicts_block = verdicts_match.group(1)
    # Match verdicts with optional inline explanation e.g. "Q1: A (reasoning...)"
    for m in re.finditer(r'^Q(\d+):\s*(A|B|Tie)\b', verdicts_block, re.MULTILINE | re.IGNORECASE):
        q_num = int(m.group(1))
        verdict = m.group(2).strip().upper()  # Normalize to uppercase
        if verdict == "TIE":
            verdict = "Tie"
        result["verdicts"][q_num] = verdict

    # Find ### Final block
    final_match = re.search(
        r'###\s*Final\s*\n(.*?)(?=\Z)',
        raw, re.DOTALL | re.IGNORECASE
    )
    if final_match:
        final_block = final_match.group(1)
        winner_match = re.search(
            r'Winner:\s*(A|B|Tie)\s*$',
            final_block, re.MULTILINE | re.IGNORECASE
        )
        if winner_match:
            result["winner"] = winner_match.group(1).strip().upper()
            if result["winner"] == "TIE":
                result["winner"] = "Tie"

    # Fallback: search whole output for winner line
    if result["winner"] is None:
        winner_match = re.search(
            r'Winner:\s*(A|B|Tie)\s*$',
            raw, re.MULTILINE | re.IGNORECASE
        )
        if winner_match:
            result["winner"] = winner_match.group(1).strip().upper()
            if result["winner"] == "TIE":
                result["winner"] = "Tie"

    result["n_questions"] = len(result["checklist"])
    result["n_verdicts"] = len(result["verdicts"])
    result["checklist_matched"] = (
        result["n_questions"] > 0
        and result["n_questions"] == result["n_verdicts"]
    )

    if result["n_questions"] == 0:
        result["parse_error"] = "no checklist questions found"

    return result


# ── Data loading ──

def load_pairs(tier: str) -> pd.DataFrame:
    """Load pairs from the reasoning parquet.

    Filters ``swap_flag == False`` and requires columns:
    sample_id, context, response_a, response_b, domain, winner.
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
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(sampled) > n:
        sampled = sampled.head(n)
    elif len(sampled) < n:
        extra = pairs.drop(sampled.index, errors="ignore").sample(
            n - len(sampled), random_state=seed
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.reset_index(drop=True)


# ── Prompt building ──

def build_self_checklist_teacher_prompt(row, gold_winner: str) -> str:
    """Format the self-checklist teacher prompt (includes gold winner) for one pair."""
    return SELF_CHECKLIST_TEACHER_PROMPT.format(
        gold_winner=gold_winner,
        context=row["context"],
        response_a=row["response_a"],
        response_b=row["response_b"],
    )


def build_self_checklist_student_prompt(row) -> str:
    """Format the self-checklist student prompt (no gold winner) for one pair.

    This is the prompt stored in the ``messages`` column for SFT training.
    It omits the gold winner so the student does not learn to rely on a
    signal unavailable at inference time.
    """
    return SELF_CHECKLIST_STUDENT_PROMPT.format(
        context=row["context"],
        response_a=row["response_a"],
        response_b=row["response_b"],
    )


# ── Build rows ──

def build_rows(
    pairs: pd.DataFrame,
    teacher_model,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[pd.DataFrame, dict]:
    """For each pair, build teacher prompt (with gold winner), run inference,
    parse the self-checklist trace, and keep rows where:
    - parse_error is None
    - checklist_matched is True
    - winner is not None
    - winner matches gold_winner
    """
    prompts: list[list[dict]] = []
    metas: list[dict] = []

    for _, r in pairs.iterrows():
        gold_winner = r["winner"]
        teacher_prompt = build_self_checklist_teacher_prompt(r, gold_winner)
        student_prompt = build_self_checklist_student_prompt(r)
        prompts.append([{"role": "user", "content": teacher_prompt}])
        metas.append({
            "sample_id": r["sample_id"],
            "domain": r["domain"],
            "winner": gold_winner,
            "student_prompt": student_prompt,
        })

    log.info("Built %d teacher prompts", len(prompts))

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
    n_not_matched = 0
    n_no_winner = 0
    n_winner_mismatch = 0
    n_native_think = 0
    n_injected_think = 0

    for m, raw in zip(metas, raws):
        parsed = parse_self_checklist_trace(raw)

        # Drop rows with parse errors
        if parsed["parse_error"] is not None:
            n_parse_fail += 1
            continue

        # Drop rows where checklist and verdict counts don't match
        if not parsed["checklist_matched"]:
            n_not_matched += 1
            continue

        # Drop rows with no winner
        if parsed["winner"] is None:
            n_no_winner += 1
            continue

        # Drop rows where winner doesn't match gold
        if parsed["winner"] != m["winner"]:
            n_winner_mismatch += 1
            continue

        # Auto-inject <think> tags if model omitted them (common with NVFP4 models
        # that jump straight to ### Checklist). Wrap everything before ### Final.
        has_think = "<think>" in raw and "</think>" in raw
        if has_think:
            n_native_think += 1
            target = raw
        else:
            n_injected_think += 1
            final_pos = raw.rfind("### Final")
            if final_pos != -1:
                target = "<think>\n" + raw[:final_pos].rstrip() + "\n</think>\n\n" + raw[final_pos:].lstrip()
            else:
                target = "<think>\n" + raw + "\n</think>"

        messages = [{"role": "user", "content": m["student_prompt"]}]
        rows.append({
            "sample_id": m["sample_id"],
            "domain": m["domain"],
            "winner": m["winner"],
            "n_questions": parsed["n_questions"],
            "n_verdicts": parsed["n_verdicts"],
            "messages": json.dumps(messages, ensure_ascii=False),
            "target_output": target,
        })

    stats = {
        "n_prompts": len(prompts),
        "n_parse_fail": n_parse_fail,
        "n_not_matched": n_not_matched,
        "n_no_winner": n_no_winner,
        "n_winner_mismatch": n_winner_mismatch,
        "n_native_think": n_native_think,
        "n_injected_think": n_injected_think,
        "n_rows_kept": len(rows),
        "teacher_inference_seconds": elapsed,
    }
    log.info(
        "Teacher parse results: kept=%d  parse_fail=%d  not_matched=%d  no_winner=%d  mismatch=%d  native_think=%d  injected_think=%d",
        len(rows), n_parse_fail, n_not_matched, n_no_winner, n_winner_mismatch, n_native_think, n_injected_think,
    )
    return pd.DataFrame(rows), stats


# ── Reporting ──

def print_summary(df: pd.DataFrame, stats: dict, output_path: Path) -> None:
    table = Table(title="Self-Checklist SFT Summary (teacher-labelled)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Rows kept", f"{len(df):,}")
    table.add_row("Prompts sent", f"{stats.get('n_prompts', 0):,}")
    table.add_row("Parse failures", f"{stats.get('n_parse_fail', 0):,}")
    table.add_row("Checklist/verdict mismatch", f"{stats.get('n_not_matched', 0):,}")
    table.add_row("No winner", f"{stats.get('n_no_winner', 0):,}")
    table.add_row("Winner != gold", f"{stats.get('n_winner_mismatch', 0):,}")
    table.add_row("Native <think> tags", f"{stats.get('n_native_think', 0):,}")
    table.add_row("Injected <think> tags", f"{stats.get('n_injected_think', 0):,}")

    if len(df):
        table.add_row("Avg questions", f"{df['n_questions'].mean():.1f}")
        table.add_row("Median questions", f"{float(df['n_questions'].median()):.0f}")
        table.add_row("Min questions", f"{int(df['n_questions'].min())}")
        table.add_row("Max questions", f"{int(df['n_questions'].max())}")

        winner_counts = Counter(df["winner"])
        for w in ["A", "B", "Tie"]:
            table.add_row(f"Winner={w}", f"{winner_counts.get(w, 0):,}")

        domain_counts = Counter(df["domain"])
        for d in sorted(domain_counts):
            table.add_row(f"Domain={d}", f"{domain_counts[d]:,}")

    table.add_row("Output", str(output_path))
    console.print(table)


# ── Main ──

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="tier_10k")
    parser.add_argument("--n-samples", type=int, default=1500,
                        help="Number of pairs to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one example prompt and exit without inference.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Override output parquet path")
    # Teacher model args
    parser.add_argument("--teacher-model-id", type=str, default=DEFAULT_TEACHER_MODEL,
                        help=f"Teacher LLM id/path (default: {DEFAULT_TEACHER_MODEL})")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
    parser.add_argument("--max-model-len", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int,
                        default=cfg.VLLM_ENGINE_KWARGS.get("max_num_batched_tokens", 12288))
    # MTP (Multi-Token Prediction) speculative decoding
    parser.add_argument("--enable-mtp", action="store_true",
                        help="Enable vLLM MTP speculative decoding.")
    parser.add_argument("--mtp-method", type=str, default="mtp",
                        help="vLLM speculative_config method (default: mtp).")
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=1,
                        help="MTP speculative depth (default: 1).")

    args = parser.parse_args()

    output_path = (
        Path(args.output_path)
        if args.output_path
        else cfg.JUDGE_SFT_DIR / f"train_{args.tier}_selfcheck.parquet"
    )

    # ── load + sample pairs ──
    pairs = load_pairs(args.tier)
    log.info("Loaded %d pairs from %s", len(pairs), args.tier)

    pairs = stratified_sample(pairs, args.n_samples, seed=args.seed)
    log.info("Sampled %d pairs (seed=%d)", len(pairs), args.seed)
    console.print("  winner counts: ", dict(Counter(pairs["winner"])))
    console.print("  domain counts: ", dict(Counter(pairs["domain"])))

    # ── dry-run ──
    if args.dry_run:
        r = pairs.iloc[0]
        prompt = build_self_checklist_teacher_prompt(r, r["winner"])
        console.print(f"\n[cyan]sample_id[/cyan]: {r['sample_id']}  winner={r['winner']}")
        console.print(f"[cyan]prompt (first 800c)[/cyan]:\n{prompt[:800]}...")
        console.print("\n[yellow]--dry-run set: skipping teacher inference.[/yellow]")
        return

    # ── load teacher ──
    quantization = args.quantization
    speculative_config = {
        "method": args.mtp_method,
        "num_speculative_tokens": args.mtp_num_speculative_tokens,
    } if args.enable_mtp else None
    log.info("Teacher: %s  (quantization=%s  mtp=%s)",
             args.teacher_model_id, quantization or "auto", bool(args.enable_mtp))

    teacher = load_judge_model(
        model_id=args.teacher_model_id,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        quantization=quantization,
        speculative_config=speculative_config,
    )

    # ── run teacher and collect rows ──
    sft_df, stats = build_rows(
        pairs,
        teacher,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print_summary(sft_df, stats, output_path)

    if sft_df.empty:
        raise SystemExit("No self-checklist SFT rows produced.")

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
