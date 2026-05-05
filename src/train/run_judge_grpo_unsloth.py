#!/usr/bin/env python3
"""
GRPO fine-tune the self-checklist judge with Unsloth + TRL.

This is the vLLM-free path for Qwen3.5-style models: Unsloth loads the policy
with ``fast_inference=False`` and TRL handles generation through Transformers.

Dataset:
    python -m src.data_process.prepare_judge_grpo --tier tier_10k

Train:
    python -m src.train.run_judge_grpo_unsloth --tier tier_10k
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_DATA_PROCESS_DIR = _SRC_DIR / "data_process"
for _p in (_SRC_DIR, _DATA_PROCESS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def completion_to_text(completion: Any) -> str:
    """Normalize TRL chat/string completion payloads to raw assistant text."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if not completion:
            return ""
        if isinstance(completion[0], dict):
            return str(completion[0].get("content", ""))
        return "\n".join(str(x) for x in completion)
    return str(completion)


def _normalize_winner(value: Any) -> str:
    winner = str(value).strip().upper()
    return "Tie" if winner == "TIE" else winner


def parse_self_checklist_trace(raw: str) -> dict[str, Any]:
    """Lightweight copy of the self-checklist parser used by judge SFT prep."""
    result: dict[str, Any] = {
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

    think_end = raw.rfind("</think>")
    search_text = raw[think_end + len("</think>"):] if think_end != -1 else raw

    checklist_match = re.search(
        r"###\s*Checklist\s*\n(.*?)(?=###\s*Item\s*Verdicts|###\s*Final|\Z)",
        search_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not checklist_match:
        result["parse_error"] = "missing ### Checklist section"
        return result
    checklist_block = checklist_match.group(1)
    for match in re.finditer(r"^Q(\d+):\s*(.+)$", checklist_block, re.MULTILINE):
        result["checklist"].append(match.group(2).strip())

    verdicts_match = re.search(
        r"###\s*Item\s*Verdicts\s*\n(.*?)(?=###\s*Final|\Z)",
        search_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not verdicts_match:
        result["parse_error"] = "missing ### Item Verdicts section"
        return result
    verdicts_block = verdicts_match.group(1)
    for match in re.finditer(
        r"^Q(\d+):\s*(A|B|Tie)\b",
        verdicts_block,
        re.MULTILINE | re.IGNORECASE,
    ):
        verdict = _normalize_winner(match.group(2))
        result["verdicts"][int(match.group(1))] = verdict

    final_match = re.search(
        r"###\s*Final\s*\n(.*?)(?=\Z)",
        search_text,
        re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        winner_match = re.search(
            r"Winner:\s*(A|B|Tie)\s*$",
            final_match.group(1),
            re.MULTILINE | re.IGNORECASE,
        )
        if winner_match:
            result["winner"] = _normalize_winner(winner_match.group(1))

    if result["winner"] is None:
        winner_match = re.search(
            r"Winner:\s*(A|B|Tie)\s*$",
            raw,
            re.MULTILINE | re.IGNORECASE,
        )
        if winner_match:
            result["winner"] = _normalize_winner(winner_match.group(1))

    result["n_questions"] = len(result["checklist"])
    result["n_verdicts"] = len(result["verdicts"])
    result["checklist_matched"] = (
        result["n_questions"] > 0
        and result["n_questions"] == result["n_verdicts"]
    )
    if result["n_questions"] == 0:
        result["parse_error"] = "no checklist questions found"
    return result


def judge_selfcheck_winner_reward(
    completions: list[Any],
    winner: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    if winner is None:
        raise RuntimeError(
            "judge_selfcheck_winner_reward needs a `winner` dataset column. "
            "Rebuild with python -m src.data_process.prepare_judge_grpo."
        )
    correct = _env_float("JUDGE_GRPO_CORRECT_REWARD", 1.0)
    wrong = _env_float("JUDGE_GRPO_WRONG_AB_PENALTY", -1.0)
    parse_fail = _env_float("JUDGE_GRPO_PARSE_FAIL_PENALTY", -0.5)
    tie_on_ab = _env_float("JUDGE_GRPO_TIE_ON_AB_PENALTY", -1.0)

    rewards: list[float] = []
    for comp, gold in zip(completions, winner):
        gold_norm = _normalize_winner(gold)
        parsed = parse_self_checklist_trace(completion_to_text(comp))
        pred = parsed["winner"]
        if pred is None:
            rewards.append(parse_fail)
            continue
        pred_norm = _normalize_winner(pred)
        if gold_norm in ("A", "B") and pred_norm == "Tie":
            rewards.append(tie_on_ab)
        elif pred_norm == gold_norm:
            rewards.append(correct)
        else:
            rewards.append(wrong)
    return rewards


def judge_selfcheck_format_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    min_q = _env_int("JUDGE_GRPO_MIN_Q", 6)
    max_q = _env_int("JUDGE_GRPO_MAX_Q", 25)

    rewards: list[float] = []
    for comp in completions:
        raw = completion_to_text(comp)
        parsed = parse_self_checklist_trace(raw)
        score = 0.0
        if "</think>" in raw:
            score += 0.25
        if parsed["n_questions"] > 0:
            score += 0.25
        if parsed["checklist_matched"] and min_q <= parsed["n_questions"] <= max_q:
            score += 0.25
        if parsed["winner"] is not None:
            score += 0.25
        rewards.append(score)
    return rewards


def judge_selfcheck_combined_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    winner_rewards = judge_selfcheck_winner_reward(completions, **kwargs)
    format_rewards = judge_selfcheck_format_reward(completions, **kwargs)
    w_winner = _env_float("JUDGE_GRPO_W_WINNER", 0.85)
    w_format = _env_float("JUDGE_GRPO_W_FORMAT", 0.15)
    return [
        w_winner * winner_score + w_format * format_score
        for winner_score, format_score in zip(winner_rewards, format_rewards)
    ]


REWARD_FUNCS = {
    "winner": judge_selfcheck_winner_reward,
    "format": judge_selfcheck_format_reward,
    "combined": judge_selfcheck_combined_reward,
}


def _coerce_messages(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages).__name__}")
    return [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in messages
    ]


def load_grpo_dataset(path: Path, max_samples: int | None = None):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")

    def to_trl_prompt(row: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": _coerce_messages(row["messages"])}

    ds = ds.map(to_trl_prompt, remove_columns=["messages"])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default=os.environ.get("TIER", "tier_10k"))
    p.add_argument("--model-name", default=None)
    p.add_argument("--dataset-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--tag", default=os.environ.get("TAG", "unsloth_grpo"))
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--fast-inference", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=10240)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--max-completion-length", type=int, default=6144)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument(
        "--reward-funcs",
        nargs="+",
        default=["winner"],
        choices=sorted(REWARD_FUNCS),
    )
    p.add_argument("--report-to", nargs="*", default=["tensorboard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-merged-16bit", action="store_true")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_name is None:
        model_name = os.environ.get("MODEL_PATH")
        if model_name is None:
            model_name = str(_PROJECT_ROOT / "models" / os.environ.get("MODEL_NAME", "Qwen3.5-4B"))
        args.model_name = model_name

    if args.dataset_path is None:
        args.dataset_path = _PROJECT_ROOT / "data" / "judge_sft" / f"grpo_{args.tier}_selfcheck.jsonl"

    if args.output_dir is None:
        model_tag = Path(str(args.model_name).rstrip("/")).name
        run_name = (
            f"judge_grpo_unsloth_{model_tag}_{args.tier}_{args.tag}"
            f"_lr{args.learning_rate}_b{args.beta}"
        )
        args.output_dir = _PROJECT_ROOT / "checkpoints" / run_name
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = resolve_paths(parse_args())

    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_path}\n"
            f"Build it with: python -m src.data_process.prepare_judge_grpo --tier {args.tier}"
        )

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_grpo_dataset(args.dataset_path, args.max_samples)
    log.info("Loaded %d GRPO rows from %s", len(dataset), args.dataset_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model_name),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.output_dir.name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        beta=args.beta,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        max_grad_norm=1.0,
        seed=args.seed,
    )

    reward_funcs = [REWARD_FUNCS[name] for name in args.reward_funcs]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_lora"))
    tokenizer.save_pretrained(str(args.output_dir / "final_lora"))

    if args.save_merged_16bit:
        merged_dir = args.output_dir / "final_merged_16bit"
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        log.info("Saved merged 16-bit model -> %s", merged_dir)


if __name__ == "__main__":
    main()
