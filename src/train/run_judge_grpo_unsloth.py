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
import time
from collections import Counter
from datetime import date
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


SELF_CHECKLIST_EVAL_PROMPT_THINKING = """\
<Task Overview>
You will evaluate two candidate responses to a user request. Your task is to:
1. Generate a checklist of specific quality criteria for comparing these two responses.
2. For each criterion, decide which response is better (A, B, or Tie).
3. Based on your evaluation, output the final winner.

<Thinking Phase (free reasoning)>
Use the thinking block to reason freely. Read the conversation and both responses,
brainstorm what dimensions matter for this specific pair, and work out which
response wins on each dimension. Format inside the thinking block does not matter.

<Final Answer Phase - STRICT FORMAT>
After you finish thinking, output the following blocks EXACTLY in this order,
with these exact headers (no extra prose, no extra blocks):

### Checklist
Q1: [comparison question 1]
Q2: [comparison question 2]
...
Q8 to Q20: [more questions as needed; aim for 8 to 20]

### Item Verdicts
Q1: A
Q2: Tie
...

### Final
Winner: A

Constraints:
- Each verdict must be exactly one of A, B, or Tie (case-insensitive).
- The number of Q lines under "### Item Verdicts" must equal the number of Q
  lines under "### Checklist".
- The Winner line must read "Winner: A", "Winner: B", or "Winner: Tie".

# Conversation History #
{context}

# Response A #
{response_a}

# Response B #
{response_b}

# Your Evaluation #
"""


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


def patch_transformers_cache_compat() -> None:
    """Restore a Transformers 4.x symbol still imported by old optional deps."""
    try:
        from transformers.utils import hub
    except Exception:
        return
    if hasattr(hub, "TRANSFORMERS_CACHE"):
        return
    hub.TRANSFORMERS_CACHE = os.environ.get(
        "TRANSFORMERS_CACHE",
        os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        ),
    )


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


def build_eval_prompts(df, enable_thinking: bool) -> tuple[list[list[dict[str, str]]], list[int]]:
    template = SELF_CHECKLIST_EVAL_PROMPT_THINKING if enable_thinking else SELF_CHECKLIST_EVAL_PROMPT
    messages_list: list[list[dict[str, str]]] = []
    keep_idx: list[int] = []
    n_swapped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if "swap_flag" in df.columns and row.get("swap_flag") is True:
            n_swapped += 1
            continue
        prompt = template.format(
            context=row["context"],
            response_a=row["response_a"],
            response_b=row["response_b"],
        )
        messages_list.append([{"role": "user", "content": prompt}])
        keep_idx.append(i)
    if n_swapped:
        log.info("[selfcheck-eval] skipped %d swap_flag rows", n_swapped)
    return messages_list, keep_idx


def generate_eval_outputs(
    model,
    tokenizer,
    messages_list: list[list[dict[str, str]]],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> list[str]:
    import torch

    outputs: list[str] = []
    do_sample = temperature > 0.0
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in range(0, len(messages_list), batch_size):
            batch_messages = messages_list[start:start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for messages in batch_messages
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            device = getattr(model, "device", None)
            if device is None:
                device = next(model.parameters()).device
            inputs = inputs.to(device)
            with torch.no_grad():
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generate_kwargs["temperature"] = max(temperature, 1e-5)
                generated = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for row in generated[:, prompt_len:]:
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
    finally:
        if was_training:
            model.train()
    return outputs


def run_selfcheck_eval(
    model,
    tokenizer,
    args: argparse.Namespace,
    *,
    label: str,
    step: int | None = None,
) -> dict[str, Any]:
    from utils import compute_metrics, load_eval_data, save_results

    df = load_eval_data(args.eval_split, args.eval_subset)
    if args.eval_max_samples:
        df = df.head(args.eval_max_samples).reset_index(drop=True)

    messages_list, keep_idx = build_eval_prompts(
        df,
        enable_thinking=args.eval_enable_thinking,
    )
    if not keep_idx:
        raise RuntimeError("No evaluable samples found for self-checklist eval.")

    log.info(
        "[selfcheck-eval] %s: samples=%d/%d split=%s subset=%s",
        label,
        len(messages_list),
        len(df),
        args.eval_split,
        args.eval_subset,
    )
    t0 = time.time()
    raw_outputs = generate_eval_outputs(
        model,
        tokenizer,
        messages_list,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.eval_max_new_tokens,
        temperature=args.eval_temperature,
        enable_thinking=args.eval_enable_thinking,
    )
    elapsed = time.time() - t0

    predicted_winners: list[str | None] = []
    n_questions: list[int | None] = []
    item_tie_rates: list[float | None] = []
    trace_parse_ok = 0
    for raw in raw_outputs:
        parsed = parse_self_checklist_trace(raw)
        predicted_winners.append(parsed["winner"])
        nq = int(parsed.get("n_questions", 0) or 0)
        nv = int(parsed.get("n_verdicts", 0) or 0)
        n_questions.append(nq if nq > 0 else None)
        if nq > 0:
            trace_parse_ok += 1
        if nv > 0:
            n_tie = sum(1 for v in parsed["verdicts"].values() if v == "Tie")
            item_tie_rates.append(n_tie / nv)
        else:
            item_tie_rates.append(None)

    df_eval = df.iloc[keep_idx].reset_index(drop=True).copy()
    df_eval["raw_output"] = raw_outputs
    df_eval["predicted_winner"] = predicted_winners
    df_eval["parse_ok"] = [w is not None for w in predicted_winners]
    df_eval["n_checklist_questions"] = n_questions
    df_eval["item_tie_rate"] = item_tie_rates

    metrics = compute_metrics(
        y_true=df_eval["winner"].tolist(),
        y_pred=df_eval["predicted_winner"].tolist(),
        domains=df_eval["domain"].tolist(),
    )
    n_eval = len(messages_list)
    metrics["inference_time_s"] = elapsed
    metrics["n_samples_total"] = len(df)
    metrics["n_samples_evaluable"] = n_eval
    metrics["parse_rate"] = float(df_eval["parse_ok"].mean()) if n_eval else 0.0
    metrics["tie_rate"] = (
        sum(1 for w in predicted_winners if w == "Tie") / n_eval
        if n_eval else 0.0
    )
    valid_nq = [v for v in n_questions if v is not None]
    valid_tie = [v for v in item_tie_rates if v is not None]
    metrics["avg_checklist_length"] = (
        sum(valid_nq) / len(valid_nq) if valid_nq else None
    )
    metrics["item_tie_rate"] = (
        sum(valid_tie) / len(valid_tie) if valid_tie else None
    )
    metrics["trace_parse_rate"] = trace_parse_ok / n_eval if n_eval else 0.0
    metrics["judge_adapter"] = str(args.output_dir / "final_lora")
    metrics["judge_mode"] = "unsloth_hf"
    metrics["base_model"] = args.model_name
    metrics["eval_label"] = label
    if step is not None:
        metrics["train_step"] = step

    split_tag = args.eval_subset or args.eval_split
    exp_name = f"selfchecklist_unsloth_{args.output_dir.name}_{split_tag}_{label}"
    save_results(df_eval, metrics, exp_name)

    winner_counts = Counter(w for w in predicted_winners if w is not None)
    log.info(
        "[selfcheck-eval] %s: acc=%s macro_f1=%s parse=%.4f tie=%.4f "
        "trace_parse=%.4f winners=%s time=%.1fs",
        label,
        f"{metrics.get('accuracy', 0):.4f}" if "accuracy" in metrics else "N/A",
        f"{metrics.get('macro_f1', 0):.4f}" if "macro_f1" in metrics else "N/A",
        metrics["parse_rate"],
        metrics["tie_rate"],
        metrics["trace_parse_rate"],
        {k: winner_counts.get(k, 0) for k in ["A", "B", "Tie"]},
        elapsed,
    )
    return metrics


class SelfCheckEvalCallback:
    def __init__(self, model, tokenizer, args: argparse.Namespace) -> None:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def on_step_end(self, trainer_args, state, control, **kwargs):
                if args.eval_steps <= 0 or state.global_step <= 0:
                    return
                if state.global_step % args.eval_steps != 0:
                    return
                try:
                    if not state.is_world_process_zero:
                        return
                except AttributeError:
                    pass
                run_selfcheck_eval(
                    model,
                    tokenizer,
                    args,
                    label=f"step_{state.global_step}",
                    step=state.global_step,
                )

        self.callback = _Callback()


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
    p.add_argument("--no-final-eval", action="store_true")
    p.add_argument("--eval-before-train", action="store_true")
    p.add_argument("--eval-steps", type=int, default=0)
    p.add_argument("--eval-split", default="dev_600")
    p.add_argument("--eval-subset", default=None)
    p.add_argument("--eval-max-samples", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-enable-thinking", action="store_true", default=True)
    p.add_argument("--eval-no-thinking", dest="eval_enable_thinking", action="store_false")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_name is None:
        model_name = os.environ.get("MODEL_PATH")
        if model_name is None:
            model_name = str(_PROJECT_ROOT / "models" / os.environ.get("MODEL_NAME", "Qwen3.5-4B"))
        args.model_name = model_name

    if args.dataset_path is None:
        default_path = _PROJECT_ROOT / "data" / "judge_sft" / f"grpo_{args.tier}_selfcheck.jsonl"
        fallback_path = _PROJECT_ROOT / "data" / "judge_sft" / f"grpo_train_{args.tier}_selfcheck.jsonl"
        args.dataset_path = fallback_path if fallback_path.exists() and not default_path.exists() else default_path

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

    patch_transformers_cache_compat()

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
    tokenizer.padding_side = "left"

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
    if args.eval_steps > 0:
        trainer.add_callback(SelfCheckEvalCallback(model, tokenizer, args).callback)
    if args.eval_before_train:
        run_selfcheck_eval(model, tokenizer, args, label="before_train")
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_lora"))
    tokenizer.save_pretrained(str(args.output_dir / "final_lora"))

    if not args.no_final_eval:
        run_selfcheck_eval(model, tokenizer, args, label=f"final_{date.today()}")

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
