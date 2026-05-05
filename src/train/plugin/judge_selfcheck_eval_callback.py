"""
ms-swift plugin: in-training self-checklist eval callback.

Mirrors the eval loop from ``src/train/run_judge_grpo_unsloth.py`` but plugs
into ms-swift's GRPO trainer via ``swift.plugin.extra_callbacks``.

Activated by passing this file in ``--external_plugins`` and setting env vars:

    SELFCHECK_EVAL_STEPS=100              # 0 disables in-training eval
    SELFCHECK_EVAL_SPLIT=dev_600
    SELFCHECK_EVAL_SUBSET=
    SELFCHECK_EVAL_MAX_SAMPLES=200
    SELFCHECK_EVAL_BATCH_SIZE=1
    SELFCHECK_EVAL_MAX_NEW_TOKENS=2048
    SELFCHECK_EVAL_TEMPERATURE=0.0
    SELFCHECK_EVAL_ENABLE_THINKING=true
    SELFCHECK_EVAL_BEFORE_TRAIN=false
    SELFCHECK_EVAL_LABEL_PREFIX=swift_grpo

Eval generation runs through HuggingFace ``model.generate`` (vLLM colocate
engine sleeps during gradient steps; we don't try to wake it for eval).
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
_DATA_PROCESS_DIR = _SRC_DIR / "data_process"
for _p in (_SRC_DIR, _DATA_PROCESS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_winner(value: Any) -> str:
    winner = str(value).strip().upper()
    return "Tie" if winner == "TIE" else winner


def parse_self_checklist_trace(raw: str) -> dict[str, Any]:
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
        search_text, re.DOTALL | re.IGNORECASE,
    )
    if not checklist_match:
        result["parse_error"] = "missing ### Checklist section"
        return result
    for match in re.finditer(r"^Q(\d+):\s*(.+)$", checklist_match.group(1), re.MULTILINE):
        result["checklist"].append(match.group(2).strip())
    verdicts_match = re.search(
        r"###\s*Item\s*Verdicts\s*\n(.*?)(?=###\s*Final|\Z)",
        search_text, re.DOTALL | re.IGNORECASE,
    )
    if not verdicts_match:
        result["parse_error"] = "missing ### Item Verdicts section"
        return result
    for match in re.finditer(
        r"^Q(\d+):\s*(A|B|Tie)\b",
        verdicts_match.group(1), re.MULTILINE | re.IGNORECASE,
    ):
        result["verdicts"][int(match.group(1))] = _normalize_winner(match.group(2))
    final_match = re.search(
        r"###\s*Final\s*\n(.*?)(?=\Z)", search_text, re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        winner_match = re.search(
            r"Winner:\s*(A|B|Tie)\s*$",
            final_match.group(1), re.MULTILINE | re.IGNORECASE,
        )
        if winner_match:
            result["winner"] = _normalize_winner(winner_match.group(1))
    if result["winner"] is None:
        winner_match = re.search(
            r"Winner:\s*(A|B|Tie)\s*$", raw, re.MULTILINE | re.IGNORECASE,
        )
        if winner_match:
            result["winner"] = _normalize_winner(winner_match.group(1))
    result["n_questions"] = len(result["checklist"])
    result["n_verdicts"] = len(result["verdicts"])
    result["checklist_matched"] = (
        result["n_questions"] > 0 and result["n_questions"] == result["n_verdicts"]
    )
    if result["n_questions"] == 0:
        result["parse_error"] = "no checklist questions found"
    return result


def _build_eval_prompts(df, enable_thinking: bool):
    template = SELF_CHECKLIST_EVAL_PROMPT_THINKING if enable_thinking else SELF_CHECKLIST_EVAL_PROMPT
    messages_list: list[list[dict[str, str]]] = []
    keep_idx: list[int] = []
    n_swapped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if "swap_flag" in df.columns and row.get("swap_flag") is True:
            n_swapped += 1
            continue
        prompt = template.format(
            context=row["context"], response_a=row["response_a"], response_b=row["response_b"],
        )
        messages_list.append([{"role": "user", "content": prompt}])
        keep_idx.append(i)
    if n_swapped:
        log.info("[selfcheck-eval] skipped %d swap_flag rows", n_swapped)
    return messages_list, keep_idx


def _generate_eval_outputs(
    model, tokenizer, messages_list, *,
    batch_size: int, max_new_tokens: int, temperature: float, enable_thinking: bool,
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
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for m in batch_messages
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            device = getattr(model, "device", None) or next(model.parameters()).device
            inputs = inputs.to(device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            for row in generated[:, prompt_len:]:
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
    finally:
        if was_training:
            model.train()
    return outputs


def _run_eval(model, tokenizer, *, label: str, step: int | None) -> dict[str, Any] | None:
    try:
        from utils import compute_metrics, load_eval_data, save_results
    except Exception as exc:
        log.warning("[selfcheck-eval] cannot import eval utils: %s", exc)
        return None

    split = _env_str("SELFCHECK_EVAL_SPLIT", "dev_600")
    subset = _env_str("SELFCHECK_EVAL_SUBSET", "")
    max_samples = _env_int("SELFCHECK_EVAL_MAX_SAMPLES", 200)
    batch_size = _env_int("SELFCHECK_EVAL_BATCH_SIZE", 1)
    max_new_tokens = _env_int("SELFCHECK_EVAL_MAX_NEW_TOKENS", 2048)
    temperature = _env_float("SELFCHECK_EVAL_TEMPERATURE", 0.0)
    enable_thinking = _env_bool("SELFCHECK_EVAL_ENABLE_THINKING", True)
    label_prefix = _env_str("SELFCHECK_EVAL_LABEL_PREFIX", "swift_grpo")

    df = load_eval_data(split, subset or None)
    if max_samples:
        df = df.head(max_samples).reset_index(drop=True)

    messages_list, keep_idx = _build_eval_prompts(df, enable_thinking=enable_thinking)
    if not keep_idx:
        log.warning("[selfcheck-eval] no evaluable samples for %s/%s", split, subset)
        return None

    log.info(
        "[selfcheck-eval] %s: samples=%d/%d split=%s subset=%s step=%s",
        label, len(messages_list), len(df), split, subset or "-", step,
    )
    t0 = time.time()
    raw_outputs = _generate_eval_outputs(
        model, tokenizer, messages_list,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        temperature=temperature, enable_thinking=enable_thinking,
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
        sum(1 for w in predicted_winners if w == "Tie") / n_eval if n_eval else 0.0
    )
    valid_nq = [v for v in n_questions if v is not None]
    valid_tie = [v for v in item_tie_rates if v is not None]
    metrics["avg_checklist_length"] = sum(valid_nq) / len(valid_nq) if valid_nq else None
    metrics["item_tie_rate"] = sum(valid_tie) / len(valid_tie) if valid_tie else None
    metrics["trace_parse_rate"] = trace_parse_ok / n_eval if n_eval else 0.0
    metrics["judge_mode"] = "swift_grpo_hf"
    metrics["eval_label"] = label
    if step is not None:
        metrics["train_step"] = step

    split_tag = subset or split
    exp_name = f"selfchecklist_{label_prefix}_{split_tag}_{label}"
    try:
        save_results(df_eval, metrics, exp_name)
    except Exception as exc:
        log.warning("[selfcheck-eval] save_results failed: %s", exc)

    winner_counts = Counter(w for w in predicted_winners if w is not None)
    log.info(
        "[selfcheck-eval] %s: acc=%s macro_f1=%s parse=%.4f tie=%.4f trace_parse=%.4f winners=%s time=%.1fs",
        label,
        f"{metrics.get('accuracy', 0):.4f}" if "accuracy" in metrics else "N/A",
        f"{metrics.get('macro_f1', 0):.4f}" if "macro_f1" in metrics else "N/A",
        metrics["parse_rate"], metrics["tie_rate"], metrics["trace_parse_rate"],
        {k: winner_counts.get(k, 0) for k in ["A", "B", "Tie"]},
        elapsed,
    )
    return metrics


def _resolve_model_and_tokenizer(trainer, kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
    if model is None and trainer is not None:
        model = getattr(trainer, "model", None)
    if tokenizer is None and trainer is not None:
        tokenizer = (
            getattr(trainer, "tokenizer", None)
            or getattr(trainer, "processing_class", None)
        )
    return model, tokenizer


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object  # type: ignore[misc, assignment]


class SelfCheckEvalCallback(TrainerCallback):
    """Periodically run a self-checklist eval inside ms-swift's GRPO loop."""

    _trainer = None  # populated lazily via on_train_begin

    def _is_main(self, state) -> bool:
        try:
            return bool(getattr(state, "is_world_process_zero", True))
        except Exception:
            return True

    def on_train_begin(self, args, state, control, **kwargs):
        self._trainer = kwargs.get("trainer", self._trainer)
        if not self._is_main(state):
            return
        if not _env_bool("SELFCHECK_EVAL_BEFORE_TRAIN", False):
            return
        model, tokenizer = _resolve_model_and_tokenizer(self._trainer, kwargs)
        if model is None or tokenizer is None:
            log.warning("[selfcheck-eval] missing model/tokenizer at train_begin")
            return
        try:
            _run_eval(model, tokenizer, label="step_0", step=0)
        except Exception as exc:
            log.exception("[selfcheck-eval] before-train eval failed: %s", exc)

    def on_step_end(self, args, state, control, **kwargs):
        eval_steps = _env_int("SELFCHECK_EVAL_STEPS", 0)
        if eval_steps <= 0:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or step % eval_steps != 0:
            return
        if not self._is_main(state):
            return
        self._trainer = kwargs.get("trainer", self._trainer)
        model, tokenizer = _resolve_model_and_tokenizer(self._trainer, kwargs)
        if model is None or tokenizer is None:
            log.warning("[selfcheck-eval] missing model/tokenizer at step %d", step)
            return
        try:
            _run_eval(model, tokenizer, label=f"step_{step}", step=step)
        except Exception as exc:
            log.exception("[selfcheck-eval] step %d eval failed: %s", step, exc)


_EVAL_STEPS = _env_int("SELFCHECK_EVAL_STEPS", 0)
_EVAL_BEFORE = _env_bool("SELFCHECK_EVAL_BEFORE_TRAIN", False)

if _EVAL_STEPS > 0 or _EVAL_BEFORE:
    try:
        from swift.plugin import extra_callbacks  # type: ignore
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[selfcheck-eval] swift.plugin.extra_callbacks unavailable (%s); "
            "callback will not be registered.",
            exc,
        )
    else:
        extra_callbacks.append(SelfCheckEvalCallback())
        log.info(
            "[selfcheck-eval] registered SelfCheckEvalCallback "
            "(every %d steps, before_train=%s)",
            _EVAL_STEPS, _EVAL_BEFORE,
        )
