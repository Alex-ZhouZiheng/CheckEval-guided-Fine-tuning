"""
CheckEval judge-in-the-loop reward plugin for ms-swift GRPO.

The generator being trained produces a checklist; a FROZEN judge (base Qwen3.5
+ optional LoRA adapter) answers that checklist pointwise for side A and side
B; `compare_checklists_pairwise` aggregates into a predicted winner; reward is
1.0 if the predicted winner matches the ground-truth winner, else 0.0. A
second ORM provides a small format reward so cold-start generators that emit
no valid sections get a gradient signal.

Two judge deployment modes, selectable via env var ``JUDGE_MODE``:

  * ``http`` (default) — judge runs as an OpenAI-compatible vLLM server; we
    call ``/v1/chat/completions``. Zero GPU contention with the colocate
    generator vLLM. See ``src/train/run_judge_vllm_serve.sh``.
      Env: JUDGE_URL (default http://127.0.0.1:8000/v1)
           JUDGE_MODEL  (model name registered at server start; adapter name
                         if you passed --lora-modules)
           JUDGE_API_KEY (default "EMPTY")

  * ``hf`` — same-process HuggingFace transformers. Simple but slow; good for
    smoke tests on a single GPU.
      Env: JUDGE_MODEL_PATH   (default: cfg.JUDGE_MODEL_ID)
           JUDGE_ADAPTER_PATH (optional PEFT adapter dir)

Common env:
  JUDGE_MAX_NEW_TOKENS  (default 512)
  JUDGE_TEMPERATURE     (default 0.0)
  TIE_DELTA             (default 0.0)

This plugin registers two reward functions with ms-swift's `orms`:
  * ``checkeval_pairwise`` — 0/1 judge-agreement reward
  * ``checklist_format``   — 0/1 any-domain-parsed reward
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List

# Make src/ importable so we can reuse existing helpers.
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent  # .../src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SRC_DIR / "data_process") not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "data_process"))
if str(_SRC_DIR / "eval") not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "eval"))

import config as cfg 
from data_process.prepare_judge_sft import build_pointwise_prompt
from eval.run_generator_infer import parse_generated_checklist
from utils import ( 
    compare_checklists_pairwise,
    parse_checkeval_output,
)

from swift.rewards import ORM, orms  # noqa: E402

log = logging.getLogger(__name__)


# ─────────────────────────── judge backends ───────────────────────────


class _HttpJudge:
    """Call a vLLM OpenAI-compatible server. One shared client per process."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.url = os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1")
        self.model = os.environ["JUDGE_MODEL"]  # must be set
        self.api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
        self.max_new_tokens = int(os.environ.get("JUDGE_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
        self.client = OpenAI(base_url=self.url, api_key=self.api_key)
        log.info("HttpJudge → %s  model=%s", self.url, self.model)

    def generate(self, prompts: List[str]) -> List[str]:
        out: List[str] = []
        for p in prompts:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": p}],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            out.append(resp.choices[0].message.content or "")
        return out


class _HfJudge:
    """Same-process HF transformers judge. Loaded once."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.environ.get("JUDGE_MODEL_PATH", str(cfg.JUDGE_MODEL_ID))
        adapter_path = os.environ.get("JUDGE_ADAPTER_PATH") or None
        self.max_new_tokens = int(os.environ.get("JUDGE_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

        log.info("HfJudge loading %s (adapter=%s)", model_path, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompts: List[str]) -> List[str]:
        import torch

        out: List[str] = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                )
            new_tokens = gen[0, inputs["input_ids"].shape[1]:]
            out.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return out


_JUDGE_SINGLETON: Any = None


def _get_judge() -> Any:
    global _JUDGE_SINGLETON
    if _JUDGE_SINGLETON is not None:
        return _JUDGE_SINGLETON
    mode = os.environ.get("JUDGE_MODE", "http").lower()
    if mode == "http":
        _JUDGE_SINGLETON = _HttpJudge()
    elif mode == "hf":
        _JUDGE_SINGLETON = _HfJudge()
    else:
        raise ValueError(f"Unknown JUDGE_MODE={mode!r}; expected 'http' or 'hf'.")
    return _JUDGE_SINGLETON


# ─────────────────────────── reward functions ─────────────────────────


class CheckEvalPairwise(ORM):
    """1.0 if judge(generated checklist) picks the ground-truth winner, else 0.0."""

    def __init__(self) -> None:
        self.tie_delta = float(os.environ.get("TIE_DELTA", "0.0"))

    def __call__(
        self,
        completions: List[str],
        winner: List[str] | None = None,
        context: List[Any] | None = None,
        response_a: List[str] | None = None,
        response_b: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        if winner is None or context is None or response_a is None or response_b is None:
            raise RuntimeError(
                "CheckEvalPairwise reward needs winner/context/response_a/response_b "
                "columns in the dataset. Use prepare_grpo_pairwise.py."
            )

        # 1) Parse checklists; keep track of which completions have non-empty ones.
        pointwise_prompts: List[str] = []
        meta: List[tuple[int, int] | None] = []  # (idx, expected_n) or None if skipped
        for i, comp in enumerate(completions):
            per_domain = parse_generated_checklist(comp or "")
            if not any(per_domain.values()):
                meta.append(None)
                continue
            row = {
                "context": context[i],
                "response_a": response_a[i],
                "response_b": response_b[i],
            }
            prompt_a, n_q = build_pointwise_prompt(row, per_domain, "A")
            prompt_b, _ = build_pointwise_prompt(row, per_domain, "B")
            if n_q == 0:
                meta.append(None)
                continue
            pointwise_prompts.append(prompt_a)
            pointwise_prompts.append(prompt_b)
            meta.append((i, n_q))

        # 2) Single batched judge call (both sides concatenated).
        rewards = [0.0] * len(completions)
        if not pointwise_prompts:
            return rewards

        judge = _get_judge()
        raw = judge.generate(pointwise_prompts)

        # 3) Parse pairwise verdicts, compute 0/1 reward.
        cursor = 0
        for m in meta:
            if m is None:
                continue
            i, n_q = m
            ra = raw[cursor]
            rb = raw[cursor + 1]
            cursor += 2
            parsed_a = parse_checkeval_output(ra, expected_n=n_q)
            parsed_b = parse_checkeval_output(rb, expected_n=n_q)
            cmp = compare_checklists_pairwise(
                parsed_a, parsed_b, expected_n=n_q, tie_delta=self.tie_delta
            )
            if cmp is None:
                continue
            gt = str(winner[i]).strip().upper()
            if cmp["winner"] == gt:
                rewards[i] = 1.0
        return rewards


class ChecklistFormat(ORM):
    """0.5 if the completion parses into at least one non-empty domain section."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        out: List[float] = []
        for comp in completions:
            per_domain = parse_generated_checklist(comp or "")
            out.append(0.5 if any(per_domain.values()) else 0.0)
        return out


orms["checkeval_pairwise"] = CheckEvalPairwise
orms["checklist_format"] = ChecklistFormat
