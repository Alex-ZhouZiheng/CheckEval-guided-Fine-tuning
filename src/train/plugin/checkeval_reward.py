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
import math
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
from evaluation.run_generator_infer import parse_generated_checklist
from utils import (
    aggregate_checklist_score,
    parse_checkeval_output,
)

from swift.rewards import ORM, orms  # noqa: E402
from swift.callbacks import TrainerCallback, callbacks_map  # noqa: E402

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


def _continuous_reward(s_a: float, s_b: float, c_a: float, c_b: float, p: int) -> float:
    """HelpSteer3-aware reward combining direction / magnitude / coverage / co-tie penalty.

    Inputs:
      s_a, s_b ∈ [0, 1]  per-side checklist scores from aggregate_checklist_score
      c_a, c_b ∈ [0, 1]  per-side coverage (n_answered / expected_n)
      p ∈ {-3,…,3}       overall_preference (neg → A wins, pos → B wins, 0 → tie)

    Design notes:
      * Non-tie weights (0.333 / 0.500 / 0.167) sum to 1.0, so the non-tie
        branch can reach the same upper bound as the tie branch.
      * ``direction`` is clipped at 0, so ``delta = 0`` (no judgment) scores
        the same as the wrong direction — removes the hedging safe-default.
      * ``co_tie`` depends on ``(1 − |delta|)`` instead of the mean score, so
        "both low" is penalized just as heavily as "both high" when ``p ≠ 0``.
    """
    delta = s_b - s_a
    c = min(c_a, c_b)
    if p == 0:
        return 0.80 * math.exp(-(delta * delta) / 0.05) + 0.20 * c
    t = p / 3.0
    sign_t = 1.0 if t > 0 else -1.0
    direction = 0.333 * max(0.0, sign_t * delta)
    magnitude = 0.500 * math.exp(-((delta - t) ** 2) / 0.15)
    if delta * t < 0:
        magnitude *= 0.5
    coverage = 0.167 * c
    co_tie = 0.20 * abs(t) * (1.0 - abs(delta)) * math.exp(-(delta * delta) / (0.12 ** 2))
    r = direction + magnitude + coverage - co_tie
    return max(0.0, min(1.0, r))


class CheckEvalPairwise(ORM):
    """Continuous HelpSteer3-aware reward based on per-side CheckEval scores."""

    def __init__(self, *args, **kwargs) -> None:
        self.na_policy = os.environ.get("CHECKEVAL_NA_POLICY", "strict")
        self.coverage_threshold = float(
            os.environ.get("CHECKEVAL_COVERAGE_THRESHOLD", "0.8")
        )

    def __call__(
        self,
        completions: List[str],
        winner: List[str] | None = None,
        context: List[Any] | None = None,
        response_a: List[str] | None = None,
        response_b: List[str] | None = None,
        overall_preference: List[Any] | None = None,
        **kwargs,
    ) -> List[float]:
        if (
            winner is None
            or context is None
            or response_a is None
            or response_b is None
            or overall_preference is None
        ):
            raise RuntimeError(
                "CheckEvalPairwise reward needs winner/context/response_a/response_b/"
                "overall_preference columns. Rebuild with prepare_grpo_pairwise.py."
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
            agg_a = aggregate_checklist_score(
                parsed_a,
                na_policy=self.na_policy,
                coverage_threshold=self.coverage_threshold,
                expected_n=n_q,
            )
            agg_b = aggregate_checklist_score(
                parsed_b,
                na_policy=self.na_policy,
                coverage_threshold=self.coverage_threshold,
                expected_n=n_q,
            )
            if agg_a is None or agg_b is None:
                continue
            s_a = float(agg_a["score"])
            s_b = float(agg_b["score"])
            c_a = float(agg_a.get("coverage") or 0.0)
            c_b = float(agg_b.get("coverage") or 0.0)
            try:
                p = int(overall_preference[i])
            except (TypeError, ValueError):
                continue
            rewards[i] = _continuous_reward(s_a, s_b, c_a, c_b, p)
        return rewards


class ChecklistFormat(ORM):
    """0.5 if the completion parses into at least one non-empty domain section."""
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        out: List[float] = []
        for comp in completions:
            per_domain = parse_generated_checklist(comp or "")
            out.append(0.5 if any(per_domain.values()) else 0.0)
        return out


orms["checkeval_pairwise"] = CheckEvalPairwise
orms["checklist_format"] = ChecklistFormat


# ─────────────────────── periodic pipeline-eval callback ──────────────────────


class PipelineEvalCallback(TrainerCallback):
    """Every ``PIPELINE_EVAL_STEPS`` training steps, save the current LoRA
    adapter to a temp dir and run ``src/evaluation/run_pipeline_eval.py`` on
    ``PIPELINE_EVAL_SUBSET`` (default ``dev_600``) as a blocking subprocess.

    Env:
      PIPELINE_EVAL_STEPS         default 100
      PIPELINE_EVAL_SUBSET        default dev_600
      PIPELINE_EVAL_SPLIT         default dev
      PIPELINE_EVAL_JUDGE_ADAPTER optional; path to judge final_adapter
      PIPELINE_EVAL_CUDA_DEVICES  optional; overrides CUDA_VISIBLE_DEVICES
                                   for the eval subprocess (e.g. "1")
      PIPELINE_EVAL_BATCH_SIZE    default 16
      PIPELINE_EVAL_TIE_DELTA     default 0.0
    """

    def __init__(self, args, trainer) -> None:
        super().__init__()
        self.trainer = trainer
        self.args = args
        self.every = int(os.environ.get("PIPELINE_EVAL_STEPS", "100"))
        self.subset = os.environ.get("PIPELINE_EVAL_SUBSET", "dev_600")
        self.split = os.environ.get("PIPELINE_EVAL_SPLIT", "dev")
        self.judge_adapter = os.environ.get("PIPELINE_EVAL_JUDGE_ADAPTER") or None
        self.eval_cuda = os.environ.get("PIPELINE_EVAL_CUDA_DEVICES") or None
        self.batch_size = os.environ.get("PIPELINE_EVAL_BATCH_SIZE", "16")
        self.tie_delta = os.environ.get("PIPELINE_EVAL_TIE_DELTA", "0.0")
        self.generator_base = os.environ.get(
            "PIPELINE_EVAL_GENERATOR_BASE", str(cfg.GENERATOR_MODEL_ID)
        )
        self.judge_base = os.environ.get(
            "PIPELINE_EVAL_JUDGE_BASE", str(cfg.JUDGE_MODEL_ID)
        )
        log.info(
            "PipelineEvalCallback enabled: every=%d subset=%s cuda=%s",
            self.every, self.subset, self.eval_cuda,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.every <= 0 or state.global_step <= 0:
            return
        if state.global_step % self.every != 0:
            return
        # Only main process.
        try:
            if not self.trainer.is_world_process_zero():
                return
        except Exception:
            pass

        step = state.global_step
        out_dir = Path(args.output_dir) / f"pipeline_eval_adapter_step_{step}"
        try:
            self.trainer.save_model(str(out_dir))
        except Exception as e:
            log.warning("[pipeline_eval] save_model failed at step %d: %s", step, e)
            return

        project_root = Path(__file__).resolve().parents[3]
        script = project_root / "src" / "evaluation" / "run_pipeline_eval.py"
        cmd = [
            sys.executable, str(script),
            "--generator-base", self.generator_base,
            "--judge-base", self.judge_base,
            "--generator-adapter", str(out_dir),
            "--eval-split", self.split,
            "--subset", self.subset,
            "--batch-size", str(self.batch_size),
            "--tie-delta", str(self.tie_delta),
            "--experiment-suffix", f"grpo_step_{step}",
        ]
        if self.judge_adapter:
            cmd += ["--judge-adapter", self.judge_adapter]

        env = os.environ.copy()
        if self.eval_cuda:
            env["CUDA_VISIBLE_DEVICES"] = self.eval_cuda
        # Avoid the child reusing parent's distributed-training env.
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
                  "MASTER_PORT", "LOCAL_WORLD_SIZE"):
            env.pop(k, None)

        log.info("[pipeline_eval] step=%d  launching: %s", step, " ".join(cmd))
        import subprocess
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            log.warning("[pipeline_eval] step=%d failed (rc=%s)", step, e.returncode)


callbacks_map["pipeline_eval"] = PipelineEvalCallback
