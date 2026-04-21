"""
CheckEval judge-in-the-loop reward plugin for ms-swift GRPO.

The generator being trained produces a checklist; a frozen judge answers that
checklist pointwise for side A and side B; rewards are computed from the
aggregated per-side scores.

Registered reward functions:
  * ``checkeval_pairwise`` - compatibility alias to the legacy continuous reward
  * ``checkeval_pairwise_continuous_legacy`` - prior continuous reward
  * ``checkeval_pairwise_r1`` - direction-first R1-style reward
  * ``checklist_format`` - 0.5 if a completion parses into any non-empty domain
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
from swift.callbacks import TrainerCallback, callbacks_map  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402
from utils import (
    aggregate_checklist_score,
    compare_checklists_pairwise,
    parse_checkeval_output,
)

log = logging.getLogger(__name__)


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def get_r1_reward_config() -> dict[str, float]:
    return {
        "tie_delta": _env_float("CHECKEVAL_TIE_DELTA", "0.05"),
        "margin_sigma": _env_float("CHECKEVAL_MARGIN_SIGMA", "0.25"),
        "margin_weight": _env_float("CHECKEVAL_MARGIN_WEIGHT", "0.2"),
        "coverage_penalty_weight": _env_float("CHECKEVAL_COV_PEN_WEIGHT", "0.3"),
        "safe_tie_credit": _env_float("CHECKEVAL_SAFE_TIE_CREDIT", "0.3"),
    }


class _HttpJudge:
    """Call a vLLM OpenAI-compatible server. One shared client per process."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.url = os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1")
        self.model = os.environ["JUDGE_MODEL"]
        self.api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
        self.max_new_tokens = int(os.environ.get("JUDGE_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
        self.client = OpenAI(base_url=self.url, api_key=self.api_key)
        log.info("HttpJudge -> %s model=%s", self.url, self.model)

    def generate(self, prompts: List[str]) -> List[str]:
        out: List[str] = []
        for prompt in prompts:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
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
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
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


def compute_continuous_reward(
    s_a: float,
    s_b: float,
    c_a: float,
    c_b: float,
    p: int,
) -> float:
    """Legacy HelpSteer3-aware reward kept for compatibility."""
    delta = s_b - s_a
    coverage = min(c_a, c_b)
    if p == 0:
        return 0.80 * math.exp(-(delta * delta) / 0.05) + 0.20 * coverage
    target = p / 3.0
    sign_t = 1.0 if target > 0 else -1.0
    direction = 0.333 * max(0.0, sign_t * delta)
    magnitude = 0.500 * math.exp(-((delta - target) ** 2) / 0.15)
    if delta * target < 0:
        magnitude *= 0.5
    coverage_term = 0.167 * coverage
    co_tie = (
        0.20
        * abs(target)
        * (1.0 - abs(delta))
        * math.exp(-(delta * delta) / (0.12 ** 2))
    )
    reward = direction + magnitude + coverage_term - co_tie
    return max(0.0, min(1.0, reward))


def direction_reward(
    delta: float,
    p: int,
    tie_delta: float,
    safe_tie_credit: float,
) -> float:
    """Verifiable pairwise-direction reward."""
    checklist_says_tie = abs(delta) <= tie_delta
    human_says_tie = (p == 0)

    if human_says_tie and checklist_says_tie:
        return 1.0
    if human_says_tie and not checklist_says_tie:
        return 0.0
    if checklist_says_tie:
        return safe_tie_credit
    return 1.0 if (delta * p > 0) else 0.0


def margin_shaping(
    delta: float,
    p: int,
    margin_sigma: float,
) -> float:
    """Light shaping that only fires when direction is already correct."""
    if p == 0 or delta * p <= 0:
        return 0.0
    target = abs(p) / 3.0
    diff = abs(delta) - target
    return math.exp(-(diff * diff) / (margin_sigma * margin_sigma))


def compute_reward_components(
    s_a: float,
    s_b: float,
    c_a: float,
    c_b: float,
    p: int,
    coverage_threshold: float,
    *,
    tie_delta: float,
    margin_sigma: float,
    margin_weight: float,
    coverage_penalty_weight: float,
    safe_tie_credit: float,
) -> dict[str, float | bool]:
    delta = s_b - s_a
    checklist_says_tie = abs(delta) <= tie_delta
    human_says_tie = (p == 0)
    direction_correct = bool(human_says_tie and checklist_says_tie) or bool(
        (not human_says_tie) and (not checklist_says_tie) and (delta * p > 0)
    )
    r_dir = direction_reward(delta, p, tie_delta, safe_tie_credit)
    r_margin = margin_shaping(delta, p, margin_sigma)
    cov_pen = 1.0 if min(c_a, c_b) < coverage_threshold else 0.0
    reward_total = (
        r_dir
        + margin_weight * r_margin
        - coverage_penalty_weight * cov_pen
    )
    return {
        "delta": delta,
        "r_dir": r_dir,
        "r_margin": r_margin,
        "cov_pen": cov_pen,
        "reward_total": reward_total,
        "checklist_says_tie": checklist_says_tie,
        "human_says_tie": human_says_tie,
        "direction_correct": direction_correct,
    }


def compute_reward(
    s_a: float,
    s_b: float,
    c_a: float,
    c_b: float,
    p: int,
    coverage_threshold: float,
    *,
    tie_delta: float,
    margin_sigma: float,
    margin_weight: float,
    coverage_penalty_weight: float,
    safe_tie_credit: float,
) -> float:
    return float(
        compute_reward_components(
            s_a,
            s_b,
            c_a,
            c_b,
            p,
            coverage_threshold,
            tie_delta=tie_delta,
            margin_sigma=margin_sigma,
            margin_weight=margin_weight,
            coverage_penalty_weight=coverage_penalty_weight,
            safe_tie_credit=safe_tie_credit,
        )["reward_total"]
    )


def gold_winner_from_preference(p: int) -> str:
    if p > 0:
        return "B"
    if p < 0:
        return "A"
    return "Tie"


def prepare_completion_pointwise_prompts(
    completion: str,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    per_domain = parse_generated_checklist(completion or "")
    if not any(per_domain.values()):
        return None
    prompt_a, n_q = build_pointwise_prompt(row, per_domain, "A")
    prompt_b, _ = build_pointwise_prompt(row, per_domain, "B")
    if n_q == 0:
        return None
    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "expected_n": n_q,
        "per_domain": per_domain,
        "format_valid": True,
        "n_questions": sum(len(v) for v in per_domain.values()),
    }


def summarize_judge_pair(
    raw_a: str,
    raw_b: str,
    *,
    expected_n: int,
    na_policy: str,
    coverage_threshold: float,
    tie_delta: float,
) -> dict[str, Any]:
    parsed_a = parse_checkeval_output(raw_a, expected_n=expected_n)
    parsed_b = parse_checkeval_output(raw_b, expected_n=expected_n)
    agg_a = aggregate_checklist_score(
        parsed_a,
        na_policy=na_policy,
        coverage_threshold=coverage_threshold,
        expected_n=expected_n,
    )
    agg_b = aggregate_checklist_score(
        parsed_b,
        na_policy=na_policy,
        coverage_threshold=coverage_threshold,
        expected_n=expected_n,
    )
    coverage_a = (
        float(parsed_a.get("n_questions_parsed", 0)) / expected_n
        if expected_n > 0 else 0.0
    )
    coverage_b = (
        float(parsed_b.get("n_questions_parsed", 0)) / expected_n
        if expected_n > 0 else 0.0
    )
    summary: dict[str, Any] = {
        "parse_ok": False,
        "parsed_a": parsed_a,
        "parsed_b": parsed_b,
        "agg_a": agg_a,
        "agg_b": agg_b,
        "score_a": None,
        "score_b": None,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "n_answered_a": int(parsed_a.get("n_questions_parsed", 0) or 0),
        "n_answered_b": int(parsed_b.get("n_questions_parsed", 0) or 0),
        "na_count_a": int(parsed_a.get("n_na", 0) or 0),
        "na_count_b": int(parsed_b.get("n_na", 0) or 0),
        "signed_delta": None,
        "abs_delta": None,
        "pred_winner": None,
        "n_aligned": 0,
    }
    if agg_a is None or agg_b is None:
        return summary

    score_a = float(agg_a["score"])
    score_b = float(agg_b["score"])
    coverage_a = float(agg_a.get("coverage") or 0.0)
    coverage_b = float(agg_b.get("coverage") or 0.0)
    cmp = compare_checklists_pairwise(
        parsed_a,
        parsed_b,
        expected_n=expected_n,
        tie_delta=tie_delta,
    )
    signed_delta = score_b - score_a
    summary.update(
        {
            "parse_ok": True,
            "score_a": score_a,
            "score_b": score_b,
            "coverage_a": coverage_a,
            "coverage_b": coverage_b,
            "n_answered_a": int(agg_a.get("n_answered", 0) or 0),
            "n_answered_b": int(agg_b.get("n_answered", 0) or 0),
            "na_count_a": int(agg_a.get("n_na", 0) or 0),
            "na_count_b": int(agg_b.get("n_na", 0) or 0),
            "signed_delta": signed_delta,
            "abs_delta": abs(signed_delta),
            "pred_winner": cmp["winner"] if cmp is not None else None,
            "n_aligned": int(cmp.get("n_aligned", 0) if cmp is not None else 0),
        }
    )
    return summary


class _BaseCheckEvalPairwise(ORM):
    def __init__(self, *args, **kwargs) -> None:
        self.na_policy = os.environ.get("CHECKEVAL_NA_POLICY", "as_no")
        self.coverage_threshold = float(
            os.environ.get("CHECKEVAL_COVERAGE_THRESHOLD", "0.8")
        )

    def score_summary(self, summary: dict[str, Any], preference: int) -> float:
        raise NotImplementedError

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

        pointwise_prompts: List[str] = []
        meta: List[tuple[int, int] | None] = []
        for i, comp in enumerate(completions):
            row = {
                "context": context[i],
                "response_a": response_a[i],
                "response_b": response_b[i],
            }
            prepared = prepare_completion_pointwise_prompts(comp or "", row)
            if prepared is None:
                meta.append(None)
                continue
            pointwise_prompts.append(prepared["prompt_a"])
            pointwise_prompts.append(prepared["prompt_b"])
            meta.append((i, int(prepared["expected_n"])))

        rewards = [0.0] * len(completions)
        if not pointwise_prompts:
            return rewards

        judge = _get_judge()
        raw = judge.generate(pointwise_prompts)
        tie_delta = get_r1_reward_config()["tie_delta"]

        cursor = 0
        for m in meta:
            if m is None:
                continue
            i, expected_n = m
            raw_a = raw[cursor]
            raw_b = raw[cursor + 1]
            cursor += 2
            summary = summarize_judge_pair(
                raw_a,
                raw_b,
                expected_n=expected_n,
                na_policy=self.na_policy,
                coverage_threshold=self.coverage_threshold,
                tie_delta=tie_delta,
            )
            if not summary["parse_ok"]:
                continue
            try:
                preference = int(overall_preference[i])
            except (TypeError, ValueError):
                continue
            rewards[i] = self.score_summary(summary, preference)
        return rewards


class CheckEvalPairwiseContinuousLegacy(_BaseCheckEvalPairwise):
    """Legacy continuous HelpSteer3-aware reward based on per-side CheckEval scores."""

    def score_summary(self, summary: dict[str, Any], preference: int) -> float:
        return compute_continuous_reward(
            float(summary["score_a"]),
            float(summary["score_b"]),
            float(summary["coverage_a"]),
            float(summary["coverage_b"]),
            preference,
        )


class CheckEvalPairwiseR1(_BaseCheckEvalPairwise):
    """Direction-first R1-style reward with light margin shaping."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.r1_cfg = get_r1_reward_config()

    def score_summary(self, summary: dict[str, Any], preference: int) -> float:
        return compute_reward(
            float(summary["score_a"]),
            float(summary["score_b"]),
            float(summary["coverage_a"]),
            float(summary["coverage_b"]),
            preference,
            self.coverage_threshold,
            tie_delta=float(self.r1_cfg["tie_delta"]),
            margin_sigma=float(self.r1_cfg["margin_sigma"]),
            margin_weight=float(self.r1_cfg["margin_weight"]),
            coverage_penalty_weight=float(self.r1_cfg["coverage_penalty_weight"]),
            safe_tie_credit=float(self.r1_cfg["safe_tie_credit"]),
        )


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


CheckEvalPairwise = CheckEvalPairwiseContinuousLegacy


orms["checkeval_pairwise"] = CheckEvalPairwiseContinuousLegacy
orms["checkeval_pairwise_continuous_legacy"] = CheckEvalPairwiseContinuousLegacy
orms["checkeval_pairwise_r1"] = CheckEvalPairwiseR1
orms["checklist_format"] = ChecklistFormat


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
        super().__init__(args, trainer)
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
        try:
            if not self.trainer.is_world_process_zero():
                return
        except Exception:
            pass

        step = state.global_step
        out_dir = Path(args.output_dir) / f"pipeline_eval_adapter_step_{step}"
        try:
            self.trainer.save_model(str(out_dir))
        except Exception as exc:
            log.warning("[pipeline_eval] save_model failed at step %d: %s", step, exc)
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
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_WORLD_SIZE",
        ):
            env.pop(key, None)

        rollout_engine = None
        for attr in ("engine", "vllm_engine", "llm_engine"):
            rollout_engine = getattr(self.trainer, attr, None)
            if rollout_engine is not None:
                break
        slept = False
        if rollout_engine is not None and hasattr(rollout_engine, "sleep"):
            try:
                rollout_engine.sleep(level=2)
                slept = True
                log.info("[pipeline_eval] rollout vLLM sleeping (level=2)")
            except Exception as exc:
                log.warning("[pipeline_eval] rollout sleep failed: %s", exc)

        log.info("[pipeline_eval] step=%d launching: %s", step, " ".join(cmd))
        import subprocess

        try:
            subprocess.run(cmd, check=True, env=env, cwd=str(project_root))
        except Exception as exc:
            log.warning("[pipeline_eval] eval failed at step %d: %s", step, exc)
        finally:
            if slept and rollout_engine is not None and hasattr(rollout_engine, "wake_up"):
                try:
                    rollout_engine.wake_up()
                    log.info("[pipeline_eval] rollout vLLM woken up")
                except Exception as exc:
                    log.warning("[pipeline_eval] rollout wake_up failed: %s", exc)


callbacks_map["pipeline_eval"] = PipelineEvalCallback
