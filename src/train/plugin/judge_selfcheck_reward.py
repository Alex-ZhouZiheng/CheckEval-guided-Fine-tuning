"""
GRPO reward plugin for the self-checklist judge.

Registered reward functions (use any combination via --reward_funcs):

  * ``judge_selfcheck_winner``  - main correctness signal.
        +1.0 if parsed winner == gold winner
        -1.0 if parsed winner is the wrong A/B
        -0.5 if winner not parseable
        Tie predictions on A/B golds count as wrong (-1.0).

  * ``judge_selfcheck_format``  - structural signal in [0, 1].
        Returns 1.0 only if parser succeeds: <think>...</think> closed,
        ### Checklist + ### Item Verdicts + ### Final blocks present,
        n_questions == n_verdicts, n_questions in [MIN_Q, MAX_Q].
        Otherwise partial credit:
            +0.25 if </think> closed
            +0.25 if checklist parsed (n_q > 0)
            +0.25 if verdict count matches checklist count
            +0.25 if winner parsed

  * ``judge_selfcheck_combined`` - convenience: 0.85 * winner + 0.15 * format
        in a single function. Use when you want one scalar reward.

Env knobs:
    JUDGE_GRPO_MIN_Q (default 6)
    JUDGE_GRPO_MAX_Q (default 25)
    JUDGE_GRPO_PARSE_FAIL_PENALTY (default -0.5)
    JUDGE_GRPO_WRONG_AB_PENALTY (default -1.0)
    JUDGE_GRPO_CORRECT_REWARD (default 1.0)
    JUDGE_GRPO_TIE_ON_AB_PENALTY (default -1.0)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent  # .../src
for _p in (_SRC_DIR, _SRC_DIR / "data_process"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_process.prepare_self_checklist_sft import parse_self_checklist_trace
from swift.rewards import ORM, orms  # noqa: E402

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


class JudgeSelfCheckWinner(ORM):
    """Correctness reward: gold winner vs parsed winner."""

    def __init__(self, *args, **kwargs) -> None:
        self.correct = _env_float("JUDGE_GRPO_CORRECT_REWARD", 1.0)
        self.wrong = _env_float("JUDGE_GRPO_WRONG_AB_PENALTY", -1.0)
        self.parse_fail = _env_float("JUDGE_GRPO_PARSE_FAIL_PENALTY", -0.5)
        self.tie_on_ab = _env_float("JUDGE_GRPO_TIE_ON_AB_PENALTY", -1.0)

    def __call__(
        self,
        completions: List[str],
        winner: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        if winner is None:
            raise RuntimeError(
                "judge_selfcheck_winner needs `winner` column. Rebuild dataset "
                "with prepare_judge_grpo.py."
            )
        out: List[float] = []
        for comp, gold in zip(completions, winner):
            gold_norm = str(gold).strip().upper()
            if gold_norm == "TIE":
                gold_norm = "Tie"
            parsed = parse_self_checklist_trace(comp or "")
            pred = parsed["winner"]
            if pred is None:
                out.append(self.parse_fail)
                continue
            pred_norm = pred.upper() if pred != "Tie" else "Tie"
            if gold_norm in ("A", "B") and pred_norm == "Tie":
                out.append(self.tie_on_ab)
            elif pred_norm == gold_norm:
                out.append(self.correct)
            else:
                out.append(self.wrong)
        return out


class JudgeSelfCheckFormat(ORM):
    """Format reward in [0, 1]. Encourages <think>, ### blocks, count match."""

    def __init__(self, *args, **kwargs) -> None:
        self.min_q = _env_int("JUDGE_GRPO_MIN_Q", 6)
        self.max_q = _env_int("JUDGE_GRPO_MAX_Q", 25)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        out: List[float] = []
        for comp in completions:
            raw = comp or ""
            score = 0.0
            if "</think>" in raw:
                score += 0.25
            parsed = parse_self_checklist_trace(raw)
            if parsed["n_questions"] > 0:
                score += 0.25
            if (
                parsed["checklist_matched"]
                and self.min_q <= parsed["n_questions"] <= self.max_q
            ):
                score += 0.25
            if parsed["winner"] is not None:
                score += 0.25
            out.append(score)
        return out


class JudgeSelfCheckCombined(ORM):
    """Convenience: 0.85 * winner + 0.15 * format in one scalar."""

    def __init__(self, *args, **kwargs) -> None:
        self.winner_fn = JudgeSelfCheckWinner()
        self.format_fn = JudgeSelfCheckFormat()
        self.w_winner = _env_float("JUDGE_GRPO_W_WINNER", 0.85)
        self.w_format = _env_float("JUDGE_GRPO_W_FORMAT", 0.15)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        w = self.winner_fn(completions, **kwargs)
        f = self.format_fn(completions, **kwargs)
        return [self.w_winner * a + self.w_format * b for a, b in zip(w, f)]


orms["judge_selfcheck_winner"] = JudgeSelfCheckWinner
orms["judge_selfcheck_format"] = JudgeSelfCheckFormat
orms["judge_selfcheck_combined"] = JudgeSelfCheckCombined
