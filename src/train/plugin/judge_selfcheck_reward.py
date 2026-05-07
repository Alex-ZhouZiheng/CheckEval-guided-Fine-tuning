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

  * ``judge_selfcheck_quality`` - quality proxy + winner reward.
        0.10 * parse_ok + 0.60 * winner
        + parse_ok * (0.15 * diversity + 0.15 * discriminative)

  * ``judge_selfcheck_margin`` - dense item-verdict margin + winner reward.
        0.55 * gated_winner + 0.20 * item_margin * partial_fmt + 0.25 * partial_fmt
        + dominance penalty when nearly all item verdicts choose one side.
        gated_winner = winner * partial_fmt when winner > 0 else winner
        (correct answers only rewarded if checklist is also well-formed)

Env knobs:
    JUDGE_GRPO_MIN_Q (default 6)
    JUDGE_GRPO_MAX_Q (default 25)
    JUDGE_GRPO_PARSE_FAIL_PENALTY (default -0.5)
    JUDGE_GRPO_WRONG_AB_PENALTY (default -1.0)
    JUDGE_GRPO_CORRECT_REWARD (default 1.0)
    JUDGE_GRPO_TIE_ON_AB_PENALTY (default -1.0)
    JUDGE_GRPO_QUALITY_W_PARSE (default 0.10)
    JUDGE_GRPO_QUALITY_W_WINNER (default 0.60)
    JUDGE_GRPO_QUALITY_W_DIVERSITY (default 0.15)
    JUDGE_GRPO_QUALITY_W_DISCRIMINATIVE (default 0.15)
    JUDGE_GRPO_MARGIN_W_WINNER (default 0.55)
    JUDGE_GRPO_MARGIN_W_ITEM (default 0.20)
    JUDGE_GRPO_MARGIN_W_FORMAT (default 0.25)
    JUDGE_GRPO_MARGIN_DOMINANCE_THRESHOLD (default 0.90)
    JUDGE_GRPO_MARGIN_DOMINANCE_PENALTY (default -0.10)
    JUDGE_GRPO_DIVERSITY_MODEL (default sentence-transformers/all-MiniLM-L6-v2)
    JUDGE_GRPO_DIVERSITY_DEVICE (default cpu)
    JUDGE_GRPO_QUALITY_LOG (optional jsonl diagnostics path)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent  # .../src
for _p in (_SRC_DIR, _SRC_DIR / "data_process"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_process.prepare_self_checklist_sft import parse_self_checklist_trace

try:
    from swift.rewards import ORM, orms  # noqa: E402
except Exception:  # pragma: no cover - only used in local analysis envs without ms-swift.
    class ORM:  # type: ignore[no-redef]
        pass

    orms = {}

log = logging.getLogger(__name__)


# Imported lazily inside _DiversityScorer.get() to avoid loading the heavy
# package on module import. Tests monkeypatch this name to inject a stub.
SentenceTransformer = None


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


def _parse_ok_score(parsed: dict) -> float:
    """Strict binary parse_ok signal for the quality reward."""
    min_q = _env_int("JUDGE_GRPO_MIN_Q", 6)
    max_q = _env_int("JUDGE_GRPO_MAX_Q", 25)
    checklist = parsed.get("checklist") or []
    verdicts = parsed.get("verdicts") or {}
    n = parsed.get("n_questions", 0) or 0
    if not checklist:
        return 0.0
    if not verdicts:
        return 0.0
    if parsed.get("winner") is None:
        return 0.0
    if not parsed.get("checklist_matched"):
        return 0.0
    if n < min_q or n > max_q:
        return 0.0
    expected_qids = set(range(1, int(n) + 1))
    if set(verdicts.keys()) != expected_qids:
        return 0.0
    return 1.0


def _partial_format_score(parsed: dict, raw: str = "") -> float:
    """Partial format score in [0, 1] with four 0.25-point components."""
    min_q = _env_int("JUDGE_GRPO_MIN_Q", 6)
    max_q = _env_int("JUDGE_GRPO_MAX_Q", 25)
    score = 0.0
    if "</think>" in raw:
        score += 0.25
    n = parsed.get("n_questions", 0) or 0
    if n > 0:
        score += 0.25
    if parsed.get("checklist_matched") and min_q <= n <= max_q:
        score += 0.25
    if parsed.get("winner") is not None:
        score += 0.25
    return score


def _discriminative_score(parsed: dict) -> float:
    """Fraction of verdicts that take an A/B stance instead of Tie."""
    verdicts = parsed.get("verdicts") or {}
    n_total = len(verdicts)
    if n_total == 0:
        return 0.0
    n_ab = sum(1 for v in verdicts.values() if v in ("A", "B"))
    return n_ab / n_total


def _item_margin_score(parsed: dict, gold: str) -> float:
    """Dense support from item verdict margin, in [-1, 1]."""
    verdicts = parsed.get("verdicts") or {}
    n_total = len(verdicts)
    if n_total == 0:
        return 0.0
    counts = {
        "A": sum(1 for v in verdicts.values() if v == "A"),
        "B": sum(1 for v in verdicts.values() if v == "B"),
        "Tie": sum(1 for v in verdicts.values() if v == "Tie"),
    }
    gold_norm = str(gold).strip().upper()
    if gold_norm == "A":
        return (counts["A"] - counts["B"]) / n_total
    if gold_norm == "B":
        return (counts["B"] - counts["A"]) / n_total
    if gold_norm == "TIE":
        return (2.0 * counts["Tie"] / n_total) - 1.0
    return 0.0


def _dominance_penalty(parsed: dict, threshold: float, penalty: float) -> float:
    """Penalize degenerate all-A/all-B item verdict patterns."""
    verdicts = parsed.get("verdicts") or {}
    n_total = len(verdicts)
    if n_total == 0:
        return 0.0
    n_a = sum(1 for v in verdicts.values() if v == "A")
    n_b = sum(1 for v in verdicts.values() if v == "B")
    if max(n_a, n_b) / n_total > threshold:
        return penalty
    return 0.0


def _winner_score(
    completion: str,
    gold: str,
    parsed: dict,
    *,
    correct: float | None = None,
    wrong: float | None = None,
    parse_fail: float | None = None,
    tie_on_ab: float | None = None,
) -> float:
    """Per-sample winner-correctness score."""
    del completion
    correct = _env_float("JUDGE_GRPO_CORRECT_REWARD", 1.0) if correct is None else correct
    wrong = _env_float("JUDGE_GRPO_WRONG_AB_PENALTY", -1.0) if wrong is None else wrong
    parse_fail = _env_float("JUDGE_GRPO_PARSE_FAIL_PENALTY", -0.5) if parse_fail is None else parse_fail
    tie_on_ab = _env_float("JUDGE_GRPO_TIE_ON_AB_PENALTY", -1.0) if tie_on_ab is None else tie_on_ab

    gold_norm = str(gold).strip().upper()
    if gold_norm == "TIE":
        gold_norm = "Tie"
    pred = parsed.get("winner")
    if pred is None:
        return parse_fail
    pred_norm = pred.upper() if pred != "Tie" else "Tie"
    if gold_norm in ("A", "B") and pred_norm == "Tie":
        return tie_on_ab
    if pred_norm == gold_norm:
        return correct
    return wrong


class _DiversityScorer:
    """Lazy singleton wrapper around a CPU sentence-transformers encoder."""

    _model = None

    @classmethod
    def get(cls):
        if cls._model is not None:
            return cls._model
        global SentenceTransformer
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as _ST

            SentenceTransformer = _ST
        name = os.environ.get(
            "JUDGE_GRPO_DIVERSITY_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        device = os.environ.get("JUDGE_GRPO_DIVERSITY_DEVICE", "cpu")
        log.info("Loading diversity encoder: %s on %s", name, device)
        cls._model = SentenceTransformer(name, device=device)
        return cls._model


def _diversity_score(parsed: dict, encoder) -> float:
    """Mean pairwise (1 - cosine) over inlined checklist questions."""
    qs = parsed.get("checklist") or []
    n = len(qs)
    if n <= 1:
        return 1.0
    emb = encoder.encode(qs, normalize_embeddings=True, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)
    sim = emb @ emb.T
    mask = ~np.eye(n, dtype=bool)
    diversity = 1.0 - float(sim[mask].mean())
    return max(0.0, min(1.0, diversity))


class JudgeSelfCheckWinner(ORM):
    """Correctness reward: gold winner vs parsed winner."""

    def __init__(self, *args, **kwargs) -> None:
        self.correct = _env_float("JUDGE_GRPO_CORRECT_REWARD", 1.0)
        self.parse_ok=_env_float("JUDGE_GRPO_CORRECT_PARSE",0.2)
        self.wrong = _env_float("JUDGE_GRPO_WRONG_AB_PENALTY", -1.0)
        self.parse_fail = _env_float("JUDGE_GRPO_PARSE_FAIL_PENALTY", -1.0)
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
            parsed = parse_self_checklist_trace(comp or "")
            out.append(
                _winner_score(
                    comp or "",
                    gold,
                    parsed,
                    correct=self.correct,
                    wrong=self.wrong,
                    parse_fail=self.parse_fail,
                    tie_on_ab=self.tie_on_ab,
                )
            )
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


class JudgeSelfCheckQuality(ORM):
    """Combined proxy + outcome reward for inlined self-checklists."""

    def __init__(self, *args, **kwargs) -> None:
        self.w_parse = _env_float("JUDGE_GRPO_QUALITY_W_PARSE", 0.10)
        self.w_winner = _env_float("JUDGE_GRPO_QUALITY_W_WINNER", 0.60)
        self.w_div = _env_float("JUDGE_GRPO_QUALITY_W_DIVERSITY", 0.15)
        self.w_disc = _env_float("JUDGE_GRPO_QUALITY_W_DISCRIMINATIVE", 0.15)
        self.quality_log = os.environ.get("JUDGE_GRPO_QUALITY_LOG")

    def __call__(
        self,
        completions: List[str],
        winner: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        if winner is None:
            raise RuntimeError(
                "judge_selfcheck_quality needs `winner` column. Rebuild dataset "
                "with prepare_judge_grpo.py."
            )
        encoder = _DiversityScorer.get()
        out: List[float] = []
        diagnostics = []
        for comp, gold in zip(completions, winner):
            raw = comp or ""
            parsed = parse_self_checklist_trace(raw)
            p_ok = _parse_ok_score(parsed)
            disc = _discriminative_score(parsed)
            div = _diversity_score(parsed, encoder) if p_ok > 0 else 0.0
            win = _winner_score(raw, gold, parsed)
            reward = (
                self.w_parse * p_ok
                + self.w_winner * win
                + p_ok * (self.w_div * div + self.w_disc * disc)
            )
            out.append(reward)
            if self.quality_log:
                diagnostics.append(
                    {
                        "gold": str(gold),
                        "pred": parsed.get("winner"),
                        "parse_ok": p_ok,
                        "winner": win,
                        "diversity": div,
                        "discriminative": disc,
                        "reward": reward,
                        "n_questions": parsed.get("n_questions", 0),
                    }
                )
        if diagnostics:
            path = Path(self.quality_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for row in diagnostics:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out


class JudgeSelfCheckMargin(ORM):
    """Dense reward: gated winner + format-weighted item margin.

    Correct answers are only rewarded if the checklist is also well-formed:
        gated_winner = winner * partial_fmt  (winner > 0)
                     = winner               (winner <= 0, full penalty always)
    This prevents reward hacking via garbage checklists with correct final labels.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.w_winner = _env_float("JUDGE_GRPO_MARGIN_W_WINNER", 0.55)
        self.w_item = _env_float("JUDGE_GRPO_MARGIN_W_ITEM", 0.20)
        self.w_format = _env_float("JUDGE_GRPO_MARGIN_W_FORMAT", 0.25)
        self.parse_fail = _env_float("JUDGE_GRPO_PARSE_FAIL_PENALTY", -0.5)
        self.dominance_threshold = _env_float(
            "JUDGE_GRPO_MARGIN_DOMINANCE_THRESHOLD",
            0.90,
        )
        self.dominance_penalty = _env_float(
            "JUDGE_GRPO_MARGIN_DOMINANCE_PENALTY",
            -0.10,
        )
        self.margin_log = os.environ.get("JUDGE_GRPO_MARGIN_LOG")

    def __call__(
        self,
        completions: List[str],
        winner: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        if winner is None:
            raise RuntimeError(
                "judge_selfcheck_margin needs `winner` column. Rebuild dataset "
                "with prepare_judge_grpo.py."
            )
        out: List[float] = []
        diagnostics = []
        for comp, gold in zip(completions, winner):
            raw = comp or ""
            parsed = parse_self_checklist_trace(raw)
            win = _winner_score(raw, gold, parsed, parse_fail=self.parse_fail)
            fmt = _partial_format_score(parsed, raw)
            gated_win = win * fmt if win > 0 else win
            item = _item_margin_score(parsed, gold)
            dom = _dominance_penalty(
                parsed,
                self.dominance_threshold,
                self.dominance_penalty,
            )
            reward = (
                self.w_winner * gated_win
                + self.w_item * item * fmt
                + self.w_format * fmt
                + dom
            )
            out.append(reward)
            if self.margin_log:
                diagnostics.append(
                    {
                        "gold": str(gold),
                        "pred": parsed.get("winner"),
                        "winner": win,
                        "gated_winner": gated_win,
                        "partial_fmt": fmt,
                        "item_margin": item,
                        "dominance_penalty": dom,
                        "reward": reward,
                        "n_questions": parsed.get("n_questions", 0),
                    }
                )
        if diagnostics:
            path = Path(self.margin_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for row in diagnostics:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out


orms["judge_selfcheck_winner"] = JudgeSelfCheckWinner
orms["judge_selfcheck_format"] = JudgeSelfCheckFormat
orms["judge_selfcheck_combined"] = JudgeSelfCheckCombined
orms["judge_selfcheck_quality"] = JudgeSelfCheckQuality
orms["judge_selfcheck_margin"] = JudgeSelfCheckMargin
