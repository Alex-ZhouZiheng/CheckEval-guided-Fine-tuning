import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data_process"))

from data_process.prepare_self_checklist_sft import parse_self_checklist_trace
from train.plugin.judge_selfcheck_reward import (
    JudgeSelfCheckQuality,
    JudgeSelfCheckWinner,
    _DiversityScorer,
    _discriminative_score,
    _diversity_score,
    _parse_ok_score,
    _winner_score,
    orms,
)


SAMPLE_GOOD_COMPLETION = """<think>
Comparing both responses on factual accuracy, completeness, and clarity.
</think>

### Checklist
Q1: Does the response correctly identify the capital of France?
Q2: Does the response provide accurate population figures?
Q3: Is the response formatted as a bulleted list as requested?
Q4: Does the response cite a credible source?
Q5: Does the response avoid speculative claims about future events?
Q6: Does the response address the user's follow-up question about climate?

### Item Verdicts
Q1: A
Q2: B
Q3: B
Q4: Tie
Q5: B
Q6: A

### Final
Winner: B
"""

SAMPLE_ALL_TIE = """<think>...</think>

### Checklist
Q1: Is response helpful?
Q2: Is response clear?
Q3: Is response accurate?
Q4: Is response complete?
Q5: Is response well-formatted?
Q6: Is response polite?

### Item Verdicts
Q1: Tie
Q2: Tie
Q3: Tie
Q4: Tie
Q5: Tie
Q6: Tie

### Final
Winner: Tie
"""

SAMPLE_REPEATED_QUESTIONS = """<think>...</think>

### Checklist
Q1: Is the response complete?
Q2: Is the response complete?
Q3: Is the response complete and thorough?
Q4: Is the response complete in coverage?
Q5: Does the response provide complete information?
Q6: Is the answer complete?
Q7: Is the response complete and detailed?
Q8: Does the response cover all aspects completely?

### Item Verdicts
Q1: A
Q2: A
Q3: A
Q4: A
Q5: A
Q6: A
Q7: A
Q8: A

### Final
Winner: A
"""

SAMPLE_DISTINCT_QUESTIONS = """<think>...</think>

### Checklist
Q1: Does the response correctly identify the chemical formula of water?
Q2: Is the response formatted as numbered bullet points?
Q3: Does the response cite a peer-reviewed source for the population claim?
Q4: Does the response avoid repeating the same example twice in section three?
Q5: Is the tone appropriate for a formal legal context?
Q6: Does the response explicitly handle an empty input edge case?
Q7: Are units consistent throughout the answer?
Q8: Does the response avoid unsupported future predictions?

### Item Verdicts
Q1: A
Q2: A
Q3: A
Q4: A
Q5: A
Q6: A
Q7: A
Q8: A

### Final
Winner: A
"""

SAMPLE_PARSE_FAIL = "<think>I will compare the responses.</think>\n\nThe winner is A."

SAMPLE_TOO_FEW = """<think>...</think>

### Checklist
Q1: Is response good?

### Item Verdicts
Q1: A

### Final
Winner: A
"""


class _StubEncoder:
    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self.mapping = mapping or {}

    def encode(self, sentences, normalize_embeddings=True, batch_size=64, **kwargs):
        del batch_size, kwargs
        out = []
        for i, s in enumerate(sentences):
            if s in self.mapping:
                v = np.array(self.mapping[s], dtype=np.float32)
            else:
                v = np.zeros(8, dtype=np.float32)
                v[i % len(v)] = 1.0
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v)
        return np.stack(out)


@pytest.fixture(autouse=True)
def _reset_diversity_model():
    _DiversityScorer._model = None
    yield
    _DiversityScorer._model = None


def _install_encoder(encoder):
    _DiversityScorer._model = encoder


def test_parse_ok_well_formed_returns_one():
    parsed = parse_self_checklist_trace(SAMPLE_GOOD_COMPLETION)
    assert _parse_ok_score(parsed) == 1.0


def test_parse_ok_parse_fail_returns_zero():
    parsed = parse_self_checklist_trace(SAMPLE_PARSE_FAIL)
    assert _parse_ok_score(parsed) == 0.0


def test_parse_ok_too_few_questions_returns_zero():
    parsed = parse_self_checklist_trace(SAMPLE_TOO_FEW)
    assert _parse_ok_score(parsed) == 0.0


def test_parse_ok_rejects_misaligned_verdict_qids():
    parsed = {
        "checklist": [f"question {i}" for i in range(1, 7)],
        "verdicts": {i: "A" for i in range(7, 13)},
        "winner": "A",
        "n_questions": 6,
        "checklist_matched": True,
    }
    assert _parse_ok_score(parsed) == 0.0


def test_discriminative_all_tie_returns_zero():
    parsed = parse_self_checklist_trace(SAMPLE_ALL_TIE)
    assert _discriminative_score(parsed) == 0.0


def test_discriminative_mixed_verdicts():
    parsed = parse_self_checklist_trace(SAMPLE_GOOD_COMPLETION)
    assert _discriminative_score(parsed) == pytest.approx(5.0 / 6.0)


def test_discriminative_empty_verdicts_returns_zero():
    parsed = parse_self_checklist_trace(SAMPLE_PARSE_FAIL)
    assert _discriminative_score(parsed) == 0.0


def test_discriminative_all_ab_returns_one():
    parsed = parse_self_checklist_trace(SAMPLE_REPEATED_QUESTIONS)
    assert _discriminative_score(parsed) == 1.0


def test_winner_helper_correct_match():
    parsed = parse_self_checklist_trace(SAMPLE_GOOD_COMPLETION)
    assert _winner_score(SAMPLE_GOOD_COMPLETION, "B", parsed) == 1.0


def test_winner_helper_wrong_ab():
    parsed = parse_self_checklist_trace(SAMPLE_GOOD_COMPLETION)
    assert _winner_score(SAMPLE_GOOD_COMPLETION, "A", parsed) == -1.0


def test_winner_helper_tie_on_ab():
    parsed = parse_self_checklist_trace(SAMPLE_ALL_TIE)
    assert _winner_score(SAMPLE_ALL_TIE, "A", parsed) == -1.0


def test_winner_helper_parse_fail():
    parsed = parse_self_checklist_trace(SAMPLE_PARSE_FAIL)
    assert _winner_score(SAMPLE_PARSE_FAIL, "A", parsed) == -0.5


def test_winner_orm_class_uses_helper_and_matches():
    fn = JudgeSelfCheckWinner()
    rewards = fn(
        completions=[SAMPLE_GOOD_COMPLETION, SAMPLE_ALL_TIE, SAMPLE_PARSE_FAIL],
        winner=["B", "A", "A"],
    )
    assert rewards == [1.0, -1.0, -0.5]


def test_diversity_identical_questions_returns_zero():
    parsed = parse_self_checklist_trace(SAMPLE_REPEATED_QUESTIONS)
    enc = _StubEncoder({q: [1.0, 0.0, 0.0, 0.0] for q in parsed["checklist"]})
    assert _diversity_score(parsed, enc) == pytest.approx(0.0, abs=1e-6)


def test_diversity_orthogonal_questions_returns_one():
    parsed = parse_self_checklist_trace(SAMPLE_GOOD_COMPLETION)
    basis = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    enc = _StubEncoder({q: basis[i] for i, q in enumerate(parsed["checklist"])})
    assert _diversity_score(parsed, enc) == pytest.approx(1.0, abs=1e-6)


def test_diversity_single_or_zero_question_returns_one():
    enc = _StubEncoder()
    assert _diversity_score({"checklist": ["only question"]}, enc) == 1.0
    assert _diversity_score({"checklist": []}, enc) == 1.0


def test_diversity_scorer_singleton_is_lazy(monkeypatch):
    calls = {"n": 0}

    def fake_loader(name, device=None, **kwargs):
        del name, device, kwargs
        calls["n"] += 1
        return _StubEncoder()

    monkeypatch.setattr(
        "train.plugin.judge_selfcheck_reward.SentenceTransformer",
        fake_loader,
        raising=False,
    )
    a = _DiversityScorer.get()
    b = _DiversityScorer.get()
    assert a is b
    assert calls["n"] == 1


def test_T1_parse_fail_gold_A():
    _install_encoder(_StubEncoder())
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_PARSE_FAIL], winner=["A"])[0]
    assert r == pytest.approx(-0.30, abs=1e-6)


def test_T2_correct_winner_all_tie_low_div():
    parsed = parse_self_checklist_trace(SAMPLE_ALL_TIE)
    qs = parsed["checklist"]
    v1 = [1.0, 0.0]
    v2 = [0.4, (1.0 - 0.4**2) ** 0.5]
    mapping = {q: v1 for q in qs[:5]}
    mapping[qs[5]] = v2
    _install_encoder(_StubEncoder(mapping))
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_ALL_TIE], winner=["Tie"])[0]
    assert r == pytest.approx(0.73, abs=1e-3)


def test_T3_wrong_winner_perfect_div_disc():
    parsed = parse_self_checklist_trace(SAMPLE_DISTINCT_QUESTIONS)
    basis = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)]
    _install_encoder(_StubEncoder({q: basis[i] for i, q in enumerate(parsed["checklist"])}))
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_DISTINCT_QUESTIONS], winner=["B"])[0]
    assert r == pytest.approx(-0.20, abs=1e-3)


def test_T4_correct_winner_repeated_questions_disc_one():
    parsed = parse_self_checklist_trace(SAMPLE_REPEATED_QUESTIONS)
    mapping = {q: [1.0, 0.0, 0.0, 0.0] for q in parsed["checklist"]}
    _install_encoder(_StubEncoder(mapping))
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_REPEATED_QUESTIONS], winner=["A"])[0]
    assert r == pytest.approx(0.85, abs=1e-3)


def test_T5_best_case():
    parsed = parse_self_checklist_trace(SAMPLE_DISTINCT_QUESTIONS)
    basis = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)]
    _install_encoder(_StubEncoder({q: basis[i] for i, q in enumerate(parsed["checklist"])}))
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_DISTINCT_QUESTIONS], winner=["A"])[0]
    assert r == pytest.approx(1.00, abs=1e-3)


def test_T6_too_few_questions_winner_correct():
    _install_encoder(_StubEncoder())
    fn = JudgeSelfCheckQuality()
    r = fn(completions=[SAMPLE_TOO_FEW], winner=["A"])[0]
    assert r == pytest.approx(0.60, abs=1e-3)


def test_quality_requires_winner_column():
    _install_encoder(_StubEncoder())
    fn = JudgeSelfCheckQuality()
    with pytest.raises(RuntimeError, match="winner"):
        fn(completions=[SAMPLE_GOOD_COMPLETION])


def test_quality_registered_in_orms():
    assert "judge_selfcheck_quality" in orms
    assert orms["judge_selfcheck_quality"] is JudgeSelfCheckQuality
