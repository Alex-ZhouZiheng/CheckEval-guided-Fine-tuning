import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))

from review_helpers import _extract_questions, _diff_answers, _render_parsed


SAMPLE_PROMPT = """
Some preamble text.

Q1: Does the response directly address the user's question?
Q2: Is the explanation clear and easy to follow?
Q3: Does the response avoid unnecessary repetition?
Q10: Is the response free of factual errors?
"""


def test_extract_questions_basic():
    result = _extract_questions(SAMPLE_PROMPT)
    assert result == {
        1: "Does the response directly address the user's question?",
        2: "Is the explanation clear and easy to follow?",
        3: "Does the response avoid unnecessary repetition?",
        10: "Is the response free of factual errors?",
    }


def test_extract_questions_empty():
    assert _extract_questions("No questions here.") == {}


def test_extract_questions_ignores_non_q_lines():
    prompt = "Q1: First question?\nsome other line\nQ2: Second question?"
    result = _extract_questions(prompt)
    assert result == {1: "First question?", 2: "Second question?"}


def test_diff_answers_finds_disagreements():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}, {"q": 3, "answer": "yes"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "yes"}, {"q": 3, "answer": "no"}],
        "na_answers": [],
    }
    result = _diff_answers(parsed_a, parsed_b)
    assert result == [(2, "no", "yes"), (3, "yes", "no")]


def test_diff_answers_na_treated_as_na():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}],
        "na_answers": [{"q": 2}],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    result = _diff_answers(parsed_a, parsed_b)
    assert result == [(2, "N/A", "no")]


def test_diff_answers_no_diff_returns_empty():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    assert _diff_answers(parsed_a, parsed_b) == []


def test_render_parsed_missing_score_uses_placeholder():
    rendered = _render_parsed({})
    assert rendered.startswith("score=?")


def test_diff_answers_missing_q_treated_as_dash():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [],
        "na_answers": [],
    }
    result = _diff_answers(parsed_a, parsed_b)
    assert result == [(1, "yes", "—")]
