import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data_process"))

from build_human_relevance import build_extractor_prompt


def test_prompt_includes_all_qids_and_reasoning():
    qmeta = [
        {"qid": 1, "dimension": "Correctness", "question_text": "Are the facts accurate?"},
        {"qid": 2, "dimension": "Clarity", "question_text": "Is the prose clear?"},
        {"qid": 7, "dimension": "Coding", "question_text": "Is the code correct?"},
    ]
    reasoning = "Response 1 is better because the facts are wrong in Response 2."
    prompt = build_extractor_prompt(reasoning=reasoning, qmeta=qmeta)
    # All qids surfaced
    assert "Q1" in prompt and "Q2" in prompt and "Q7" in prompt
    # Reasoning text passed through verbatim
    assert reasoning in prompt
    # Output schema referenced
    assert "mentioned_qids" in prompt
