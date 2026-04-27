import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data_process"))

from build_human_relevance import build_extractor_prompt, parse_extractor_response


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


def test_parse_strict_json():
    raw = '{"mentioned_qids": [1, 7, 12]}'
    out = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert out == ([1, 7, 12], False)  # qids, fallback_used


def test_parse_with_prose_wrapping():
    raw = 'Sure! Here you go: {"mentioned_qids": [3, 4]} done.'
    out = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert out == ([3, 4], False)


def test_parse_regex_fallback():
    raw = "I think Q5 and Q19 apply, plus q42."
    qids, fallback = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert sorted(qids) == [5, 19, 42]
    assert fallback is True


def test_parse_regex_fallback_accepts_bare_qid_numbers():
    raw = "The relevant checklist items are 5 and 19."
    qids, fallback = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert sorted(qids) == [5, 19]
    assert fallback is True


def test_parse_filters_out_of_range_qids():
    raw = '{"mentioned_qids": [1, 99, 200]}'
    qids, _ = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert qids == [1]


def test_parse_total_failure_returns_empty():
    raw = "completely unparseable response with no Qs at all"
    qids, fallback = parse_extractor_response(raw, valid_qids=set(range(1, 62)))
    assert qids == []
    assert fallback is True


from build_human_relevance import aggregate_h


def test_aggregate_h_basic():
    # 3 annotators on one sample; valid bank = qids 1..5
    per_annotator_qids = [
        {"qids": [1, 2], "fallback": False, "ok": True},
        {"qids": [2, 3], "fallback": True,  "ok": True},
        {"qids": [],     "fallback": True,  "ok": False},  # parse total failure -> excluded
    ]
    rows = aggregate_h(
        sample_id="s1", prompt_id="p1",
        per_annotator=per_annotator_qids,
        valid_qids=set(range(1, 6)),
    )
    # Two usable annotators -> qid1 1/2=0.5, qid2 2/2=1.0, qid3 1/2=0.5
    by_qid = {r["qid"]: r["h"] for r in rows}
    assert by_qid == {1: 0.5, 2: 1.0, 3: 0.5}
    assert all(r["n_annotators"] == 2 for r in rows)
    assert all(r["sample_id"] == "s1" and r["prompt_id"] == "p1" for r in rows)


def test_aggregate_h_drops_zero_usable():
    rows = aggregate_h(
        sample_id="s2", prompt_id="p2",
        per_annotator=[{"qids": [], "fallback": True, "ok": False}],
        valid_qids=set(range(1, 6)),
    )
    assert rows == []
