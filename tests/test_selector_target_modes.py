import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

import pandas as pd
import torch

from run_selector_train import _prepare_oracle_tensors


def _toy_oracle():
    # 2 samples, 3 qids, all parseable
    q = pd.DataFrame([
        {"sample_id": "s1", "qid": 1, "u1_answerable": 1, "u2_abs_contrib": 0.8, "u3_decisive": 1.0, "dim": "D", "question_text": "q1", "parse_fail": False},
        {"sample_id": "s1", "qid": 2, "u1_answerable": 1, "u2_abs_contrib": 0.4, "u3_decisive": 0.0, "dim": "D", "question_text": "q2", "parse_fail": False},
        {"sample_id": "s1", "qid": 3, "u1_answerable": 1, "u2_abs_contrib": 0.2, "u3_decisive": 0.0, "dim": "D", "question_text": "q3", "parse_fail": False},
        {"sample_id": "s2", "qid": 1, "u1_answerable": 1, "u2_abs_contrib": 0.6, "u3_decisive": 1.0, "dim": "D", "question_text": "q1", "parse_fail": False},
        {"sample_id": "s2", "qid": 2, "u1_answerable": 1, "u2_abs_contrib": 0.3, "u3_decisive": 0.0, "dim": "D", "question_text": "q2", "parse_fail": False},
        {"sample_id": "s2", "qid": 3, "u1_answerable": 1, "u2_abs_contrib": 0.1, "u3_decisive": 0.0, "dim": "D", "question_text": "q3", "parse_fail": False},
    ])
    s = pd.DataFrame([
        {"sample_id": "s1", "context": "c1", "response_a": "ra1", "response_b": "rb1"},
        {"sample_id": "s2", "context": "c2", "response_a": "ra2", "response_b": "rb2"},
    ])
    return q, s


def _toy_human():
    # s1: q1 mentioned by 2/3, q2 by 1/3, q3 absent
    # s2: q1 absent, q2 absent, q3 mentioned by 3/3
    return pd.DataFrame([
        {"sample_id": "s1", "qid": 1, "h": 0.667, "n_annotators": 3},
        {"sample_id": "s1", "qid": 2, "h": 0.333, "n_annotators": 3},
        {"sample_id": "s2", "qid": 3, "h": 1.0,   "n_annotators": 3},
    ])


def test_oracle_baseline_unchanged():
    q, s = _toy_oracle()
    t = _prepare_oracle_tensors(
        q_df=q, s_df=s, alpha=1.0, beta=1.0, gamma=0.2,
        only_oracle_correct_for_ranking=False,
        target_mode="oracle_baseline",
    )
    # y[s1, q1] = 1*0.8 + 1*1.0 + 0.2*1 = 2.0
    assert torch.isclose(t.rank_target[t.sample_ids.index("s1"), 0], torch.tensor(2.0))


def test_pure_human_replaces_rank_target():
    q, s = _toy_oracle()
    h = _toy_human()
    t = _prepare_oracle_tensors(
        q_df=q, s_df=s, alpha=1.0, beta=1.0, gamma=0.2,
        only_oracle_correct_for_ranking=False,
        target_mode="pure_human", human_relevance_df=h, oracle_fallback_eps=0.1,
    )
    s1 = t.sample_ids.index("s1")
    s2 = t.sample_ids.index("s2")
    # rank_target should now equal h verbatim
    assert torch.isclose(t.rank_target[s1, 0], torch.tensor(0.667), atol=1e-3)
    assert torch.isclose(t.rank_target[s1, 1], torch.tensor(0.333), atol=1e-3)
    assert torch.isclose(t.rank_target[s1, 2], torch.tensor(0.0))
    assert torch.isclose(t.rank_target[s2, 2], torch.tensor(1.0))
    # ans_target unchanged (= u1)
    assert torch.isclose(t.ans_target[s1, 0], torch.tensor(1.0))


def test_human_oracle_fallback_uses_eps_u2_when_h_zero():
    q, s = _toy_oracle()
    h = _toy_human()
    t = _prepare_oracle_tensors(
        q_df=q, s_df=s, alpha=1.0, beta=1.0, gamma=0.2,
        only_oracle_correct_for_ranking=False,
        target_mode="human_oracle_fallback", human_relevance_df=h, oracle_fallback_eps=0.1,
    )
    s1 = t.sample_ids.index("s1")
    # h>0 -> use h
    assert torch.isclose(t.rank_target[s1, 0], torch.tensor(0.667), atol=1e-3)
    # h==0 -> eps * u2 = 0.1 * 0.2 = 0.02
    assert torch.isclose(t.rank_target[s1, 2], torch.tensor(0.02), atol=1e-4)


def test_pure_human_requires_human_df():
    q, s = _toy_oracle()
    try:
        _prepare_oracle_tensors(
            q_df=q, s_df=s, alpha=1.0, beta=1.0, gamma=0.2,
            only_oracle_correct_for_ranking=False,
            target_mode="pure_human", human_relevance_df=None,
        )
    except ValueError as e:
        assert "human_relevance_df" in str(e)
    else:
        raise AssertionError("expected ValueError")
