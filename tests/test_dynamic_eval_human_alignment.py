import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))

import pandas as pd

from run_dynamic_eval import compute_human_alignment


def test_recall_at_k_basic():
    human_df = pd.DataFrame(
        [
            {"sample_id": "s1", "qid": 1, "h": 0.5},
            {"sample_id": "s1", "qid": 2, "h": 0.5},
            {"sample_id": "s2", "qid": 3, "h": 1.0},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {"sample_id": "s1", "selected_qids": [1, 4, 7], "asked_qids": [1, 4, 7, 2]},
            {"sample_id": "s2", "selected_qids": [3], "asked_qids": [3]},
        ]
    )

    out = compute_human_alignment(pred_df, human_df)

    assert abs(out["recall_human_selected"] - 0.75) < 1e-6
    assert abs(out["recall_human_asked"] - 1.0) < 1e-6
    assert out["n_evaluated"] == 2


def test_recall_at_k_excludes_empty_yes_set():
    human_df = pd.DataFrame([{"sample_id": "s1", "qid": 1, "h": 0.5}])
    pred_df = pd.DataFrame(
        [
            {"sample_id": "s1", "selected_qids": [1], "asked_qids": [1]},
            {"sample_id": "s2", "selected_qids": [9], "asked_qids": [9]},
        ]
    )

    out = compute_human_alignment(pred_df, human_df)

    assert out["n_evaluated"] == 1
    assert out["recall_human_selected"] == 1.0
