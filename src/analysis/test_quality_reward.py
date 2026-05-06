#!/usr/bin/env python3
"""Pre-training validation gate for judge_selfcheck_quality reward."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SRC_ROOT / "data_process") not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT / "data_process"))

from data_process.prepare_self_checklist_sft import parse_self_checklist_trace  # noqa: E402
from train.plugin.judge_selfcheck_reward import (  # noqa: E402
    JudgeSelfCheckQuality,
    _DiversityScorer,
    _discriminative_score,
    _diversity_score,
    _parse_ok_score,
    _winner_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


class _StubEncoder:
    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self.mapping = mapping or {}

    def encode(self, sentences, normalize_embeddings=True, batch_size=64, **kwargs):
        del batch_size, kwargs
        out = []
        for i, s in enumerate(sentences):
            if s in self.mapping:
                v = np.asarray(self.mapping[s], dtype=np.float32)
            else:
                v = np.zeros(16, dtype=np.float32)
                v[i % len(v)] = 1.0
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v)
        return np.stack(out)


def _make_completion(questions: list[str], verdicts: list[str], winner: str) -> str:
    body = ["<think>Comparing carefully.</think>", "", "### Checklist"]
    body += [f"Q{i + 1}: {q}" for i, q in enumerate(questions)]
    body += ["", "### Item Verdicts"]
    body += [f"Q{i + 1}: {v}" for i, v in enumerate(verdicts)]
    body += ["", "### Final", f"Winner: {winner}"]
    return "\n".join(body)


def _distinct_questions() -> list[str]:
    return [
        "Does the response correctly identify the chemical formula of water?",
        "Is the response formatted as numbered bullet points?",
        "Does the response cite a peer-reviewed source for the population claim?",
        "Does the response avoid repeating the same example twice in section three?",
        "Is the tone appropriate for a formal legal context?",
        "Does the response explicitly handle an empty input edge case?",
        "Are units consistent throughout the answer?",
        "Does the response avoid unsupported future predictions?",
    ]


def _all_tie_questions() -> list[str]:
    return [f"Generic question {i + 1} about helpfulness?" for i in range(6)]


def _orthogonal_encoder(questions: list[str]) -> _StubEncoder:
    n = len(questions)
    basis = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return _StubEncoder({q: basis[i] for i, q in enumerate(questions)})


def _low_div_encoder(questions: list[str]) -> _StubEncoder:
    v1 = [1.0, 0.0]
    v2 = [0.4, (1.0 - 0.4**2) ** 0.5]
    mapping = {q: v1 for q in questions[:5]}
    mapping[questions[5]] = v2
    return _StubEncoder(mapping)


def build_synthetic_fixtures() -> list[dict[str, Any]]:
    distinct = _distinct_questions()
    all_tie = _all_tie_questions()
    return [
        {
            "name": "parse_fail",
            "completion": "<think>I will compare.</think>\n\nThe winner is A.",
            "gold": "A",
            "encoder": _StubEncoder(),
            "expected_R": -0.30,
        },
        {
            "name": "perfect",
            "completion": _make_completion(distinct, ["A"] * 8, "A"),
            "gold": "A",
            "encoder": _orthogonal_encoder(distinct),
            "expected_R": 1.00,
        },
        {
            "name": "all_tie_correct_winner",
            "completion": _make_completion(all_tie, ["Tie"] * 6, "Tie"),
            "gold": "Tie",
            "encoder": _low_div_encoder(all_tie),
            "expected_R": 0.73,
        },
        {
            "name": "wrong_winner_perfect_proxy",
            "completion": _make_completion(distinct, ["A"] * 8, "A"),
            "gold": "B",
            "encoder": _orthogonal_encoder(distinct),
            "expected_R": -0.20,
        },
    ]


def score_one(fn: JudgeSelfCheckQuality, completion: str, gold: str) -> dict[str, float]:
    parsed = parse_self_checklist_trace(completion)
    p_ok = _parse_ok_score(parsed)
    disc = _discriminative_score(parsed)
    encoder = _DiversityScorer.get()
    div = _diversity_score(parsed, encoder) if p_ok > 0 else 0.0
    win = _winner_score(completion, gold, parsed)
    reward = fn(completions=[completion], winner=[gold])[0]
    return {
        "parse_ok": p_ok,
        "discriminative": disc,
        "diversity": div,
        "winner": win,
        "R": reward,
    }


def run_synthetic(out_csv: Path) -> bool:
    rows = []
    all_ok = True
    for fx in build_synthetic_fixtures():
        _DiversityScorer._model = fx["encoder"]
        fn = JudgeSelfCheckQuality()
        components = score_one(fn, fx["completion"], fx["gold"])
        delta = abs(components["R"] - fx["expected_R"])
        ok = delta <= 0.01
        rows.append(
            {
                "name": fx["name"],
                "expected_R": fx["expected_R"],
                **components,
                "delta": delta,
                "pass": ok,
            }
        )
        log.info(
            "[synth] %-30s R=%+.4f expected=%+.2f delta=%.4f pass=%s",
            fx["name"],
            components["R"],
            fx["expected_R"],
            delta,
            ok,
        )
        all_ok = all_ok and ok
    _DiversityScorer._model = None
    _write_csv(out_csv, rows)
    return all_ok


def _write_csv(out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", out_csv)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data", [])
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def _extract_from_messages(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for msg in reversed(value):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    if value and isinstance(value[-1], dict):
        return str(value[-1].get("content") or "")
    return ""


def _infer_col(rows: list[dict[str, Any]], requested: str, candidates: list[str]) -> str:
    if requested != "auto":
        return requested
    keys = set(rows[0].keys())
    for cand in candidates:
        if cand in keys:
            return cand
    if "messages" in keys:
        return "__messages_last__"
    raise KeyError(f"Could not infer column from candidates={candidates}. Available={sorted(keys)}")


def _get_value(row: dict[str, Any], col: str) -> Any:
    if col == "__messages_last__":
        return _extract_from_messages(row.get("messages"))
    return row.get(col)


def run_real(
    data_path: Path,
    n: int,
    out_csv: Path,
    *,
    completion_col: str,
    winner_col: str,
    min_parse_ok: float,
    min_std: float,
    min_winner_acc: float,
    max_mean: float,
) -> bool:
    records = _read_records(data_path)
    if not records:
        raise RuntimeError(f"No rows found in {data_path}")
    completion_col = _infer_col(
        records,
        completion_col,
        ["completion", "target", "target_output", "output", "response", "raw_output"],
    )
    winner_col = _infer_col(records, winner_col, ["winner", "gold_winner", "winner_label"])
    fn = JudgeSelfCheckQuality()
    rows = []
    for row in records[:n]:
        comp = str(_get_value(row, completion_col) or "")
        gold = str(_get_value(row, winner_col) or "")
        parsed = parse_self_checklist_trace(comp)
        comps = score_one(fn, comp, gold)
        rows.append(
            {
                "completion_preview": comp[:80].replace("\n", " "),
                "gold": gold,
                "pred": parsed.get("winner"),
                **comps,
                "n_questions": parsed.get("n_questions", 0),
            }
        )
    _write_csv(out_csv, rows)

    rewards = [r["R"] for r in rows]
    parse_ok_rate = sum(1 for r in rows if r["parse_ok"] == 1.0) / len(rows)
    winner_acc = sum(1 for r in rows if r["winner"] == 1.0) / len(rows)
    tie_rate = sum(1 for r in rows if r["pred"] == "Tie") / len(rows)
    ab_rate = sum(1 for r in rows if r["pred"] in ("A", "B")) / len(rows)
    reward_mean = statistics.mean(rewards)
    reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    log.info(
        "[real] n=%d R_mean=%+.4f R_std=%.4f parse_ok=%.3f winner_acc=%.3f ab_rate=%.3f tie_rate=%.3f cols=(%s,%s)",
        len(rows),
        reward_mean,
        reward_std,
        parse_ok_rate,
        winner_acc,
        ab_rate,
        tie_rate,
        completion_col,
        winner_col,
    )

    fails = []
    if parse_ok_rate < min_parse_ok:
        fails.append(f"parse_ok_rate {parse_ok_rate:.3f} < {min_parse_ok:.3f}")
    if reward_std < min_std:
        fails.append(f"R_std {reward_std:.4f} < {min_std:.4f}")
    if winner_acc < min_winner_acc:
        fails.append(f"winner_acc {winner_acc:.3f} < {min_winner_acc:.3f}")
    if reward_mean > max_mean:
        fails.append(f"R_mean {reward_mean:+.4f} > {max_mean:+.4f} (saturated)")
    if ab_rate == 0.0:
        fails.append("ab_rate is 0.0")
    for msg in fails:
        log.error("[real] FAIL: %s", msg)
    return not fails


def run_encoder_smoke() -> bool:
    _DiversityScorer._model = None
    encoder = _DiversityScorer.get()
    emb = encoder.encode(["hello world"], normalize_embeddings=True)
    shape = tuple(np.asarray(emb).shape)
    ok = shape == (1, 384)
    log.info("[encoder] shape=%s pass=%s", shape, ok)
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--encoder-smoke", action="store_true")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--completion-col", default="auto")
    parser.add_argument("--winner-col", default="auto")
    parser.add_argument("--min-parse-ok", type=float, default=0.70)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--min-winner-acc", type=float, default=0.35)
    parser.add_argument("--max-mean", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    t0 = time.time()
    ok = True
    if args.synthetic:
        ok = run_synthetic(args.out) and ok
    if args.encoder_smoke:
        ok = run_encoder_smoke() and ok
    if args.data is not None:
        ok = run_real(
            args.data,
            args.n,
            args.out,
            completion_col=args.completion_col,
            winner_col=args.winner_col,
            min_parse_ok=args.min_parse_ok,
            min_std=args.min_std,
            min_winner_acc=args.min_winner_acc,
            max_mean=args.max_mean,
        ) and ok
    log.info("Total wall time: %.2fs", time.time() - t0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
