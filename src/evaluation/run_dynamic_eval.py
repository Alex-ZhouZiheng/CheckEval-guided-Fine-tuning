#!/usr/bin/env python3
"""Dynamic checklist evaluation with top-k selection and escalation policies.

Policies:
- static_v3
- random_k
- domain_fixed_k
- learned_topk
- learned_topk_escalate
- learned_topk_fallback

Usage:
    python src/evaluation/run_dynamic_eval.py \
        --selector results/checkpoints/selector_v1 \
        --bank checklists/v3_frozen \
        --split dev_600 --policy learned_topk --k 20 \
        --out results/dynamic_dev_600/p3_k20
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import ast
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
import yaml

import config as cfg
from evaluation.selector_infer import (
    active_qids_for_domain,
    build_sample_texts,
    load_eval_pairs,
    load_selector_bundle,
    score_samples_with_bundle,
    select_topk_with_quota,
)
from utils import (
    aggregate_checklist_score,
    build_pointwise_prompt_from_qids,
    compare_checklists_pairwise,
    compute_metrics,
    generate_batch,
    load_judge_model,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _winner_from_margin(margin: float, tie_delta: float) -> str:
    if margin > tie_delta:
        return "A"
    if margin < -tie_delta:
        return "B"
    return "Tie"


def _load_bank(bank_dir: Path) -> pd.DataFrame:
    path = bank_dir / "bank_index.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_parquet(path).sort_values("qid", kind="stable").reset_index(drop=True)

    needed = {"qid", "dimension", "question_text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"bank_index missing columns: {sorted(missing)}")

    if "definition" not in df.columns:
        df["definition"] = ""
    return df


def _load_definitions(bank_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for yaml_path in sorted(bank_dir.glob("*_filtered.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dim = data.get("dimension", yaml_path.stem)
        out[dim] = data.get("definition", "")
    return out


def _build_prompt_from_qids(
    row: pd.Series,
    qids: list[int],
    qmeta: dict[int, dict[str, str]],
    side: str,
) -> str:
    return build_pointwise_prompt_from_qids(row=row, qids=qids, qmeta=qmeta, side=side)


def _load_dotenv_if_available() -> Path | None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return None

    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return None

    load_dotenv(dotenv_path, override=False)
    return Path(dotenv_path)


def _http_judge_generate(
    messages_list: list[list[dict[str, str]]],
    url: str,
    model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    concurrency: int = 32,
    extra_body_mode: str = "qwen-thinking-off",
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key=api_key)
    extra_body = None
    if extra_body_mode == "qwen-thinking-off":
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif extra_body_mode == "deepseek-thinking-on":
        extra_body = {"thinking": {"type": "enabled"}}
    elif extra_body_mode == "deepseek-thinking-off":
        extra_body = {"thinking": {"type": "disabled"}}
    elif extra_body_mode != "none":
        raise ValueError(f"Unknown HTTP extra body mode: {extra_body_mode}")

    def dump_obj(obj: Any) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    def one(msgs: list[dict[str, str]]) -> dict[str, Any]:
        kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", "") or ""
        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "finish_reason": getattr(choice, "finish_reason", None),
            "content_len": len(content),
            "reasoning_len": len(reasoning_content),
            "usage": dump_obj(getattr(resp, "usage", None)),
            "message": dump_obj(msg),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(
            tqdm(ex.map(one, messages_list), total=len(messages_list), desc="Judge HTTP")
        )


def _local_judge_response(raw: str) -> dict[str, Any]:
    return {
        "content": raw,
        "reasoning_content": "",
        "finish_reason": None,
        "content_len": len(raw),
        "reasoning_len": 0,
        "usage": None,
        "message": None,
    }


def _generate_local(
    model,
    all_messages: list[list[dict[str, str]]],
    batch_size: int,
    max_new_tokens: int,
    prompt_chunk_size: int,
    seed: int | None,
) -> list[str]:
    out: list[str] = []
    for start in range(0, len(all_messages), prompt_chunk_size):
        chunk = all_messages[start : start + prompt_chunk_size]
        out.extend(
            generate_batch(
                model,
                chunk,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        )
    return out


def _save_raw_output_debug(
    raw_outputs: dict[str, Any],
    prefix: str,
    response: dict[str, Any],
) -> None:
    raw_outputs[prefix] = response.get("content", "")
    raw_outputs[f"{prefix}_reasoning_content"] = response.get("reasoning_content", "")
    raw_outputs[f"{prefix}_finish_reason"] = response.get("finish_reason")
    raw_outputs[f"{prefix}_content_len"] = response.get("content_len")
    raw_outputs[f"{prefix}_reasoning_len"] = response.get("reasoning_len")
    raw_outputs[f"{prefix}_usage"] = json.dumps(
        response.get("usage"), ensure_ascii=False, default=str
    )


def _parse_stage_labels(raw: str, qids: list[int]) -> tuple[dict[int, str], dict]:
    parsed = parse_checkeval_output(raw, expected_n=len(qids))
    labels: dict[int, str] = {}

    for a in parsed.get("answers", []):
        local_q = int(a["q"])
        if 1 <= local_q <= len(qids):
            labels[qids[local_q - 1]] = str(a["answer"])

    for a in parsed.get("na_answers", []):
        local_q = int(a["q"])
        if 1 <= local_q <= len(qids):
            labels[qids[local_q - 1]] = "na"

    return labels, parsed


def _labels_to_parsed(labels: dict[int, str], asked_qids: list[int]) -> dict[str, Any]:
    """Convert global-qid labels into the parsed checklist shape used by utils."""
    answers: list[dict[str, Any]] = []
    na_answers: list[dict[str, Any]] = []
    for local_q, qid in enumerate(asked_qids, 1):
        label = str(labels.get(qid, "na")).lower()
        if label == "na":
            na_answers.append({"q": local_q})
        elif label in {"yes", "no"}:
            answers.append({"q": local_q, "answer": label})
        else:
            na_answers.append({"q": local_q})

    n_yes = sum(1 for a in answers if a["answer"] == "yes")
    n_no = sum(1 for a in answers if a["answer"] == "no")
    n_answered = len(answers)
    n_na = len(na_answers)
    expected_n = len(asked_qids)

    return {
        "answers": answers,
        "na_answers": na_answers,
        "n_questions_parsed": n_answered,
        "n_yes": n_yes,
        "n_no": n_no,
        "n_na": n_na,
        "na_qnums": [a["q"] for a in na_answers],
        "score": n_yes / n_answered if n_answered else 0.0,
        "complete": n_answered == expected_n and n_na == 0,
        "complete_with_na": (n_answered + n_na) == expected_n,
    }


def _score_checklists(
    labels_a: dict[int, str],
    labels_b: dict[int, str],
    asked_qids: list[int],
    *,
    score_method: str,
    tie_delta: float,
    aggregate_na_policy: str,
    aggregate_coverage_threshold: float,
) -> dict[str, Any]:
    if not asked_qids:
        return {
            "margin": 0.0,
            "winner": "Tie",
            "n_aligned": 0,
            "score_a": None,
            "score_b": None,
        }

    parsed_a = _labels_to_parsed(labels_a, asked_qids)
    parsed_b = _labels_to_parsed(labels_b, asked_qids)
    expected_n = len(asked_qids)

    if score_method == "compare_checklists_pairwise":
        cmp = compare_checklists_pairwise(
            parsed_a,
            parsed_b,
            expected_n=expected_n,
            tie_delta=tie_delta,
        )
        if cmp is None:
            return {
                "margin": 0.0,
                "winner": "Tie",
                "n_aligned": 0,
                "score_a": None,
                "score_b": None,
            }
        return {
            "margin": float(cmp["margin"]),
            "winner": cmp["winner"],
            "n_aligned": int(cmp["n_aligned"]),
            "score_a": None,
            "score_b": None,
        }

    if score_method == "aggregate_checklist_score":
        agg_a = aggregate_checklist_score(
            parsed_a,
            na_policy=aggregate_na_policy,
            coverage_threshold=aggregate_coverage_threshold,
            expected_n=expected_n,
        )
        agg_b = aggregate_checklist_score(
            parsed_b,
            na_policy=aggregate_na_policy,
            coverage_threshold=aggregate_coverage_threshold,
            expected_n=expected_n,
        )
        n_aligned = sum(
            1
            for qid in asked_qids
            if str(labels_a.get(qid, "na")).lower() != "na"
            and str(labels_b.get(qid, "na")).lower() != "na"
        )
        if agg_a is None or agg_b is None:
            return {
                "margin": 0.0,
                "winner": "Tie",
                "n_aligned": int(n_aligned),
                "score_a": None,
                "score_b": None,
            }

        score_a = float(agg_a["score"])
        score_b = float(agg_b["score"])
        margin = score_a - score_b
        return {
            "margin": float(margin),
            "winner": _winner_from_margin(float(margin), tie_delta=tie_delta),
            "n_aligned": int(n_aligned),
            "score_a": score_a,
            "score_b": score_b,
        }

    raise ValueError(f"unknown score_method: {score_method}")


def _run_stage(
    stage_name: str,
    df_by_sid: dict[str, pd.Series],
    qids_by_sid: dict[str, list[int]],
    qmeta: dict[int, dict[str, str]],
    judge_mode: str,
    model,
    batch_size: int,
    max_new_tokens: int,
    prompt_chunk_size: int,
    seed: int,
    judge_url: str,
    judge_model: str | None,
    judge_api_key: str,
    http_concurrency: int,
    http_extra_body: str,
    http_reasoning_effort: str | None,
) -> tuple[dict[str, dict[str, Any]], float]:
    metas: list[dict[str, Any]] = []
    messages_a: list[list[dict[str, str]]] = []
    messages_b: list[list[dict[str, str]]] = []

    for sid, qids in qids_by_sid.items():
        if not qids:
            continue
        row = df_by_sid[sid]
        pa = _build_prompt_from_qids(row, qids, qmeta, side="A")
        pb = _build_prompt_from_qids(row, qids, qmeta, side="B")

        messages_a.append([{"role": "user", "content": pa}])
        messages_b.append([{"role": "user", "content": pb}])
        metas.append({"sid": sid, "qids": qids})

    if not metas:
        return {}, 0.0

    all_messages = messages_a + messages_b
    t0 = time.time()
    if judge_mode == "http":
        if not judge_model:
            raise SystemExit("--judge-model is required for --judge-mode http")
        responses = _http_judge_generate(
            all_messages,
            url=judge_url,
            model=judge_model,
            api_key=judge_api_key,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            concurrency=http_concurrency,
            extra_body_mode=http_extra_body,
            reasoning_effort=http_reasoning_effort,
        )
    else:
        raw = _generate_local(
            model=model,
            all_messages=all_messages,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            prompt_chunk_size=prompt_chunk_size,
            seed=seed,
        )
        responses = [_local_judge_response(x) for x in raw]
    elapsed = time.time() - t0

    n = len(metas)
    raw_a = responses[:n]
    raw_b = responses[n:]

    out: dict[str, dict[str, Any]] = {}
    for meta, resp_a, resp_b in zip(metas, raw_a, raw_b):
        sid = meta["sid"]
        qids = meta["qids"]
        r_a = str(resp_a.get("content", ""))
        r_b = str(resp_b.get("content", ""))

        labels_a, parsed_a = _parse_stage_labels(r_a, qids)
        labels_b, parsed_b = _parse_stage_labels(r_b, qids)

        out[sid] = {
            "labels_a": labels_a,
            "labels_b": labels_b,
            "parse_ok": (not parsed_a.get("_raw_fallback")) and (not parsed_b.get("_raw_fallback")),
            "raw_a": r_a,
            "raw_b": r_b,
            "resp_a": resp_a,
            "resp_b": resp_b,
            "stage": stage_name,
        }

    return out, elapsed


def _stable_shuffle(values: list[int], seed: int, key: str) -> list[int]:
    payload = f"{seed}:{key}".encode("utf-8")
    sid_seed = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    rng = __import__("random").Random(sid_seed)
    arr = list(values)
    rng.shuffle(arr)
    return arr


def _build_domain_fixed_rankings(
    oracle_train_path: Path,
    bank_df: pd.DataFrame,
) -> dict[str, list[int]]:
    if not oracle_train_path.exists():
        raise FileNotFoundError(
            f"{oracle_train_path} not found. Provide --oracle-train for domain_fixed_k policy."
        )

    df = pd.read_parquet(oracle_train_path)
    required = {"domain", "qid", "u2_abs_contrib"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{oracle_train_path} missing columns: {sorted(missing)}")

    mean_u2 = (
        df.groupby(["domain", "qid"], as_index=False)["u2_abs_contrib"]
        .mean()
        .rename(columns={"u2_abs_contrib": "mean_u2"})
    )

    out: dict[str, list[int]] = {}
    for domain in sorted(mean_u2["domain"].astype(str).unique().tolist()):
        active = active_qids_for_domain(bank_df, domain)
        scores = mean_u2[mean_u2["domain"] == domain].set_index("qid")["mean_u2"].to_dict()
        ranked = sorted(active, key=lambda q: (scores.get(q, 0.0), -q), reverse=True)
        out[domain] = ranked

    return out


def _parse_qid_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = ast.literal_eval(text)
        return _parse_qid_list(parsed)
    raise TypeError(f"Unsupported qid list value: {type(value).__name__}")


def _load_selector_picks(path: Path, df: pd.DataFrame) -> dict[str, list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)

    picks = pd.read_parquet(path)
    if "sample_id" not in picks.columns:
        raise ValueError(f"{path} missing required column: sample_id")

    qid_col = "ranked_qids" if "ranked_qids" in picks.columns else "selected_qids"
    if qid_col not in picks.columns:
        raise ValueError(f"{path} must contain ranked_qids or selected_qids")

    pick_map = {
        str(row["sample_id"]): _parse_qid_list(row[qid_col])
        for _, row in picks.drop_duplicates(subset=["sample_id"], keep="first").iterrows()
    }

    missing = [str(sid) for sid in df["sample_id"].astype(str).tolist() if str(sid) not in pick_map]
    if missing:
        raise ValueError(
            f"{path} is missing selector picks for {len(missing)} eval samples "
            f"(first 5: {missing[:5]})"
        )

    log.info("Loaded selector picks from %s using %s", path, qid_col)
    return pick_map


def compute_human_alignment(
    pred_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> dict[str, float | int | None]:
    """Recall of selected/asked qids against the human yes-set ({qid: h > 0})."""
    yes_by_sample: dict[str, set[int]] = {}
    for _, row in human_df.iterrows():
        if float(row["h"]) > 0:
            yes_by_sample.setdefault(str(row["sample_id"]), set()).add(int(row["qid"]))

    recalls_selected: list[float] = []
    recalls_asked: list[float] = []
    for _, row in pred_df.iterrows():
        sid = str(row["sample_id"])
        yes_set = yes_by_sample.get(sid)
        if not yes_set:
            continue

        selected = set(_parse_qid_list(row.get("selected_qids")))
        asked = set(_parse_qid_list(row.get("asked_qids")))
        recalls_selected.append(len(selected & yes_set) / len(yes_set))
        recalls_asked.append(len(asked & yes_set) / len(yes_set))

    return {
        "recall_human_selected": (
            float(sum(recalls_selected) / len(recalls_selected)) if recalls_selected else None
        ),
        "recall_human_asked": (
            float(sum(recalls_asked) / len(recalls_asked)) if recalls_asked else None
        ),
        "n_evaluated": int(len(recalls_selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=[
            "static_v3",
            "random_k",
            "domain_fixed_k",
            "learned_topk",
            "learned_topk_escalate",
            "learned_topk_fallback",
        ],
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--selector", type=Path, default=None)
    parser.add_argument(
        "--selector-picks",
        type=Path,
        default=None,
        help="Optional selector_infer parquet with ranked_qids/selected_qids; skips loading selector checkpoint.",
    )
    parser.add_argument("--oracle-train", type=Path, default=Path("data/oracle/train_oracle_v3.parquet"))
    parser.add_argument(
        "--human-relevance",
        type=Path,
        default=None,
        help=(
            "Optional human-relevance parquet; when set, emit "
            "recall_human_{selected,asked} in metrics.json."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument(
        "--score-method",
        choices=["compare_checklists_pairwise", "aggregate_checklist_score"],
        default="compare_checklists_pairwise",
        help=(
            "How to convert parsed checklist answers into A/B winner. "
            "compare_checklists_pairwise uses utils.compare_checklists_pairwise; "
            "aggregate_checklist_score uses utils.aggregate_checklist_score for each side."
        ),
    )
    parser.add_argument(
        "--aggregate-na-policy",
        choices=["strict", "as_no", "skip", "partial"],
        default="as_no",
        help="N/A policy used only when --score-method aggregate.",
    )
    parser.add_argument(
        "--aggregate-coverage-threshold",
        type=float,
        default=0.8,
        help="Coverage threshold used only with --score-method aggregate --aggregate-na-policy partial.",
    )
    parser.add_argument("--tie-delta", type=float, default=0.05)
    parser.add_argument("--tau-escalate", type=float, default=0.05)
    parser.add_argument("--tau-hard", type=float, default=0.03)
    parser.add_argument("--n-min", type=int, default=8)

    parser.add_argument("--no-dim-quota", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-chunk-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--selector-batch-size", type=int, default=64)
    parser.add_argument("--selector-device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    parser.add_argument("--judge-mode", choices=["local", "http"], default="local")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["llamacpp", "vllm"],
                        help="Inference backend for local judge; defaults to cfg.INFERENCE_BACKEND.")
    parser.add_argument("--judge-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default=None,
        help="Read the HTTP judge API key from this environment variable.",
    )
    parser.add_argument("--http-concurrency", type=int, default=32)
    parser.add_argument(
        "--http-extra-body",
        choices=["qwen-thinking-off", "deepseek-thinking-on", "deepseek-thinking-off", "none"],
        default="qwen-thinking-off",
        help=(
            "Extra OpenAI-compatible request body for HTTP judge. "
            "Use 'deepseek-thinking-off' for DeepSeek V4."
        ),
    )
    parser.add_argument(
        "--http-reasoning-effort",
        choices=["high", "max"],
        default=None,
        help="DeepSeek V4 reasoning effort when thinking mode is enabled.",
    )
    parser.add_argument("--base-model", type=str, default=str(cfg.JUDGE_MODEL_ID))

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"],
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_model_len"],
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"],
    )
    parser.add_argument("--save-raw-outputs", action="store_true")
    args = parser.parse_args()

    dotenv_path = _load_dotenv_if_available()
    if dotenv_path is not None:
        log.info("Loaded environment variables from %s", dotenv_path)

    bank_dir = args.bank.resolve()
    judge_api_key = args.judge_api_key
    if args.judge_api_key_env:
        judge_api_key = _os.environ.get(args.judge_api_key_env)
        if not judge_api_key:
            raise SystemExit(f"{args.judge_api_key_env} is not set")
    elif args.judge_mode == "http" and "deepseek.com" in args.judge_url and judge_api_key == "EMPTY":
        judge_api_key = _os.environ.get("DEEPSEEK_API_KEY")
        if not judge_api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY is not set. Put it in .env, export it, or pass --judge-api-key."
            )

    bank_df = _load_bank(bank_dir)
    _ = _load_definitions(bank_dir)
    qmeta = {
        int(r["qid"]): {
            "dimension": str(r["dimension"]),
            "question_text": str(r["question_text"]),
            "definition": str(r.get("definition", "") or ""),
        }
        for _, r in bank_df.iterrows()
    }

    df = load_eval_pairs(
        split=args.split,
        subset=args.subset,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    df_by_sid = {str(r["sample_id"]): r for _, r in df.iterrows()}

    # Ranking candidates per sample.
    ranking_by_sid: dict[str, list[int]] = {}
    selector_elapsed = 0.0

    if args.policy in {"learned_topk", "learned_topk_escalate", "learned_topk_fallback"}:
        if args.selector_picks is not None:
            ranking_by_sid = _load_selector_picks(args.selector_picks.resolve(), df)
        else:
            if args.selector is None:
                raise SystemExit("--selector or --selector-picks is required for learned policies")

            import torch

            selector_device = torch.device(args.selector_device)
            bundle = load_selector_bundle(args.selector, selector_device)
            texts = build_sample_texts(df)

            t0_sel = time.time()
            score_matrix = score_samples_with_bundle(bundle, texts, batch_size=args.selector_batch_size)
            selector_elapsed = time.time() - t0_sel

            for i, row in df.iterrows():
                sid = str(row["sample_id"])
                active = active_qids_for_domain(bank_df, str(row["domain"]))
                active = [q for q in active if q in bundle.qid_to_idx]
                if not active:
                    active = active_qids_for_domain(bank_df, str(row["domain"]))
                    ranking_by_sid[sid] = active
                    continue

                idx = torch.tensor([bundle.qid_to_idx[q] for q in active], dtype=torch.long)
                s = score_matrix[i, idx]
                order = torch.argsort(s, descending=True)
                ranking_by_sid[sid] = [active[int(j)] for j in order.tolist()]

    elif args.policy == "random_k":
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            active = active_qids_for_domain(bank_df, str(row["domain"]))
            ranking_by_sid[sid] = _stable_shuffle(active, args.seed, sid)

    elif args.policy == "domain_fixed_k":
        domain_rankings = _build_domain_fixed_rankings(args.oracle_train.resolve(), bank_df)
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            domain = str(row["domain"])
            active = active_qids_for_domain(bank_df, domain)
            ranked = domain_rankings.get(domain, active)
            ranked = [q for q in ranked if q in set(active)]
            leftover = [q for q in active if q not in set(ranked)]
            ranking_by_sid[sid] = ranked + leftover

    else:  # static_v3
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            ranking_by_sid[sid] = active_qids_for_domain(bank_df, str(row["domain"]))

    # State per sample.
    state: dict[str, dict[str, Any]] = {}
    stage_latency_s: dict[str, float] = {str(r["sample_id"]): 0.0 for _, r in df.iterrows()}

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        domain = str(row["domain"])
        active = active_qids_for_domain(bank_df, domain)
        ranking = ranking_by_sid[sid]

        ranking = [q for q in ranking if q in set(active)]
        ranking += [q for q in active if q not in set(ranking)]

        if args.policy == "static_v3":
            initial = list(ranking)
        else:
            initial = select_topk_with_quota(
                ranked_active_qids=ranking,
                bank_df=bank_df,
                domain=domain,
                k=args.k,
                enforce_quota=(not args.no_dim_quota),
            )

        state[sid] = {
            "ranking": ranking,
            "active_qids": active,
            "initial_qids": list(initial),
            "asked_qids": [],
            "labels_a": {},
            "labels_b": {},
            "parse_ok_all": True,
            "escalated": False,
            "fallback": False,
            "margin_initial": None,
            "n_aligned_initial": None,
            "score_a_initial": None,
            "score_b_initial": None,
            "margin_final": None,
            "n_aligned_final": None,
            "score_a_final": None,
            "score_b_final": None,
            "winner_final": "Tie",
            "raw_outputs": {},
        }

    # Judge model (single load for local mode).
    model = None
    if args.judge_mode == "local":
        model = load_judge_model(
            model_id=args.base_model,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    # Stage 1: initial top-k (or static full bank).
    stage1_q = {sid: st["initial_qids"] for sid, st in state.items()}
    stage1_out, t_stage1 = _run_stage(
        stage_name="stage1",
        df_by_sid=df_by_sid,
        qids_by_sid=stage1_q,
        qmeta=qmeta,
        judge_mode=args.judge_mode,
        model=model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt_chunk_size=args.prompt_chunk_size,
        seed=args.seed,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        judge_api_key=judge_api_key,
        http_concurrency=args.http_concurrency,
        http_extra_body=args.http_extra_body,
        http_reasoning_effort=args.http_reasoning_effort,
    )
    if stage1_out:
        stage1_avg = t_stage1 / len(stage1_out)
        for sid, result in stage1_out.items():
            st = state[sid]
            st["asked_qids"].extend(st["initial_qids"])
            st["labels_a"].update(result["labels_a"])
            st["labels_b"].update(result["labels_b"])
            st["parse_ok_all"] = st["parse_ok_all"] and result["parse_ok"]
            stage_latency_s[sid] += stage1_avg
            if args.save_raw_outputs:
                _save_raw_output_debug(st["raw_outputs"], "stage1_a", result["resp_a"])
                _save_raw_output_debug(st["raw_outputs"], "stage1_b", result["resp_b"])

    for sid, st in state.items():
        cmp = _score_checklists(
            st["labels_a"],
            st["labels_b"],
            st["asked_qids"],
            score_method=args.score_method,
            tie_delta=args.tie_delta,
            aggregate_na_policy=args.aggregate_na_policy,
            aggregate_coverage_threshold=args.aggregate_coverage_threshold,
        )
        st["margin_initial"] = cmp["margin"]
        st["n_aligned_initial"] = cmp["n_aligned"]
        st["score_a_initial"] = cmp["score_a"]
        st["score_b_initial"] = cmp["score_b"]
        st["margin_final"] = cmp["margin"]
        st["n_aligned_final"] = cmp["n_aligned"]
        st["score_a_final"] = cmp["score_a"]
        st["score_b_final"] = cmp["score_b"]
        st["winner_final"] = cmp["winner"]

    # Stage 2: escalation
    do_escalate = args.policy in {"learned_topk_escalate", "learned_topk_fallback"}
    stage2_q: dict[str, list[int]] = {}
    if do_escalate:
        for sid, st in state.items():
            need = (abs(float(st["margin_final"])) < args.tau_escalate) or (
                int(st["n_aligned_final"]) < args.n_min
            )
            if not need:
                continue
            remaining = [q for q in st["ranking"] if q not in set(st["asked_qids"])]
            if not remaining:
                continue
            add_q = remaining[: args.k]
            stage2_q[sid] = add_q
            st["escalated"] = True

    if stage2_q:
        stage2_out, t_stage2 = _run_stage(
            stage_name="stage2_escalate",
            df_by_sid=df_by_sid,
            qids_by_sid=stage2_q,
            qmeta=qmeta,
            judge_mode=args.judge_mode,
            model=model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            prompt_chunk_size=args.prompt_chunk_size,
            seed=args.seed,
            judge_url=args.judge_url,
            judge_model=args.judge_model,
            judge_api_key=judge_api_key,
            http_concurrency=args.http_concurrency,
            http_extra_body=args.http_extra_body,
            http_reasoning_effort=args.http_reasoning_effort,
        )

        stage2_avg = t_stage2 / len(stage2_out) if stage2_out else 0.0
        for sid, result in stage2_out.items():
            st = state[sid]
            st["asked_qids"].extend(stage2_q[sid])
            st["labels_a"].update(result["labels_a"])
            st["labels_b"].update(result["labels_b"])
            st["parse_ok_all"] = st["parse_ok_all"] and result["parse_ok"]
            stage_latency_s[sid] += stage2_avg
            if args.save_raw_outputs:
                _save_raw_output_debug(st["raw_outputs"], "stage2_a", result["resp_a"])
                _save_raw_output_debug(st["raw_outputs"], "stage2_b", result["resp_b"])

            cmp = _score_checklists(
                st["labels_a"],
                st["labels_b"],
                st["asked_qids"],
                score_method=args.score_method,
                tie_delta=args.tie_delta,
                aggregate_na_policy=args.aggregate_na_policy,
                aggregate_coverage_threshold=args.aggregate_coverage_threshold,
            )
            st["margin_final"] = cmp["margin"]
            st["n_aligned_final"] = cmp["n_aligned"]
            st["score_a_final"] = cmp["score_a"]
            st["score_b_final"] = cmp["score_b"]
            st["winner_final"] = cmp["winner"]

    # Stage 3: hard fallback to full bank.
    do_fallback = args.policy == "learned_topk_fallback"
    stage3_q: dict[str, list[int]] = {}
    if do_fallback:
        for sid, st in state.items():
            if abs(float(st["margin_final"])) >= args.tau_hard:
                continue
            remaining = [q for q in st["active_qids"] if q not in set(st["asked_qids"])]
            if not remaining:
                continue
            stage3_q[sid] = remaining
            st["fallback"] = True

    if stage3_q:
        stage3_out, t_stage3 = _run_stage(
            stage_name="stage3_fallback",
            df_by_sid=df_by_sid,
            qids_by_sid=stage3_q,
            qmeta=qmeta,
            judge_mode=args.judge_mode,
            model=model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            prompt_chunk_size=args.prompt_chunk_size,
            seed=args.seed,
            judge_url=args.judge_url,
            judge_model=args.judge_model,
            judge_api_key=judge_api_key,
            http_concurrency=args.http_concurrency,
            http_extra_body=args.http_extra_body,
            http_reasoning_effort=args.http_reasoning_effort,
        )

        stage3_avg = t_stage3 / len(stage3_out) if stage3_out else 0.0
        for sid, result in stage3_out.items():
            st = state[sid]
            st["asked_qids"].extend(stage3_q[sid])
            st["labels_a"].update(result["labels_a"])
            st["labels_b"].update(result["labels_b"])
            st["parse_ok_all"] = st["parse_ok_all"] and result["parse_ok"]
            stage_latency_s[sid] += stage3_avg
            if args.save_raw_outputs:
                _save_raw_output_debug(st["raw_outputs"], "stage3_a", result["resp_a"])
                _save_raw_output_debug(st["raw_outputs"], "stage3_b", result["resp_b"])

            cmp = _score_checklists(
                st["labels_a"],
                st["labels_b"],
                st["asked_qids"],
                score_method=args.score_method,
                tie_delta=args.tie_delta,
                aggregate_na_policy=args.aggregate_na_policy,
                aggregate_coverage_threshold=args.aggregate_coverage_threshold,
            )
            st["margin_final"] = cmp["margin"]
            st["n_aligned_final"] = cmp["n_aligned"]
            st["score_a_final"] = cmp["score_a"]
            st["score_b_final"] = cmp["score_b"]
            st["winner_final"] = cmp["winner"]

    # selector overhead is shared across all samples.
    if selector_elapsed > 0 and len(state) > 0:
        sel_avg = selector_elapsed / len(state)
        for sid in state:
            stage_latency_s[sid] += sel_avg

    rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    domains: list[str] = []

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        st = state[sid]
        y_true.append(str(row["winner"]))
        y_pred.append(str(st["winner_final"]))
        domains.append(str(row["domain"]))

        r = {
            "sample_id": sid,
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "winner": row["winner"],
            "predicted_winner": st["winner_final"],
            "margin_initial": st["margin_initial"],
            "margin_final": st["margin_final"],
            "n_aligned_initial": st["n_aligned_initial"],
            "n_aligned_final": st["n_aligned_final"],
            "score_a_initial": st["score_a_initial"],
            "score_b_initial": st["score_b_initial"],
            "score_a_final": st["score_a_final"],
            "score_b_final": st["score_b_final"],
            "selected_qids": st["initial_qids"],
            "asked_qids": st["asked_qids"],
            "k_selected": len(st["initial_qids"]),
            "k_after_escalation": len(st["asked_qids"]),
            "escalated": bool(st["escalated"]),
            "fallback": bool(st["fallback"]),
            "parse_ok": bool(st["parse_ok_all"]),
            "latency_s_estimate": float(stage_latency_s[sid]),
        }
        if args.save_raw_outputs:
            r.update(st["raw_outputs"])
        rows.append(r)

    pred_df = pd.DataFrame(rows)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, domains=domains)
    metrics["policy"] = args.policy
    metrics["k"] = args.k
    metrics["score_method"] = args.score_method
    metrics["tie_delta"] = args.tie_delta
    metrics["aggregate_na_policy"] = args.aggregate_na_policy
    metrics["aggregate_coverage_threshold"] = args.aggregate_coverage_threshold
    metrics["tau_escalate"] = args.tau_escalate
    metrics["tau_hard"] = args.tau_hard
    metrics["n_min"] = args.n_min
    metrics["avg_k"] = float(pred_df["k_after_escalation"].mean()) if len(pred_df) else 0.0
    metrics["tie_rate"] = float((pred_df["predicted_winner"] == "Tie").mean()) if len(pred_df) else 0.0
    metrics["parse_ok_rate"] = float(pred_df["parse_ok"].mean()) if len(pred_df) else 0.0
    metrics["selector_inference_time_s"] = selector_elapsed
    metrics["selector_picks"] = str(args.selector_picks) if args.selector_picks else None
    metrics["judge_mode"] = args.judge_mode
    metrics["judge_url"] = args.judge_url if args.judge_mode == "http" else None
    metrics["judge_model"] = args.judge_model if args.judge_mode == "http" else args.base_model
    metrics["http_extra_body"] = args.http_extra_body if args.judge_mode == "http" else None
    metrics["http_reasoning_effort"] = args.http_reasoning_effort if args.judge_mode == "http" else None

    if args.human_relevance is not None:
        if not args.human_relevance.exists():
            raise SystemExit(f"--human-relevance not found: {args.human_relevance}")
        human_df = pd.read_parquet(args.human_relevance)
        required = {"sample_id", "qid", "h"}
        missing = required - set(human_df.columns)
        if missing:
            raise SystemExit(f"human_relevance parquet missing columns: {sorted(missing)}")
        metrics.update(compute_human_alignment(pred_df, human_df))
        metrics["human_relevance"] = str(args.human_relevance)

    if len(pred_df):
        metrics["latency_p50_s"] = float(pred_df["latency_s_estimate"].quantile(0.5))
        metrics["latency_p90_s"] = float(pred_df["latency_s_estimate"].quantile(0.9))
        total_latency = float(pred_df["latency_s_estimate"].sum())
        metrics["samples_per_second_estimate"] = float(len(pred_df) / total_latency) if total_latency > 0 else None

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.parquet"
    metrics_path = out_dir / "metrics.json"
    pred_df.to_parquet(pred_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    log.info("Saved predictions -> %s", pred_path)
    log.info("Saved metrics     -> %s", metrics_path)


if __name__ == "__main__":
    main()
