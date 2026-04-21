#!/usr/bin/env python3
"""
Offline replay-pool audit for the GRPO reward.

Subcommands:
  replay_pool  Build K sampled checklist completions per prompt.
  score        Score a replay pool with the frozen judge + R1 reward.
  analyze      Compute group-level and summary audit metrics.
  perturb      Apply counterfactual perturbations and rescore them.
  run_all      Run replay_pool -> score -> analyze -> perturb end to end.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SRC_ROOT / "data_process") not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT / "data_process"))
if str(_SRC_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT / "evaluation"))

import pandas as pd
from vllm.lora.request import LoRARequest

import config as cfg
from data_process.prepare_generator_sft import (
    build_generator_messages,
    format_checklist_target,
)
from evaluation.run_generator_infer import parse_generated_checklist
from evaluation.run_judge_eval import _http_judge_generate
from train.plugin.checkeval_reward import (
    compute_reward_components,
    get_r1_reward_config,
    gold_winner_from_preference,
    prepare_completion_pointwise_prompts,
    summarize_judge_pair,
)
from utils import compute_metrics, generate_batch, load_judge_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = cfg.RESULTS_DIR / "reward_audit"


def _path_or_none(raw: str | None) -> Path | None:
    return Path(raw).resolve() if raw else None


def _safe_str(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    return text or fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _default_subset(eval_split: str) -> str:
    preferred = [f"grpo_{eval_split}_600.jsonl", "grpo_dev_600.jsonl", f"grpo_{eval_split}.jsonl"]
    for name in preferred:
        if (cfg.GENERATOR_SFT_DIR / name).exists():
            return name.removeprefix("grpo_").removesuffix(".jsonl")
    candidates = sorted(cfg.GENERATOR_SFT_DIR.glob("grpo_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No GRPO source file found under {cfg.GENERATOR_SFT_DIR}."
        )
    return candidates[0].name.removeprefix("grpo_").removesuffix(".jsonl")


def _resolve_subset(eval_split: str, subset: str | None) -> str:
    return subset or _default_subset(eval_split)


def _candidate_generator_source_paths(
    base_dir: Path,
    eval_split: str,
    subset: str,
) -> list[Path]:
    tags: list[str] = []
    for tag in [subset, eval_split, f"{eval_split}_600", "dev_600", "dev"]:
        if tag and tag not in tags:
            tags.append(tag)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for tag in tags:
        for name in (
            f"grpo_{tag}.jsonl",
            f"{tag}.jsonl",
            f"{tag}.parquet",
            f"train_{tag}.parquet",
        ):
            path = base_dir / name
            if path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_source_rows(
    rows: list[dict[str, Any]],
    source_path: Path,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source_path} contains a non-dict record: {type(row)!r}")
        normalized.append(dict(row))

    if not normalized:
        return normalized

    has_messages = any(row.get("messages") is not None for row in normalized)
    has_context = any(row.get("context") is not None for row in normalized)
    has_responses = all(("response_a" in row and "response_b" in row) for row in normalized)
    if not (has_messages or has_context):
        raise ValueError(
            f"{source_path} must provide either a 'messages' field or raw 'context' text."
        )
    if not has_responses:
        raise ValueError(
            f"{source_path} must include 'response_a' and 'response_b' for reward scoring."
        )
    return normalized


def _load_generator_source_rows(
    eval_split: str,
    subset: str,
    max_samples: int | None,
    generator_data_dir: str | None,
) -> tuple[list[dict[str, Any]], Path]:
    base_dir = Path(generator_data_dir).resolve() if generator_data_dir else cfg.GENERATOR_SFT_DIR
    candidate_paths = _candidate_generator_source_paths(base_dir, eval_split, subset)
    source_path = next((path for path in candidate_paths if path.exists()), None)
    if source_path is None:
        searched = "\n".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "No generator-source file found. Tried:\n"
            f"{searched}"
        )

    if source_path.suffix == ".jsonl":
        rows = _load_jsonl_records(source_path)
    elif source_path.suffix == ".parquet":
        rows = pd.read_parquet(source_path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported source format: {source_path}")

    rows = _normalize_source_rows(rows, source_path)
    if max_samples:
        rows = rows[:max_samples]
    return rows, source_path


def _messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = row.get("messages")
    if raw_messages is not None:
        parsed = raw_messages
        if isinstance(raw_messages, str):
            try:
                parsed = json.loads(raw_messages)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, list) and parsed:
            return parsed

    if row.get("context") is None:
        raise ValueError("Source row is missing both 'messages' and 'context'.")
    return build_generator_messages(row)


def _resolve_run_dir(
    output_dir: str | None,
    run_name: str | None,
    path_hint: Path | None = None,
) -> Path:
    if path_hint is not None and output_dir is None and run_name is None:
        return path_hint.resolve().parent
    base_dir = Path(output_dir).resolve() if output_dir else DEFAULT_AUDIT_DIR
    if run_name:
        return base_dir / _sanitize_name(run_name)
    return base_dir / "default_run"


def _read_adapter_rank(adapter_path: Path | None) -> int:
    if adapter_path is None:
        return 16
    with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("r", 16))


def _make_lora_request(adapter_path: Path | None) -> LoRARequest | None:
    if adapter_path is None:
        return None
    return LoRARequest(
        lora_name=adapter_path.name,
        lora_int_id=1,
        lora_path=str(adapter_path),
    )


def _approx_tokens(text: str) -> int:
    return len((text or "").split())


def _load_generator_model(
    model_id: str,
    adapter_path: Path | None,
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    lora_rank = _read_adapter_rank(adapter_path)
    return load_judge_model(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=adapter_path is not None,
        max_lora_rank=max(lora_rank, 16) if adapter_path is not None else None,
        max_loras=1 if adapter_path is not None else None,
    )


class JudgeRunner:
    def __init__(
        self,
        *,
        mode: str,
        url: str,
        model_name: str | None,
        api_key: str,
        base_model: str,
        adapter_path: Path | None,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.mode = mode
        self.url = url
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.adapter_path = adapter_path
        self._local_model = None
        self._local_lora_request = None
        if mode == "local":
            self._local_model = _load_generator_model(
                base_model,
                adapter_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self._local_lora_request = _make_lora_request(adapter_path)

    def generate(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        if not messages_list:
            return []
        if self.mode == "http":
            if not self.model_name:
                raise ValueError("HTTP judge mode requires --judge-model or JUDGE_MODEL.")
            return _http_judge_generate(
                messages_list,
                url=self.url,
                model=self.model_name,
                api_key=self.api_key,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        return generate_batch(
            self._local_model,
            messages_list,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            lora_request=self._local_lora_request,
        )


def _load_eval_rows(
    eval_split: str,
    subset: str,
    max_samples: int | None,
    generator_data_dir: str | None,
) -> tuple[list[dict[str, Any]], Path]:
    return _load_generator_source_rows(eval_split, subset, max_samples, generator_data_dir)


def replay_pool_command(args) -> Path:
    subset = _resolve_subset(args.eval_split, args.subset)
    run_name = args.run_name or f"{subset}_k{args.k}"
    run_dir = _resolve_run_dir(args.output_dir, run_name)
    rows, source_path = _load_eval_rows(
        args.eval_split,
        subset,
        args.max_samples,
        args.generator_data_dir,
    )
    adapter_path = _path_or_none(args.generator_adapter)
    model = _load_generator_model(
        args.generator_base,
        adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    lora_request = _make_lora_request(adapter_path)
    messages_list = [_messages_from_row(row) for row in rows]
    records: list[dict[str, Any]] = []
    seeds = []

    log.info(
        "Building replay pool: subset=%s rows=%d k=%d temp=%.2f source=%s",
        subset, len(rows), args.k, args.temperature, source_path,
    )
    for k_idx in range(args.k):
        seed_value = (args.seed + k_idx) if args.seed is not None else None
        seeds.append(seed_value)
        outputs = generate_batch(
            model,
            messages_list,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=seed_value,
            lora_request=lora_request,
        )
        for row, raw_completion in zip(rows, outputs):
            per_domain = parse_generated_checklist(raw_completion or "")
            generated_checklist = format_checklist_target(per_domain)
            prompt_id = _safe_str(row.get("prompt_id"), _safe_str(row.get("sample_id"), "unknown"))
            sample_id = _safe_str(row.get("sample_id"), prompt_id)
            winner = _safe_str(row.get("winner"), "")
            if "overall_preference" in row and row.get("overall_preference") is not None:
                overall_preference = int(row["overall_preference"])
            elif winner in ("A", "B") and row.get("preference_strength") is not None:
                strength = int(abs(int(row["preference_strength"])))
                overall_preference = -strength if winner == "A" else strength
            else:
                overall_preference = 0
            gold_winner = winner or gold_winner_from_preference(overall_preference)
            n_questions = sum(len(v) for v in per_domain.values())
            record = {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "group_id": prompt_id,
                "completion_id": f"{prompt_id}__k{k_idx}",
                "k_idx": k_idx,
                "domain": _safe_str(row.get("domain"), "unknown"),
                "winner": winner or gold_winner,
                "gold_winner": gold_winner,
                "overall_preference": overall_preference,
                "preference_strength": row.get("preference_strength"),
                "context": row.get("context"),
                "response_a": row.get("response_a"),
                "response_b": row.get("response_b"),
                "generator_model": args.generator_base,
                "generator_adapter": str(adapter_path) if adapter_path else "base",
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "seed": seed_value,
                "raw_completion": raw_completion,
                "generated_checklist": generated_checklist,
                "completion_chars": len(raw_completion or ""),
                "completion_tokens_approx": _approx_tokens(raw_completion or ""),
                "n_domains": sum(1 for qs in per_domain.values() if qs),
                "n_questions": n_questions,
                "format_valid": bool(n_questions > 0),
                "variant_type": "original",
                "source_completion_id": f"{prompt_id}__k{k_idx}",
                "replicate_id": 0,
            }
            records.append(record)

    replay_df = pd.DataFrame(records)
    parquet_path = run_dir / "replay_pool.parquet"
    jsonl_path = run_dir / "replay_pool.jsonl"
    _save_frame(replay_df, parquet_path)
    _write_jsonl(jsonl_path, replay_df.to_dict(orient="records"))
    _write_json(
        run_dir / "replay_pool_meta.json",
        {
            "subset": subset,
            "n_rows": len(rows),
            "k": args.k,
            "n_completions": len(records),
            "source_path": str(source_path),
            "generator_model": args.generator_base,
            "generator_adapter": str(adapter_path) if adapter_path else "base",
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seeds": seeds,
        },
    )
    log.info("Replay pool saved to %s", parquet_path)
    return parquet_path


def _build_judge_runner(args) -> JudgeRunner:
    adapter_path = _path_or_none(args.judge_adapter)
    return JudgeRunner(
        mode=args.judge_mode,
        url=args.judge_url,
        model_name=args.judge_model,
        api_key=args.judge_api_key,
        base_model=args.judge_base,
        adapter_path=adapter_path,
        batch_size=args.batch_size,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def score_pool_df(df: pd.DataFrame, args) -> pd.DataFrame:
    judge = _build_judge_runner(args)
    reward_cfg = get_r1_reward_config()
    scored_records: list[dict[str, Any]] = []

    for replicate_id in range(args.n_rescore_repeats):
        messages: list[list[dict[str, str]]] = []
        meta: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            record = row.to_dict()
            record["replicate_id"] = replicate_id
            meta.append({"record": record, "prepared": None})
            prepared = prepare_completion_pointwise_prompts(
                str(record.get("raw_completion") or ""),
                {
                    "context": record.get("context"),
                    "response_a": record.get("response_a"),
                    "response_b": record.get("response_b"),
                },
            )
            if prepared is None:
                continue
            meta[-1]["prepared"] = prepared
            messages.append([{"role": "user", "content": prepared["prompt_a"]}])
            messages.append([{"role": "user", "content": prepared["prompt_b"]}])

        raw_outputs = judge.generate(messages)
        cursor = 0
        for item in meta:
            record = dict(item["record"])
            record.setdefault("variant_type", "original")
            record.setdefault("source_completion_id", record.get("completion_id"))
            record.setdefault("timeout_flag", False)
            record.setdefault("exception_type", None)
            record.setdefault("exception_message", None)
            record.setdefault("judge_raw_a", None)
            record.setdefault("judge_raw_b", None)
            record.setdefault("parse_ok", False)
            record.setdefault("score_A", None)
            record.setdefault("score_B", None)
            record.setdefault("signed_delta", None)
            record.setdefault("abs_delta", None)
            record.setdefault("pred_winner", None)
            record.setdefault("winner_correct", False)
            record.setdefault("n_answered_A", 0)
            record.setdefault("n_answered_B", 0)
            record.setdefault("na_count_A", 0)
            record.setdefault("na_count_B", 0)
            record.setdefault("na_count", 0)
            record.setdefault("coverage_A", 0.0)
            record.setdefault("coverage_B", 0.0)
            record.setdefault("r_dir", 0.0)
            record.setdefault("r_margin", 0.0)
            record.setdefault("cov_pen", 0.0)
            record.setdefault("reward_total", 0.0)
            record.setdefault("checklist_says_tie", False)
            record.setdefault("human_says_tie", bool(int(record.get("overall_preference", 0)) == 0))
            record.setdefault("direction_correct", False)

            prepared = item["prepared"]
            if prepared is None:
                scored_records.append(record)
                continue

            raw_a = raw_outputs[cursor]
            raw_b = raw_outputs[cursor + 1]
            cursor += 2
            record["judge_raw_a"] = raw_a
            record["judge_raw_b"] = raw_b
            try:
                summary = summarize_judge_pair(
                    raw_a,
                    raw_b,
                    expected_n=int(prepared["expected_n"]),
                    na_policy=os.environ.get("CHECKEVAL_NA_POLICY", "as_no"),
                    coverage_threshold=float(os.environ.get("CHECKEVAL_COVERAGE_THRESHOLD", "0.8")),
                    tie_delta=float(reward_cfg["tie_delta"]),
                )
                preference = int(record.get("overall_preference", 0))
                gold_winner = _safe_str(record.get("gold_winner"), gold_winner_from_preference(preference))
                record["gold_winner"] = gold_winner
                record["parse_ok"] = bool(summary["parse_ok"])
                record["score_A"] = summary["score_a"]
                record["score_B"] = summary["score_b"]
                record["signed_delta"] = summary["signed_delta"]
                record["abs_delta"] = summary["abs_delta"]
                record["pred_winner"] = summary["pred_winner"]
                record["winner_correct"] = bool(summary["pred_winner"] == gold_winner)
                record["n_answered_A"] = int(summary["n_answered_a"])
                record["n_answered_B"] = int(summary["n_answered_b"])
                record["na_count_A"] = int(summary["na_count_a"])
                record["na_count_B"] = int(summary["na_count_b"])
                record["na_count"] = int(summary["na_count_a"]) + int(summary["na_count_b"])
                record["coverage_A"] = float(summary["coverage_a"])
                record["coverage_B"] = float(summary["coverage_b"])
                if summary["parse_ok"]:
                    reward_parts = compute_reward_components(
                        float(summary["score_a"]),
                        float(summary["score_b"]),
                        float(summary["coverage_a"]),
                        float(summary["coverage_b"]),
                        preference,
                        float(os.environ.get("CHECKEVAL_COVERAGE_THRESHOLD", "0.8")),
                        tie_delta=float(reward_cfg["tie_delta"]),
                        margin_sigma=float(reward_cfg["margin_sigma"]),
                        margin_weight=float(reward_cfg["margin_weight"]),
                        coverage_penalty_weight=float(reward_cfg["coverage_penalty_weight"]),
                        safe_tie_credit=float(reward_cfg["safe_tie_credit"]),
                    )
                    record["r_dir"] = float(reward_parts["r_dir"])
                    record["r_margin"] = float(reward_parts["r_margin"])
                    record["cov_pen"] = float(reward_parts["cov_pen"])
                    record["reward_total"] = float(reward_parts["reward_total"])
                    record["checklist_says_tie"] = bool(reward_parts["checklist_says_tie"])
                    record["human_says_tie"] = bool(reward_parts["human_says_tie"])
                    record["direction_correct"] = bool(reward_parts["direction_correct"])
                else:
                    record["checklist_says_tie"] = False
            except Exception as exc:
                record["exception_type"] = type(exc).__name__
                record["exception_message"] = str(exc)
            scored_records.append(record)

    return pd.DataFrame(scored_records)


def score_command(args) -> Path:
    replay_path = Path(args.replay_pool).resolve()
    run_dir = _resolve_run_dir(args.output_dir, args.run_name, replay_path)
    replay_df = pd.read_parquet(replay_path)
    scored_df = score_pool_df(replay_df, args)
    scored_path = run_dir / "scored_completions.parquet"
    _save_frame(scored_df, scored_path)
    _write_json(
        run_dir / "score_meta.json",
        {
            "replay_pool": str(replay_path),
            "n_rows": len(replay_df),
            "n_scored_rows": len(scored_df),
            "judge_mode": args.judge_mode,
            "judge_model": args.judge_model if args.judge_mode == "http" else args.judge_base,
            "judge_adapter": args.judge_adapter or "base",
            "n_rescore_repeats": args.n_rescore_repeats,
        },
    )
    log.info("Scored completions saved to %s", scored_path)
    return scored_path


def _select_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "variant_type" in work.columns:
        work = work[work["variant_type"].fillna("original") == "original"].copy()
    if "replicate_id" in work.columns and (work["replicate_id"] == 0).any():
        work = work[work["replicate_id"] == 0].copy()
    elif "replicate_id" in work.columns:
        work = (
            work.sort_values(["completion_id", "replicate_id"])
            .drop_duplicates("completion_id", keep="first")
            .copy()
        )
    return work


def _series_stats(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float | None:
    sub = df[[left, right]].copy()
    sub[left] = pd.to_numeric(sub[left], errors="coerce")
    sub[right] = pd.to_numeric(sub[right], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 2:
        return None
    if sub[left].nunique() < 2 or sub[right].nunique() < 2:
        return None
    return float(sub[left].corr(sub[right], method="spearman"))


def _coverage_min(row: pd.Series) -> float:
    return float(min(row.get("coverage_A", 0.0) or 0.0, row.get("coverage_B", 0.0) or 0.0))


def _gold_proxy_key(row: pd.Series) -> tuple[int, float, float]:
    return (
        1 if bool(row.get("winner_correct", False)) else 0,
        float(row.get("abs_delta") or 0.0),
        _coverage_min(row),
    )


def _pairwise_ranking_accuracy(group: pd.DataFrame) -> float | None:
    eligible = 0
    correct = 0
    rows = [row for _, row in group.iterrows()]
    for left, right in combinations(rows, 2):
        left_key = _gold_proxy_key(left)
        right_key = _gold_proxy_key(right)
        if left_key == right_key:
            continue
        eligible += 1
        gold_sign = 1 if left_key > right_key else -1
        left_reward = float(left.get("reward_total") or 0.0)
        right_reward = float(right.get("reward_total") or 0.0)
        reward_sign = 1 if left_reward > right_reward else (-1 if left_reward < right_reward else 0)
        if reward_sign == gold_sign:
            correct += 1
    if eligible == 0:
        return None
    return correct / eligible


def _build_group_metrics(primary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in primary_df.groupby("group_id", sort=False):
        reward_std = float(group["reward_total"].std(ddof=0)) if len(group) else 0.0
        reward_min = float(group["reward_total"].min()) if len(group) else 0.0
        reward_max = float(group["reward_total"].max()) if len(group) else 0.0
        ranked = group.assign(
            coverage_min=group.apply(_coverage_min, axis=1),
        ).sort_values(
            ["reward_total", "abs_delta", "coverage_min", "completion_id"],
            ascending=[False, False, False, True],
        )
        best = ranked.iloc[0]
        rows.append(
            {
                "group_id": group_id,
                "group_size": int(len(group)),
                "reward_mean": float(group["reward_total"].mean()),
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_range": reward_max - reward_min,
                "zero_variance_group": bool(abs(reward_std) <= 1e-12),
                "n_valid": int(group["parse_ok"].astype(bool).sum()),
                "valid_rate": float(group["parse_ok"].astype(bool).mean()),
                "best_reward_completion_id": best["completion_id"],
                "best_reward_correct": bool(best["winner_correct"]),
                "best_reward_pred_winner": best["pred_winner"],
                "any_correct_in_group": bool(group["winner_correct"].astype(bool).any()),
                "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(group),
            }
        )
    return pd.DataFrame(rows)


def _compute_stability_metrics(df: pd.DataFrame) -> dict[str, Any]:
    original = df[df["variant_type"].fillna("original") == "original"].copy()
    if "completion_id" not in original.columns:
        return {}
    rows = []
    for completion_id, group in original.groupby("completion_id", sort=False):
        rewards = pd.to_numeric(group["reward_total"], errors="coerce").dropna()
        if len(rewards) < 2:
            continue
        span = float(rewards.max() - rewards.min())
        rows.append(
            {
                "completion_id": completion_id,
                "n_scores": int(len(rewards)),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                "reward_span": span,
                "exact_match": bool(span <= 1e-12),
            }
        )
    if not rows:
        return {}
    stab_df = pd.DataFrame(rows)
    return {
        "n_completions_with_repeats": int(len(stab_df)),
        "exact_match_rate": float(stab_df["exact_match"].mean()),
        "reward_span": _series_stats(stab_df["reward_span"]),
    }


def _compute_perturbation_metrics(scored_df: pd.DataFrame) -> dict[str, Any]:
    original = _select_primary_rows(scored_df)[["completion_id", "reward_total"]].rename(
        columns={"completion_id": "source_completion_id", "reward_total": "base_reward"}
    )
    perturbed = scored_df[scored_df["variant_type"].fillna("original") != "original"].copy()
    if perturbed.empty:
        return {}
    joined = perturbed.merge(original, on="source_completion_id", how="left")
    joined["reward_delta"] = joined["reward_total"] - joined["base_reward"]
    summary: dict[str, Any] = {}
    for variant_type, group in joined.groupby("variant_type", sort=False):
        summary[variant_type] = {
            "count": int(len(group)),
            "reward_delta": _series_stats(group["reward_delta"]),
            "parse_rate": float(group["parse_ok"].astype(bool).mean()),
        }
    if "length_only" in summary:
        mean_delta = summary["length_only"]["reward_delta"]["mean"]
        summary["length_only_warning"] = bool(mean_delta is not None and mean_delta > 0.05)
    return summary


def analyze_command(args) -> Path:
    scored_path = Path(args.scored).resolve()
    run_dir = _resolve_run_dir(args.output_dir, args.run_name, scored_path)
    scored_df = pd.read_parquet(scored_path)
    primary_df = _select_primary_rows(scored_df)
    group_df = _build_group_metrics(primary_df)
    group_path = run_dir / "group_metrics.parquet"
    _save_frame(group_df, group_path)
    group_df.to_csv(run_dir / "group_metrics.csv", index=False)

    pipeline_metrics = compute_metrics(
        y_true=primary_df["gold_winner"].tolist(),
        y_pred=primary_df["pred_winner"].tolist(),
        domains=primary_df["domain"].tolist(),
    )
    summary = {
        "n_primary_rows": int(len(primary_df)),
        "parse_rate": float(primary_df["parse_ok"].astype(bool).mean()) if len(primary_df) else None,
        "valid_rate": float(primary_df["parse_ok"].astype(bool).mean()) if len(primary_df) else None,
        "exception_rate": float(primary_df["exception_type"].notna().mean()) if "exception_type" in primary_df else 0.0,
        "nan_rate": float(primary_df["reward_total"].isna().mean()) if len(primary_df) else None,
        "timeout_rate": float(primary_df["timeout_flag"].astype(bool).mean()) if "timeout_flag" in primary_df else 0.0,
        "zero_variance_group_rate": float(group_df["zero_variance_group"].mean()) if len(group_df) else None,
        "group_reward_std": _series_stats(group_df["reward_std"]) if len(group_df) else {},
        "group_reward_range": _series_stats(group_df["reward_range"]) if len(group_df) else {},
        "reward_selected_candidate_accuracy": float(group_df["best_reward_correct"].mean()) if len(group_df) else None,
        "best_of_k_accuracy": float(group_df["any_correct_in_group"].mean()) if len(group_df) else None,
        "pairwise_ranking_accuracy": _series_stats(group_df["pairwise_ranking_accuracy"]) if len(group_df) else {},
        "reward_length_corr": _safe_corr(primary_df, "reward_total", "completion_tokens_approx"),
        "reward_format_corr": _safe_corr(
            primary_df.assign(format_valid_num=primary_df["format_valid"].astype(int)),
            "reward_total",
            "format_valid_num",
        ),
        "reward_n_questions_corr": _safe_corr(primary_df, "reward_total", "n_questions"),
        "pipeline_metrics": pipeline_metrics,
        "stability": _compute_stability_metrics(scored_df),
        "perturbations": _compute_perturbation_metrics(scored_df),
    }
    summary_path = run_dir / "summary_metrics.json"
    _write_json(summary_path, summary)
    if summary["perturbations"]:
        _write_json(run_dir / "perturbation_metrics.json", summary["perturbations"])
    log.info("Analysis saved to %s", summary_path)
    return summary_path


def _format_variant_text(raw_completion: str, variant_idx: int) -> str:
    per_domain = parse_generated_checklist(raw_completion or "")
    if not any(per_domain.values()):
        return raw_completion or ""
    bullet = "*" if variant_idx % 2 == 0 else "1."
    lines: list[str] = []
    for domain, questions in per_domain.items():
        if not questions:
            continue
        lines.append(f"## {domain}")
        lines.append("")
        for question in questions:
            lines.append(f"{bullet} {question}")
        lines.append("")
    return "\n".join(lines).strip()


def _length_variant_text(raw_completion: str, variant_idx: int) -> str:
    notes = [
        "Checklist note: keep these questions conservative and literal.",
        "Checklist note: prefer applying these questions consistently across both responses.",
        "Checklist note: this scaffold is intentionally verbose for auditing only.",
    ]
    suffix = notes[variant_idx % len(notes)]
    base = (raw_completion or "").rstrip()
    return f"{base}\n\nAudit note:\n{suffix}\n"


def _content_damage_variant_text(raw_completion: str, variant_idx: int) -> str:
    per_domain = parse_generated_checklist(raw_completion or "")
    if not any(per_domain.values()):
        return raw_completion or ""
    damaged: dict[str, list[str]] = {}
    for domain, questions in per_domain.items():
        if not questions:
            continue
        new_questions = list(questions)
        if len(new_questions) > 1:
            new_questions = new_questions[:-1]
        if new_questions:
            new_questions[0] = "Is the response generally acceptable?"
        if variant_idx % 2 == 1 and len(new_questions) > 1:
            new_questions = [new_questions[0]]
        damaged[domain] = new_questions or ["Is the response generally acceptable?"]
    return format_checklist_target(damaged)


def perturb_command(args) -> Path:
    scored_path = Path(args.scored).resolve()
    run_dir = _resolve_run_dir(args.output_dir, args.run_name, scored_path)
    scored_df = pd.read_parquet(scored_path)
    base_df = _select_primary_rows(scored_df)
    perturb_rows: list[dict[str, Any]] = []
    for _, row in base_df.iterrows():
        base = row.to_dict()
        source_completion_id = base["completion_id"]
        for variant_idx in range(args.n_perturb_per_type):
            variants = {
                "format_only": _format_variant_text(base.get("raw_completion") or "", variant_idx),
                "length_only": _length_variant_text(base.get("raw_completion") or "", variant_idx),
                "content_damage": _content_damage_variant_text(base.get("raw_completion") or "", variant_idx),
            }
            for variant_type, raw_text in variants.items():
                per_domain = parse_generated_checklist(raw_text or "")
                record = dict(base)
                record["completion_id"] = f"{source_completion_id}__{variant_type}_{variant_idx}"
                record["variant_type"] = variant_type
                record["source_completion_id"] = source_completion_id
                record["raw_completion"] = raw_text
                record["generated_checklist"] = format_checklist_target(per_domain)
                record["completion_chars"] = len(raw_text or "")
                record["completion_tokens_approx"] = _approx_tokens(raw_text or "")
                record["n_domains"] = sum(1 for qs in per_domain.values() if qs)
                record["n_questions"] = sum(len(qs) for qs in per_domain.values())
                record["format_valid"] = bool(record["n_questions"] > 0)
                record["replicate_id"] = 0
                perturb_rows.append(record)

    perturb_pool_df = pd.DataFrame(perturb_rows)
    perturb_pool_path = run_dir / "perturb_pool.parquet"
    _save_frame(perturb_pool_df, perturb_pool_path)
    score_args = argparse.Namespace(**vars(args))
    score_args.n_rescore_repeats = 1
    perturbed_scored_df = score_pool_df(perturb_pool_df, score_args)
    perturbed_scored_path = run_dir / "perturbed_scored_completions.parquet"
    _save_frame(perturbed_scored_df, perturbed_scored_path)
    combined_df = pd.concat([scored_df, perturbed_scored_df], ignore_index=True)
    combined_path = run_dir / "combined_scored_completions.parquet"
    _save_frame(combined_df, combined_path)
    perturb_metrics = _compute_perturbation_metrics(combined_df)
    _write_json(run_dir / "perturbation_metrics.json", perturb_metrics)
    log.info("Perturbation analysis saved to %s", run_dir / "perturbation_metrics.json")
    return perturbed_scored_path


def run_all_command(args) -> None:
    replay_path = replay_pool_command(args)
    score_args = argparse.Namespace(**vars(args))
    score_args.replay_pool = str(replay_path)
    scored_path = score_command(score_args)
    analyze_args = argparse.Namespace(output_dir=args.output_dir, run_name=args.run_name, scored=str(scored_path))
    analyze_command(analyze_args)
    perturb_args = argparse.Namespace(**vars(args))
    perturb_args.scored = str(scored_path)
    perturb_command(perturb_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_run_args(p):
        p.add_argument("--output-dir", type=str, default=None)
        p.add_argument("--run-name", type=str, default=None)

    def add_vllm_args(p):
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--tensor-parallel-size", type=int, default=cfg.VLLM_ENGINE_KWARGS["tensor_parallel_size"])
        p.add_argument("--max-model-len", type=int, default=cfg.VLLM_ENGINE_KWARGS["max_model_len"])
        p.add_argument("--gpu-memory-utilization", type=float, default=cfg.VLLM_ENGINE_KWARGS["gpu_memory_utilization"])

    replay = subparsers.add_parser("replay_pool")
    replay.add_argument("--eval-split", type=str, default="dev")
    replay.add_argument("--subset", type=str, default=None)
    replay.add_argument("--generator-data-dir", type=str, default=str(cfg.GENERATOR_SFT_DIR))
    replay.add_argument("--max-samples", type=int, default=None)
    replay.add_argument("--generator-base", type=str, default=str(cfg.GENERATOR_MODEL_ID))
    replay.add_argument("--generator-adapter", type=str, default=None)
    replay.add_argument("--k", type=int, default=8)
    replay.add_argument("--temperature", type=float, default=1.0)
    replay.add_argument("--max-new-tokens", type=int, default=1024)
    replay.add_argument("--seed", type=int, default=1234)
    add_vllm_args(replay)
    add_shared_run_args(replay)

    score = subparsers.add_parser("score")
    score.add_argument("--replay-pool", type=str, required=True)
    score.add_argument("--judge-mode", choices=["http", "local"], default="http")
    score.add_argument("--judge-url", type=str, default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1"))
    score.add_argument("--judge-model", type=str, default=os.environ.get("JUDGE_MODEL"))
    score.add_argument("--judge-api-key", type=str, default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    score.add_argument("--judge-base", type=str, default=str(cfg.JUDGE_MODEL_ID))
    score.add_argument("--judge-adapter", type=str, default=None)
    score.add_argument("--judge-max-new-tokens", type=int, default=512)
    score.add_argument("--judge-temperature", type=float, default=0.0)
    score.add_argument("--n-rescore-repeats", type=int, default=1)
    add_vllm_args(score)
    add_shared_run_args(score)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--scored", type=str, required=True)
    add_shared_run_args(analyze)

    perturb = subparsers.add_parser("perturb")
    perturb.add_argument("--scored", type=str, required=True)
    perturb.add_argument("--judge-mode", choices=["http", "local"], default="http")
    perturb.add_argument("--judge-url", type=str, default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1"))
    perturb.add_argument("--judge-model", type=str, default=os.environ.get("JUDGE_MODEL"))
    perturb.add_argument("--judge-api-key", type=str, default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    perturb.add_argument("--judge-base", type=str, default=str(cfg.JUDGE_MODEL_ID))
    perturb.add_argument("--judge-adapter", type=str, default=None)
    perturb.add_argument("--judge-max-new-tokens", type=int, default=512)
    perturb.add_argument("--judge-temperature", type=float, default=0.0)
    perturb.add_argument("--n-perturb-per-type", type=int, default=1)
    add_vllm_args(perturb)
    add_shared_run_args(perturb)

    run_all = subparsers.add_parser("run_all")
    run_all.add_argument("--eval-split", type=str, default="dev")
    run_all.add_argument("--subset", type=str, default=None)
    run_all.add_argument("--generator-data-dir", type=str, default=str(cfg.GENERATOR_SFT_DIR))
    run_all.add_argument("--max-samples", type=int, default=None)
    run_all.add_argument("--generator-base", type=str, default=str(cfg.GENERATOR_MODEL_ID))
    run_all.add_argument("--generator-adapter", type=str, default=None)
    run_all.add_argument("--k", type=int, default=8)
    run_all.add_argument("--temperature", type=float, default=1.0)
    run_all.add_argument("--max-new-tokens", type=int, default=1024)
    run_all.add_argument("--seed", type=int, default=1234)
    run_all.add_argument("--judge-mode", choices=["http", "local"], default="http")
    run_all.add_argument("--judge-url", type=str, default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1"))
    run_all.add_argument("--judge-model", type=str, default=os.environ.get("JUDGE_MODEL"))
    run_all.add_argument("--judge-api-key", type=str, default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    run_all.add_argument("--judge-base", type=str, default=str(cfg.JUDGE_MODEL_ID))
    run_all.add_argument("--judge-adapter", type=str, default=None)
    run_all.add_argument("--judge-max-new-tokens", type=int, default=512)
    run_all.add_argument("--judge-temperature", type=float, default=0.0)
    run_all.add_argument("--n-rescore-repeats", type=int, default=1)
    run_all.add_argument("--n-perturb-per-type", type=int, default=1)
    add_vllm_args(run_all)
    add_shared_run_args(run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "replay_pool":
        replay_pool_command(args)
    elif args.command == "score":
        score_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "perturb":
        perturb_command(args)
    elif args.command == "run_all":
        run_all_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
