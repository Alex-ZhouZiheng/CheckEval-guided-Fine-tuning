#!/usr/bin/env python3
"""Sweep dynamic-eval k/tie_delta grids with selector-first execution.

This wrapper keeps the GPU-heavy steps separated:
1. run selector_infer.py once to create selector picks, unless --selector-picks is given;
2. run run_dynamic_eval.py for each k/tie_delta pair;
3. collect all metrics.json files into one summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import compute_metrics  # noqa: E402


def _split_csv(text: str, cast):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast(part))
    if not out:
        raise ValueError(f"empty list: {text!r}")
    return out


def _slug_float(value: float) -> str:
    text = f"{value:.6g}".replace("-", "m").replace(".", "p")
    return text


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _metric_row(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        m = json.load(f)

    n_total = int(m.get("n_total", 0) or 0)
    n_valid = int(m.get("n_valid", 0) or 0)
    valid_acc = float(m.get("accuracy", 0.0) or 0.0)
    correct_non_tie = valid_acc * n_valid
    effective_acc = (correct_non_tie / n_total) if n_total else 0.0

    row = {
        "policy": m.get("policy"),
        "k": m.get("k"),
        "tie_delta": m.get("tie_delta"),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_tie": m.get("n_tie"),
        "valid_accuracy": valid_acc,
        "effective_accuracy": effective_acc,
        "macro_f1": m.get("macro_f1"),
        "tie_rate": m.get("tie_rate"),
        "parse_ok_rate": m.get("parse_ok_rate"),
        "avg_k": m.get("avg_k"),
        "selector_picks": m.get("selector_picks"),
        "judge_mode": m.get("judge_mode"),
        "judge_model": m.get("judge_model"),
        "judge_url": m.get("judge_url"),
        "http_extra_body": m.get("http_extra_body"),
        "http_reasoning_effort": m.get("http_reasoning_effort"),
        "out_dir": str(metrics_path.parent),
    }

    per_domain = m.get("per_domain") or {}
    for domain, stats in per_domain.items():
        row[f"{domain}_n"] = stats.get("n")
        row[f"{domain}_accuracy"] = stats.get("accuracy")
        row[f"{domain}_macro_f1"] = stats.get("macro_f1")

    return row


def _winner_from_margin(margin: float, tie_delta: float) -> str:
    if margin > tie_delta:
        return "A"
    if margin < -tie_delta:
        return "B"
    return "Tie"


def _recompute_metrics(
    pred_path: Path,
    base_metrics_path: Path,
    tie_delta: float,
    out_dir: Path,
) -> Path:
    pred = pd.read_parquet(pred_path)
    pred = pred.copy()
    pred["predicted_winner"] = pred["margin_final"].map(
        lambda x: _winner_from_margin(float(x), tie_delta=tie_delta)
    )

    y_true = pred["winner"].astype(str).tolist()
    y_pred = pred["predicted_winner"].astype(str).tolist()
    domains = pred["domain"].astype(str).tolist()

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, domains=domains)

    with base_metrics_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    metrics["policy"] = base.get("policy")
    metrics["k"] = base.get("k")
    metrics["tie_delta"] = tie_delta
    metrics["tau_escalate"] = base.get("tau_escalate")
    metrics["tau_hard"] = base.get("tau_hard")
    metrics["n_min"] = base.get("n_min")
    metrics["avg_k"] = float(pred["k_after_escalation"].mean()) if len(pred) else 0.0
    metrics["tie_rate"] = float((pred["predicted_winner"] == "Tie").mean()) if len(pred) else 0.0
    metrics["parse_ok_rate"] = float(pred["parse_ok"].mean()) if len(pred) else 0.0
    metrics["selector_inference_time_s"] = base.get("selector_inference_time_s", 0.0)
    metrics["selector_picks"] = base.get("selector_picks")
    metrics["judge_mode"] = base.get("judge_mode")
    metrics["judge_out_dir"] = str(pred_path.parent)

    if len(pred):
        metrics["latency_p50_s"] = float(pred["latency_s_estimate"].quantile(0.5))
        metrics["latency_p90_s"] = float(pred["latency_s_estimate"].quantile(0.9))
        total_latency = float(pred["latency_s_estimate"].sum())
        metrics["samples_per_second_estimate"] = (
            float(len(pred) / total_latency) if total_latency > 0 else None
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred_path = out_dir / "predictions.parquet"
    out_metrics_path = out_dir / "metrics.json"
    pred.to_parquet(out_pred_path, index=False)
    with out_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    return out_metrics_path


def _write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            -float(r.get("effective_accuracy") or 0.0),
            -float(r.get("valid_accuracy") or 0.0),
            float(r.get("tie_rate") or 1.0),
        ),
    )

    json_path = out_root / "sweep_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    csv_path = out_root / "sweep_summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep summary -> {csv_path}", flush=True)
    print(f"Saved sweep summary -> {json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", type=str, default="dev_600")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--selector", type=Path, default=None)
    parser.add_argument("--selector-picks", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--policy", type=str, default="learned_topk")
    parser.add_argument("--k-list", type=str, default="8,10,12,15")
    parser.add_argument("--tie-deltas", type=str, default="0,0.03,0.05,0.08,0.10")
    parser.add_argument("--selector-device", type=str, default="cuda")
    parser.add_argument("--selector-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-dim-quota", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--judge-mode", choices=["local", "http"], default="local")
    parser.add_argument("--backend", choices=["llamacpp", "vllm"], default=None)
    parser.add_argument("--judge-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument("--judge-api-key-env", type=str, default=None)
    parser.add_argument("--http-concurrency", type=int, default=32)
    parser.add_argument(
        "--http-extra-body",
        choices=["qwen-thinking-off", "deepseek-thinking-on", "deepseek-thinking-off", "none"],
        default="qwen-thinking-off",
    )
    parser.add_argument("--http-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--prompt-chunk-size", type=int, default=1000)
    parser.add_argument("--max-model-len", type=int, default=10000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ks = _split_csv(args.k_list, int)
    tie_deltas = _split_csv(args.tie_deltas, float)
    out_root = args.out_root.resolve()
    env = os.environ.copy()
    if args.offline:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"

    selector_picks = args.selector_picks
    if selector_picks is None:
        if args.selector is None:
            raise SystemExit("--selector is required when --selector-picks is not provided")

        max_k = max(ks)
        selector_picks = out_root / "selector" / f"{args.split}_top{max_k}.parquet"
        if args.overwrite or not selector_picks.exists():
            cmd = [
                sys.executable,
                "src/evaluation/selector_infer.py",
                "--selector",
                str(args.selector),
                "--split",
                args.split,
                "--k",
                str(max_k),
                "--device",
                args.selector_device,
                "--batch-size",
                str(args.selector_batch_size),
                "--out",
                str(selector_picks),
            ]
            if args.subset:
                cmd.extend(["--subset", args.subset])
            if args.input_path:
                cmd.extend(["--input-path", str(args.input_path)])
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.no_dim_quota:
                cmd.append("--no-dim-quota")
            _run(cmd, env=env)
        else:
            print(f"Reuse selector picks -> {selector_picks}", flush=True)

    rows: list[dict[str, Any]] = []
    for k in ks:
        judge_dir = out_root / "judge_runs" / f"{args.policy}_k{k}"
        judge_metrics_path = judge_dir / "metrics.json"
        judge_pred_path = judge_dir / "predictions.parquet"

        if args.overwrite or not judge_metrics_path.exists() or not judge_pred_path.exists():
            cmd = [
                sys.executable,
                "src/evaluation/run_dynamic_eval.py",
                "--bank",
                str(args.bank),
                "--split",
                args.split,
                "--policy",
                args.policy,
                "--selector-picks",
                str(selector_picks),
                "--k",
                str(k),
                "--tie-delta",
                "0.0",
                "--judge-mode",
                args.judge_mode,
                "--batch-size",
                str(args.batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--prompt-chunk-size",
                str(args.prompt_chunk_size),
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--seed",
                str(args.seed),
                "--out",
                str(judge_dir),
            ]
            if args.subset:
                cmd.extend(["--subset", args.subset])
            if args.input_path:
                cmd.extend(["--input-path", str(args.input_path)])
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.no_dim_quota:
                cmd.append("--no-dim-quota")
            if args.backend:
                cmd.extend(["--backend", args.backend])
            if args.base_model:
                cmd.extend(["--base-model", args.base_model])
            if args.judge_mode == "http":
                cmd.extend(
                    [
                        "--judge-url",
                        args.judge_url,
                        "--judge-api-key",
                        args.judge_api_key,
                        "--http-concurrency",
                        str(args.http_concurrency),
                        "--http-extra-body",
                        args.http_extra_body,
                    ]
                )
                if args.http_reasoning_effort:
                    cmd.extend(["--http-reasoning-effort", args.http_reasoning_effort])
                if args.judge_api_key_env:
                    cmd.extend(["--judge-api-key-env", args.judge_api_key_env])
                if args.judge_model:
                    cmd.extend(["--judge-model", args.judge_model])
            _run(cmd, env=env)
        else:
            print(f"Reuse judge run -> {judge_dir}", flush=True)

        for td in tie_deltas:
            out_dir = out_root / f"{args.policy}_k{k}_td{_slug_float(td)}"
            metrics_path = out_dir / "metrics.json"
            if args.overwrite or not metrics_path.exists():
                metrics_path = _recompute_metrics(
                    pred_path=judge_pred_path,
                    base_metrics_path=judge_metrics_path,
                    tie_delta=td,
                    out_dir=out_dir,
                )
                print(f"Recomputed metrics -> {metrics_path}", flush=True)
            else:
                print(f"Reuse metrics -> {metrics_path}", flush=True)

            rows.append(_metric_row(metrics_path))

    _write_summary(rows, out_root)


if __name__ == "__main__":
    main()
