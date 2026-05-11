"""Normalize judge-eval metric JSONs to canonical schema.

Reads raw `results/*_metrics.json`, emits unified-schema records to
`results/normalized/<basename>.json`. Backfills hashes where input
artifacts are present locally (split parquets, checklist banks). Marks
unrecoverable fields (adapter_sha, base_sha, run-time git_sha) as null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NORM_DIR = RESULTS / "normalized"
SPLITS = ROOT / "data" / "splits"
CHECKLISTS = ROOT / "checklists"


def sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class MethodInfo:
    method: str
    family: str   # zeroshot | ft_vanilla | ft_checkeval | selfcheck | pipeline
    base_id: str | None
    adapter_path: str | None
    bank: str | None         # bank dir name (e.g. v3_frozen) or "self" / None
    checklist_source: str    # "none" | "static" | "self" | "generator"


def classify(raw: dict[str, Any], filename: str) -> MethodInfo:
    name = filename.lower()
    keys = set(raw.keys())

    if "generated_source" in keys or raw.get("judge_format") == "comparative" or "pipeline" in name:
        return MethodInfo(
            method="pipeline_cmp",
            family="pipeline",
            base_id=raw.get("base_model"),
            adapter_path=raw.get("judge_adapter"),
            bank=None,
            checklist_source="generator",
        )
    if "judge_adapter" in keys and "base_model" in keys:
        adapter = raw.get("judge_adapter")
        base = raw.get("base_model")
        m = re.search(r"checkpoint-\d+", str(adapter or ""))
        if m:
            tag = m.group(0)
        elif not adapter or str(adapter) == "base":
            base_short = re.search(r"Qwen[\d\.]+-\d+B", str(base or ""))
            tag = f"base_{base_short.group(0)}" if base_short else "base"
        else:
            tag = "adapter"
        return MethodInfo(
            method=f"selfcheck_{tag}",
            family="selfcheck",
            base_id=base,
            adapter_path=adapter,
            bank="self",
            checklist_source="self",
        )
    if "eval_mode" in keys and raw.get("eval_mode") == "checkeval":
        return MethodInfo(
            method="ft_checkeval",
            family="ft_checkeval",
            base_id=raw.get("model_id"),
            adapter_path=raw.get("adapter_path"),
            bank="v3_frozen",
            checklist_source="static",
        )
    if "adapter_path" in keys and raw.get("eval_mode") in (None, "vanilla"):
        return MethodInfo(
            method="ft_vanilla",
            family="ft_vanilla",
            base_id=raw.get("model_id"),
            adapter_path=raw.get("adapter_path"),
            bank=None,
            checklist_source="none",
        )
    if "model_id" in keys and "adapter_path" not in keys:
        return MethodInfo(
            method="vanilla_zeroshot",
            family="zeroshot",
            base_id=raw.get("model_id"),
            adapter_path=None,
            bank=None,
            checklist_source="none",
        )
    return MethodInfo(
        method="unknown",
        family="unknown",
        base_id=None,
        adapter_path=None,
        bank=None,
        checklist_source="unknown",
    )


def detect_split(filename: str) -> str:
    name = filename.lower()
    for s in ("dev_600", "test", "dev"):
        if f"_{s}_" in name or name.endswith(f"_{s}_metrics.json"):
            return s
    if "_test_" in name:
        return "test"
    return "unknown"


def normalize(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    info = classify(raw, path.name)
    split = detect_split(path.name)

    n_total = int(raw.get("n_total") or raw.get("n_samples_total") or 0)
    n_valid = int(raw.get("n_valid") or 0)
    n_tie = int(raw.get("n_tie") or 0)
    n_unparseable = int(raw.get("n_unparseable") or 0)
    parse_rate = raw.get("parse_rate")
    if parse_rate is None and n_total:
        parse_rate = n_valid / n_total

    # Schema drift: legacy files have only `accuracy` (= valid_accuracy).
    # Current files have both `accuracy` (= real) + `valid_accuracy`.
    acc = raw.get("accuracy")
    valid_acc_field = raw.get("valid_accuracy")
    if valid_acc_field is not None:
        valid_acc = float(valid_acc_field)
        real_acc = float(acc) if acc is not None else (
            valid_acc * n_valid / n_total if n_total else None
        )
    else:
        valid_acc = float(acc) if acc is not None else None
        real_acc = (valid_acc * n_valid / n_total) if (valid_acc is not None and n_total) else None

    split_path = SPLITS / f"{split}.parquet"
    split_sha = sha256_file(split_path) if split != "unknown" else None

    bank_sha = None
    n_questions = raw.get("n_checklist_questions")
    if info.bank and info.bank not in ("self",):
        bank_dir = CHECKLISTS / info.bank
        idx = bank_dir / "index.json"
        bank_sha = sha256_file(idx) if idx.exists() else None

    norm = {
        "schema_version": SCHEMA_VERSION,
        "source_file": (
            str(path.relative_to(ROOT)).replace("\\", "/")
            if str(path).startswith(str(ROOT)) else str(path).replace("\\", "/")
        ),
        "method": info.method,
        "family": info.family,
        "split": {
            "name": split,
            "n_total": n_total,
            "split_sha256": split_sha,
        },
        "model": {
            "base_id": info.base_id,
            "base_sha256": None,
            "adapter_path": info.adapter_path,
            "adapter_sha256": None,
            "backend": raw.get("backend"),
            "max_model_len": raw.get("max_model_len"),
            "tensor_parallel_size": raw.get("tensor_parallel_size"),
            "enable_thinking": raw.get("enable_thinking"),
        },
        "prompt": {
            "template_id": raw.get("judge_format") or raw.get("eval_mode"),
            "na_policy": raw.get("na_policy") or os.environ.get("CHECKEVAL_NA_POLICY"),
            "tie_delta": raw.get("tie_delta"),
            "scoring": raw.get("scoring"),
        },
        "checklist": {
            "source": info.checklist_source,
            "bank": info.bank,
            "bank_sha256": bank_sha,
            "n_questions": n_questions,
            "avg_length": raw.get("avg_checklist_length"),
        },
        "seed": raw.get("seed"),
        "counts": {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_tie": n_tie,
            "n_unparseable": n_unparseable,
        },
        "metrics": {
            "valid_accuracy": float(valid_acc) if valid_acc is not None else None,
            "real_accuracy": real_acc,
            "macro_f1": raw.get("macro_f1"),
            "parse_rate": float(parse_rate) if parse_rate is not None else None,
            "position_bias_A": raw.get("position_bias_A"),
        },
        "per_domain": raw.get("per_domain"),
        "classification_report": raw.get("classification_report"),
        "pred_distribution": raw.get("pred_distribution"),
        "timing": {
            "inference_time_s": raw.get("inference_time_s"),
            "samples_per_second": raw.get("samples_per_second"),
        },
        "provenance": {
            "git_sha_at_normalize": git_head(),
            "git_sha_at_run": None,
            "normalized_at": None,
        },
    }
    return norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="explicit metric files; else glob results/*_metrics.json")
    ap.add_argument("--out-dir", default=str(NORM_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        files = [Path(p).resolve() for p in args.inputs]
    else:
        files = sorted(RESULTS.glob("*_metrics.json"))

    written = 0
    skipped = 0
    for p in files:
        if not p.is_file():
            print(f"MISSING {p}", file=sys.stderr)
            skipped += 1
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"PARSE_FAIL {p}: {e}", file=sys.stderr)
            skipped += 1
            continue
        norm = normalize(raw, p)
        out = out_dir / p.name
        out.write_text(json.dumps(norm, indent=2), encoding="utf-8")
        written += 1

    print(f"normalized={written} skipped={skipped} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
