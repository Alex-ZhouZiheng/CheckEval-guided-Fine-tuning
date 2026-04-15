#!/usr/bin/env python3
"""
Generate checklist evaluation SFT data for joint DPO + checklist training.

For each pairwise training sample, creates checklist evaluation prompts for
both responses (A and B), paired with target checklist answers.

Two modes:
- **synthetic**: heuristic answers derived from preference labels (fast, no GPU)
- **teacher**: teacher model generates checklist answers via vLLM, ChatGPT,
  or the OpenAI Batch API

Usage:
    # Quick synthetic data for pipeline testing
    python prepare_checklist_sft.py --tier debug_5k --mode synthetic

    # Teacher-generated data with local vLLM model
    python prepare_checklist_sft.py --tier debug_5k --mode teacher

    # Teacher-generated data with ChatGPT
    python prepare_checklist_sft.py --tier debug_5k --mode teacher \
        --teacher-backend openai --model-id gpt-4o-mini

    # Teacher-generated data with OpenAI Batch API
    python prepare_checklist_sft.py --tier tier_10k --mode teacher \
        --teacher-backend openai_batch --model-id gpt-4o-mini \
        --stage run_all --resume

    # Different checklist source
    python prepare_checklist_sft.py --tier tier_10k --mode teacher \
        --checklist-dir ../checklists/v2

    # Custom output directory (for control experiments)
    python prepare_checklist_sft.py --tier tier_10k --mode teacher \
        --output-dir ../data/checklist_sft_v2
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import openai
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config as cfg
from utils import (
    build_checkeval_prompt,
    expected_question_count,
    load_checklists,
    parse_checkeval_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
console = Console()

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
OPENAI_BATCH_MAX_FILE_BYTES = 200 * 1024 * 1024
OPENAI_BATCH_TARGET_FILE_BYTES = 190 * 1024 * 1024
OPENAI_BATCH_TARGET_ESTIMATED_TOKENS = 200_000
OPENAI_BATCH_PREP_VERSION = 2


class TokenLimitExceededError(RuntimeError):
    """Raised when OpenAI Batch rejects a shard due to enqueued token limits."""


# synthetic mode


def generate_synthetic_answers(
    n_questions: int,
    is_chosen: bool,
    preference_strength: int,
    seed: int,
) -> str:
    """Generate synthetic checklist answers based on preference signal."""
    rng = np.random.RandomState(seed)
    strength = abs(preference_strength)

    if is_chosen:
        p_yes = 0.65 + 0.05 * strength
    else:
        p_yes = 0.35 - 0.05 * strength

    lines = []
    for i in range(1, n_questions + 1):
        ans = "yes" if rng.random() < p_yes else "no"
        lines.append(f"Q{i}: {ans}")
    return "\n".join(lines)


def build_synthetic_sft(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
) -> pd.DataFrame:
    """Build checklist SFT data using synthetic (heuristic) answers."""
    records = []
    for idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Synthetic SFT")
    ):
        domain = row["domain"]
        winner = row["winner"]
        pref = row.get("preference_strength", 2)
        n_q = expected_question_count(domain, checklists)

        for side in ("A", "B"):
            prompt_text = build_checkeval_prompt(
                row, checklists, definitions, domain=domain, side=side,
            )
            is_chosen = side == winner
            completion = generate_synthetic_answers(
                n_q, is_chosen, pref, seed=idx * 2 + (0 if side == "A" else 1),
            )
            records.append({
                "prompt_text": prompt_text,
                "completion_text": completion,
                "domain": domain,
                "side": side,
                "is_chosen": is_chosen,
                "parse_valid": True,
                "n_questions": n_q,
            })

    return pd.DataFrame(records)


# teacher mode


def _infer_teacher_backend(model_id: str, teacher_backend: str) -> str:
    """Resolve the teacher backend from CLI args."""
    if teacher_backend != "auto":
        return teacher_backend

    openai_prefixes = ("gpt-", "chatgpt-", "o1", "o3", "o4")
    if model_id.startswith(openai_prefixes):
        return "openai"
    return "vllm"


def _get_openai_client():
    """Create an OpenAI client from environment configuration."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI backend requires `pip install openai`.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in your environment or project .env."
        )

    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
        log.info("Using custom OpenAI base URL: %s", base_url)

    return OpenAI(**kwargs)


def generate_openai_batch(
    prompts: list[str],
    model_id: str,
    max_concurrent: int = 2,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Generate checklist answers with an OpenAI-compatible chat model."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = _get_openai_client()
    results = [""] * len(prompts)
    errors = 0

    def _retry_after_seconds(exc: Exception) -> float | None:
        text = str(exc)
        match = re.search(r"try again in ([0-9.]+)s", text, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0
        return None

    def _call(idx: int, prompt: str) -> tuple[int, str]:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                text = response.choices[0].message.content or ""
                return idx, text.strip()
            except Exception as exc:
                if attempt == 2:
                    log.warning("OpenAI teacher call failed (idx=%d): %s", idx, exc)
                    return idx, ""
                wait_s = _retry_after_seconds(exc)
                if wait_s is None:
                    wait_s = 2 ** attempt
                log.info("Rate-limited / transient error on idx=%d, sleeping %.1fs", idx, wait_s)
                time.sleep(wait_s)
        return idx, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts),
                           desc="Teacher inference"):
            idx, text = future.result()
            results[idx] = text
            if not text:
                errors += 1

    if errors:
        log.warning("OpenAI teacher: %d/%d calls failed", errors, len(prompts))
    return results


def build_teacher_sft(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    model_id: str = cfg.JUDGE_MODEL_ID,
    teacher_backend: str = "auto",
    batch_size: int = 16,
    max_concurrent: int = 2,
) -> pd.DataFrame:
    """Build checklist SFT data using a teacher model via vLLM or OpenAI."""
    from utils import generate_batch, load_judge_model

    resolved_backend = _infer_teacher_backend(str(model_id), teacher_backend)
    if resolved_backend == "openai_batch":
        raise ValueError("Use the staged OpenAI Batch flow for teacher-backend=openai_batch.")
    log.info("Teacher backend: %s", resolved_backend)
    log.info("Teacher model: %s", model_id)

    all_prompts: list[str] = []
    meta: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
        domain = row["domain"]
        n_q = expected_question_count(domain, checklists)
        winner = row["winner"]

        for side in ("A", "B"):
            prompt_text = build_checkeval_prompt(
                row, checklists, definitions, domain=domain, side=side,
            )
            all_prompts.append(prompt_text)
            meta.append({
                "domain": domain,
                "side": side,
                "is_chosen": side == winner,
                "n_questions": n_q,
            })

    log.info("Running teacher inference on %d prompts ...", len(all_prompts))
    if resolved_backend == "openai":
        raw_outputs = generate_openai_batch(
            all_prompts,
            model_id=str(model_id),
            max_concurrent=max_concurrent,
            max_new_tokens=512,
        )
    else:
        llm = load_judge_model(model_id)
        messages_list = [[{"role": "user", "content": p}] for p in all_prompts]
        raw_outputs = generate_batch(
            llm, messages_list, batch_size=batch_size, max_new_tokens=2048,
        )

    records = []
    n_valid = 0
    for prompt_text, raw, m in zip(all_prompts, raw_outputs, meta):
        parsed = parse_checkeval_output(raw, expected_n=m["n_questions"])
        valid = not parsed.get("_raw_fallback", False)
        if valid:
            n_valid += 1

        records.append({
            "prompt_text": prompt_text,
            "completion_text": raw.strip(),
            "domain": m["domain"],
            "side": m["side"],
            "is_chosen": m["is_chosen"],
            "parse_valid": valid,
            "n_questions": m["n_questions"],
        })

    log.info(
        "Teacher parse rate: %d/%d (%.1f%%)",
        n_valid,
        len(records),
        100 * n_valid / len(records) if records else 0.0,
    )

    if resolved_backend == "vllm":
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return pd.DataFrame(records)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object line in {path}")
            records.append(obj)
    return records


def _safe_model_name(model_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_id).strip())
    return safe.strip("._") or "model"


def _resolve_train_path(tier: str) -> Path:
    if tier == "full":
        return cfg.SPLITS_DIR / "train.parquet"
    return cfg.SPLITS_DIR / f"train_{tier}.parquet"


def _load_train_df(tier: str) -> tuple[pd.DataFrame, Path]:
    src_path = _resolve_train_path(tier)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}. Run prepare_data.py first.")
    df = pd.read_parquet(src_path)
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    return df, src_path


def _make_custom_id(sample_idx: int, side: str) -> str:
    return f"sample-{sample_idx:06d}-{side}"


def _sdk_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
        if isinstance(data, dict):
            return data
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {key: value for key, value in vars(obj).items() if not key.startswith("_")}
    return {"value": str(obj)}


def _get_batch_work_dir(
    output_dir: Path,
    batch_output_dir: Path | None,
    model_id: str,
    tier: str,
) -> Path:
    if batch_output_dir is not None:
        work_dir = batch_output_dir
    else:
        work_dir = output_dir / "batch_jobs" / _safe_model_name(model_id) / tier
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _final_output_name(
    tier: str,
    mode: str,
    teacher_backend: str | None = None,
) -> str:
    prefix = f"train_{tier}" if tier != "full" else "train"
    if mode == "synthetic":
        return f"{prefix}_synthetic.parquet"
    if mode == "teacher" and teacher_backend == "openai_batch":
        return f"{prefix}_teacher_openai_batch.parquet"
    return f"{prefix}.parquet"


def _jsonl_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def _jsonl_line_bytes(line: str) -> int:
    return len(line.encode("utf-8"))


def _estimate_request_tokens_from_line(line: str) -> int:
    # Estimate input tokens from byte length, then add max_completion_tokens
    # (OpenAI Batch API counts input + max_completion_tokens toward enqueued quota).
    input_tokens = max(1, math.ceil(len(line.encode("utf-8")) / 4.0))
    try:
        body = json.loads(line).get("body", {})
        max_output = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    except (json.JSONDecodeError, AttributeError):
        max_output = 0
    return input_tokens + max_output


def _batch_error_messages(batch_info: dict[str, Any]) -> list[str]:
    errors = batch_info.get("errors") or {}
    data = errors.get("data") if isinstance(errors, dict) else None
    if not isinstance(data, list):
        return []

    messages: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        code = item.get("code") or "unknown"
        message = item.get("message") or "unknown error"
        line = item.get("line")
        if line is not None:
            messages.append(f"{code} (line {line}): {message}")
        else:
            messages.append(f"{code}: {message}")
    return messages


def _batch_has_token_limit_error(batch_info: dict[str, Any]) -> bool:
    errors = batch_info.get("errors") or {}
    data = errors.get("data") if isinstance(errors, dict) else None
    if not isinstance(data, list):
        return False
    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").lower()
        message = str(item.get("message") or "").lower()
        if code == "token_limit_exceeded":
            return True
        if "enqueued token limit reached" in message:
            return True
    return False


def _is_failed_batch_status(status: str | None) -> bool:
    return status in {"failed", "expired", "cancelled"}


def _single_or_none(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else None


def _load_shard_index(work_dir: Path) -> dict[str, Any]:
    shard_index_path = work_dir / "shard_index.json"
    if shard_index_path.exists():
        data = _read_json(shard_index_path)
        shards = data.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Invalid shard_index.json in {work_dir}")
        return data

    batch_input_path = work_dir / "batch_input.jsonl"
    manifest_path = work_dir / "manifest.jsonl"
    if batch_input_path.exists() and manifest_path.exists():
        return {
            "n_shards": 1,
            "n_requests": len(_read_jsonl(manifest_path)),
            "legacy_single_shard": True,
            "shards": [{
                "shard_id": "shard_00000",
                "work_dir": str(work_dir),
                "batch_input_path": str(batch_input_path),
                "manifest_path": str(manifest_path),
                "n_requests": len(_read_jsonl(manifest_path)),
            }],
        }

    raise FileNotFoundError(f"No shard_index.json or legacy batch files found in {work_dir}")


def _shard_index_is_current(shard_index: dict[str, Any]) -> bool:
    if int(shard_index.get("prep_version", 0)) < OPENAI_BATCH_PREP_VERSION:
        return False
    if "target_estimated_tokens" not in shard_index:
        return False
    return True


def _iter_shards(work_dir: Path) -> list[dict[str, Any]]:
    shard_index = _load_shard_index(work_dir)
    shards = []
    for shard in shard_index["shards"]:
        shard_copy = dict(shard)
        shard_copy["work_dir"] = str(Path(shard_copy["work_dir"]))
        shards.append(shard_copy)
    return shards


def _cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _cancel_remote_batch(shard_dir: Path) -> None:
    """Cancel the remote OpenAI batch job associated with a shard, if any."""
    meta_path = shard_dir / "batch_meta.json"
    if not meta_path.exists():
        return
    meta = _read_json(meta_path)
    batch_id = meta.get("batch_id")
    if not batch_id:
        return
    try:
        client = _get_openai_client()
        batch_info = _sdk_to_dict(client.batches.retrieve(batch_id))
        if _is_active_batch_status(batch_info.get("status")):
            client.batches.cancel(batch_id)
            log.info("Cancelled remote batch %s (was %s)", batch_id, batch_info.get("status"))
    except Exception as exc:
        log.warning("Failed to cancel remote batch %s: %s", batch_id, exc)


def _reset_single_shard_state(shard_dir: Path) -> None:
    _cancel_remote_batch(shard_dir)
    for filename in [
        "batch_meta.json",
        "batch_status.json",
        "batch_output.jsonl",
        "batch_error.jsonl",
    ]:
        _cleanup_path(shard_dir / filename)


def _read_jsonl_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def _is_active_batch_status(status: str | None) -> bool:
    return status in {"validating", "in_progress", "finalizing", "cancelling"}


def _count_completed_prefix(shard_specs: list[dict[str, Any]]) -> int:
    completed_count = 0
    for shard in shard_specs:
        shard_status_path = Path(shard["work_dir"]) / "batch_status.json"
        if not shard_status_path.exists():
            break
        shard_status = _read_json(shard_status_path)
        if shard_status.get("status") != "completed":
            break
        completed_count += 1
    return completed_count


def _flush_prepared_shard(
    shard_specs: list[dict[str, Any]],
    shard_root: Path,
    shard_start_index: int,
    request_lines: list[str],
    manifest_lines: list[str],
    shard_bytes: int,
    shard_estimated_tokens: int,
) -> None:
    if not request_lines:
        return

    shard_id = f"shard_{shard_start_index + len(shard_specs):05d}"
    shard_dir = shard_root / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_batch_input_path = shard_dir / "batch_input.jsonl"
    shard_manifest_path = shard_dir / "manifest.jsonl"
    with open(shard_batch_input_path, "w", encoding="utf-8") as f_req:
        f_req.writelines(request_lines)
    with open(shard_manifest_path, "w", encoding="utf-8") as f_manifest:
        f_manifest.writelines(manifest_lines)

    first_manifest = json.loads(manifest_lines[0])
    last_manifest = json.loads(manifest_lines[-1])
    shard_specs.append({
        "shard_id": shard_id,
        "work_dir": str(shard_dir),
        "batch_input_path": str(shard_batch_input_path),
        "manifest_path": str(shard_manifest_path),
        "n_requests": len(request_lines),
        "batch_input_bytes": shard_bytes,
        "estimated_tokens": shard_estimated_tokens,
        "first_custom_id": first_manifest["custom_id"],
        "last_custom_id": last_manifest["custom_id"],
    })


def _write_shard_group(
    request_lines: list[str],
    manifest_lines: list[str],
    shard_root: Path,
    shard_start_index: int,
    target_bytes: int,
    target_estimated_tokens: int,
) -> list[dict[str, Any]]:
    shard_specs: list[dict[str, Any]] = []
    current_request_lines: list[str] = []
    current_manifest_lines: list[str] = []
    current_bytes = 0
    current_estimated_tokens = 0

    for request_line, manifest_line in zip(request_lines, manifest_lines):
        request_bytes = _jsonl_line_bytes(request_line)
        request_estimated_tokens = _estimate_request_tokens_from_line(request_line)
        custom_id = json.loads(manifest_line)["custom_id"]

        if request_bytes > target_bytes:
            raise ValueError(
                f"Single request {custom_id} exceeds shard size budget "
                f"({request_bytes} > {target_bytes})."
            )
        if request_estimated_tokens > target_estimated_tokens:
            raise ValueError(
                f"Single request {custom_id} exceeds token shard budget "
                f"({request_estimated_tokens} > {target_estimated_tokens})."
            )

        if current_request_lines and (
            current_bytes + request_bytes > target_bytes
            or current_estimated_tokens + request_estimated_tokens > target_estimated_tokens
        ):
            _flush_prepared_shard(
                shard_specs=shard_specs,
                shard_root=shard_root,
                shard_start_index=shard_start_index,
                request_lines=current_request_lines,
                manifest_lines=current_manifest_lines,
                shard_bytes=current_bytes,
                shard_estimated_tokens=current_estimated_tokens,
            )
            current_request_lines = []
            current_manifest_lines = []
            current_bytes = 0
            current_estimated_tokens = 0

        current_request_lines.append(request_line)
        current_manifest_lines.append(manifest_line)
        current_bytes += request_bytes
        current_estimated_tokens += request_estimated_tokens

    _flush_prepared_shard(
        shard_specs=shard_specs,
        shard_root=shard_root,
        shard_start_index=shard_start_index,
        request_lines=current_request_lines,
        manifest_lines=current_manifest_lines,
        shard_bytes=current_bytes,
        shard_estimated_tokens=current_estimated_tokens,
    )
    return shard_specs


def _maybe_reshard_pending_tail(
    work_dir: Path,
    shard_index: dict[str, Any],
    target_estimated_tokens: int,
) -> dict[str, Any]:
    current_target = int(shard_index.get("target_estimated_tokens", 0) or 0)
    if current_target and current_target <= target_estimated_tokens:
        return shard_index

    shard_specs = _iter_shards(work_dir)
    completed_prefix_count = _count_completed_prefix(shard_specs)

    active_shards = []
    for shard in shard_specs[completed_prefix_count:]:
        shard_status_path = Path(shard["work_dir"]) / "batch_status.json"
        if not shard_status_path.exists():
            continue
        shard_status = _read_json(shard_status_path)
        if _is_active_batch_status(shard_status.get("status")):
            active_shards.append(shard["shard_id"])
    if active_shards:
        log.warning(
            "Skipping pending-tail reshard because active shard(s) exist: %s",
            ", ".join(active_shards),
        )
        return shard_index

    batch_input_path = work_dir / "batch_input.jsonl"
    manifest_path = work_dir / "manifest.jsonl"
    shard_index_path = work_dir / "shard_index.json"
    shards_root = work_dir / "shards"
    request_lines = _read_jsonl_lines(batch_input_path)
    manifest_lines = _read_jsonl_lines(manifest_path)
    if len(request_lines) != len(manifest_lines):
        raise ValueError(
            f"Combined batch input / manifest length mismatch in {work_dir}: "
            f"{len(request_lines)} vs {len(manifest_lines)}"
        )

    preserved_specs = [dict(shard) for shard in shard_specs[:completed_prefix_count]]
    preserved_request_count = sum(int(shard.get("n_requests", 0)) for shard in preserved_specs)

    for shard in shard_specs[completed_prefix_count:]:
        _cleanup_path(Path(shard["work_dir"]))

    rebuilt_specs = _write_shard_group(
        request_lines=request_lines[preserved_request_count:],
        manifest_lines=manifest_lines[preserved_request_count:],
        shard_root=shards_root,
        shard_start_index=completed_prefix_count,
        target_bytes=OPENAI_BATCH_TARGET_FILE_BYTES,
        target_estimated_tokens=target_estimated_tokens,
    )

    updated_shards = preserved_specs + rebuilt_specs
    updated_index = {
        "prep_version": OPENAI_BATCH_PREP_VERSION,
        "n_shards": len(updated_shards),
        "n_requests": len(request_lines),
        "target_batch_input_bytes": OPENAI_BATCH_TARGET_FILE_BYTES,
        "max_batch_input_bytes": OPENAI_BATCH_MAX_FILE_BYTES,
        "target_estimated_tokens": target_estimated_tokens,
        "migrated_from_target_estimated_tokens": current_target or None,
        "completed_prefix_preserved": completed_prefix_count,
        "shards": updated_shards,
    }
    _write_json(shard_index_path, updated_index)

    for filename in [
        "batch_meta.json",
        "batch_status.json",
        "batch_output.jsonl",
        "batch_error.jsonl",
        "raw_outputs.json",
        "error_summary.json",
    ]:
        _cleanup_path(work_dir / filename)
    for path in work_dir.glob("train*_teacher_openai_batch.parquet"):
        _cleanup_path(path)
    for path in work_dir.glob("train*_teacher_openai_batch.meta.json"):
        _cleanup_path(path)

    log.info(
        "Resharded pending tail in %s: preserved %d completed shard(s), rebuilt %d shard(s), new total=%d, target_tokens=%d",
        work_dir,
        completed_prefix_count,
        len(rebuilt_specs),
        len(updated_shards),
        target_estimated_tokens,
    )
    return updated_index


def _cleanup_batch_workspace(work_dir: Path) -> None:
    generated_paths = [
        work_dir / "shards",
        work_dir / "batch_input.jsonl",
        work_dir / "manifest.jsonl",
        work_dir / "shard_index.json",
        work_dir / "batch_meta.json",
        work_dir / "batch_status.json",
        work_dir / "batch_output.jsonl",
        work_dir / "batch_error.jsonl",
        work_dir / "raw_outputs.json",
        work_dir / "error_summary.json",
    ]
    for path in generated_paths:
        _cleanup_path(path)
    for path in work_dir.glob("train*_teacher_openai_batch.parquet"):
        _cleanup_path(path)
    for path in work_dir.glob("train*_teacher_openai_batch.meta.json"):
        _cleanup_path(path)


def prepare_openai_batch_job(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    tier: str,
    model_id: str,
    work_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare sharded OpenAI Batch API request files for checklist SFT generation."""
    batch_input_path = work_dir / "batch_input.jsonl"
    manifest_path = work_dir / "manifest.jsonl"
    shard_index_path = work_dir / "shard_index.json"
    shards_root = work_dir / "shards"

    if resume and shard_index_path.exists() and batch_input_path.exists() and manifest_path.exists():
        shard_index = _read_json(shard_index_path)
        if _shard_index_is_current(shard_index):
            shard_index = _maybe_reshard_pending_tail(
                work_dir=work_dir,
                shard_index=shard_index,
                target_estimated_tokens=OPENAI_BATCH_TARGET_ESTIMATED_TOKENS,
            )
            _build_aggregate_batch_meta(work_dir, tier=tier, model_id=str(model_id))
            _build_aggregate_batch_status(work_dir, tier=tier, model_id=str(model_id))
            log.info(
                "Prepare resume: using existing shard plan in %s (%d shards)",
                work_dir,
                int(shard_index.get("n_shards", 0)),
            )
            return {
                "work_dir": str(work_dir),
                "batch_input_path": str(batch_input_path),
                "manifest_path": str(manifest_path),
                "shard_index_path": str(shard_index_path),
                "n_requests": int(shard_index.get("n_requests", 0)),
                "n_shards": int(shard_index.get("n_shards", 0)),
            }
        log.info("Existing shard plan is outdated; rebuilding with token-aware sharding.")

    _cleanup_batch_workspace(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shards_root.mkdir(parents=True, exist_ok=True)

    all_request_lines: list[str] = []
    all_manifest_lines: list[str] = []
    total_requests = 0

    with open(batch_input_path, "w", encoding="utf-8") as f_all_req, open(
        manifest_path, "w", encoding="utf-8"
    ) as f_all_manifest:
        for sample_idx, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Preparing batch prompts")
        ):
            domain = row["domain"]
            winner = row["winner"]
            n_questions = expected_question_count(domain, checklists)

            for side in ("A", "B"):
                custom_id = _make_custom_id(sample_idx, side)
                prompt_text = build_checkeval_prompt(
                    row, checklists, definitions, domain=domain, side=side,
                )
                request_record = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "temperature": 0.0,
                        "max_completion_tokens": 2048,
                    },
                }
                manifest_record = {
                    "custom_id": custom_id,
                    "sample_idx": sample_idx,
                    "side": side,
                    "domain": domain,
                    "winner": winner,
                    "is_chosen": side == winner,
                    "n_questions": n_questions,
                    "prompt_text": prompt_text,
                }

                request_line = _jsonl_line(request_record)
                manifest_line = _jsonl_line(manifest_record)
                request_bytes = _jsonl_line_bytes(request_line)
                request_estimated_tokens = _estimate_request_tokens_from_line(request_line)

                if request_bytes > OPENAI_BATCH_TARGET_FILE_BYTES:
                    raise ValueError(
                        f"Single request {custom_id} exceeds shard size budget "
                        f"({request_bytes} > {OPENAI_BATCH_TARGET_FILE_BYTES})."
                    )
                if request_estimated_tokens > OPENAI_BATCH_TARGET_ESTIMATED_TOKENS:
                    raise ValueError(
                        f"Single request {custom_id} exceeds token shard budget "
                        f"({request_estimated_tokens} > {OPENAI_BATCH_TARGET_ESTIMATED_TOKENS})."
                    )

                if current_request_lines and (
                    current_bytes + request_bytes > OPENAI_BATCH_TARGET_FILE_BYTES
                    or current_estimated_tokens + request_estimated_tokens > OPENAI_BATCH_TARGET_ESTIMATED_TOKENS
                ):
                    _flush_shard()

                f_all_req.write(request_line)
                f_all_manifest.write(manifest_line)
                all_request_lines.append(request_line)
                all_manifest_lines.append(manifest_line)
                total_requests += 1

    shard_specs = _write_shard_group(
        request_lines=all_request_lines,
        manifest_lines=all_manifest_lines,
        shard_root=shards_root,
        shard_start_index=0,
        target_bytes=OPENAI_BATCH_TARGET_FILE_BYTES,
        target_estimated_tokens=OPENAI_BATCH_TARGET_ESTIMATED_TOKENS,
    )

    shard_index = {
        "prep_version": OPENAI_BATCH_PREP_VERSION,
        "n_shards": len(shard_specs),
        "n_requests": total_requests,
        "target_batch_input_bytes": OPENAI_BATCH_TARGET_FILE_BYTES,
        "max_batch_input_bytes": OPENAI_BATCH_MAX_FILE_BYTES,
        "target_estimated_tokens": OPENAI_BATCH_TARGET_ESTIMATED_TOKENS,
        "shards": shard_specs,
    }
    _write_json(shard_index_path, shard_index)

    log.info("Prepared %d batch requests across %d shard(s)", total_requests, len(shard_specs))
    log.info("Prepared combined batch input -> %s", batch_input_path)
    log.info("Prepared combined manifest -> %s", manifest_path)
    log.info("Prepared shard plan -> %s", shard_index_path)
    return {
        "work_dir": str(work_dir),
        "batch_input_path": str(batch_input_path),
        "manifest_path": str(manifest_path),
        "shard_index_path": str(shard_index_path),
        "n_requests": total_requests,
        "n_shards": len(shard_specs),
    }


def submit_openai_batch_job(
    work_dir: Path,
    tier: str,
    model_id: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Upload shard inputs and create one OpenAI batch job per shard."""
    shard_specs = _iter_shards(work_dir)
    for shard in shard_specs:
        shard_dir = Path(shard["work_dir"])
        shard_meta_path = shard_dir / "batch_meta.json"
        shard_meta: dict[str, Any] = _read_json(shard_meta_path) if shard_meta_path.exists() else {}
        if resume and shard_meta.get("batch_id"):
            log.info("Submit resume: %s -> existing batch %s", shard["shard_id"], shard_meta["batch_id"])
            continue
        _submit_single_shard_batch(shard, tier=tier, model_id=model_id, resume=resume)

    return _build_aggregate_batch_meta(work_dir, tier=tier, model_id=model_id)


def _build_aggregate_batch_meta(
    work_dir: Path,
    tier: str,
    model_id: str,
) -> dict[str, Any]:
    shard_specs = _iter_shards(work_dir)
    shard_metas: list[dict[str, Any]] = []
    for shard in shard_specs:
        shard_meta_path = Path(shard["work_dir"]) / "batch_meta.json"
        if shard_meta_path.exists():
            shard_metas.append(_read_json(shard_meta_path))
        else:
            shard_metas.append(dict(shard))

    aggregate_meta = {
        "tier": tier,
        "teacher_backend": "openai_batch",
        "model_id": str(model_id),
        "n_shards": len(shard_metas),
        "batch_id": _single_or_none([s.get("batch_id") for s in shard_metas if s.get("batch_id")]),
        "input_file_id": _single_or_none([s.get("input_file_id") for s in shard_metas if s.get("input_file_id")]),
        "batch_ids": [s.get("batch_id") for s in shard_metas if s.get("batch_id")],
        "input_file_ids": [s.get("input_file_id") for s in shard_metas if s.get("input_file_id")],
        "shards": shard_metas,
    }
    _write_json(work_dir / "batch_meta.json", aggregate_meta)
    return aggregate_meta


def _build_aggregate_batch_status(
    work_dir: Path,
    tier: str,
    model_id: str,
) -> dict[str, Any]:
    shard_specs = _iter_shards(work_dir)
    shard_statuses: list[dict[str, Any]] = []
    total_completed = 0
    total_failed = 0
    total_requests = 0
    all_completed = True

    for shard in shard_specs:
        shard_dir = Path(shard["work_dir"])
        shard_status_path = shard_dir / "batch_status.json"
        if not shard_status_path.exists():
            shard_statuses.append({
                "shard_id": shard["shard_id"],
                "work_dir": str(shard_dir),
                "batch_id": None,
                "status": "pending",
                "input_file_id": None,
                "output_file_id": None,
                "error_file_id": None,
                "request_counts": {"completed": 0, "failed": 0, "total": 0},
                "errors": None,
            })
            all_completed = False
            continue

        shard_status = _read_json(shard_status_path)
        counts = shard_status.get("request_counts") or {}
        total_completed += int(counts.get("completed", 0))
        total_failed += int(counts.get("failed", 0))
        total_requests += int(counts.get("total", 0))
        all_completed = all_completed and shard_status.get("status") == "completed"
        shard_statuses.append({
            "shard_id": shard["shard_id"],
            "work_dir": str(shard_dir),
            "batch_id": shard_status.get("id"),
            "status": shard_status.get("status"),
            "input_file_id": shard_status.get("input_file_id"),
            "output_file_id": shard_status.get("output_file_id"),
            "error_file_id": shard_status.get("error_file_id"),
            "request_counts": counts,
            "errors": shard_status.get("errors"),
        })

    failed_shards = [s for s in shard_statuses if _is_failed_batch_status(s.get("status"))]
    if failed_shards:
        aggregate_status_name = "failed"
    elif all_completed:
        aggregate_status_name = "completed"
    else:
        aggregate_status_name = "running"

    aggregate_status = {
        "tier": tier,
        "teacher_backend": "openai_batch",
        "model_id": str(model_id),
        "n_shards": len(shard_statuses),
        "status": aggregate_status_name,
        "batch_id": _single_or_none([s["batch_id"] for s in shard_statuses if s.get("batch_id")]),
        "batch_ids": [s["batch_id"] for s in shard_statuses if s.get("batch_id")],
        "output_file_id": _single_or_none([s["output_file_id"] for s in shard_statuses if s.get("output_file_id")]),
        "output_file_ids": [s["output_file_id"] for s in shard_statuses if s.get("output_file_id")],
        "error_file_id": _single_or_none([s["error_file_id"] for s in shard_statuses if s.get("error_file_id")]),
        "error_file_ids": [s["error_file_id"] for s in shard_statuses if s.get("error_file_id")],
        "request_counts": {
            "completed": total_completed,
            "failed": total_failed,
            "total": total_requests,
        },
        "shards": shard_statuses,
    }
    _write_json(work_dir / "batch_status.json", aggregate_status)
    return aggregate_status


def _retry_on_auth_error(fn, *, retries: int = 3, delay: float = 20.0, label: str = ""):
    """Call *fn()* with retry on ``openai.AuthenticationError``.

    OpenAI occasionally returns 401 for scoped keys due to transient
    permission-propagation delays.  We sleep and retry up to *retries* times
    before re-raising.
    """
    for attempt in range(retries):
        try:
            return fn()
        except openai.AuthenticationError:
            if attempt < retries - 1:
                log.warning(
                    "%sAuthenticationError, retrying in %.0fs (%d/%d)…",
                    f"{label}: " if label else "",
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
            else:
                raise


def _submit_single_shard_batch(
    shard: dict[str, Any],
    tier: str,
    model_id: str,
    resume: bool = False,
) -> dict[str, Any]:
    client = _get_openai_client()
    shard_dir = Path(shard["work_dir"])
    shard_meta_path = shard_dir / "batch_meta.json"
    shard_meta: dict[str, Any] = _read_json(shard_meta_path) if shard_meta_path.exists() else {}

    if resume and shard_meta.get("batch_id"):
        return shard_meta

    batch_input_path = Path(shard["batch_input_path"])
    if not batch_input_path.exists():
        raise FileNotFoundError(f"Missing shard batch input file: {batch_input_path}")

    input_file_id = shard_meta.get("input_file_id")
    file_info: dict[str, Any] = shard_meta.get("input_file") or {}
    if not input_file_id:
        with open(batch_input_path, "rb") as f:
            input_file = _retry_on_auth_error(
                lambda: client.files.create(file=f, purpose="batch"),
                label=f"{shard['shard_id']} files.create",
            )
        input_file = _retry_on_auth_error(
            lambda: client.files.wait_for_processing(input_file.id, poll_interval=2.0),
            label=f"{shard['shard_id']} files.wait_for_processing",
        )
        input_file_id = input_file.id
        file_info = _sdk_to_dict(input_file)
        log.info("Uploaded %s input file: %s", shard["shard_id"], input_file_id)

    batch = _retry_on_auth_error(
        lambda: client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "task": "checklist_sft",
                "tier": tier,
                "model_id": str(model_id),
                "shard_id": shard["shard_id"],
            },
        ),
        label=f"{shard['shard_id']} batches.create",
    )
    batch_info = _sdk_to_dict(batch)
    shard_meta = {
        **shard,
        "tier": tier,
        "teacher_backend": "openai_batch",
        "model_id": str(model_id),
        "input_file_id": input_file_id,
        "input_file": file_info,
        "batch_id": batch_info.get("id"),
        "batch": batch_info,
    }
    _write_json(shard_meta_path, shard_meta)
    log.info("Created %s batch job: %s", shard["shard_id"], shard_meta["batch_id"])
    return shard_meta


def _poll_single_shard_batch(
    shard: dict[str, Any],
    poll_interval: float = 30.0,
) -> dict[str, Any]:
    client = _get_openai_client()
    shard_dir = Path(shard["work_dir"])
    shard_meta_path = shard_dir / "batch_meta.json"
    shard_status_path = shard_dir / "batch_status.json"
    if not shard_meta_path.exists():
        raise FileNotFoundError(f"Missing shard metadata: {shard_meta_path}")

    shard_meta = _read_json(shard_meta_path)
    batch_id = shard_meta.get("batch_id")
    if not batch_id:
        raise ValueError(f"batch_id missing in {shard_meta_path}")

    while True:
        batch = _retry_on_auth_error(
            lambda: client.batches.retrieve(batch_id),
            label=f"{shard['shard_id']} batches.retrieve",
        )
        batch_info = _sdk_to_dict(batch)
        _write_json(shard_status_path, batch_info)

        counts = batch_info.get("request_counts") or {}
        log.info(
            "%s batch %s status=%s completed=%s failed=%s total=%s",
            shard["shard_id"],
            batch_id,
            batch_info.get("status"),
            counts.get("completed", 0),
            counts.get("failed", 0),
            counts.get("total", 0),
        )

        if batch_info.get("status") == "completed":
            return batch_info
        if _is_failed_batch_status(batch_info.get("status")):
            joined = "; ".join(_batch_error_messages(batch_info)) or "batch failed without detailed error"
            if _batch_has_token_limit_error(batch_info):
                raise TokenLimitExceededError(f"{shard['shard_id']} failed ({batch_id}): {joined}")
            raise RuntimeError(f"{shard['shard_id']} failed ({batch_id}): {joined}")
        time.sleep(poll_interval)


def _cancel_stale_batches(work_dir: Path) -> None:
    """Cancel any non-completed batch jobs from previous runs to free enqueued token quota.

    Skips shards that are already completed (safe for --resume).
    """
    shard_specs = _iter_shards(work_dir)
    for shard in shard_specs:
        shard_dir = Path(shard["work_dir"])
        status_path = shard_dir / "batch_status.json"
        if status_path.exists():
            status = _read_json(status_path)
            if status.get("status") == "completed":
                continue
        # Cancel and reset any shard that has a batch_id but is not completed
        meta_path = shard_dir / "batch_meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            if meta.get("batch_id"):
                _cancel_remote_batch(shard_dir)
                _reset_single_shard_state(shard_dir)


def _cancel_all_org_active_batches() -> None:
    """Cancel ALL in-progress/validating/cancelling batch jobs in the org to free enqueued token quota."""
    QUOTA_CONSUMING_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
    client = _get_openai_client()
    cancelled = 0
    still_active = []
    try:
        batches = client.batches.list(limit=100)
        for batch in batches.data:
            if batch.status in QUOTA_CONSUMING_STATUSES:
                if batch.status != "cancelling":
                    try:
                        client.batches.cancel(batch.id)
                        cancelled += 1
                        log.info(
                            "Cancelled org-wide batch %s (status=%s, created=%s)",
                            batch.id, batch.status, batch.created_at,
                        )
                    except Exception as exc:
                        log.warning("Failed to cancel batch %s: %s", batch.id, exc)
                still_active.append(batch.id)
    except Exception as exc:
        log.warning("Failed to list batches: %s", exc)

    if not still_active:
        log.info("No active org-wide batches found.")
        return

    # Wait for all cancelling batches to reach a terminal state
    log.info(
        "Waiting for %d batch(es) to fully release quota (cancelled %d new)...",
        len(still_active), cancelled,
    )
    for _ in range(60):  # up to 5 minutes
        time.sleep(5)
        remaining = []
        for bid in still_active:
            try:
                b = client.batches.retrieve(bid)
                if b.status in QUOTA_CONSUMING_STATUSES:
                    remaining.append(bid)
            except Exception:
                pass
        if not remaining:
            log.info("All batches fully cancelled/completed. Quota released.")
            return
        still_active = remaining
    log.warning(
        "%d batch(es) still not terminal after waiting: %s",
        len(still_active), ", ".join(still_active),
    )


def run_openai_batch_job_all_sequential(
    work_dir: Path,
    tier: str,
    model_id: str,
    poll_interval: float = 30.0,
    resume: bool = False,
    token_limit_retry_wait: float = 60.0,
    max_token_limit_retries: int = 100,
) -> None:
    """Run submit+poll sequentially per shard to avoid enqueued-token spikes."""
    # Cancel ALL active batches in the org (not just this work_dir) to free token quota
    _cancel_all_org_active_batches()
    _cancel_stale_batches(work_dir)
    shard_specs = _iter_shards(work_dir)
    for shard in shard_specs:
        shard_dir = Path(shard["work_dir"])
        shard_status_path = shard_dir / "batch_status.json"
        token_limit_retries = 0

        while True:
            if resume and shard_status_path.exists():
                shard_status = _read_json(shard_status_path)
                status = shard_status.get("status")
                if status == "completed":
                    log.info("Run-all resume: %s already completed, skipping submit/poll", shard["shard_id"])
                    break
                if _is_failed_batch_status(status):
                    if _batch_has_token_limit_error(shard_status):
                        token_limit_retries += 1
                        if token_limit_retries > max_token_limit_retries:
                            raise RuntimeError(
                                f"{shard['shard_id']} exceeded max token-limit retries "
                                f"({max_token_limit_retries})."
                            )
                        log.warning(
                            "%s previously failed due to token queue limit. Waiting %.0fs before retry %d/%d.",
                            shard["shard_id"],
                            token_limit_retry_wait,
                            token_limit_retries,
                            max_token_limit_retries,
                        )
                        _reset_single_shard_state(shard_dir)
                        _build_aggregate_batch_meta(work_dir, tier=tier, model_id=model_id)
                        _build_aggregate_batch_status(work_dir, tier=tier, model_id=model_id)
                        time.sleep(token_limit_retry_wait)
                    else:
                        joined = "; ".join(_batch_error_messages(shard_status)) or "batch failed without detailed error"
                        raise RuntimeError(f"{shard['shard_id']} has a non-retryable failed batch: {joined}")

            try:
                _submit_single_shard_batch(shard, tier=tier, model_id=model_id, resume=resume)
                _build_aggregate_batch_meta(work_dir, tier=tier, model_id=model_id)
                _poll_single_shard_batch(shard, poll_interval=poll_interval)
                _build_aggregate_batch_status(work_dir, tier=tier, model_id=model_id)
                break
            except TokenLimitExceededError as exc:
                token_limit_retries += 1
                if token_limit_retries > max_token_limit_retries:
                    raise RuntimeError(
                        f"{shard['shard_id']} exceeded max token-limit retries "
                        f"({max_token_limit_retries}). Last error: {exc}"
                    ) from exc
                log.warning(
                    "%s hit token queue limit. Waiting %.0fs before retry %d/%d.",
                    shard["shard_id"],
                    token_limit_retry_wait,
                    token_limit_retries,
                    max_token_limit_retries,
                )
                _reset_single_shard_state(shard_dir)
                _build_aggregate_batch_meta(work_dir, tier=tier, model_id=model_id)
                _build_aggregate_batch_status(work_dir, tier=tier, model_id=model_id)
                time.sleep(token_limit_retry_wait)

    _build_aggregate_batch_meta(work_dir, tier=tier, model_id=model_id)
    _build_aggregate_batch_status(work_dir, tier=tier, model_id=model_id)


def poll_openai_batch_job(
    work_dir: Path,
    poll_interval: float = 30.0,
) -> dict[str, Any]:
    """Poll all shard batch jobs until completion; fail fast on terminal errors."""
    meta_path = work_dir / "batch_meta.json"
    status_path = work_dir / "batch_status.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing batch metadata: {meta_path}")

    meta = _read_json(meta_path)
    shard_specs = _iter_shards(work_dir)
    client = _get_openai_client()

    while True:
        shard_statuses: list[dict[str, Any]] = []
        total_completed = 0
        total_failed = 0
        total_requests = 0
        all_completed = True

        for shard in shard_specs:
            shard_dir = Path(shard["work_dir"])
            shard_meta_path = shard_dir / "batch_meta.json"
            shard_meta = _read_json(shard_meta_path)
            batch_id = shard_meta.get("batch_id")
            if not batch_id:
                raise ValueError(f"batch_id missing in {shard_meta_path}")

            batch = client.batches.retrieve(batch_id)
            batch_info = _sdk_to_dict(batch)
            _write_json(shard_dir / "batch_status.json", batch_info)

            counts = batch_info.get("request_counts") or {}
            total_completed += int(counts.get("completed", 0))
            total_failed += int(counts.get("failed", 0))
            total_requests += int(counts.get("total", 0))
            all_completed = all_completed and batch_info.get("status") == "completed"

            shard_statuses.append({
                "shard_id": shard["shard_id"],
                "work_dir": str(shard_dir),
                "batch_id": batch_id,
                "status": batch_info.get("status"),
                "input_file_id": batch_info.get("input_file_id"),
                "output_file_id": batch_info.get("output_file_id"),
                "error_file_id": batch_info.get("error_file_id"),
                "request_counts": counts,
                "errors": batch_info.get("errors"),
            })

        aggregate_status = _build_aggregate_batch_status(
            work_dir,
            tier=str(meta.get("tier")),
            model_id=str(meta.get("model_id")),
        )

        failed_shards = [s for s in shard_statuses if _is_failed_batch_status(s["status"])]
        if failed_shards:
            messages = []
            for shard in failed_shards:
                shard_disk_status = _read_json(Path(shard["work_dir"]) / "batch_status.json")
                error_messages = _batch_error_messages(shard_disk_status)
                joined = "; ".join(error_messages) if error_messages else "batch failed without detailed error file"
                messages.append(f"{shard['shard_id']} ({shard['batch_id']}): {joined}")
            raise RuntimeError("OpenAI batch failed: " + " | ".join(messages))

        log.info(
            "Batch shards: %d/%d completed, requests completed=%d failed=%d total=%d",
            sum(1 for s in shard_statuses if s["status"] == "completed"),
            len(shard_statuses),
            total_completed,
            total_failed,
            total_requests,
        )

        if all_completed:
            return aggregate_status
        time.sleep(poll_interval)


def _download_batch_file(client: Any, file_id: str, target_path: Path) -> None:
    content = client.files.content(file_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content.write_to_file(target_path)


def _extract_text_segments(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)
    return str(content)


def _extract_chat_completion_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    if not isinstance(message, dict):
        return ""
    return _extract_text_segments(message.get("content")).strip()


def _summarize_error_file(error_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[str, int] = {}
    by_message: dict[str, int] = {}
    custom_ids: list[str] = []

    for record in error_records:
        custom_id = record.get("custom_id")
        if isinstance(custom_id, str):
            custom_ids.append(custom_id)

        error_obj = record.get("error")
        if not isinstance(error_obj, dict):
            response = record.get("response") or {}
            if isinstance(response, dict):
                body = response.get("body") or {}
                if isinstance(body, dict) and isinstance(body.get("error"), dict):
                    error_obj = body["error"]
                else:
                    error_obj = {}
            else:
                error_obj = {}

        code = (
            error_obj.get("code")
            or error_obj.get("type")
            or str((record.get("response") or {}).get("status_code") or "unknown")
        )
        message = error_obj.get("message") or "unknown"
        by_code[code] = by_code.get(code, 0) + 1
        by_message[message] = by_message.get(message, 0) + 1

    return {
        "n_errors": len(error_records),
        "by_code": by_code,
        "by_message": by_message,
        "custom_ids": custom_ids,
    }


def download_openai_batch_results(
    work_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Download shard outputs, merge them, and align by custom_id."""
    meta_path = work_dir / "batch_meta.json"
    status_path = work_dir / "batch_status.json"
    output_path = work_dir / "batch_output.jsonl"
    error_path = work_dir / "batch_error.jsonl"
    raw_outputs_path = work_dir / "raw_outputs.json"
    error_summary_path = work_dir / "error_summary.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing batch metadata: {meta_path}")

    meta = _read_json(meta_path)
    aggregate_status = _read_json(status_path) if status_path.exists() else {}
    shard_specs = _iter_shards(work_dir)

    if aggregate_status:
        failed_shards = [s for s in aggregate_status.get("shards", []) if _is_failed_batch_status(s.get("status"))]
        if failed_shards:
            raise RuntimeError("Cannot download outputs because at least one batch shard failed.")
        pending_shards = [s for s in aggregate_status.get("shards", []) if s.get("status") != "completed"]
        if pending_shards:
            raise RuntimeError("Cannot download outputs because batch shards are not completed yet.")

    if resume and raw_outputs_path.exists():
        log.info("Download resume: using existing raw outputs from %s", raw_outputs_path)
        return {
            "batch_id": meta.get("batch_id"),
            "batch_ids": meta.get("batch_ids", []),
            "raw_outputs_path": str(raw_outputs_path),
        }

    client = _get_openai_client()
    raw_outputs: dict[str, str] = {}
    merged_error_records: list[dict[str, Any]] = []
    shard_results: list[dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as f_output:
        for shard in shard_specs:
            shard_dir = Path(shard["work_dir"])
            shard_status_path = shard_dir / "batch_status.json"
            if not shard_status_path.exists():
                raise FileNotFoundError(f"Missing shard status file: {shard_status_path}")
            shard_status = _read_json(shard_status_path)
            if shard_status.get("status") != "completed":
                raise RuntimeError(
                    f"Cannot download shard {shard['shard_id']} because status is {shard_status.get('status')}"
                )

            output_file_id = shard_status.get("output_file_id")
            if not output_file_id:
                raise RuntimeError(f"Shard {shard['shard_id']} completed without output_file_id")

            shard_output_path = shard_dir / "batch_output.jsonl"
            if not (resume and shard_output_path.exists()):
                _download_batch_file(client, output_file_id, shard_output_path)
                log.info("Downloaded %s output -> %s", shard["shard_id"], shard_output_path)

            with open(shard_output_path, "r", encoding="utf-8") as f_shard_output:
                for line in f_shard_output:
                    if not line.strip():
                        continue
                    f_output.write(line)
                    record = json.loads(line)
                    custom_id = record.get("custom_id")
                    if not isinstance(custom_id, str):
                        continue
                    response = record.get("response") or {}
                    body = response.get("body") if isinstance(response, dict) else {}
                    if isinstance(body, dict):
                        if custom_id in raw_outputs:
                            raise ValueError(f"Duplicate custom_id in merged outputs: {custom_id}")
                        raw_outputs[custom_id] = _extract_chat_completion_text(body)

            shard_error_file_id = shard_status.get("error_file_id")
            if shard_error_file_id:
                shard_error_path = shard_dir / "batch_error.jsonl"
                if not (resume and shard_error_path.exists()):
                    _download_batch_file(client, shard_error_file_id, shard_error_path)
                    log.info("Downloaded %s error file -> %s", shard["shard_id"], shard_error_path)
                merged_error_records.extend(_read_jsonl(shard_error_path))

            shard_results.append({
                "shard_id": shard["shard_id"],
                "work_dir": str(shard_dir),
                "output_file_id": output_file_id,
                "error_file_id": shard_status.get("error_file_id"),
            })

    _write_json(raw_outputs_path, raw_outputs)
    log.info("Saved aligned raw outputs -> %s", raw_outputs_path)

    if merged_error_records:
        _write_jsonl(error_path, merged_error_records)
        _write_json(error_summary_path, _summarize_error_file(merged_error_records))
        log.info("Saved merged error summary -> %s", error_summary_path)
    else:
        _cleanup_path(error_path)
        _cleanup_path(error_summary_path)

    result = {
        "batch_id": meta.get("batch_id"),
        "batch_ids": meta.get("batch_ids", []),
        "output_file_id": _single_or_none([s["output_file_id"] for s in shard_results if s.get("output_file_id")]),
        "output_file_ids": [s["output_file_id"] for s in shard_results if s.get("output_file_id")],
        "error_file_id": _single_or_none([s["error_file_id"] for s in shard_results if s.get("error_file_id")]),
        "error_file_ids": [s["error_file_id"] for s in shard_results if s.get("error_file_id")],
        "raw_outputs_path": str(raw_outputs_path),
        "shards": shard_results,
    }
    return result


def finalize_openai_batch_job(
    df: pd.DataFrame,
    checklists: dict[str, list[str]],
    definitions: dict[str, str],
    tier: str,
    mode: str,
    model_id: str,
    checklist_dir: Path,
    output_dir: Path,
    work_dir: Path,
    filter_valid: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Finalize OpenAI batch outputs into the standard checklist SFT parquet."""
    manifest_path = work_dir / "manifest.jsonl"
    raw_outputs_path = work_dir / "raw_outputs.json"
    meta_path = work_dir / "batch_meta.json"
    status_path = work_dir / "batch_status.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not raw_outputs_path.exists():
        raise FileNotFoundError(f"Missing raw outputs: {raw_outputs_path}")

    manifest = _read_jsonl(manifest_path)
    raw_outputs = _read_json(raw_outputs_path)
    batch_meta = _read_json(meta_path) if meta_path.exists() else {}
    batch_status = _read_json(status_path) if status_path.exists() else {}

    if batch_status:
        failed_shards = [s for s in batch_status.get("shards", []) if _is_failed_batch_status(s.get("status"))]
        if failed_shards:
            raise RuntimeError("Cannot finalize because at least one batch shard failed.")
        pending_shards = [s for s in batch_status.get("shards", []) if s.get("status") != "completed"]
        if pending_shards:
            raise RuntimeError("Cannot finalize because batch shards are not completed yet.")

    records: list[dict[str, Any]] = []
    n_missing_outputs = 0

    for item in tqdm(manifest, total=len(manifest), desc="Finalizing batch outputs"):
        custom_id = str(item["custom_id"])
        sample_idx = int(item["sample_idx"])
        side = str(item["side"])
        manifest_domain = str(item["domain"])
        manifest_winner = str(item["winner"])
        is_chosen = bool(item["is_chosen"])
        n_questions = int(item["n_questions"])
        prompt_text = str(item["prompt_text"])

        row = df.iloc[sample_idx]
        if row["domain"] != manifest_domain or row["winner"] != manifest_winner:
            log.warning(
                "Manifest mismatch for %s (sample_idx=%d): domain/winner changed; using manifest values",
                custom_id,
                sample_idx,
            )

        rebuilt_prompt = build_checkeval_prompt(
            row, checklists, definitions, domain=manifest_domain, side=side,
        )
        if prompt_text != rebuilt_prompt:
            log.warning(
                "Prompt mismatch for %s; using manifest prompt_text to preserve prepared request",
                custom_id,
            )

        raw = raw_outputs.get(custom_id, "")
        if not isinstance(raw, str):
            raw = str(raw)
        if custom_id not in raw_outputs:
            n_missing_outputs += 1

        parsed = parse_checkeval_output(raw, expected_n=n_questions)
        parse_valid = not parsed.get("_raw_fallback", False)

        records.append({
            "prompt_text": prompt_text,
            "completion_text": raw.strip(),
            "domain": manifest_domain,
            "side": side,
            "is_chosen": is_chosen,
            "parse_valid": parse_valid,
            "n_questions": n_questions,
        })

    sft_df = pd.DataFrame(records)
    if filter_valid:
        before = len(sft_df)
        sft_df = sft_df[sft_df["parse_valid"]].reset_index(drop=True)
        log.info("Filtered: %d -> %d valid samples", before, len(sft_df))

    final_name = _final_output_name(tier=tier, mode=mode, teacher_backend="openai_batch")
    final_root_path = output_dir / final_name
    final_work_path = work_dir / final_name
    final_meta = {
        "tier": tier,
        "mode": mode,
        "teacher_backend": "openai_batch",
        "model_id": str(model_id),
        "n_samples": len(sft_df),
        "n_valid": int(sft_df["parse_valid"].sum()) if len(sft_df) else 0,
        "parse_rate": float(sft_df["parse_valid"].mean()) if len(sft_df) else 0.0,
        "batch_id": batch_meta.get("batch_id"),
        "input_file_id": batch_meta.get("input_file_id"),
        "output_file_id": batch_status.get("output_file_id"),
        "error_file_id": batch_status.get("error_file_id"),
        "batch_ids": batch_meta.get("batch_ids", []),
        "input_file_ids": batch_meta.get("input_file_ids", []),
        "output_file_ids": batch_status.get("output_file_ids", []),
        "error_file_ids": batch_status.get("error_file_ids", []),
        "n_shards": int(batch_meta.get("n_shards", 1)),
        "n_missing_outputs": n_missing_outputs,
        "checklist_dir": str(checklist_dir),
        "manifest_path": str(manifest_path),
        "raw_outputs_path": str(raw_outputs_path),
        "work_dir": str(work_dir),
        "filter_valid": filter_valid,
    }

    final_work_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(final_work_path, index=False)
    _write_json(final_work_path.with_suffix(".meta.json"), final_meta)

    if final_root_path != final_work_path:
        final_root_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_work_path, final_root_path)
        shutil.copy2(
            final_work_path.with_suffix(".meta.json"),
            final_root_path.with_suffix(".meta.json"),
        )

    log.info("Saved final parquet -> %s", final_work_path)
    if final_root_path != final_work_path:
        log.info("Saved canonical parquet copy -> %s", final_root_path)

    return sft_df, final_meta, final_root_path


# summary


def print_summary(sft_df: pd.DataFrame) -> None:
    table = Table(title="Checklist SFT Data Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    parse_valid = int(sft_df["parse_valid"].sum()) if len(sft_df) else 0
    parse_rate = 100 * float(sft_df["parse_valid"].mean()) if len(sft_df) else 0.0

    table.add_row("Total samples", f"{len(sft_df):,}")
    table.add_row("Chosen side", f"{int(sft_df['is_chosen'].sum()) if len(sft_df) else 0:,}")
    table.add_row("Rejected side", f"{int((~sft_df['is_chosen']).sum()) if len(sft_df) else 0:,}")
    table.add_row("Parse valid", f"{parse_valid:,}")
    table.add_row("Parse rate", f"{parse_rate:.1f}%")

    if len(sft_df):
        for domain in sorted(sft_df["domain"].unique()):
            n = int((sft_df["domain"] == domain).sum())
            table.add_row(f"  {domain}", f"{n:,}")

    console.print(table)


# main


def main():
    parser = argparse.ArgumentParser(
        description="Generate checklist SFT data for joint training"
    )
    parser.add_argument(
        "--tier", type=str, default="debug_5k",
        choices=["debug_5k", "tier_10k", "tier_20k", "full"],
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "teacher"],
        help="synthetic = fast heuristic; teacher = vLLM, ChatGPT, or OpenAI Batch teacher model",
    )
    parser.add_argument(
        "--checklist-dir", type=str, default=None,
        help="Override checklist directory (default: checklists/filtered)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: data/checklist_sft)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model-id", type=str, default=str(cfg.JUDGE_MODEL_ID),
        help="Teacher model path/name. Examples: local Qwen path or gpt-4o-mini.",
    )
    parser.add_argument(
        "--teacher-backend", type=str, default="auto",
        choices=["auto", "vllm", "openai", "openai_batch"],
        help="Teacher inference backend. 'auto' maps gpt/o-series names to OpenAI.",
    )
    parser.add_argument(
        "--stage", type=str, default="run_all",
        choices=["prepare", "submit", "poll", "download", "finalize", "run_all"],
        help="Execution stage for teacher-backend=openai_batch.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="Polling interval in seconds for --stage poll / run_all.",
    )
    parser.add_argument(
        "--token-limit-retry-wait", type=float, default=60.0,
        help="Seconds to wait before retrying a shard after token_limit_exceeded.",
    )
    parser.add_argument(
        "--max-token-limit-retries", type=int, default=100,
        help="Maximum retries per shard for token_limit_exceeded during run_all.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing batch artifacts instead of recreating them.",
    )
    parser.add_argument(
        "--batch-output-dir", type=str, default=None,
        help="Override the OpenAI batch working directory.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max concurrent OpenAI requests when teacher-backend=openai.",
    )
    parser.add_argument(
        "--filter-valid", action="store_true",
        help="Only keep samples that parsed successfully (teacher mode)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checklist_dir = Path(args.checklist_dir) if args.checklist_dir else cfg.CHECKLISTS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else cfg.CHECKLIST_SFT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir = Path(args.batch_output_dir) if args.batch_output_dir else None

    log.info("Loading checklists from %s", checklist_dir)
    checklists, definitions = load_checklists(checklist_dir)
    total_q = sum(len(qs) for qs in checklists.values())
    log.info("  %d dimensions, %d total questions", len(checklists), total_q)

    df, src_path = _load_train_df(args.tier)
    log.info("Loaded %d pairwise samples from %s (ties excluded)", len(df), src_path.name)

    resolved_teacher_backend = None
    if args.mode == "synthetic":
        sft_df = build_synthetic_sft(df, checklists, definitions)
    else:
        resolved_teacher_backend = _infer_teacher_backend(args.model_id, args.teacher_backend)
        if resolved_teacher_backend == "openai_batch":
            work_dir = _get_batch_work_dir(
                output_dir=output_dir,
                batch_output_dir=batch_output_dir,
                model_id=args.model_id,
                tier=args.tier,
            )

            if args.dry_run:
                log.warning("--dry-run is ignored for teacher-backend=openai_batch stages.")

            if args.stage in {"prepare", "run_all"}:
                prepare_openai_batch_job(
                    df=df,
                    checklists=checklists,
                    definitions=definitions,
                    tier=args.tier,
                    model_id=str(args.model_id),
                    work_dir=work_dir,
                    resume=args.resume,
                )

            if args.stage == "run_all":
                run_openai_batch_job_all_sequential(
                    work_dir=work_dir,
                    tier=args.tier,
                    model_id=str(args.model_id),
                    poll_interval=args.poll_interval,
                    resume=args.resume,
                    token_limit_retry_wait=args.token_limit_retry_wait,
                    max_token_limit_retries=args.max_token_limit_retries,
                )
            elif args.stage == "submit":
                submit_openai_batch_job(
                    work_dir=work_dir,
                    tier=args.tier,
                    model_id=str(args.model_id),
                    resume=args.resume,
                )

            if args.stage == "poll":
                poll_openai_batch_job(
                    work_dir=work_dir,
                    poll_interval=args.poll_interval,
                )

            if args.stage in {"download", "run_all"}:
                download_openai_batch_results(
                    work_dir=work_dir,
                    resume=args.resume,
                )

            if args.stage in {"finalize", "run_all"}:
                sft_df, _, out_path = finalize_openai_batch_job(
                    df=df,
                    checklists=checklists,
                    definitions=definitions,
                    tier=args.tier,
                    mode=args.mode,
                    model_id=str(args.model_id),
                    checklist_dir=checklist_dir,
                    output_dir=output_dir,
                    work_dir=work_dir,
                    filter_valid=args.filter_valid,
                )
                print_summary(sft_df)
                console.print(f"\n[bold green]Done. Checklist SFT data -> {out_path}[/bold green]")
            else:
                console.print(f"\n[bold green]Done. OpenAI batch stage '{args.stage}' -> {work_dir}[/bold green]")
            return

        sft_df = build_teacher_sft(
            df,
            checklists,
            definitions,
            model_id=args.model_id,
            teacher_backend=args.teacher_backend,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
        )

    if args.filter_valid:
        before = len(sft_df)
        sft_df = sft_df[sft_df["parse_valid"]].reset_index(drop=True)
        log.info("Filtered: %d -> %d valid samples", before, len(sft_df))

    print_summary(sft_df)

    if args.dry_run:
        log.info("Dry run - not saving.")
        sample = sft_df.iloc[0]
        console.print("\n[bold]Sample:[/bold]")
        console.print(f"  [cyan]domain[/cyan]: {sample['domain']}")
        console.print(f"  [cyan]side[/cyan]: {sample['side']}")
        console.print(f"  [cyan]is_chosen[/cyan]: {sample['is_chosen']}")
        console.print(f"  [cyan]prompt (first 200 chars)[/cyan]:\n{sample['prompt_text'][:200]}...")
        console.print(f"  [cyan]completion[/cyan]:\n{sample['completion_text'][:300]}")
        return

    out_name = _final_output_name(
        tier=args.tier,
        mode=args.mode,
        teacher_backend=resolved_teacher_backend,
    )
    out_path = output_dir / out_name
    sft_df.to_parquet(out_path, index=False)
    log.info("Saved %d SFT samples to %s", len(sft_df), out_path)

    meta = {
        "tier": args.tier,
        "mode": args.mode,
        "teacher_backend": resolved_teacher_backend,
        "model_id": args.model_id if args.mode == "teacher" else None,
        "checklist_dir": str(checklist_dir),
        "n_samples": len(sft_df),
        "n_valid": int(sft_df["parse_valid"].sum()),
        "n_dimensions": len(checklists),
        "n_total_questions": total_q,
        "filter_valid": args.filter_valid,
    }
    meta_path = out_path.with_suffix(".meta.json")
    _write_json(meta_path, meta)
    log.info("Saved metadata to %s", meta_path)

    console.print(f"\n[bold green]Done. Checklist SFT data -> {out_path}[/bold green]")


if __name__ == "__main__":
    main()
