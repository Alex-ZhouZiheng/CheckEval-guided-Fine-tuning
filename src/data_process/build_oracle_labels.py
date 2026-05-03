#!/usr/bin/env python3
"""Build oracle supervision labels for per-question checklist utility.

This script runs the judge on the full frozen bank and emits:
1) per-(sample, qid) labels for selector training
2) per-sample aggregate outcomes for filtering/calibration

Usage:
    python src/data_process/build_oracle_labels.py \
        --bank checklists/v3_frozen \
        --split train --tier tier_10k \
        --out data/oracle/train_oracle_v3.parquet
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(
    0,
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data_process"),
)

import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from tqdm import tqdm
import yaml

import config as cfg
from prepare_data_reasoning import make_sample_id
from utils import (
    _PAIRWISE_TABLE,
    _select_dimensions,
    aggregate_checklist_score,
    build_pointwise_prompt_from_qids,
    compare_checklists_pairwise,
    compute_per_question_decisiveness,
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


def _load_bank_index(bank_dir: Path) -> pd.DataFrame:
    bank_index = bank_dir / "bank_index.parquet"
    if not bank_index.exists():
        raise FileNotFoundError(
            f"{bank_index} not found. Run build_bank_index.py first."
        )
    df = pd.read_parquet(bank_index).sort_values("qid", kind="stable").reset_index(drop=True)

    required = {"qid", "dimension", "question_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bank_index missing columns: {sorted(missing)}")

    if "definition" not in df.columns:
        df["definition"] = ""
    if "sub_aspect" not in df.columns:
        df["sub_aspect"] = ""
    return df


def _load_dimension_definitions(bank_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for yaml_path in sorted(bank_dir.glob("*_filtered.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dim = data.get("dimension", yaml_path.stem)
        out[dim] = data.get("definition", "")
    return out


def _active_qids_for_domain(bank_df: pd.DataFrame, domain: str) -> list[int]:
    allowed = _select_dimensions(str(domain))
    allowed_lower = {d.lower() for d in allowed}

    active = bank_df[
        bank_df["dimension"].map(lambda d: d in allowed or str(d).lower() in allowed_lower)
    ]
    if active.empty:
        # Fallback for non-standard dimension names.
        active = bank_df
    return active["qid"].astype(int).tolist()


def _build_prompt(
    row: pd.Series,
    qrows: pd.DataFrame,
    definitions: dict[str, str],
    side: str,
) -> str:
    qids = qrows["qid"].astype(int).tolist()
    qmeta = {
        int(r["qid"]): {
            "dimension": str(r["dimension"]),
            "question_text": str(r["question_text"]),
            "definition": definitions.get(str(r["dimension"]), ""),
        }
        for _, r in qrows.iterrows()
    }
    return build_pointwise_prompt_from_qids(row, qids=qids, qmeta=qmeta, side=side)


def _load_pairs(
    split: str,
    tier: str | None,
    input_path: Path | None,
    max_samples: int | None,
) -> pd.DataFrame:
    if input_path is not None:
        path = input_path
    elif split == "train" and tier and tier != "full":
        path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    else:
        path = cfg.SPLITS_DIR / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    required = {"prompt_id", "domain", "context", "response_a", "response_b", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)

    if "sample_id" not in df.columns:
        df["sample_id"] = df.apply(
            lambda r: make_sample_id(
                prompt_id=r["prompt_id"],
                response_a=r["response_a"],
                response_b=r["response_b"],
                winner=r["winner"],
            ),
            axis=1,
        )

    log.info("Loaded %s rows from %s", f"{len(df):,}", path)
    return df.reset_index(drop=True)


def _http_judge_generate(
    messages_list: list[list[dict[str, str]]],
    url: str,
    model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    concurrency: int = 32,
    extra_body_mode: str = "auto",
    reasoning_effort: str | None = None,
    request_keys: list[str] | None = None,
    resume_cache: Path | None = None,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key=api_key)
    extra_body = _http_extra_body(extra_body_mode, url)
    if request_keys is None:
        request_keys = [str(i) for i in range(len(messages_list))]
    if len(request_keys) != len(messages_list):
        raise ValueError("request_keys length must match messages_list length")

    signature = _http_request_signature(
        url=url,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        extra_body=extra_body,
        reasoning_effort=reasoning_effort,
    )
    prompt_hashes = [_prompt_hash(msgs) for msgs in messages_list]
    cached = _load_http_resume_cache(resume_cache, signature)
    out: list[dict[str, Any] | None] = [None] * len(messages_list)
    pending: list[int] = []
    for idx, (key, prompt_sha) in enumerate(zip(request_keys, prompt_hashes)):
        response = cached.get((key, prompt_sha))
        if response is None:
            pending.append(idx)
        else:
            out[idx] = response

    if resume_cache is not None:
        log.info(
            "HTTP resume cache %s: %d cached, %d pending",
            resume_cache,
            len(messages_list) - len(pending),
            len(pending),
        )
        resume_cache.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()

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
            "usage": _dump_obj(getattr(resp, "usage", None)),
            "message": _dump_obj(msg),
        }

    def save_cache(idx: int, response: dict[str, Any]) -> None:
        if resume_cache is None:
            return
        record = {
            "request_key": request_keys[idx],
            "prompt_sha256": prompt_hashes[idx],
            "signature": signature,
            "response": response,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with write_lock:
            with resume_cache.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(one, messages_list[idx]): idx for idx in pending}
            errors: list[tuple[int, BaseException]] = []
            with tqdm(total=len(pending), desc="Judge HTTP") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        response = fut.result()
                    except BaseException as exc:
                        errors.append((idx, exc))
                    else:
                        out[idx] = response
                        save_cache(idx, response)
                    pbar.update(1)
            if errors:
                first_idx, first_error = errors[0]
                log.error(
                    "HTTP judge failed for %d/%d pending requests; completed requests were cached.",
                    len(errors),
                    len(pending),
                )
                raise RuntimeError(
                    f"HTTP request failed for {request_keys[first_idx]}"
                ) from first_error

    missing = [request_keys[idx] for idx, response in enumerate(out) if response is None]
    if missing:
        raise RuntimeError(f"Missing HTTP responses after generation: {missing[:5]}")
    return [response for response in out if response is not None]


def _http_request_signature(
    url: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None,
    reasoning_effort: str | None,
) -> str:
    payload = {
        "url": url,
        "model": model,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "extra_body": extra_body,
        "reasoning_effort": reasoning_effort,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_http_resume_cache(
    cache_path: Path | None,
    signature: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}

    out: dict[tuple[str, str], dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed cache line %s in %s", line_no, cache_path)
                continue
            if record.get("signature") != signature:
                continue
            key = record.get("request_key")
            prompt_sha = record.get("prompt_sha256")
            response = record.get("response")
            if not key or not prompt_sha or not isinstance(response, dict):
                continue
            out[(str(key), str(prompt_sha))] = response
    return out


def _http_extra_body(extra_body_mode: str, judge_url: str) -> dict[str, Any] | None:
    mode = extra_body_mode
    if mode == "auto":
        mode = "deepseek-thinking-on" if "deepseek.com" in judge_url else "qwen-thinking-off"

    if mode == "qwen-thinking-off":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if mode == "deepseek-thinking-on":
        return {"thinking": {"type": "enabled"}}
    if mode == "deepseek-thinking-off":
        return {"thinking": {"type": "disabled"}}
    if mode == "none":
        return None
    raise ValueError(f"Unknown HTTP extra body mode: {extra_body_mode}")


def _dump_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


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


def _response_parse_text(response: dict[str, Any]) -> str:
    content = str(response.get("content") or "")
    if content.strip():
        return content
    return str(response.get("reasoning_content") or "")


def _resolve_http_api_key(judge_url: str, judge_api_key: str, judge_api_key_env: str | None) -> str:
    if judge_api_key_env:
        resolved = _os.environ.get(judge_api_key_env)
        if not resolved:
            raise SystemExit(f"{judge_api_key_env} is not set")
        return resolved

    if judge_api_key != "EMPTY":
        return judge_api_key

    if "deepseek.com" in judge_url:
        resolved = _os.environ.get("DEEPSEEK_API_KEY")
        if not resolved:
            raise SystemExit(
                "DEEPSEEK_API_KEY is not set. Put it in .env, export it, or pass --judge-api-key."
            )
        return resolved

    return _os.environ.get("OPENAI_API_KEY", judge_api_key)


def build_vllm_speculative_config(
    enable_mtp: bool,
    mtp_method: str,
    num_speculative_tokens: int,
) -> dict | None:
    if not enable_mtp:
        return None
    if num_speculative_tokens <= 0:
        raise ValueError("--mtp-num-speculative-tokens must be > 0 when --enable-mtp is set")
    return {
        "method": mtp_method,
        "num_speculative_tokens": int(num_speculative_tokens),
    }


def _generate_local(
    all_messages: list[list[dict[str, str]]],
    base_model: str,
    batch_size: int,
    max_new_tokens: int,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prompt_chunk_size: int,
    seed: int | None,
    backend: str | None = None,
    speculative_config: dict | None = None,
) -> tuple[list[str], float]:
    model = load_judge_model(
        model_id=base_model,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_config=speculative_config,
    )

    outputs: list[str] = []
    t0 = time.time()
    for start in range(0, len(all_messages), prompt_chunk_size):
        chunk = all_messages[start : start + prompt_chunk_size]
        outputs.extend(
            generate_batch(
                model,
                chunk,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        )
    elapsed = time.time() - t0
    return outputs, elapsed


def _attach_individual_preference(results: pd.DataFrame) -> tuple[pd.DataFrame, int, str | None]:
    """Join raw HelpSteer3 ``individual_preference`` via prompt_id sha256."""
    import hashlib as _hashlib

    def _ctx_to_text(ctx) -> str:
        return "\n\n".join(f"[{t['role']}]\n{t['content']}" for t in ctx)

    def _pid(text: str) -> str:
        return _hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    needed = set(results["prompt_id"].dropna())
    raw_dir = cfg.DATA_DIR / "raw"
    out = results.copy()
    best_source = None
    best_filled = 0

    for fname in ("helpsteer3_train.parquet", "helpsteer3_test.parquet"):
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_parquet(fpath, columns=["context", "individual_preference"])
        if "individual_preference" not in raw.columns:
            continue
        raw_pids = raw["context"].apply(lambda c: _pid(_ctx_to_text(c)))
        mask = raw_pids.isin(needed)
        lookup = dict(zip(raw_pids[mask], raw.loc[mask, "individual_preference"]))
        candidate = out["prompt_id"].map(lookup)
        filled = int(candidate.notna().sum())
        if filled > best_filled:
            out["individual_preference"] = candidate
            best_filled = filled
            best_source = fname
        if filled == len(out):
            break

    return out, best_filled, best_source


def _local_label_map(parsed: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for a in parsed.get("answers", []):
        out[int(a["q"])] = str(a["answer"])
    for a in parsed.get("na_answers", []):
        out[int(a["q"])] = "na"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="Frozen bank dir with bank_index.parquet")
    parser.add_argument("--split", type=str, default="train", help="Split name under data/splits")
    parser.add_argument("--tier", type=str, default=None, help="Optional train tier, e.g. tier_10k")
    parser.add_argument("--input-path", type=Path, default=None, help="Optional explicit parquet input")
    parser.add_argument("--qid-subset", type=Path, default=None,
        help="Optional parquet with column 'qid'. Restrict per-sample question set to intersection of domain-active qids and this list.")
    parser.add_argument("--out", type=Path, required=True, help="Per-question oracle parquet output")
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=None,
        help="Optional per-sample output path (default: <out_stem>_sample.parquet)",
    )
    parser.add_argument("--base-model", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tie-delta", type=float, default=0.05)
    parser.add_argument("--prompt-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--judge-mode",
        choices=["local", "http", "vllm"],
        default="local",
        help="`local` dispatches via cfg.INFERENCE_BACKEND; `vllm` forces in-process vLLM "
             "(supports --enable-mtp); `http` uses an OpenAI-compatible endpoint.",
    )
    parser.add_argument("--judge-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", type=str, default=None, help="Required in --judge-mode http")
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default=None,
        help="Read the HTTP judge API key from this environment variable.",
    )
    parser.add_argument(
        "--http-extra-body",
        choices=["auto", "qwen-thinking-off", "deepseek-thinking-on", "deepseek-thinking-off", "none"],
        default="auto",
        help="Provider-specific extra_body for HTTP judge calls.",
    )
    parser.add_argument(
        "--http-reasoning-effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Optional reasoning_effort for HTTP judge calls when supported.",
    )
    parser.add_argument("--http-concurrency", type=int, default=32)
    parser.add_argument(
        "--http-resume-cache",
        type=Path,
        default=None,
        help="JSONL cache for completed HTTP requests (default: <out_stem>.http_cache.jsonl).",
    )
    parser.add_argument(
        "--no-http-resume",
        action="store_true",
        help="Disable HTTP request resume caching.",
    )

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
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=cfg.VLLM_ENGINE_KWARGS["max_num_seqs"],
        help="vLLM max_num_seqs for local judge mode.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="vLLM max_num_batched_tokens for local judge mode.",
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Enable vLLM MTP speculative decoding (only with --judge-mode vllm).",
    )
    parser.add_argument(
        "--mtp-method",
        type=str,
        default="mtp",
        help="vLLM speculative_config method, e.g. mtp or qwen3_next_mtp.",
    )
    parser.add_argument(
        "--mtp-num-speculative-tokens",
        type=int,
        default=1,
        help="vLLM MTP speculative depth.",
    )
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Include raw judge outputs in the per-sample parquet (large files).",
    )
    parser.add_argument(
        "--review-out",
        type=Path,
        default=None,
        help="If set, dump a review parquet for samples where the true winner "
             "received zero 'yes' answers. Compatible with src/evaluation/review_app.py.",
    )
    parser.add_argument(
        "--review-na-policy",
        type=str,
        default="skip",
        choices=["strict", "as_no", "skip", "partial"],
        help="na_policy for aggregating diagnostic scores in the review parquet.",
    )
    parser.add_argument(
        "--review-coverage-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--rerun-out",
        type=Path,
        default=None,
        help="If set, dump samples where the oracle disagreed with ground truth "
             "(wrong winner, Tie, or parse_fail) as a parquet usable directly as "
             "--input-path for a phase-2 stronger-model rerun. Sample_id is "
             "preserved so merge_oracle.py can splice strong answers back in.",
    )
    args = parser.parse_args()

    bank_dir = args.bank.resolve()
    bank_df = _load_bank_index(bank_dir)
    definitions = _load_dimension_definitions(bank_dir)

    qid_filter: set[int] | None = None
    if args.qid_subset is not None:
        sub = pd.read_parquet(args.qid_subset)
        qid_filter = set(sub["qid"].astype(int).tolist())
        log.info("Restricting to %d qids from --qid-subset (%s)", len(qid_filter), args.qid_subset)

    pairs = _load_pairs(
        split=args.split,
        tier=args.tier,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )

    # Build prompts once.
    messages_a: list[list[dict[str, str]]] = []
    messages_b: list[list[dict[str, str]]] = []
    metas: list[dict[str, Any]] = []

    bank_by_qid = bank_df.set_index("qid")

    for _, row in pairs.iterrows():
        active_qids = _active_qids_for_domain(bank_df, row["domain"])
        if qid_filter is not None:
            active_qids = [q for q in active_qids if q in qid_filter]
            if not active_qids:
                continue
        qrows = bank_df[bank_df["qid"].isin(active_qids)].sort_values("qid", kind="stable")
        n_q = len(qrows)
        if n_q == 0:
            continue

        prompt_a = _build_prompt(row, qrows, definitions, side="A")
        prompt_b = _build_prompt(row, qrows, definitions, side="B")

        messages_a.append([{"role": "user", "content": prompt_a}])
        messages_b.append([{"role": "user", "content": prompt_b}])
        metas.append(
            {
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "domain": row["domain"],
                "winner_gt": row["winner"],
                "context": row["context"],
                "response_a": row["response_a"],
                "response_b": row["response_b"],
                "active_qids": qrows["qid"].astype(int).tolist(),
                "n_questions": n_q,
            }
        )

    if not metas:
        raise SystemExit("No evaluable rows after bank/domain filtering.")

    all_messages = messages_a + messages_b

    if args.enable_mtp and args.judge_mode != "vllm":
        raise SystemExit("--enable-mtp requires --judge-mode vllm")

    http_resume_cache = None

    if args.judge_mode == "http":
        if not args.judge_model:
            raise SystemExit("--judge-model is required when --judge-mode http")
        judge_api_key = _resolve_http_api_key(
            judge_url=args.judge_url,
            judge_api_key=args.judge_api_key,
            judge_api_key_env=args.judge_api_key_env,
        )
        if not args.no_http_resume:
            http_resume_cache = (
                args.http_resume_cache.resolve()
                if args.http_resume_cache is not None
                else args.out.with_name(f"{args.out.stem}.http_cache.jsonl").resolve()
            )
        request_keys = (
            [f"{meta['sample_id']}:A" for meta in metas]
            + [f"{meta['sample_id']}:B" for meta in metas]
        )
        t0 = time.time()
        all_responses = _http_judge_generate(
            all_messages,
            url=args.judge_url,
            model=args.judge_model,
            api_key=judge_api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            concurrency=args.http_concurrency,
            extra_body_mode=args.http_extra_body,
            reasoning_effort=args.http_reasoning_effort,
            request_keys=request_keys,
            resume_cache=http_resume_cache,
        )
        all_outputs = [_response_parse_text(r) for r in all_responses]
        infer_elapsed = time.time() - t0
    else:
        backend_override = "vllm" if args.judge_mode == "vllm" else None
        speculative_config = build_vllm_speculative_config(
            enable_mtp=args.enable_mtp,
            mtp_method=args.mtp_method,
            num_speculative_tokens=args.mtp_num_speculative_tokens,
        )
        all_outputs, infer_elapsed = _generate_local(
            all_messages,
            base_model=args.base_model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prompt_chunk_size=args.prompt_chunk_size,
            seed=args.seed,
            backend=backend_override,
            speculative_config=speculative_config,
        )
        all_responses = [_local_judge_response(raw) for raw in all_outputs]

    n = len(metas)
    raw_a = all_outputs[:n]
    raw_b = all_outputs[n:]
    resp_a = all_responses[:n]
    resp_b = all_responses[n:]

    question_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    n_parse_ok = 0

    prompts_a_all = [m[0]["content"] for m in messages_a]
    prompts_b_all = [m[0]["content"] for m in messages_b]

    for idx, (meta, out_a, out_b, meta_a, meta_b) in enumerate(
        tqdm(zip(metas, raw_a, raw_b, resp_a, resp_b), total=n, desc="Parse oracle")
    ):
        qids = meta["active_qids"]
        n_q = meta["n_questions"]

        parsed_a = parse_checkeval_output(out_a, expected_n=n_q)
        parsed_b = parse_checkeval_output(out_b, expected_n=n_q)

        cmp = compare_checklists_pairwise(
            parsed_a,
            parsed_b,
            expected_n=n_q,
            tie_delta=args.tie_delta,
        )
        decisive = compute_per_question_decisiveness(
            parsed_a,
            parsed_b,
            expected_n=n_q,
            tie_delta=args.tie_delta,
        )

        if cmp is not None:
            n_parse_ok += 1
            winner_pred = cmp["winner"]
            margin_full = float(cmp["margin"])
            n_aligned_full = int(cmp["n_aligned"])
        else:
            winner_pred = None
            margin_full = None
            n_aligned_full = None

        local_a = _local_label_map(parsed_a)
        local_b = _local_label_map(parsed_b)

        # When either side failed to parse, `decisive` is None. Mark u3 as
        # NaN on those rows so downstream training can filter them out
        # instead of treating "not decisive" and "unknown" as identical.
        decisive_available = decisive is not None
        decisive_by_local_q: dict[int, bool] = {}
        if decisive_available:
            decisive_by_local_q = {
                int(q): bool(v.get("decisive", False))
                for q, v in decisive.get("per_question", {}).items()
            }

        for local_q, global_qid in enumerate(qids, start=1):
            if parsed_a.get("_raw_fallback"):
                ans_a = "parse_fail"
            else:
                ans_a = local_a.get(local_q, "na")

            if parsed_b.get("_raw_fallback"):
                ans_b = "parse_fail"
            else:
                ans_b = local_b.get(local_q, "na")

            parse_fail = (ans_a == "parse_fail") or (ans_b == "parse_fail")

            if not parse_fail:
                pair_contrib: float | None = float(_PAIRWISE_TABLE[(ans_a, ans_b)])
                u1: int | None = int((ans_a != "na") or (ans_b != "na"))
                u2: float | None = float(abs(pair_contrib))
            else:
                pair_contrib = None
                u1 = None
                u2 = None

            if decisive_available and not parse_fail:
                u3: float | None = float(bool(decisive_by_local_q.get(local_q, False)))
            else:
                u3 = None

            question_rows.append(
                {
                    "sample_id": meta["sample_id"],
                    "prompt_id": meta["prompt_id"],
                    "domain": meta["domain"],
                    "qid": int(global_qid),
                    "ans_a": ans_a,
                    "ans_b": ans_b,
                    "parse_fail": bool(parse_fail),
                    "pair_contrib": pair_contrib,
                    "u1_answerable": u1,
                    "u2_abs_contrib": u2,
                    "u3_decisive": u3,
                    "dim": str(bank_by_qid.loc[global_qid, "dimension"]),
                    "sub_aspect": str(bank_by_qid.loc[global_qid, "sub_aspect"]),
                    "question_text": str(bank_by_qid.loc[global_qid, "question_text"]),
                }
            )

        sample_row = {
            "sample_id": meta["sample_id"],
            "prompt_id": meta["prompt_id"],
            "domain": meta["domain"],
            "winner_gt": meta["winner_gt"],
            "winner_pred_full": winner_pred,
            "margin_full": margin_full,
            "n_aligned_full": n_aligned_full,
            "n_questions": n_q,
            "oracle_agrees_gt": (
                bool(winner_pred == meta["winner_gt"]) if winner_pred in {"A", "B", "Tie"} else None
            ),
            "context": meta["context"],
            "response_a": meta["response_a"],
            "response_b": meta["response_b"],
        }
        if args.save_raw_outputs:
            sample_row["raw_output_a"] = out_a
            sample_row["raw_output_b"] = out_b
            sample_row["raw_content_a"] = meta_a.get("content", "")
            sample_row["raw_content_b"] = meta_b.get("content", "")
            sample_row["raw_reasoning_content_a"] = meta_a.get("reasoning_content", "")
            sample_row["raw_reasoning_content_b"] = meta_b.get("reasoning_content", "")
            sample_row["raw_finish_reason_a"] = meta_a.get("finish_reason")
            sample_row["raw_finish_reason_b"] = meta_b.get("finish_reason")
            sample_row["raw_content_len_a"] = meta_a.get("content_len")
            sample_row["raw_content_len_b"] = meta_b.get("content_len")
            sample_row["raw_reasoning_len_a"] = meta_a.get("reasoning_len")
            sample_row["raw_reasoning_len_b"] = meta_b.get("reasoning_len")
            sample_row["raw_usage_a"] = json.dumps(meta_a.get("usage"), ensure_ascii=False, default=str)
            sample_row["raw_usage_b"] = json.dumps(meta_b.get("usage"), ensure_ascii=False, default=str)
            sample_row["raw_message_a"] = json.dumps(meta_a.get("message"), ensure_ascii=False, default=str)
            sample_row["raw_message_b"] = json.dumps(meta_b.get("message"), ensure_ascii=False, default=str)

        sample_rows.append(sample_row)

        # ── review parquet: true winner with zero "yes" answers ──
        if args.review_out is not None:
            winner_gt = meta["winner_gt"]
            parse_fail_either = bool(parsed_a.get("_raw_fallback")) or bool(parsed_b.get("_raw_fallback"))
            n_yes_winner = None
            if not parse_fail_either and winner_gt in {"A", "B"}:
                parsed_winner = parsed_a if winner_gt == "A" else parsed_b
                n_yes_winner = int(parsed_winner.get("n_yes", 0))

            if n_yes_winner == 0:
                agg_a = aggregate_checklist_score(
                    parsed_a,
                    na_policy=args.review_na_policy,
                    coverage_threshold=args.review_coverage_threshold,
                    expected_n=n_q,
                )
                agg_b = aggregate_checklist_score(
                    parsed_b,
                    na_policy=args.review_na_policy,
                    coverage_threshold=args.review_coverage_threshold,
                    expected_n=n_q,
                )
                score_a = agg_a["score"] if agg_a else None
                score_b = agg_b["score"] if agg_b else None
                n_yes_a = agg_a["n_yes"] if agg_a else int(parsed_a.get("n_yes", 0))
                n_yes_b = agg_b["n_yes"] if agg_b else int(parsed_b.get("n_yes", 0))

                pw_margin = margin_full
                pred = winner_pred
                if pred is None:
                    err_cat = "parse_failure"
                elif pred == "Tie":
                    err_cat = "tie"
                elif pred == winner_gt:
                    err_cat = "correct"
                else:
                    err_cat = "wrong_winner"

                review_rows.append({
                    "sample_id": meta["sample_id"],
                    "prompt_id": meta["prompt_id"],
                    "domain": meta["domain"],
                    "context": meta["context"],
                    "response_a": meta["response_a"],
                    "response_b": meta["response_b"],
                    "winner": winner_gt,
                    "predicted_winner": pred,
                    "score_a": score_a,
                    "score_b": score_b,
                    "pairwise_margin": pw_margin,
                    "n_yes_a": n_yes_a,
                    "n_yes_b": n_yes_b,
                    "n_na_a": int(parsed_a.get("n_na", 0)),
                    "n_na_b": int(parsed_b.get("n_na", 0)),
                    "na_qnums_a": parsed_a.get("na_qnums", []),
                    "na_qnums_b": parsed_b.get("na_qnums", []),
                    "expected_n_questions": n_q,
                    "error_category": err_cat,
                    "prompt_a": prompts_a_all[idx],
                    "prompt_b": prompts_b_all[idx],
                    "raw_output_a": out_a,
                    "raw_output_b": out_b,
                    "parsed_a_json": json.dumps(parsed_a, ensure_ascii=False, default=str),
                    "parsed_b_json": json.dumps(parsed_b, ensure_ascii=False, default=str),
                    "n_yes_winner": n_yes_winner,
                    "_review_split": "wrong",
                })

    q_df = pd.DataFrame(question_rows)
    s_df = pd.DataFrame(sample_rows)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_parquet(out_path, index=False)

    if args.sample_out is not None:
        sample_out = args.sample_out.resolve()
    else:
        sample_out = out_path.with_name(f"{out_path.stem}_sample.parquet")
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    s_df.to_parquet(sample_out, index=False)

    if "winner_pred_full" in s_df.columns:
        valid_mask = s_df["winner_pred_full"].isin(["A", "B"])
        n_valid = int(valid_mask.sum())
        n_tie = int((s_df["winner_pred_full"] == "Tie").sum())
        n_unparseable = int(s_df["winner_pred_full"].isna().sum())
        agree_valid = (
            float((s_df.loc[valid_mask, "winner_pred_full"] == s_df.loc[valid_mask, "winner_gt"]).mean())
            if n_valid else None
        )
    else:
        n_valid = n_tie = n_unparseable = 0
        agree_valid = None

    metrics = {
        "n_samples": int(len(s_df)),
        "n_valid": n_valid,
        "n_tie": n_tie,
        "n_unparseable": n_unparseable,
        "n_question_rows": int(len(q_df)),
        "parse_ok_rate": float(n_parse_ok / len(s_df)) if len(s_df) else 0.0,
        "avg_questions_per_sample": float(s_df["n_questions"].mean()) if len(s_df) else 0.0,
        "oracle_agreement_rate_total": float(s_df["oracle_agrees_gt"].dropna().mean()) if "oracle_agrees_gt" in s_df else None,
        "oracle_agreement_rate_valid": agree_valid,
        "inference_time_s": infer_elapsed,
        "samples_per_second": float(len(s_df) / infer_elapsed) if infer_elapsed > 0 else None,
        "bank": str(bank_dir),
        "split": args.split,
        "tier": args.tier,
        "base_model": args.base_model,
        "judge_mode": args.judge_mode,
        "http_extra_body": args.http_extra_body if args.judge_mode == "http" else None,
        "http_reasoning_effort": args.http_reasoning_effort if args.judge_mode == "http" else None,
        "http_resume_cache": str(http_resume_cache) if http_resume_cache is not None else None,
        "tie_delta": args.tie_delta,
        "enable_mtp": bool(args.enable_mtp) if args.judge_mode == "vllm" else False,
        "mtp_method": args.mtp_method if args.judge_mode == "vllm" and args.enable_mtp else None,
        "mtp_num_speculative_tokens": (
            args.mtp_num_speculative_tokens if args.judge_mode == "vllm" and args.enable_mtp else None
        ),
    }

    metrics_path = out_path.with_suffix(".meta.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("Saved oracle question rows -> %s", out_path)
    log.info("Saved oracle sample rows   -> %s", sample_out)
    log.info("Saved run metadata         -> %s", metrics_path)

    if args.rerun_out is not None:
        rerun_path = args.rerun_out.resolve()
        rerun_path.parent.mkdir(parents=True, exist_ok=True)
        rerun_records = []
        for meta, sr in zip(metas, sample_rows):
            if sr.get("oracle_agrees_gt") is True:
                continue
            rerun_records.append({
                "sample_id": meta["sample_id"],
                "prompt_id": meta["prompt_id"],
                "domain": meta["domain"],
                "context": meta["context"],
                "response_a": meta["response_a"],
                "response_b": meta["response_b"],
                "winner": meta["winner_gt"],
                "active_qids": meta["active_qids"],
                "n_questions": meta["n_questions"],
                "weak_winner_pred": sr["winner_pred_full"],
                "weak_margin": sr["margin_full"],
                "weak_oracle_agrees_gt": sr["oracle_agrees_gt"],
            })
        if not rerun_records:
            log.warning("No incorrect samples to dump for rerun. Skipping %s.", rerun_path)
        else:
            pd.DataFrame(rerun_records).to_parquet(rerun_path, index=False)
            log.info(
                "Saved rerun parquet (%d incorrect samples) -> %s",
                len(rerun_records),
                rerun_path,
            )
            log.info(
                "Phase 2: python src/data_process/build_oracle_labels.py "
                "--bank %s --input-path %s --base-model <strong-model> --out <strong_oracle.parquet>",
                bank_dir,
                rerun_path,
            )

    if args.review_out is not None:
        review_path = args.review_out.resolve()
        review_path.parent.mkdir(parents=True, exist_ok=True)
        if not review_rows:
            log.warning("No review rows (no sample had true winner with zero 'yes'). Skipping %s.", review_path)
        else:
            review_df = pd.DataFrame(review_rows)
            review_df, filled, src = _attach_individual_preference(review_df)
            if filled:
                log.info("Joined individual_preference: %d/%d rows from %s", filled, len(review_df), src)
            else:
                log.warning("individual_preference not attached (raw parquets missing in %s).", cfg.DATA_DIR / "raw")
            review_df.to_parquet(review_path, index=False)
            log.info("Saved review parquet (%d rows) -> %s", len(review_df), review_path)
            log.info(
                "Launch: streamlit run src/evaluation/review_app.py -- --results %s",
                review_path,
            )


if __name__ == "__main__":
    main()
