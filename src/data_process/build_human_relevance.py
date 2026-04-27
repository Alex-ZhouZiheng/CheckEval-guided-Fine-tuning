"""Extract human-implied yes-set from annotator reasoning over the v3 bank."""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data_process"))

import argparse
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
from prepare_data_reasoning import make_sample_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def build_extractor_prompt(reasoning: str, qmeta: list[dict]) -> str:
    lines = []
    for q in qmeta:
        lines.append(f"Q{int(q['qid'])} ({q['dimension']}): {q['question_text']}")
    questions_block = "\n".join(lines)
    n_questions = len(qmeta)

    return (
        "You are mapping a human annotator's free-text rationale onto a fixed "
        f"checklist of {n_questions} evaluation questions. Return the qids the rationale "
        "directly addresses (positively or negatively). If the rationale is "
        "ambiguous, return an empty list. Do not infer.\n\n"
        "[Checklist]\n"
        f"{questions_block}\n\n"
        "[Annotator rationale]\n"
        f"{reasoning}\n\n"
        "Return strict JSON: {\"mentioned_qids\": [int, ...]}"
    )


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\"mentioned_qids\"\s*:\s*\[[^\]]*\][^{}]*\}", re.DOTALL)
_QID_TOKEN_RE = re.compile(r"\b[Qq]?(\d{1,2})\b")


def parse_extractor_response(raw: str, valid_qids: set[int]) -> tuple[list[int], bool]:
    """Return (qids, fallback_used). Filters to valid_qids and dedupes."""
    if not isinstance(raw, str) or not raw.strip():
        return [], True

    # Strict JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("mentioned_qids"), list):
            qids = _filter(obj["mentioned_qids"], valid_qids)
            return qids, False
    except json.JSONDecodeError:
        pass

    # JSON object embedded in prose
    m = _JSON_OBJ_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            qids = _filter(obj.get("mentioned_qids", []), valid_qids)
            return qids, False
        except json.JSONDecodeError:
            pass

    # Regex fallback over Q-tokens
    matches = [int(m.group(1)) for m in _QID_TOKEN_RE.finditer(raw)]
    return _filter(matches, valid_qids), True


def _filter(seq, valid_qids: set[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for v in seq:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv in valid_qids and iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out


def aggregate_h(
    sample_id: str,
    prompt_id: str,
    per_annotator: list[dict],
    valid_qids: set[int],
) -> list[dict]:
    usable = [a for a in per_annotator if a.get("ok")]
    n = len(usable)
    if n == 0:
        return []

    counts: dict[int, int] = {}
    for a in usable:
        for qid in a["qids"]:
            counts[qid] = counts.get(qid, 0) + 1

    rows: list[dict] = []
    for qid, c in counts.items():
        if qid not in valid_qids:
            continue
        rows.append({
            "prompt_id": prompt_id,
            "sample_id": sample_id,
            "qid": int(qid),
            "h": float(c) / float(n),
            "n_annotators": int(n),
        })
    return rows


# ---------------------------------------------------------------------------
# Loaders and HTTP helpers
# ---------------------------------------------------------------------------

def _load_bank(bank_dir: Path) -> pd.DataFrame:
    p = bank_dir / "bank_index.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run build_bank_index.py first.")
    df = pd.read_parquet(p).sort_values("qid", kind="stable").reset_index(drop=True)
    return df[["qid", "dimension", "question_text"]]


def _ctx_to_text(ctx) -> str:
    return "\n\n".join(f"[{t['role']}]\n{t['content']}" for t in ctx)


def _pid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_split_with_ip(split: str, tier: str | None, input_path: Path | None) -> pd.DataFrame:
    """Load a split and join individual_preference from raw helpsteer3 parquets."""
    if input_path is not None:
        path = input_path
    elif split == "train" and tier and tier != "full":
        path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    else:
        path = cfg.SPLITS_DIR / f"{split}.parquet"

    df = pd.read_parquet(path)
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

    raw_dir = cfg.DATA_DIR / "raw"
    pid_to_ip: dict[str, np.ndarray] = {}
    for fname in ("helpsteer3_train.parquet", "helpsteer3_test.parquet"):
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_parquet(fpath, columns=["context", "individual_preference"])
        raw_pids = raw["context"].apply(lambda c: _pid(_ctx_to_text(c)))
        for pid, ip in zip(raw_pids, raw["individual_preference"]):
            pid_to_ip.setdefault(pid, ip)

    df["individual_preference"] = df["prompt_id"].map(pid_to_ip)
    n_ip = int(df["individual_preference"].notna().sum())
    log.info("Loaded %d rows from %s; joined IP for %d/%d", len(df), path, n_ip, len(df))
    return df.reset_index(drop=True)


def _http_extract(
    prompts: list[str],
    url: str,
    model: str,
    api_key: str,
    max_new_tokens: int,
    concurrency: int,
) -> list[str]:
    from openai import OpenAI

    client = OpenAI(base_url=url, api_key=api_key)

    def one(p: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p}],
                temperature=0.0,
                max_tokens=max_new_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            log.warning("HTTP call failed: %s", e)
            return ""

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(tqdm(ex.map(one, prompts), total=len(prompts), desc="Extract HTTP"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tier", type=str, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-url", type=str, default="http://127.0.0.1:8080/v1")
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument("--http-concurrency", type=int, default=4)
    args = parser.parse_args()

    bank_df = _load_bank(args.bank.resolve())
    valid_qids = set(bank_df["qid"].astype(int).tolist())
    qmeta = bank_df.to_dict(orient="records")

    df = _load_split_with_ip(args.split, args.tier, args.input_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples).reset_index(drop=True)

    # Build all (sample, annotator) prompt jobs upfront so HTTP concurrency
    # amortizes across all annotators.
    jobs: list[dict] = []
    for _, row in df.iterrows():
        ip = row.get("individual_preference")
        if ip is None or (isinstance(ip, float) and np.isnan(ip)):
            continue
        for a_idx, ann in enumerate(ip):
            reasoning = ann.get("reasoning") if isinstance(ann, dict) else None
            if not reasoning or not str(reasoning).strip():
                continue
            jobs.append({
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "annotator_idx": a_idx,
                "prompt": build_extractor_prompt(reasoning=str(reasoning), qmeta=qmeta),
            })

    log.info("Built %d (sample, annotator) extraction jobs", len(jobs))

    t0 = time.time()
    raws = _http_extract(
        [j["prompt"] for j in jobs],
        url=args.judge_url, model=args.judge_model,
        api_key=args.judge_api_key, max_new_tokens=args.max_new_tokens,
        concurrency=args.http_concurrency,
    )
    elapsed = time.time() - t0

    # Group annotator results by sample_id, then aggregate.
    by_sample: dict[str, list[dict]] = {}
    pid_by_sample: dict[str, str] = {}
    n_strict = 0
    n_fallback = 0
    n_empty = 0
    n_parse_fail = 0
    for j, raw in zip(jobs, raws):
        has_raw = bool(raw and raw.strip())
        ok = has_raw
        qids, used_fallback = parse_extractor_response(raw, valid_qids) if ok else ([], True)
        if ok and used_fallback and not qids:
            ok = False
            n_parse_fail += 1
        if not has_raw:
            n_empty += 1
        elif not ok:
            pass
        elif used_fallback:
            n_fallback += 1
        else:
            n_strict += 1
        by_sample.setdefault(j["sample_id"], []).append({
            "qids": qids, "fallback": used_fallback, "ok": ok,
        })
        pid_by_sample[j["sample_id"]] = j["prompt_id"]

    rows: list[dict] = []
    for sid, per_annotator in by_sample.items():
        rows.extend(aggregate_h(
            sample_id=sid, prompt_id=pid_by_sample[sid],
            per_annotator=per_annotator, valid_qids=valid_qids,
        ))

    out_df = pd.DataFrame(rows, columns=["prompt_id", "sample_id", "qid", "h", "n_annotators"])
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_jobs = len(jobs)
    meta = {
        "split": args.split,
        "tier": args.tier,
        "bank": str(args.bank.resolve()),
        "n_samples": int(df["sample_id"].nunique()),
        "n_jobs": n_jobs,
        "n_strict_parse": n_strict,
        "n_regex_fallback": n_fallback,
        "n_empty_response": n_empty,
        "n_parse_fail": n_parse_fail,
        "parse_strict_rate": (n_strict / n_jobs) if n_jobs else None,
        "fallback_rate": (n_fallback / n_jobs) if n_jobs else None,
        "n_h_rows": int(len(out_df)),
        "h_density_per_sample_mean": (
            float(out_df.groupby("sample_id").size().mean()) if len(out_df) else 0.0
        ),
        "elapsed_s": elapsed,
        "judge_model": args.judge_model,
        "judge_url": args.judge_url,
    }

    import json as _json
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved %d rows -> %s", len(out_df), out_path)
    log.info("Strict %d / fallback %d / empty %d (of %d jobs)",
             n_strict, n_fallback, n_empty, n_jobs)


if __name__ == "__main__":
    main()
