from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

OUT = Path("results/coverage_dev200")
INSTANCES = OUT / "dev200_instances.parquet"
INPUT = OUT / "atomic_claims_dev200_input.parquet"
CACHE = OUT / "atomic_claims_dev200_cache.jsonl"
OUTPUT = OUT / "atomic_claims_dev200.parquet"
PREVIEW = OUT / "atomic_claims_dev200.preview.jsonl"
SUMMARY = OUT / "atomic_claims_dev200_summary.json"

MODEL = os.environ.get("ATOMIC_CLAIM_MODEL", "deepseek-v4-flash")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CONCURRENCY = int(os.environ.get("ATOMIC_CLAIM_CONCURRENCY", "100"))

PROMPT = """You are decomposing a human preference rationale into atomic evaluation claims.

Task:
Given the user context, Response A, Response B, the human annotator's preferred response, and the annotator rationale, extract atomic claims that explain the preference.

Definition:
An atomic claim is one single evaluative reason that could be checked independently.
Each claim must compare Response A and Response B, or identify a concrete strength/weakness of one response that affects the pairwise preference.

Rules:
1. Split multi-part rationales into separate atomic claims.
2. Do not add reasons that are not stated or clearly implied by the rationale.
3. Do not simply say "A is better" or "B is better".
4. Keep each claim specific enough that a checklist question could address it.
5. If the rationale is vague, output the most faithful minimal claim and mark it as "vague".
6. Use Response A / Response B, not "first answer" or "second answer".
7. Follow the annotator rationale as written. Do not correct possible A/B mistakes using outside knowledge or other annotations.

Return valid JSON only.

Schema:
{{
  "claims": [
    {{
      "claim": "...",
      "target": "A" | "B" | "both" | "comparison",
      "aspect": "instruction_following" | "correctness" | "completeness" | "clarity" | "reasoning" | "safety" | "conciseness" | "format" | "other",
      "polarity": "pro_A" | "pro_B" | "neutral_or_diagnostic",
      "specificity": "specific" | "medium" | "vague"
    }}
  ]
}}

Input:
User context:
{context}

Response A:
{response_a}

Response B:
{response_b}

Human preferred response:
{gold_winner}

Annotator rationale:
{rationale_text}
"""

VALID_TARGETS = {"A", "B", "both", "comparison"}
VALID_ASPECTS = {
    "instruction_following",
    "correctness",
    "completeness",
    "clarity",
    "reasoning",
    "safety",
    "conciseness",
    "format",
    "other",
}
VALID_POLARITIES = {"pro_A", "pro_B", "neutral_or_diagnostic"}
VALID_SPEC = {"specific", "medium", "vague"}


def load_dotenv(path: Path = Path(".env")):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_input():
    df = pd.read_parquet(INSTANCES)
    rows = []
    for row in df.to_dict("records"):
        reasonings = row.get("human_reasonings")
        if reasonings is None:
            reasonings = []
        elif hasattr(reasonings, "tolist"):
            reasonings = reasonings.tolist()
        for i, txt in enumerate(reasonings, start=1):
            rows.append(
                {
                    "instance_id": row["instance_id"],
                    "domain": row["domain"],
                    "annotator_id": f"ann{i}",
                    "rationale_text": str(txt),
                    "gold_winner": row["gold_winner"],
                    "context": row["context"],
                    "response_a": row["response_a"],
                    "response_b": row["response_b"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_parquet(INPUT, index=False)
    out[["instance_id", "domain", "annotator_id", "gold_winner", "rationale_text"]].to_json(
        OUT / "atomic_claims_dev200_input.preview.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out


def cache_key(row):
    return f"{row['instance_id']}::{row['annotator_id']}"


def read_cache():
    done = {}
    if not CACHE.exists():
        return done
    with CACHE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done[rec["key"]] = rec
    return done


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def normalize_claims(obj):
    claims = obj.get("claims", []) if isinstance(obj, dict) else []
    out = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        claim = str(c.get("claim", "")).strip()
        if not claim:
            continue
        target = c.get("target")
        aspect = c.get("aspect")
        polarity = c.get("polarity")
        specificity = c.get("specificity")
        out.append(
            {
                "claim": claim,
                "target": target if target in VALID_TARGETS else "comparison",
                "aspect": aspect if aspect in VALID_ASPECTS else "other",
                "polarity": polarity if polarity in VALID_POLARITIES else "neutral_or_diagnostic",
                "specificity": specificity if specificity in VALID_SPEC else "medium",
            }
        )
    return out


def call_one(client, row, max_retries=3):
    content = PROMPT.format(**row.to_dict())
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            return raw, normalize_claims(extract_json(raw)), None
        except Exception as e:
            last_error = str(e)
            time.sleep(2 * (attempt + 1))
    return "", [], last_error


def write_outputs(df, done):
    rows = []
    errors = []
    for row in df.to_dict("records"):
        rec = done.get(f"{row['instance_id']}::{row['annotator_id']}")
        if not rec:
            continue
        if rec.get("error"):
            errors.append(rec)
        for atom_i, c in enumerate(rec.get("claims") or [], start=1):
            rows.append(
                {
                    "instance_id": row["instance_id"],
                    "domain": row["domain"],
                    "annotator_id": row["annotator_id"],
                    "rationale_text": row["rationale_text"],
                    "atom_id": f"{row['annotator_id']}_atom{atom_i}",
                    "atom_text": c["claim"],
                    "target": c["target"],
                    "aspect": c["aspect"],
                    "polarity": c["polarity"],
                    "specificity": c["specificity"],
                    "gold_winner": row["gold_winner"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT, index=False)
    out.to_json(PREVIEW, orient="records", lines=True, force_ascii=False)
    summary = {
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "input_rationales": int(len(df)),
        "instances": int(df["instance_id"].nunique()),
        "claims": int(len(out)),
        "errors": int(len(errors)),
        "claims_by_specificity": out["specificity"].value_counts().to_dict()
        if len(out)
        else {},
        "claims_by_aspect": out["aspect"].value_counts().to_dict() if len(out) else {},
        "output": str(OUTPUT),
        "input": str(INPUT),
        "error_keys": [e["key"] for e in errors[:20]],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    load_dotenv()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    df = build_input()
    done = read_cache()
    pending = [(cache_key(row), row) for _, row in df.iterrows() if cache_key(row) not in done]
    print(f"input={len(df)} cached={len(done)} pending={len(pending)} concurrency={CONCURRENCY} model={MODEL}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            future_to_item = {ex.submit(call_one, client, row): (key_, row) for key_, row in pending}
            for future in as_completed(future_to_item):
                key_, row = future_to_item[future]
                try:
                    raw, claims, error = future.result()
                except Exception as e:
                    raw, claims, error = "", [], str(e)
                rec = {
                    "key": key_,
                    "instance_id": row["instance_id"],
                    "domain": row["domain"],
                    "annotator_id": row["annotator_id"],
                    "rationale_text": row["rationale_text"],
                    "gold_winner": row["gold_winner"],
                    "raw_output": raw,
                    "claims": claims,
                    "error": error,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[key_] = rec
                print(f"[{len(done)}/{len(df)}] {key_} claims={len(claims)} error={error}")

    write_outputs(df, done)


if __name__ == "__main__":
    main()
