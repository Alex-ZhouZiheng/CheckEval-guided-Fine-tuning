from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from utils import generate_batch, load_judge_model

OUT = Path("results/coverage_dev200")
INPUT = OUT / "atomic_claims_dev20_input.parquet"
OUTPUT = OUT / "atomic_claims_dev20.parquet"
RAW_OUTPUT = OUT / "atomic_claims_dev20_raw.parquet"
PREVIEW = OUT / "atomic_claims_dev20.preview.jsonl"
SUMMARY = OUT / "atomic_claims_dev20_summary.json"

MODEL = "models/Qwen3.5-9B"

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


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        return json.loads(fence.group(1))
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


def main():
    df = pd.read_parquet(INPUT)
    messages = [
        [{"role": "user", "content": PROMPT.format(**row)}]
        for row in df.to_dict("records")
    ]
    model = load_judge_model(
        MODEL,
        backend="vllm",
        max_model_len=16384,
        max_num_seqs=16,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.85,
    )
    raw_outputs = generate_batch(
        model,
        messages,
        batch_size=16,
        max_new_tokens=1200,
        temperature=0.0,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    raw_df = df[["instance_id", "domain", "annotator_id", "rationale_text", "gold_winner"]].copy()
    raw_df["raw_output"] = raw_outputs
    raw_df.to_parquet(RAW_OUTPUT, index=False)

    rows = []
    errors = []
    for source_row, raw in zip(df.to_dict("records"), raw_outputs):
        try:
            claims = normalize_claims(extract_json(raw))
        except Exception as e:
            claims = []
            errors.append(
                {
                    "instance_id": source_row["instance_id"],
                    "annotator_id": source_row["annotator_id"],
                    "error": str(e),
                    "raw_output": raw,
                }
            )
        for atom_i, c in enumerate(claims, start=1):
            rows.append(
                {
                    "instance_id": source_row["instance_id"],
                    "domain": source_row["domain"],
                    "annotator_id": source_row["annotator_id"],
                    "rationale_text": source_row["rationale_text"],
                    "atom_id": f"{source_row['annotator_id']}_atom{atom_i}",
                    "atom_text": c["claim"],
                    "target": c["target"],
                    "aspect": c["aspect"],
                    "polarity": c["polarity"],
                    "specificity": c["specificity"],
                    "gold_winner": source_row["gold_winner"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT, index=False)
    out.to_json(PREVIEW, orient="records", lines=True, force_ascii=False)
    summary = {
        "model": MODEL,
        "input_rationales": int(len(df)),
        "instances": int(df["instance_id"].nunique()),
        "claims": int(len(out)),
        "parse_errors": int(len(errors)),
        "claims_by_specificity": out["specificity"].value_counts().to_dict()
        if len(out)
        else {},
        "claims_by_aspect": out["aspect"].value_counts().to_dict() if len(out) else {},
        "output": str(OUTPUT),
        "raw_output": str(RAW_OUTPUT),
        "errors": errors[:10],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
