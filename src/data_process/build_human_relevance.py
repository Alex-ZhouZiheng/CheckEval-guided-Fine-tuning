"""Extract human-implied yes-set from annotator reasoning over the v3 bank."""
from __future__ import annotations

import json
import re


def build_extractor_prompt(reasoning: str, qmeta: list[dict]) -> str:
    lines = []
    for q in qmeta:
        lines.append(f"Q{int(q['qid'])} ({q['dimension']}): {q['question_text']}")
    questions_block = "\n".join(lines)

    return (
        "You are mapping a human annotator's free-text rationale onto a fixed "
        "checklist of 61 evaluation questions. Return the qids the rationale "
        "directly addresses (positively or negatively). If the rationale is "
        "ambiguous, return an empty list. Do not infer.\n\n"
        "[Checklist]\n"
        f"{questions_block}\n\n"
        "[Annotator rationale]\n"
        f"{reasoning}\n\n"
        "Return strict JSON: {\"mentioned_qids\": [int, ...]}"
    )


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\"mentioned_qids\"\s*:\s*\[[^\]]*\][^{}]*\}", re.DOTALL)
_QID_TOKEN_RE = re.compile(r"\b[Qq](\d{1,2})\b")


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
