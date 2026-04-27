"""Extract human-implied yes-set from annotator reasoning over the v3 bank."""
from __future__ import annotations


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
