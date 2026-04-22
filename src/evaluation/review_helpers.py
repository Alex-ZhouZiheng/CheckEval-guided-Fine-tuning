"""Pure helper functions for review_app — no Streamlit dependency."""

import re

_Q_LINE_RE = re.compile(r"^Q(\d+):\s+(.+)$", re.MULTILINE)


def _verdict_badge(winner: str, predicted: str) -> str:
    correct = winner == predicted
    icon = "✅" if correct else "❌"
    return f"{icon} True: **{winner}** | Predicted: **{predicted}**"


def _render_parsed(parsed: dict) -> str:
    if parsed.get("_raw_fallback"):
        return f"**(parse failure)**\n\n```\n{parsed.get('raw_text', '')[:500]}\n```"
    lines = []
    for a in parsed.get("answers", []):
        lines.append(f"Q{a['q']}: {a['answer']}")
    for a in parsed.get("na_answers", []):
        lines.append(f"Q{a['q']}: N/A")
    lines_sorted = sorted(lines, key=lambda x: int(x.split(":")[0][1:]))
    meta = (
        f"score={parsed.get('score', '?'):.3f}  "
        f"yes={parsed.get('n_yes')}  no={parsed.get('n_no')}  "
        f"na={parsed.get('n_na')}  answered={parsed.get('n_questions_parsed')}"
    )
    return meta + "\n\n" + "\n".join(lines_sorted)


def _extract_questions(prompt: str) -> dict[int, str]:
    """Return {q_num: question_text} parsed from a CheckEval prompt string."""
    return {int(m.group(1)): m.group(2).strip() for m in _Q_LINE_RE.finditer(prompt)}


def _diff_answers(
    parsed_a: dict, parsed_b: dict
) -> list[tuple[int, str, str]]:
    """Return [(q_num, ans_a, ans_b)] for questions where A and B disagree."""
    def _build_map(parsed: dict) -> dict[int, str]:
        m: dict[int, str] = {}
        for a in parsed.get("answers", []):
            m[a["q"]] = a["answer"]
        for a in parsed.get("na_answers", []):
            m[a["q"]] = "N/A"
        return m

    map_a = _build_map(parsed_a)
    map_b = _build_map(parsed_b)
    all_qs = sorted(set(map_a) | set(map_b))
    return [
        (q, map_a.get(q, "—"), map_b.get(q, "—"))
        for q in all_qs
        if map_a.get(q, "—") != map_b.get(q, "—")
    ]
