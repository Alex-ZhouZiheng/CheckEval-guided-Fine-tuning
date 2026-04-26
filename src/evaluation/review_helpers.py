"""Pure helper functions for review_app — no Streamlit dependency."""

import re
import ast
from pathlib import Path
from typing import Any

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


def _load_question_meta(checklists_dir: Path | None = None) -> dict[str, tuple[str, str]]:
    """Return {question_text: (dimension, sub_aspect)} from YAML checklist files."""
    import yaml
    if checklists_dir is None:
        checklists_dir = Path(__file__).parent.parent.parent / "checklists" / "v4_frozen"
    meta: dict[str, tuple[str, str]] = {}
    for yaml_path in sorted(checklists_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dim_name = data.get("dimension", yaml_path.stem)
        for sub_name, sub_data in data.get("sub_aspects", {}).items():
            for q in sub_data.get("filtered_questions", []):
                meta[q.strip()] = (dim_name, sub_name)
    return meta


def _load_question_meta_by_qid(checklists_dir: Path | None = None) -> dict[int, dict[str, str]]:
    """Return {global_qid: metadata} from a frozen bank_index.parquet."""
    import pandas as pd

    if checklists_dir is None:
        checklists_dir = Path(__file__).parent.parent.parent / "checklists" / "v4_frozen"
    bank_path = checklists_dir / "bank_index.parquet"
    if not bank_path.exists():
        return {}

    bank_df = pd.read_parquet(bank_path)
    dim_col = "dimension" if "dimension" in bank_df.columns else "dim"
    out: dict[int, dict[str, str]] = {}
    for _, row in bank_df.iterrows():
        qid = int(row["qid"])
        out[qid] = {
            "question_text": str(row.get("question_text", "")),
            "dimension": str(row.get(dim_col, "")),
            "sub_aspect": str(row.get("sub_aspect", "")),
        }
    return out


def _parse_qid_list(value: Any) -> list[int]:
    """Parse qid list values coming from parquet/list/string cells."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _parse_qid_list(ast.literal_eval(text))
    return []


def _answer_map(parsed: dict) -> dict[int, str]:
    """Convert parsed CheckEval output to {local_qnum: yes|no|N/A}."""
    out: dict[int, str] = {}
    for a in parsed.get("answers", []):
        out[int(a["q"])] = str(a["answer"])
    for a in parsed.get("na_answers", []):
        out[int(a["q"])] = "N/A"
    return out


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
