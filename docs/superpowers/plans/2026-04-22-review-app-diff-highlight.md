# Review App Differing Questions Section — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Differing Questions" section to each example card in `review_app.py` that shows checklist questions where parsed answer A and parsed answer B disagree, with original question text.

**Architecture:** Two pure helper functions (`_extract_questions`, `_diff_answers`) added to `review_app.py` alongside existing helpers. A new sidebar checkbox controls visibility. A new rendering block appended inside each expander card after the existing parsed answers section.

**Tech Stack:** Python 3, Streamlit, standard `re` module (already imported in utils.py; add to review_app.py).

---

## File Map

| File | Action |
|------|--------|
| `src/evaluation/review_app.py` | Modify — add 2 helpers, 1 sidebar checkbox, 1 new section per card |
| `tests/test_review_app_helpers.py` | Create — pytest tests for the two pure helpers |

---

### Task 1: Add `_extract_questions` helper and tests

**Files:**
- Modify: `src/evaluation/review_app.py:1-10` (add `import re`)
- Modify: `src/evaluation/review_app.py:27-50` (add helper after `_render_parsed`)
- Create: `tests/test_review_app_helpers.py`

- [ ] **Step 1: Create test file with failing test for `_extract_questions`**

Create `tests/test_review_app_helpers.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))

# Import helpers directly — avoid triggering Streamlit at module level
import importlib, types, unittest.mock as mock

# Patch streamlit before importing review_app helpers
_st_mock = types.ModuleType("streamlit")
for _attr in ["set_page_config", "cache_data", "error", "stop", "sidebar",
              "title", "caption", "markdown", "radio", "selectbox", "checkbox",
              "divider", "columns", "expander", "text_area", "info", "metric",
              "dataframe"]:
    setattr(_st_mock, _attr, mock.MagicMock())
sys.modules["streamlit"] = _st_mock
sys.modules["pandas"] = __import__("pandas")

import review_app as ra


SAMPLE_PROMPT = """
Some preamble text.

Q1: Does the response directly address the user's question?
Q2: Is the explanation clear and easy to follow?
Q3: Does the response avoid unnecessary repetition?
Q10: Is the response free of factual errors?
"""


def test_extract_questions_basic():
    result = ra._extract_questions(SAMPLE_PROMPT)
    assert result == {
        1: "Does the response directly address the user's question?",
        2: "Is the explanation clear and easy to follow?",
        3: "Does the response avoid unnecessary repetition?",
        10: "Is the response free of factual errors?",
    }


def test_extract_questions_empty():
    assert ra._extract_questions("No questions here.") == {}


def test_extract_questions_ignores_non_q_lines():
    prompt = "Q1: First question?\nsome other line\nQ2: Second question?"
    result = ra._extract_questions(prompt)
    assert result == {1: "First question?", 2: "Second question?"}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd D:/CheckEval-guided-Fine-tuning
python -m pytest tests/test_review_app_helpers.py::test_extract_questions_basic -v
```

Expected: `ImportError` or `AttributeError: module 'review_app' has no attribute '_extract_questions'`

- [ ] **Step 3: Add `import re` to `review_app.py`**

In `src/evaluation/review_app.py`, the existing imports are at lines 9-15. Add `import re` after the existing imports:

```python
import argparse
import json
import re
import sys
from pathlib import Path
```

- [ ] **Step 4: Add `_extract_questions` helper after `_render_parsed` (after line 49)**

Insert after the closing of `_render_parsed` (after line 49, before the `# ── page config` comment):

```python
_Q_LINE_RE = re.compile(r"^Q(\d+):\s+(.+)$", re.MULTILINE)


def _extract_questions(prompt: str) -> dict[int, str]:
    """Return {q_num: question_text} parsed from a CheckEval prompt string."""
    return {int(m.group(1)): m.group(2).strip() for m in _Q_LINE_RE.finditer(prompt)}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_review_app_helpers.py::test_extract_questions_basic tests/test_review_app_helpers.py::test_extract_questions_empty tests/test_review_app_helpers.py::test_extract_questions_ignores_non_q_lines -v
```

Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/review_app.py tests/test_review_app_helpers.py
git commit -m "feat: add _extract_questions helper to review_app"
```

---

### Task 2: Add `_diff_answers` helper and tests

**Files:**
- Modify: `src/evaluation/review_app.py` (add helper after `_extract_questions`)
- Modify: `tests/test_review_app_helpers.py` (add tests)

- [ ] **Step 1: Add failing tests for `_diff_answers`**

Append to `tests/test_review_app_helpers.py`:

```python
def test_diff_answers_finds_disagreements():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}, {"q": 3, "answer": "yes"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "yes"}, {"q": 3, "answer": "no"}],
        "na_answers": [],
    }
    result = ra._diff_answers(parsed_a, parsed_b)
    assert result == [(2, "no", "yes"), (3, "yes", "no")]


def test_diff_answers_na_treated_as_na():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}],
        "na_answers": [{"q": 2}],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    result = ra._diff_answers(parsed_a, parsed_b)
    assert result == [(2, "N/A", "no")]


def test_diff_answers_no_diff_returns_empty():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [{"q": 1, "answer": "yes"}, {"q": 2, "answer": "no"}],
        "na_answers": [],
    }
    assert ra._diff_answers(parsed_a, parsed_b) == []


def test_diff_answers_missing_q_treated_as_dash():
    parsed_a = {
        "answers": [{"q": 1, "answer": "yes"}],
        "na_answers": [],
    }
    parsed_b = {
        "answers": [],
        "na_answers": [],
    }
    result = ra._diff_answers(parsed_a, parsed_b)
    assert result == [(1, "yes", "—")]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_review_app_helpers.py::test_diff_answers_finds_disagreements -v
```

Expected: `AttributeError: module 'review_app' has no attribute '_diff_answers'`

- [ ] **Step 3: Add `_diff_answers` helper after `_extract_questions`**

Insert immediately after `_extract_questions` in `src/evaluation/review_app.py`:

```python
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
```

- [ ] **Step 4: Run all helper tests**

```bash
python -m pytest tests/test_review_app_helpers.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/review_app.py tests/test_review_app_helpers.py
git commit -m "feat: add _diff_answers helper to review_app"
```

---

### Task 3: Add sidebar checkbox and differing-questions section to each card

**Files:**
- Modify: `src/evaluation/review_app.py` — sidebar block (around line 97-99) and card loop (around line 239-261)

- [ ] **Step 1: Add `show_diff` checkbox to sidebar**

In `src/evaluation/review_app.py`, find the sidebar block (lines 83-99). The last three checkboxes are:

```python
    show_prompt = st.checkbox("Show full prompts", value=False)
    show_raw = st.checkbox("Show raw model output", value=True)
    show_parsed = st.checkbox("Show parsed answers", value=True)
```

Add `show_diff` after `show_parsed`:

```python
    show_prompt = st.checkbox("Show full prompts", value=False)
    show_raw = st.checkbox("Show raw model output", value=True)
    show_parsed = st.checkbox("Show parsed answers", value=True)
    show_diff = st.checkbox("Show differing questions", value=True)
```

- [ ] **Step 2: Add differing-questions section inside each expander card**

In `src/evaluation/review_app.py`, find the end of the parsed answers block (around lines 239-261):

```python
        # ── parsed answers ──
        if show_parsed:
            st.markdown("---")
            parsed_a = json.loads(row["parsed_a_json"]) if "parsed_a_json" in row else {}
            parsed_b = json.loads(row["parsed_b_json"]) if "parsed_b_json" in row else {}
            dl, dr = st.columns(2)
            with dl:
                st.markdown("**Parsed answers A**")
                st.text_area(
                    "parsed_a",
                    value=_render_parsed(parsed_a),
                    height=300,
                    key=f"pars_a_{i}",
                    label_visibility="collapsed",
                )
            with dr:
                st.markdown("**Parsed answers B**")
                st.text_area(
                    "parsed_b",
                    value=_render_parsed(parsed_b),
                    height=300,
                    key=f"pars_b_{i}",
                    label_visibility="collapsed",
                )
```

After the closing of this `if show_parsed:` block, add:

```python
        # ── differing questions ──
        if show_diff and "parsed_a_json" in row and "parsed_b_json" in row:
            st.markdown("---")
            _parsed_a = json.loads(row["parsed_a_json"])
            _parsed_b = json.loads(row["parsed_b_json"])
            diffs = _diff_answers(_parsed_a, _parsed_b)
            st.markdown(f"**Differing Questions ({len(diffs)})**")
            if not diffs:
                st.info("No differing answers between A and B.")
            else:
                q_texts = _extract_questions(str(row.get("prompt_a", "")))
                table_rows = [
                    f"| Q{q} | {q_texts.get(q, '_unknown_')} | {ans_a} | {ans_b} |"
                    for q, ans_a, ans_b in diffs
                ]
                table = (
                    "| Q# | Question | A | B |\n"
                    "|-----|----------|---|---|\n"
                    + "\n".join(table_rows)
                )
                st.markdown(table)
```

Note: `_parsed_a` / `_parsed_b` use underscore-prefixed names to avoid shadowing the `parsed_a` / `parsed_b` variables that may exist in the `show_parsed` block above when both are visible simultaneously.

- [ ] **Step 3: Verify the app renders without error**

Run the Streamlit app against a real parquet file:

```bash
streamlit run src/evaluation/review_app.py -- --results <path-to-review-samples.parquet>
```

Expected: app loads, each card shows "Differing Questions (N)" section, table renders with question text, A/B columns. If no parquet is available, confirm at minimum that `python -c "import ast; ast.parse(open('src/evaluation/review_app.py').read())"` exits cleanly.

```bash
python -c "import ast; ast.parse(open('src/evaluation/review_app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/test_review_app_helpers.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/review_app.py
git commit -m "feat: add differing questions section to review app cards"
```

---

## Self-Review

**Spec coverage:**
- `_extract_questions` — Task 1 ✓
- `_diff_answers` — Task 2 ✓
- `show_diff` sidebar checkbox — Task 3 ✓
- New section in each expander card — Task 3 ✓
- N/A handling — Task 2 tests cover it ✓
- Missing Q as `"—"` — Task 2 tests cover it ✓
- No new data deps (uses existing parquet columns) — confirmed ✓

**Placeholder scan:** None found.

**Type consistency:**
- `_extract_questions` returns `dict[int, str]` → consumed as `q_texts.get(q, '_unknown_')` where `q` is `int` from `_diff_answers` ✓
- `_diff_answers` returns `list[tuple[int, str, str]]` → destructured as `q, ans_a, ans_b` ✓
- `_build_map` internal to `_diff_answers`, not referenced externally ✓
