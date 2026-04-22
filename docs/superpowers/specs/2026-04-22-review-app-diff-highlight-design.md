# Design: Differing Questions Section in Review App

**Date:** 2026-04-22  
**File:** `src/evaluation/review_app.py`

## Goal

In each example card, add a new "Differing Questions" section that shows only the checklist questions where parsed answer A and parsed answer B disagree, along with the original question text.

## Components

### 1. `_extract_questions(prompt: str) -> dict[int, str]`

Parses `prompt_a` for lines matching `Q\d+: <text>` (the format produced by `build_checkeval_prompt`). Returns `{q_num: question_text}`. Uses only `prompt_a` — both sides share the same question set.

### 2. `_diff_answers(parsed_a: dict, parsed_b: dict) -> list[tuple[int, str, str]]`

- Builds `{q_num: answer}` from `parsed_a["answers"]` and `parsed_b["answers"]` (answer = `"yes"` or `"no"`).
- Merges in N/A answers from `na_answers` lists as `"N/A"`.
- Missing Q treated as `"—"`.
- Returns `[(q_num, ans_a, ans_b), ...]` sorted by `q_num`, only where `ans_a != ans_b`.

### 3. Sidebar toggle

```python
show_diff = st.checkbox("Show differing questions", value=True)
```

Added alongside existing `show_parsed` checkbox.

### 4. New section in each expander card

Rendered after the parsed answers block, gated on `show_diff`:

- Header: `**Differing Questions (N)**` where N = count of differing Qs.
- If N == 0: `st.info("No differing answers between A and B.")`
- If N > 0: `st.markdown` table:

  | Q# | Question | A | B |
  |----|----------|---|---|
  | Q3 | Does the response ... | yes | no |

## Data Flow

```
row["prompt_a"]       → _extract_questions() → {q_num: text}
row["parsed_a_json"]  → json.loads()          → parsed_a dict
row["parsed_b_json"]  → json.loads()          → parsed_b dict
parsed_a, parsed_b    → _diff_answers()        → [(q_num, ans_a, ans_b)]
q_num + question dict → render markdown table
```

## Constraints

- No new data dependencies — `prompt_a`, `parsed_a_json`, `parsed_b_json` already in parquet.
- No checklist YAML reload needed.
- `show_diff` checkbox independent of `show_parsed` (can show diff without full parsed text areas).
