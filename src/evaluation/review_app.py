"""
Streamlit review app for teacher-model CheckEval outputs.

Launch:
    streamlit run src/evaluation/review_app.py -- \\
        --results results/teacher_review_dev_n100_review_samples.parquet
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from review_helpers import (
    _answer_map,
    _diff_answers,
    _extract_questions,
    _load_question_meta,
    _load_question_meta_by_qid,
    _parse_qid_list,
    _render_parsed,
    _verdict_badge,
)

# ── CLI arg for parquet path ──────────────────────────────────────────────────
def _parse_args() -> Path | None:
    import os
    # env var takes priority (useful when Streamlit swallows CLI args)
    if env := os.environ.get("REVIEW_RESULTS"):
        return Path(env)
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    p.add_argument("--results", type=Path)
    try:
        args, _ = p.parse_known_args(raw)
    except argparse.ArgumentError:
        return None
    return args.results


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Teacher Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def _cached_question_meta() -> dict:
    return _load_question_meta()


@st.cache_data
def _cached_question_meta_by_qid() -> dict:
    return _load_question_meta_by_qid()


@st.cache_data
def load_data(path: Path, mtime: float) -> pd.DataFrame:  # mtime busts cache on file change
    df = pd.read_parquet(path)
    # ensure list columns are lists not strings
    list_cols = (
        "na_qnums_a",
        "na_qnums_b",
        "selected_qids",
        "asked_qids",
        "selected_question_weights",
        "selected_question_importance_raw",
    )
    for col in list_cols:
        if col in df.columns and df[col].dtype == object:
            if col in {"selected_qids", "asked_qids"}:
                df[col] = df[col].apply(_parse_qid_list)
            elif col in {"selected_question_weights", "selected_question_importance_raw"}:
                df[col] = df[col].apply(_parse_float_list)
            else:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
    return df


def _parse_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [float(x) for x in value if x is not None]
    if isinstance(value, tuple):
        return [float(x) for x in value if x is not None]
    if hasattr(value, "tolist"):
        return _parse_float_list(value.tolist())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _parse_float_list(ast.literal_eval(text))
    return []


results_path = _parse_args()
if results_path is None:
    with st.sidebar:
        st.title("Teacher Review")
        _input = st.text_input("Results parquet path", placeholder="/path/to/file.parquet")
    if not _input:
        st.info("Enter path to results parquet in the sidebar.")
        st.stop()
    results_path = Path(_input)
if not results_path.exists():
    st.error(f"File not found: {results_path}")
    st.stop()

df = load_data(results_path, results_path.stat().st_mtime)
q_meta = _cached_question_meta()
q_meta_by_qid = _cached_question_meta_by_qid()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Teacher Review")
    st.caption(f"`{results_path.name}`")
    st.markdown(f"**{len(df)} examples** loaded")
    if "individual_preference" not in df.columns:
        st.warning(
            "This file does not include `individual_preference`, so Human Reasoning "
            "cannot be shown. If this came from `run_teacher_review.py`, prefer the "
            "`*_review_samples.parquet` file or rerun after preparing raw/with_reason data."
        )

    split_filter = st.radio(
        "Show",
        ["All", "Wrong predictions", "Correct predictions"],
        index=0,
    )

    domain_options = ["All"] + sorted(df["domain"].unique().tolist())
    domain_filter = st.selectbox("Domain", domain_options)

    show_prompt = st.checkbox("Show full prompts", value=False)
    show_raw = st.checkbox("Show raw model output", value=True)
    show_parsed = st.checkbox("Show parsed answers", value=True)
    show_selector = st.checkbox("Show selector questions", value=True)
    show_weights = st.checkbox(
        "Show question weights",
        value="selected_question_weights" in df.columns,
    )
    show_diff = st.checkbox("Show differing questions", value=True)
    show_human = st.checkbox("Show human reasoning", value=True)

# ── filter ────────────────────────────────────────────────────────────────────

view = df.copy()
if split_filter == "Wrong predictions":
    view = view[view["_review_split"] == "wrong"]
elif split_filter == "Correct predictions":
    view = view[view["_review_split"] == "correct"]
if domain_filter != "All":
    view = view[view["domain"] == domain_filter]

st.markdown(f"### Showing **{len(view)}** examples")

if len(view) == 0:
    st.info("No examples match the current filter.")
    st.stop()

# ── summary stats ─────────────────────────────────────────────────────────────

n_wrong = (view["_review_split"] == "wrong").sum()
n_correct = (view["_review_split"] == "correct").sum()
cols = st.columns(4)
cols[0].metric("Wrong", n_wrong)
cols[1].metric("Correct", n_correct)
cols[2].metric(
    "Avg score A",
    f"{view['score_a'].dropna().mean():.3f}" if "score_a" in view else "—",
)
cols[3].metric(
    "Avg score B",
    f"{view['score_b'].dropna().mean():.3f}" if "score_b" in view else "—",
)

st.divider()

# ── example cards ─────────────────────────────────────────────────────────────

for i, (_, row) in enumerate(view.iterrows()):
    winner = str(row.get("winner", "?"))
    predicted = str(row.get("predicted_winner", "?"))
    domain = str(row.get("domain", "?"))
    review_split = str(row.get("_review_split", "?"))
    score_a = row.get("score_a")
    score_b = row.get("score_b")
    margin = row.get("pairwise_margin")

    header_color = "🔴" if review_split == "wrong" else "🟢"
    header = (
        f"{header_color} **Example {i + 1}** &nbsp;|&nbsp; "
        f"domain: `{domain}` &nbsp;|&nbsp; "
        f"{_verdict_badge(winner, predicted)}"
    )
    if score_a is not None and score_b is not None:
        header += (
            f" &nbsp;|&nbsp; score A: `{score_a:.3f}` B: `{score_b:.3f}`"
        )
    if margin is not None:
        header += f" &nbsp;|&nbsp; margin: `{margin:+.3f}`"

    with st.expander(header, expanded=False):
        # ── context / conversation ──
        st.markdown("#### Conversation Context")
        st.text_area(
            "context",
            value=str(row.get("context", "")),
            height=200,
            key=f"ctx_{i}",
            label_visibility="collapsed",
        )

        left, right = st.columns(2)

        with left:
            st.markdown(f"#### Response A  (true winner: **{winner}**)")
            st.text_area(
                "response_a",
                value=str(row.get("response_a", "")),
                height=300,
                key=f"ra_{i}",
                label_visibility="collapsed",
            )

        with right:
            st.markdown(f"#### Response B")
            st.text_area(
                "response_b",
                value=str(row.get("response_b", "")),
                height=300,
                key=f"rb_{i}",
                label_visibility="collapsed",
            )

        # ── prompts ──
        if show_prompt and "prompt_a" in row and "prompt_b" in row:
            st.markdown("---")
            pl, pr = st.columns(2)
            with pl:
                st.markdown("**Prompt A (sent to judge)**")
                st.text_area(
                    "prompt_a",
                    value=str(row.get("prompt_a", "")),
                    height=400,
                    key=f"pa_{i}",
                    label_visibility="collapsed",
                )
            with pr:
                st.markdown("**Prompt B (sent to judge)**")
                st.text_area(
                    "prompt_b",
                    value=str(row.get("prompt_b", "")),
                    height=400,
                    key=f"pb_{i}",
                    label_visibility="collapsed",
                )

        # ── raw outputs ──
        if show_raw:
            st.markdown("---")
            rl, rr = st.columns(2)
            with rl:
                st.markdown("**Raw output A**")
                st.text_area(
                    "raw_a",
                    value=str(row.get("raw_output_a", "")),
                    height=300,
                    key=f"raw_a_{i}",
                    label_visibility="collapsed",
                )
            with rr:
                st.markdown("**Raw output B**")
                st.text_area(
                    "raw_b",
                    value=str(row.get("raw_output_b", "")),
                    height=300,
                    key=f"raw_b_{i}",
                    label_visibility="collapsed",
                )

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

        # ── selector questions ──
        if show_selector and ("asked_qids" in row or "selected_qids" in row):
            st.markdown("---")
            asked_qids = _parse_qid_list(row.get("asked_qids"))
            selected_qids = _parse_qid_list(row.get("selected_qids"))
            if not asked_qids:
                asked_qids = selected_qids
            selected_set = set(selected_qids)
            parsed_a_for_selector = json.loads(row["parsed_a_json"]) if "parsed_a_json" in row else {}
            parsed_b_for_selector = json.loads(row["parsed_b_json"]) if "parsed_b_json" in row else {}
            ans_a_map = _answer_map(parsed_a_for_selector)
            ans_b_map = _answer_map(parsed_b_for_selector)
            q_weights = _parse_float_list(row.get("selected_question_weights"))
            q_raw_weights = _parse_float_list(row.get("selected_question_importance_raw"))
            has_weights = show_weights and bool(q_weights)

            if not asked_qids:
                st.info("No selector qids saved for this example.")
            else:
                st.markdown(f"**Selector Questions ({len(asked_qids)})**")
                table_rows = []
                for local_q, global_qid in enumerate(asked_qids, start=1):
                    meta = q_meta_by_qid.get(int(global_qid), {})
                    q_text = meta.get("question_text", "_unknown_")
                    dim = meta.get("dimension", "?")
                    sub = meta.get("sub_aspect", "?")
                    ans_a = ans_a_map.get(local_q, "-")
                    ans_b = ans_b_map.get(local_q, "-")
                    differs = ans_a != ans_b
                    if winner == "A":
                        bad_for_winner = ans_a.lower() == "no" and ans_b.lower() in ("yes", "n/a")
                    else:
                        bad_for_winner = ans_b.lower() == "no" and ans_a.lower() in ("yes", "n/a")
                    stage = "initial" if int(global_qid) in selected_set or not selected_set else "added"
                    flag = "bad_for_winner" if bad_for_winner else ("diff" if differs else "")
                    cells = [
                        str(local_q),
                        str(global_qid),
                        stage,
                        q_text.replace("|", "\\|"),
                        str(ans_a),
                        str(ans_b),
                    ]
                    if has_weights:
                        weight = q_weights[local_q - 1] if local_q <= len(q_weights) else None
                        raw_weight = q_raw_weights[local_q - 1] if local_q <= len(q_raw_weights) else None
                        cells.extend(
                            [
                                f"{weight:.4f}" if weight is not None else "",
                                f"{raw_weight:.2f}" if raw_weight is not None else "",
                            ]
                        )
                    cells.extend(
                        [
                            dim.replace("|", "\\|"),
                            sub.replace("|", "\\|"),
                            flag,
                        ]
                    )
                    table_rows.append(
                        "| "
                        + " | ".join(cells)
                        + " |"
                    )

                header = "| Local Q | Global qid | Stage | Question | A | B |"
                align = "|---:|---:|---|---|---|---|"
                if has_weights:
                    header += " Weight | Raw importance |"
                    align += "---:|---:|"
                header += " Dimension | Sub-aspect | Flag |"
                align += "---|---|---|"
                table = header + "\n" + align + "\n" + "\n".join(table_rows)
                st.markdown(table)

        # ── differing questions ──
        if show_diff and "parsed_a_json" in row and "parsed_b_json" in row:
            st.markdown("---")
            _parsed_a = json.loads(row["parsed_a_json"])
            _parsed_b = json.loads(row["parsed_b_json"])
            diffs = _diff_answers(_parsed_a, _parsed_b)
            if not diffs:
                st.markdown("**Differing Questions (0)**")
                st.info("No differing answers between A and B.")
            else:
                q_texts = _extract_questions(str(row.get("prompt_a", "")))
                asked_qids_for_diff = _parse_qid_list(row.get("asked_qids"))
                q_weights_for_diff = _parse_float_list(row.get("selected_question_weights"))
                q_raw_for_diff = _parse_float_list(row.get("selected_question_importance_raw"))
                has_diff_weights = show_weights and bool(q_weights_for_diff)
                table_rows = []
                n_flagged = 0
                for q, ans_a, ans_b in diffs:
                    q_text = q_texts.get(q, "_unknown_")
                    dim, sub = q_meta.get(q_text, ("?", "?"))
                    if winner == "A":
                        flagged = ans_a.lower() == "no" and ans_b.lower() in ("yes", "n/a")
                    else:
                        flagged = ans_b.lower() == "no" and ans_a.lower() in ("yes", "n/a")
                    if flagged:
                        n_flagged += 1
                    flag = "⚠️" if flagged else ""
                    cells = [flag, f"Q{q}", q_text.replace("|", "\\|"), ans_a, ans_b]
                    if has_diff_weights:
                        local_idx = q - 1
                        global_qid = (
                            asked_qids_for_diff[local_idx]
                            if 0 <= local_idx < len(asked_qids_for_diff)
                            else ""
                        )
                        weight = (
                            q_weights_for_diff[local_idx]
                            if 0 <= local_idx < len(q_weights_for_diff)
                            else None
                        )
                        raw_weight = (
                            q_raw_for_diff[local_idx]
                            if 0 <= local_idx < len(q_raw_for_diff)
                            else None
                        )
                        cells.extend(
                            [
                                str(global_qid),
                                f"{weight:.4f}" if weight is not None else "",
                                f"{raw_weight:.2f}" if raw_weight is not None else "",
                            ]
                        )
                    cells.extend([dim.replace("|", "\\|"), sub.replace("|", "\\|")])
                    table_rows.append("| " + " | ".join(cells) + " |")
                st.markdown(
                    f"**Differing Questions ({len(diffs)}"
                    + (f", ⚠️ {n_flagged} bad for winner" if n_flagged else "")
                    + ")**"
                )
                header = "| | Q# | Question | A | B |"
                align = "|---|-----|----------|---|---|"
                if has_diff_weights:
                    header += " Global qid | Weight | Raw importance |"
                    align += "---:|---:|---:|"
                header += " Domain | Sub-aspect |"
                align += "--------|------------|"
                table = header + "\n" + align + "\n" + "\n".join(table_rows)
                st.markdown(table)

        # ── human reasoning ──
        if show_human and "individual_preference" in row:
            st.markdown("---")
            raw_pref = row["individual_preference"]
            if isinstance(raw_pref, str):
                prefs = json.loads(raw_pref)
            else:
                prefs = list(raw_pref) if raw_pref is not None else []
            if not prefs:
                st.info("No human reasoning available.")
            else:
                st.markdown(f"**Human Reasoning ({len(prefs)} annotator(s))**")
                for idx, p in enumerate(prefs):
                    score = p.get("score", "?")
                    reasoning = p.get("reasoning", "")
                    fb1 = p.get("feedback1", "")
                    fb2 = p.get("feedback2", "")
                    with st.expander(f"Annotator {idx + 1}  |  score: `{score}`", expanded=(idx == 0)):
                        st.markdown(f"**Overall reasoning:** {reasoning}")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Feedback on Response A**")
                            st.markdown(fb1)
                        with c2:
                            st.markdown("**Feedback on Response B**")
                            st.markdown(fb2)

        # ── diagnostics row ──
        st.markdown("---")
        diag_cols = st.columns(6)
        diag_cols[0].metric("n_yes A", row.get("n_yes_a", "—"))
        diag_cols[1].metric("n_yes B", row.get("n_yes_b", "—"))
        diag_cols[2].metric("n_na A", row.get("n_na_a", "—"))
        diag_cols[3].metric("n_na B", row.get("n_na_b", "—"))
        diag_cols[4].metric("expected Q", row.get("expected_n_questions", "—"))
        diag_cols[5].metric("error_cat", str(row.get("error_category", "—")))
