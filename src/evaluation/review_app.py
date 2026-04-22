"""
Streamlit review app for teacher-model CheckEval outputs.

Launch:
    streamlit run src/evaluation/review_app.py -- \\
        --results results/teacher_review_dev_n100_review_samples.parquet
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from review_helpers import (
    _diff_answers,
    _extract_questions,
    _render_parsed,
    _verdict_badge,
)

# ── CLI arg for parquet path ──────────────────────────────────────────────────
def _parse_args() -> Path:
    # Streamlit forwards args after "--" to the script
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--results", type=Path, required=True)
    args, _ = p.parse_known_args(raw)
    return args.results


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Teacher Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: Path, mtime: float) -> pd.DataFrame:  # mtime busts cache on file change
    df = pd.read_parquet(path)
    # ensure list columns are lists not strings
    for col in ("na_qnums_a", "na_qnums_b"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
    return df


def _context_to_text(context) -> str:
    """Serialise a multi-turn conversation list into a single string (mirrors prepare_data.py)."""
    parts = []
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _make_prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@st.cache_data
def _build_individual_pref_lookup(raw_dir: Path) -> dict:
    """Build a prompt_id -> list[dict] lookup from raw HelpSteer3 parquet files."""
    lookup: dict = {}
    for fname in ("helpsteer3_train.parquet", "helpsteer3_test.parquet"):
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_parquet(fpath)
        if "individual_preference" not in raw.columns:
            continue
        for _, row in raw.iterrows():
            ctx_text = _context_to_text(row["context"])
            pid = _make_prompt_id(ctx_text)
            if pid not in lookup:
                lookup[pid] = list(row["individual_preference"])
    return lookup


results_path = _parse_args()
if not results_path.exists():
    st.error(f"File not found: {results_path}")
    st.stop()

df = load_data(results_path, results_path.stat().st_mtime)

# Attempt to enrich with individual_preference from the raw HelpSteer3 files.
# The raw files live two levels up from results, under data/raw/.
_raw_dir = results_path.parent.parent / "data" / "raw"
if "individual_preference" not in df.columns and _raw_dir.exists():
    _pref_lookup = _build_individual_pref_lookup(_raw_dir)
    if _pref_lookup:
        df["individual_preference"] = df["prompt_id"].map(_pref_lookup)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Teacher Review")
    st.caption(f"`{results_path.name}`")
    st.markdown(f"**{len(df)} examples** loaded")

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
