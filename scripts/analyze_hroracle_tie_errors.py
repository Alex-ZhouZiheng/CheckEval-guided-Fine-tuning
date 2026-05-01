#!/usr/bin/env python
"""Systematic error analysis of HR-oracle weighted tie/error review parquet.

Usage:
    python scripts/analyze_hroracle_tie_errors.py \
        --input results/review/hroracle_weighted_tie_error_review_with_weights.parquet \
        --output results/review/tie_error_analysis
"""

import argparse, json, os, sys, textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Heuristic thresholds (tweak here, not in body)
# ---------------------------------------------------------------------------
GENERIC_SUB_ASPECTS = {
    "Relevance to User", "Task Adherence", "Coverage Adequacy",
    "Explanation Sufficiency", "Information Efficiency",
    "Practical Usability", "Completeness", "Task Completeness",
}
GENERIC_PHRASES = [
    "address the main question", "perform the task",
    "cover each explicit request", "deliver the specific artifact",
    "focus on what the user needs", "provide enough explanation",
    "specific steps, methods, or procedures",
    "sufficient detail for a user to act", "satisfies the user",
    "actionable", "directly address", "fulfil the user",
]
LOW_WEIGHT_THRESHOLD = 0.02
LOW_NONZERO_RATE_THRESHOLD = 0.3
SMALL_MARGIN_THRESHOLD = 0.05
WEAK_PREF_STRENGTH_WORDS = {"slightly", "marginally", "a bit", "tiny", "barely"}

# Task-type keyword rules
TASK_KEYWORDS = {
    "creative_writing_script_scene": [
        "script", "scenario", "scene", "story", "dialogue", "fanfiction",
        "crossover", "what if", "roleplay", "character", "narrative",
        "write a story", "write a script", "worldbuilding", "plot",
    ],
    "coding": [
        "code", "python", "javascript", "bug", "function", "implement",
        "program", "compile", "debug", "api", "library", "framework",
        "syntax", "sql", "java", "c++", "typescript", "react", "node",
        "html", "css", "bash", "docker", "algorithm", "regex",
    ],
    "math_stem": [
        "calculate", "solve", "equation", "proof", "math", "formula",
        "derivative", "integral", "theorem", "statistics", "probability",
        "physics", "chemistry", "biology", "science",
    ],
    "summarization": ["summarize", "summary", "tl;dr", "recap", "key points"],
    "translation": ["translate", "translation"],
    "editing_rewriting": [
        "rewrite", "polish", "improve", "grammar", "edit", "proofread",
        "revise", "rephrase", "paraphrase",
    ],
    "advice": [
        "advice", "should i", "recommend", "suggestion", "what would you do",
        "help me decide",
    ],
    "comparison_recommendation": [
        "compare", "versus", "vs", "better", "which one", "difference between",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trunc(text, n=200):
    s = str(text)
    return s if len(s) <= n else s[:n] + "…"


def parse_answer_json(json_str: str) -> dict:
    """Return {qnum: answer} from parsed JSON string, or empty on failure."""
    try:
        obj = json.loads(json_str)
        return {a["q"]: a["answer"] for a in obj.get("answers", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def safe_mean(series):
    return series.mean() if len(series) > 0 else float("nan")


def safe_median(series):
    return series.median() if len(series) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------

def detect_task_type(prompt: str) -> str:
    p = prompt.lower()
    for ttype, keywords in TASK_KEYWORDS.items():
        if any(kw in p for kw in keywords):
            return ttype
    return "general_explanation"


# ---------------------------------------------------------------------------
# Failure mode classifiers
# ---------------------------------------------------------------------------

def classify_failure_modes(row, generic_qids: set, task_type: str) -> list:
    """Return list of failure mode tags for a sample."""
    tags = []

    # Parse answers and basic columns
    ans_a = row.get("answers_a") or {}
    ans_b = row.get("answers_b") or {}
    sel_qids = row.get("selected_qids")
    if isinstance(sel_qids, np.ndarray):
        sel_qids = sel_qids.tolist()
    sel_qids = sel_qids or []
    weights = row.get("selected_question_weights")
    if isinstance(weights, np.ndarray):
        weights = weights.tolist()
    weights = weights or []
    margin = abs(row.get("pairwise_margin", 0))
    error_cat = row.get("error_category", "")
    prompt = str(row.get("context", ""))
    resp_a = str(row.get("response_a", ""))
    resp_b = str(row.get("response_b", ""))
    domain = row.get("domain", "")

    # Computed stats (stored with _calc_ prefix)
    n_generic_sel = row.get("_calc_n_generic_selected", 0)
    n_sel = row.get("_calc_n_selected", 0)
    nonzero_rate = row.get("_calc_nonzero_rate", 0)
    n_na = row.get("_calc_n_any_na", 0)
    n_diff = row.get("_calc_n_diff", 0)
    nonzero_qids = row.get("_calc_nonzero_qids", [])
    nonzero_weights = row.get("_calc_nonzero_weights", [])
    support_a_sum = row.get("_calc_support_a_weight_sum", 0)
    support_b_sum = row.get("_calc_support_b_weight_sum", 0)
    max_abs_contrib_qid = row.get("_calc_max_abs_contrib_qid", None)
    max_abs_contrib_weight = row.get("_calc_max_abs_contrib_weight", 0)
    predicted = row.get("predicted_winner", "Tie")
    winner = row.get("winner", "")
    n_same_yes_yes = row.get("_calc_n_yes_yes", 0)
    n_same_no_no = row.get("_calc_n_no_no", 0)
    n_yes_yes_generic = row.get("_calc_n_yes_yes_generic", 0)
    nonzero_contributions = row.get("_calc_nonzero_contributions", [])

    # FM1: generic non-discriminative
    generic_ratio = n_generic_sel / max(n_sel, 1)
    if generic_ratio >= 0.5 and nonzero_rate < LOW_NONZERO_RATE_THRESHOLD:
        tags.append("FM1_generic_non_discriminative")
    elif generic_ratio >= 0.6 and n_yes_yes_generic >= 4:
        tags.append("FM1_generic_non_discriminative")

    # FM2: missing task-specific criteria
    task_specific_dims = {
        "creative_writing_script_scene": "clarity_and_communication",
        "coding": "coding_communication_conditional",
        "math_stem": "correctness_and_completeness",
    }
    needed_dim = task_specific_dims.get(task_type)
    if needed_dim and needed_dim not in str(row.get("_calc_selected_dimensions", [])):
        if task_type in ("creative_writing_script_scene", "coding"):
            tags.append("FM2_missing_task_specific_criteria")

    # FM3: NA noise
    if n_na >= 5:
        tags.append("FM3_na_noise_or_bad_question_fit")

    # FM4: low weight on discriminative qs
    if n_diff > 0 and nonzero_rate > 0 and all(
        w < LOW_WEIGHT_THRESHOLD for w in (nonzero_weights or [])
    ):
        tags.append("FM4_low_weight_on_discriminative_questions")

    # FM5: wrong direction with high weight
    if predicted != winner and predicted != "Tie" and winner != "Tie":
        if error_cat == "wrong_winner":
            # Check if highest-weight nonzero q supported wrong side
            if max_abs_contrib_qid is not None:
                # Determine which side the max q supported
                for qid, w, contrib in zip(
                    nonzero_qids, nonzero_weights,
                    nonzero_contributions
                ):
                    if qid == max_abs_contrib_qid:
                        if contrib > 0:
                            supported = "A"
                        else:
                            supported = "B"
                        if supported != winner:
                            tags.append("FM5_wrong_direction_high_weight")
                        break

    # FM6: human reason mismatch (check individual_preference overlap)
    ip = row.get("individual_preference", [])
    prompt_words = set(prompt.lower().split()[:200])
    suspicious = False
    for p_data in ip:
        if isinstance(p_data, dict):
            reasoning = str(p_data.get("reasoning", ""))
            if len(reasoning) > 50:
                # Check if reasoning mentions things not in prompt or responses
                reason_words = set(reasoning.lower().split())
                overlap_prompt = len(reason_words & prompt_words)
                if overlap_prompt < 3 and len(reason_words) < 20:
                    suspicious = True
                    break
    if suspicious:
        tags.append("FM6_human_reason_mismatch_or_annotation_noise")

    # FM7: binary checklist insufficient
    # High yes/yes rate + human reason mentions gradient words
    same_yes = n_same_yes_yes + n_same_no_no
    same_rate = same_yes / max(n_sel, 1)
    if same_rate >= 0.7 and n_diff >= 1 and n_diff <= 3:
        tags.append("FM7_binary_checklist_insufficient")

    # FM8: relevance vs contrastiveness gap
    if n_sel > 0 and n_diff <= 2 and nonzero_rate < 0.2:
        tags.append("FM8_selector_relevance_vs_contrastiveness_gap")

    # FM9: aggregation / tie delta issue
    if margin < SMALL_MARGIN_THRESHOLD and error_cat in ("tie", "wrong_winner"):
        tags.append("FM9_aggregation_threshold_or_tie_delta_issue")

    # FM10: annotator disagreement or weak preference
    if len(ip) >= 2:
        scores = [p_data.get("score", 0) for p_data in ip if isinstance(p_data, dict)]
        nonzero = [s for s in scores if s != 0]
        if len(nonzero) >= 2:
            signs = [1 if s > 0 else -1 for s in nonzero]
            if len(set(signs)) > 1:
                tags.append("FM10_annotator_disagreement")
            elif len(nonzero) >= 3:
                abs_vals = [abs(s) for s in nonzero]
                if max(abs_vals) <= 1:
                    tags.append("FM10_weak_preference_strength")

    return tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze HR-oracle tie errors")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank-index", default="checklists/v4_frozen/bank_index.parquet")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.input)
    bank = pd.read_parquet(args.bank_index)
    qid_to_meta = {}
    for _, row in bank.iterrows():
        qid_to_meta[int(row["qid"])] = {
            "dimension": row["dimension"],
            "sub_aspect": row["sub_aspect"],
            "question_text": row["question_text"],
        }

    # Identify generic qids from bank
    generic_qids = set()
    for qid, meta in qid_to_meta.items():
        sa = meta.get("sub_aspect", "")
        qt = meta.get("question_text", "").lower()
        if sa in GENERIC_SUB_ASPECTS:
            generic_qids.add(qid)
        elif any(phrase in qt for phrase in GENERIC_PHRASES):
            generic_qids.add(qid)
    print(f"Identified {len(generic_qids)} generic qids: {sorted(generic_qids)}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "cases").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Parse answers and reconstruct per-sample contributions
    # -----------------------------------------------------------------------
    print("Parsing answers and reconstructing contributions…")

    answers_a_list = []
    answers_b_list = []
    for _, row in df.iterrows():
        answers_a_list.append(parse_answer_json(row["parsed_a_json"]))
        answers_b_list.append(parse_answer_json(row["parsed_b_json"]))

    df["answers_a"] = answers_a_list
    df["answers_b"] = answers_b_list

    # Build per-sample stats
    samples = []
    for i, (_, row) in enumerate(df.iterrows()):
        ans_a = row["answers_a"]
        ans_b = row["answers_b"]
        sel_qids = list(row["selected_qids"]) if isinstance(row["selected_qids"], np.ndarray) else row["selected_qids"]
        weights = list(row["selected_question_weights"]) if isinstance(row["selected_question_weights"], np.ndarray) else row["selected_question_weights"]
        weights = [float(w) if not (isinstance(w, float) and np.isnan(w)) else np.nan for w in weights]

        s = {"idx": i, "sample_id": row["sample_id"]}

        # Map qidx (1..k in parsed JSON) to global qid
        # parsed_a_json uses q: 1..k indexing, not global qid
        n_sel = len(sel_qids)
        s["n_selected"] = n_sel

        # Build qid→answer map for A and B
        a_answers = {}
        b_answers = {}
        for qidx, gqid in enumerate(sel_qids, 1):
            a_answers[gqid] = ans_a.get(qidx, "N/A")
            b_answers[gqid] = ans_b.get(qidx, "N/A")

        # Count categories
        n_yes_yes = n_no_no = n_na_na = n_diff = 0
        n_na_a = n_na_b = n_any_na = 0
        n_yes_a = n_yes_b = 0
        diff_qids = []
        zero_qids = []
        nonzero_qids = []
        nonzero_weights_list = []
        nonzero_contributions = []
        nonzero_directions = []
        support_a_weight = 0.0
        support_b_weight = 0.0
        abs_nonzero_weight = 0.0
        max_contrib_qid = None
        max_contrib_weight = 0.0
        n_yes_yes_generic = 0

        for qidx, gqid in enumerate(sel_qids):
            aa = a_answers.get(gqid, "N/A")
            ab = b_answers.get(gqid, "N/A")
            w = weights[qidx] if qidx < len(weights) else np.nan

            if aa == "N/A" or ab == "N/A":
                n_any_na += 1
            if aa == "N/A":
                n_na_a += 1
            if ab == "N/A":
                n_na_b += 1

            if aa == ab:
                if aa == "yes":
                    n_yes_yes += 1
                    if gqid in generic_qids:
                        n_yes_yes_generic += 1
                elif aa == "no":
                    n_no_no += 1
                elif aa == "N/A":
                    n_na_na += 1
                # Same answer = zero contribution
                zero_qids.append(gqid)
            else:
                n_diff += 1
                diff_qids.append(gqid)
                if aa == "yes" and ab == "no":
                    if not np.isnan(w):
                        support_a_weight += w
                        abs_nonzero_weight += abs(w)
                        nonzero_qids.append(gqid)
                        nonzero_weights_list.append(w)
                        nonzero_contributions.append(1)  # supports A
                        nonzero_directions.append("A")
                        if abs(w) > max_contrib_weight:
                            max_contrib_weight = abs(w)
                            max_contrib_qid = gqid
                elif aa == "no" and ab == "yes":
                    if not np.isnan(w):
                        support_b_weight += w
                        abs_nonzero_weight += abs(w)
                        nonzero_qids.append(gqid)
                        nonzero_weights_list.append(w)
                        nonzero_contributions.append(-1)  # supports B
                        nonzero_directions.append("B")
                        if abs(w) > max_contrib_weight:
                            max_contrib_weight = abs(w)
                            max_contrib_qid = gqid
                elif aa == "N/A" or ab == "N/A":
                    # NA on one side — depends on policy; count as zero for now
                    zero_qids.append(gqid)

            if aa == "yes":
                n_yes_a += 1
            if ab == "yes":
                n_yes_b += 1

        s["n_yes_yes"] = n_yes_yes
        s["n_no_no"] = n_no_no
        s["n_na_na"] = n_na_na
        s["n_diff"] = n_diff
        s["n_na_a"] = n_na_a
        s["n_na_b"] = n_na_b
        s["n_any_na"] = n_any_na
        s["n_same_yes_yes"] = n_yes_yes  # alias
        s["n_same_no_no"] = n_no_no  # alias

        nonzero_rate = len(nonzero_qids) / max(n_sel, 1)
        s["nonzero_rate"] = nonzero_rate
        s["nonzero_qids"] = nonzero_qids
        s["nonzero_weights"] = nonzero_weights_list
        s["nonzero_contributions"] = nonzero_contributions
        s["nonzero_directions"] = nonzero_directions
        s["support_a_weight_sum"] = support_a_weight
        s["support_b_weight_sum"] = support_b_weight
        s["abs_nonzero_weight"] = abs_nonzero_weight
        s["max_abs_contrib_qid"] = max_contrib_qid
        s["max_abs_contrib_weight"] = max_contrib_weight
        s["zero_qids"] = zero_qids
        s["diff_qids"] = diff_qids
        s["n_yes_a"] = n_yes_a
        s["n_yes_b"] = n_yes_b
        s["n_yes_yes_generic"] = n_yes_yes_generic

        # Generic question count
        n_generic_sel = sum(1 for q in sel_qids if q in generic_qids)
        s["n_generic_selected"] = n_generic_sel
        s["generic_ratio"] = n_generic_sel / max(n_sel, 1)
        s["share_zero_contribution"] = len(zero_qids) / max(n_sel, 1)

        # Missing weights
        n_missing_w = sum(1 for w in weights if np.isnan(w))
        s["missing_weight_count"] = n_missing_w

        # Selected dimensions
        sel_dims = [qid_to_meta.get(q, {}).get("dimension", "unknown") for q in sel_qids]
        s["selected_dimensions"] = sel_dims

        # Task type
        prompt = str(row.get("context", ""))
        task_type = detect_task_type(prompt)
        s["task_type"] = task_type

        # Get individual_preference scores
        ip = row.get("individual_preference", [])
        if isinstance(ip, np.ndarray):
            ip = ip.tolist()
        annotator_scores = []
        annotator_reasonings = []
        for p_data in ip:
            if isinstance(p_data, dict):
                s_val = p_data.get("score", 0)
                annotator_scores.append(s_val)
                annotator_reasonings.append(p_data.get("reasoning", ""))
        s["annotator_scores"] = annotator_scores
        s["annotator_reasonings"] = annotator_reasonings
        s["n_annotators"] = len(annotator_scores)

        samples.append(s)

    # Attach samples as columns
    for key in samples[0]:
        if key not in ("idx", "sample_id"):
            df[f"_calc_{key}"] = [s[key] for s in samples]

    print(f"Parsed {len(samples)} samples.")

    # -----------------------------------------------------------------------
    # 2. Question-level frequency table (A)
    # -----------------------------------------------------------------------
    print("Building question_frequency.csv…")

    qfreq_rows = []
    all_qids = sorted(set(qid_to_meta.keys()))
    for qid in all_qids:
        meta = qid_to_meta.get(qid, {})
        selected = 0
        diff_c = 0
        nonzero_c = 0
        support_a = 0
        support_b = 0
        na_c = 0
        weights_vals = []

        for s in samples:
            sel_qids = list(df.loc[s["idx"], "selected_qids"])
            ans_a = df.loc[s["idx"], "answers_a"]
            ans_b = df.loc[s["idx"], "answers_b"]
            raw_weights = list(df.loc[s["idx"], "selected_question_weights"])
            if qid in sel_qids:
                selected += 1
                pos = sel_qids.index(qid)
                w = raw_weights[pos] if pos < len(raw_weights) else np.nan
                if not np.isnan(w):
                    weights_vals.append(w)
                aa = ans_a.get(pos + 1, "N/A")
                ab = ans_b.get(pos + 1, "N/A")
                if aa != ab:
                    diff_c += 1
                    if aa == "yes" and ab == "no":
                        nonzero_c += 1
                        support_a += 1
                    elif aa == "no" and ab == "yes":
                        nonzero_c += 1
                        support_b += 1
                if aa == "N/A" or ab == "N/A":
                    na_c += 1

        qfreq_rows.append({
            "qid": qid,
            "selected_count": selected,
            "diff_count": diff_c,
            "nonzero_contribution_count": nonzero_c,
            "nonzero_rate": round(nonzero_c / max(selected, 1), 4),
            "avg_weight": round(safe_mean(pd.Series(weights_vals)), 6),
            "median_weight": round(safe_median(pd.Series(weights_vals)), 6),
            "total_abs_weight": round(sum(abs(w) for w in weights_vals), 6),
            "support_A_count": support_a,
            "support_B_count": support_b,
            "na_count": na_c,
            "dimension": meta.get("dimension", ""),
            "sub_aspect": meta.get("sub_aspect", ""),
            "question_text": meta.get("question_text", ""),
            "is_generic": qid in generic_qids,
        })

    qfreq = pd.DataFrame(qfreq_rows)
    qfreq.to_csv(out_dir / "tables" / "question_frequency.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Dimension failure stats (B)
    # -----------------------------------------------------------------------
    print("Building dimension_failure_stats.csv…")

    dim_stats_rows = []
    dim_groups = qfreq.groupby(["dimension", "sub_aspect"])
    total_sel = qfreq["selected_count"].sum()
    total_nonzero = qfreq["nonzero_contribution_count"].sum()
    # Count error samples
    error_mask = df["error_category"].isin(["tie", "wrong_winner"])
    error_sample_qids = defaultdict(int)
    for _, row in df[error_mask].iterrows():
        for qid in row["selected_qids"]:
            error_sample_qids[int(qid)] += 1

    for (dim, sa), grp in dim_groups:
        dim_stats_rows.append({
            "dimension": dim,
            "sub_aspect": sa,
            "selected_count": int(grp["selected_count"].sum()),
            "diff_count": int(grp["diff_count"].sum()),
            "nonzero_rate": round(grp["nonzero_contribution_count"].sum() / max(grp["selected_count"].sum(), 1), 4),
            "avg_weight": round(grp["avg_weight"].mean(), 6),
            "share_of_all_selected": round(grp["selected_count"].sum() / max(total_sel, 1), 4),
            "share_of_all_nonzero": round(grp["nonzero_contribution_count"].sum() / max(total_nonzero, 1), 4),
        })

    dim_stats = pd.DataFrame(dim_stats_rows)
    dim_stats.to_csv(out_dir / "tables" / "dimension_failure_stats.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Generic question overuse (C)
    # -----------------------------------------------------------------------
    print("Building generic_question_overuse.csv…")

    gen_qfreq = qfreq[qfreq["is_generic"]]
    non_gen_qfreq = qfreq[~qfreq["is_generic"]]

    generic_rows = []
    for qid in sorted(generic_qids):
        if qid in qfreq["qid"].values:
            row = qfreq[qfreq["qid"] == qid].iloc[0]
        else:
            continue
        # Error samples with this qid
        err_with_qid = 0
        yes_yes_in_errors = 0
        for s in samples:
            if df.loc[s["idx"], "error_category"] not in ("tie", "wrong_winner"):
                continue
            sel_qids = list(df.loc[s["idx"], "selected_qids"])
            if qid in sel_qids:
                err_with_qid += 1
                pos = sel_qids.index(qid)
                ans_a = df.loc[s["idx"], "answers_a"].get(pos + 1, "N/A")
                ans_b = df.loc[s["idx"], "answers_b"].get(pos + 1, "N/A")
                if ans_a == "yes" and ans_b == "yes":
                    yes_yes_in_errors += 1

        total_err = error_mask.sum()
        generic_rows.append({
            "qid": qid,
            "sub_aspect": row["sub_aspect"],
            "dimension": row["dimension"],
            "question_text": trunc(row["question_text"], 150),
            "selected_count": int(row["selected_count"]),
            "nonzero_contribution_count": int(row["nonzero_contribution_count"]),
            "nonzero_rate": row["nonzero_rate"],
            "avg_weight": row["avg_weight"],
            "presence_in_error_samples": err_with_qid,
            "error_sample_rate": round(err_with_qid / max(total_err, 1), 4),
            "yes_yes_in_error_samples": yes_yes_in_errors,
            "yes_yes_in_error_rate": round(yes_yes_in_errors / max(err_with_qid, 1), 4),
        })

    gen_overuse = pd.DataFrame(generic_rows)
    gen_overuse.to_csv(out_dir / "tables" / "generic_question_overuse.csv", index=False)

    # Overall generic stats
    total_generic_sel = gen_overuse["selected_count"].sum() if len(gen_overuse) > 0 else 0
    total_all_sel = qfreq["selected_count"].sum()
    print(f"Generic questions: {total_generic_sel}/{total_all_sel} selections "
          f"({total_generic_sel/max(total_all_sel,1):.1%})")

    # -----------------------------------------------------------------------
    # 5. Classify failure modes
    # -----------------------------------------------------------------------
    print("Classifying failure modes…")

    failure_tags = []
    for _, row in df.iterrows():
        task_type = row.get("_calc_task_type", "")
        tags = classify_failure_modes(row, generic_qids, task_type)
        failure_tags.append(tags)

    df["_failure_modes"] = failure_tags

    # Count failure modes
    fm_counter = Counter()
    for tags in failure_tags:
        for t in tags:
            fm_counter[t] += 1

    fm_counts = pd.DataFrame([
        {"failure_mode": fm, "count": c, "pct_of_samples": round(c / len(df), 4)}
        for fm, c in fm_counter.most_common()
    ])
    fm_counts.to_csv(out_dir / "tables" / "failure_mode_counts.csv", index=False)
    print("Failure mode counts:")
    print(fm_counts.to_string(index=False))

    # Co-occurrence
    fm_list = list(fm_counter.keys())
    cooc = np.zeros((len(fm_list), len(fm_list)), dtype=int)
    for tags in failure_tags:
        for i, f1 in enumerate(fm_list):
            for j, f2 in enumerate(fm_list):
                if f1 in tags and f2 in tags:
                    cooc[i, j] += 1
    cooc_df = pd.DataFrame(cooc, index=fm_list, columns=fm_list)
    cooc_df.to_csv(out_dir / "tables" / "failure_mode_cooccurrence.csv")

    # By domain
    fm_domain_rows = []
    for domain in df["domain"].unique():
        domain_df = df[df["domain"] == domain]
        for fm in fm_list:
            count = sum(fm in tags for tags in domain_df["_failure_modes"])
            fm_domain_rows.append({
                "domain": domain,
                "failure_mode": fm,
                "count": count,
                "pct": round(count / max(len(domain_df), 1), 4),
            })
    fm_domain = pd.DataFrame(fm_domain_rows)
    fm_domain.to_csv(out_dir / "tables" / "failure_mode_by_domain.csv", index=False)

    # By task type
    fm_task_rows = []
    for tt in df["_calc_task_type"].unique():
        tt_df = df[df["_calc_task_type"] == tt]
        for fm in fm_list:
            count = sum(fm in tags for tags in tt_df["_failure_modes"])
            if count > 0:
                fm_task_rows.append({
                    "task_type": tt,
                    "failure_mode": fm,
                    "count": count,
                    "pct": round(count / max(len(tt_df), 1), 4),
                })
    fm_task = pd.DataFrame(fm_task_rows)
    fm_task.to_csv(out_dir / "tables" / "failure_mode_by_task_type.csv", index=False)

    # -----------------------------------------------------------------------
    # 6. Task type stats
    # -----------------------------------------------------------------------
    print("Building task_type_stats.csv…")

    tt_rows = []
    for tt in sorted(df["_calc_task_type"].unique()):
        tt_df = df[df["_calc_task_type"] == tt]
        tt_tags = []
        for tags in tt_df["_failure_modes"]:
            tt_tags.extend(tags)
        tt_fm_counter = Counter(tt_tags)

        # Top selected dimensions
        dim_counter = Counter()
        for dims in tt_df["_calc_selected_dimensions"]:
            dim_counter.update(dims)
        top_dims = ", ".join(f"{d}({c})" for d, c in dim_counter.most_common(5))

        tt_rows.append({
            "task_type": tt,
            "n_samples": len(tt_df),
            "avg_selected_generic_questions": round(tt_df["_calc_n_generic_selected"].mean(), 2),
            "avg_nonzero_rate": round(tt_df["_calc_nonzero_rate"].mean(), 4),
            "avg_na_count": round(tt_df["_calc_n_any_na"].mean(), 2),
            "avg_diff_count": round(tt_df["_calc_n_diff"].mean(), 2),
            "avg_yes_yes_count": round(tt_df["_calc_n_yes_yes"].mean(), 2),
            "top_failure_modes": ", ".join(f"{fm}({c})" for fm, c in tt_fm_counter.most_common(3)),
            "top_dimensions": top_dims,
        })

    tt_stats = pd.DataFrame(tt_rows)
    tt_stats.to_csv(out_dir / "tables" / "task_type_stats.csv", index=False)
    print("\nTask type stats:")
    print(tt_stats.to_string(index=False))

    # -----------------------------------------------------------------------
    # 7. Sample-level error tags
    # -----------------------------------------------------------------------
    sample_tags = pd.DataFrame({
        "sample_id": df["sample_id"],
        "domain": df["domain"],
        "error_category": df["error_category"],
        "task_type": df["_calc_task_type"],
        "non_zero_rate": df["_calc_nonzero_rate"],
        "n_generic_selected": df["_calc_n_generic_selected"],
        "n_diff": df["_calc_n_diff"],
        "n_any_na": df["_calc_n_any_na"],
        "n_yes_yes": df["_calc_n_yes_yes"],
        "pairwise_margin": df["pairwise_margin"],
        "failure_modes": [", ".join(tags) for tags in failure_tags],
        "n_failure_modes": [len(tags) for tags in failure_tags],
    })
    sample_tags.to_csv(out_dir / "tables" / "sample_level_error_tags.csv", index=False)

    # -----------------------------------------------------------------------
    # 8. Representative cases per failure mode
    # -----------------------------------------------------------------------
    print("Writing case files…")

    max_cases_per_fm = 5
    for fm in fm_list:
        # Find samples with this tag
        idxs = [i for i, tags in enumerate(failure_tags) if fm in tags]
        if not idxs:
            continue

        # Sort by severity: pick ones with highest generic ratio or lowest nonzero rate
        fm_samples = df.iloc[idxs].copy()
        fm_samples["_sort_key"] = fm_samples["_calc_nonzero_rate"]
        fm_samples = fm_samples.sort_values("_sort_key")

        lines = [f"# {fm}\n", f"Samples tagged: {len(idxs)}\n\n"]
        selected = 0
        for _, row in fm_samples.iterrows():
            if selected >= max_cases_per_fm:
                break
            selected += 1

            lines.append(f"## Case {selected}: {row['sample_id']}\n")
            lines.append(f"- Domain: `{row['domain']}`")
            lines.append(f"- Task type: `{row['_calc_task_type']}`")
            lines.append(f"- True winner: `{row['winner']}`, Predicted: `{row['predicted_winner']}`")
            lines.append(f"- Error category: `{row['error_category']}`")
            sc_a = row.get("score_a", np.nan)
            sc_b = row.get("score_b", np.nan)
            lines.append(f"- Score: A={sc_a:.4f}, B={sc_b:.4f}, Margin={row['pairwise_margin']:.4f}" if not np.isnan(sc_a) else f"- Margin: {row['pairwise_margin']:.4f}")
            lines.append("")

            lines.append(f"**Prompt:** {trunc(row['context'], 400)}\n")
            lines.append(f"**Response A:** {trunc(row['response_a'], 300)}\n")
            lines.append(f"**Response B:** {trunc(row['response_b'], 300)}\n")

            # Selected questions table
            lines.append("\n| qid | dimension | sub_aspect | answer A | answer B | weight | contribution |")
            lines.append("|-----|-----------|------------|----------|----------|--------|--------------|")
            sel_qids = list(row["selected_qids"]) if isinstance(row["selected_qids"], np.ndarray) else row["selected_qids"]
            weights = list(row["selected_question_weights"]) if isinstance(row["selected_question_weights"], np.ndarray) else row["selected_question_weights"]
            ans_a = row["answers_a"]
            ans_b = row["answers_b"]
            for pos, qid in enumerate(sel_qids):
                meta = qid_to_meta.get(int(qid), {})
                aa = ans_a.get(pos + 1, "N/A")
                ab = ans_b.get(pos + 1, "N/A")
                w = weights[pos] if pos < len(weights) else np.nan
                w_str = f"{w:.4f}" if not np.isnan(w) else "nan"
                if aa == "yes" and ab == "no":
                    contr = "+A"
                elif aa == "no" and ab == "yes":
                    contr = "+B"
                else:
                    contr = "0"
                lines.append(
                    f"| {qid} | {meta.get('dimension','')[:25]} | {meta.get('sub_aspect','')[:25]} "
                    f"| {aa} | {ab} | {w_str} | {contr} |"
                )
            lines.append("")

            # Human reasoning
            ip = row.get("individual_preference", [])
            if isinstance(ip, np.ndarray):
                ip = ip.tolist()
            for i_p, p_data in enumerate(ip):
                if isinstance(p_data, dict):
                    lines.append(f"**Annotator {i_p+1}:** score={p_data.get('score','')}")
                    lines.append(f"> {trunc(p_data.get('reasoning', ''), 500)}\n")

            # Why tagged
            lines.append(f"**Why tagged {fm}:**")
            lines.append(f"- Nonzero rate: {row['_calc_nonzero_rate']:.3f}")
            lines.append(f"- N generic selected: {row['_calc_n_generic_selected']}/{row['_calc_n_selected']}")
            lines.append(f"- N diff: {row['_calc_n_diff']}, N yes-yes: {row['_calc_n_yes_yes']}, N any NA: {row['_calc_n_any_na']}")
            lines.append(f"- Task type: {row['_calc_task_type']}")
            lines.append("")
            lines.append("---\n")

        # Write case file
        with open(out_dir / "cases" / f"{fm}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # -----------------------------------------------------------------------
    # 9. Build analysis summary
    # -----------------------------------------------------------------------
    print("Building analysis_summary.md…")

    n_total = len(df)
    n_wrong = (df["error_category"] == "wrong_winner").sum()
    n_tie = (df["error_category"] == "tie").sum()
    n_correct = (df["error_category"] == "correct").sum()

    avg_nonzero = df["_calc_nonzero_rate"].mean()
    avg_generic = df["_calc_n_generic_selected"].mean()
    avg_na = df["_calc_n_any_na"].mean()
    avg_diff = df["_calc_n_diff"].mean()
    avg_yes_yes = df["_calc_n_yes_yes"].mean()

    # Top failure modes
    top_fms = fm_counter.most_common(5)

    # Top generic questions
    top_gen_qs = gen_overuse.sort_values("selected_count", ascending=False).head(5)

    # Task types most error-prone
    task_error_rates = []
    for tt in df["_calc_task_type"].unique():
        tt_df = df[df["_calc_task_type"] == tt]
        err_rate = (tt_df["error_category"] != "correct").mean()
        task_error_rates.append((tt, len(tt_df), err_rate))
    task_error_rates.sort(key=lambda x: -x[2])

    # Domain differences
    domain_stats = []
    for dom in df["domain"].unique():
        ddf = df[df["domain"] == dom]
        domain_stats.append({
            "domain": dom,
            "n": len(ddf),
            "avg_nonzero_rate": ddf["_calc_nonzero_rate"].mean(),
            "avg_yes_yes": ddf["_calc_n_yes_yes"].mean(),
            "avg_na": ddf["_calc_n_any_na"].mean(),
            "avg_generic": ddf["_calc_n_generic_selected"].mean(),
        })

    summary = f"""# HR-Oracle Weighted Tie/Error Analysis Summary

**Data:** {n_total} samples ({n_tie} ties, {n_wrong} wrong winners, {n_correct} correct)

## Overview

| Metric | Value |
|--------|-------|
| Total samples | {n_total} |
| Ties (predicted) | {n_tie} ({n_tie/n_total:.1%}) |
| Wrong winner | {n_wrong} ({n_wrong/n_total:.1%}) |
| Avg nonzero contribution rate | {avg_nonzero:.3f} |
| Avg generic questions selected | {avg_generic:.1f}/{df['_calc_n_selected'].mean():.0f} (k={df['k_selected'].iloc[0]}) |
| Avg N/A count per sample | {avg_na:.2f} |
| Avg diff questions per sample | {avg_diff:.2f} |
| Avg yes-yes per sample | {avg_yes_yes:.2f} |

## Ranked Failure Modes

"""
    for fm, cnt in top_fms:
        summary += f"- **{fm}**: {cnt} samples ({cnt/n_total:.1%})\n"

    summary += f"""
## Domain Breakdown

"""
    for ds in domain_stats:
        summary += (
            f"- **{ds['domain']}** (n={ds['n']}): "
            f"nonzero_rate={ds['avg_nonzero_rate']:.3f}, "
            f"yes_yes={ds['avg_yes_yes']:.1f}, "
            f"NA={ds['avg_na']:.1f}, "
            f"generic={ds['avg_generic']:.1f}\n"
        )

    summary += f"""
## Task Type Sensitivity

| Task type | N | Error rate | Avg nonzero | Avg generic | Avg yes-yes | Avg NA |
|-----------|------|-----------|-------------|-------------|-------------|--------|
"""
    for tt, n_tt, err_rate in task_error_rates:
        tt_df = df[df["_calc_task_type"] == tt]
        summary += (
            f"| {tt} | {n_tt} | {err_rate:.2%} | "
            f"{tt_df['_calc_nonzero_rate'].mean():.3f} | "
            f"{tt_df['_calc_n_generic_selected'].mean():.1f} | "
            f"{tt_df['_calc_n_yes_yes'].mean():.1f} | "
            f"{tt_df['_calc_n_any_na'].mean():.1f} |\n"
        )

    summary += f"""
## Generic Question Impact

Generic questions account for **{total_generic_sel}/{total_all_sel} ({total_generic_sel/max(total_all_sel,1):.1%})** of all selections.

Top 5 most-selected generic questions:
"""
    if len(top_gen_qs) > 0:
        for _, gr in top_gen_qs.iterrows():
            summary += (
                f"- qid={int(gr['qid'])} ({gr['sub_aspect']}): "
                f"selected {int(gr['selected_count'])}x, nonzero_rate={gr['nonzero_rate']:.3f}, "
                f"yes_yes in {gr['yes_yes_in_error_rate']:.1%} of error samples\n"
            )

    # Check creative writing specifically
    cw_df = df[df["_calc_task_type"] == "creative_writing_script_scene"]
    if len(cw_df) > 0:
        summary += f"""
## Creative Writing / Script Focus

Creative writing samples (n={len(cw_df)}):
- Avg nonzero rate: {cw_df['_calc_nonzero_rate'].mean():.3f}
- Avg yes-yes: {cw_df['_calc_n_yes_yes'].mean():.1f}
- Avg generic selected: {cw_df['_calc_n_generic_selected'].mean():.1f}
- Avg diff: {cw_df['_calc_n_diff'].mean():.1f}
- Top failure modes: {Counter(t for tags in cw_df['_failure_modes'] for t in tags).most_common(3)}
"""
    else:
        summary += "\n## Creative Writing / Script Focus\n\nNo creative writing samples found in this dataset.\n"

    # Key actionable recommendations
    summary += f"""
## Actionable Recommendations

1. **Primary bottleneck:** {top_fms[0][0] if top_fms else 'N/A'} affects {top_fms[0][1] if top_fms else 0} samples ({top_fms[0][1]/n_total:.1%} if top_fms else 'N/A')
2. **Selector vs bank vs judge:** Based on nonzero_rate={avg_nonzero:.3f}, the dominant issue is {'low question discrimination (yes/yes everywhere) — selector may be relevant but not contrastive' if avg_nonzero < 0.3 else 'aggregation/weight calibration' if avg_nonzero > 0.6 else 'mixed — both question selection and answer discrimination need improvement'}
3. **Generic question overuse:** {'CRITICAL — {:.0f}% of selections are generic, contributing zero discrimination'.format(100*total_generic_sel/max(total_all_sel,1)) if total_generic_sel/total_all_sel > 0.4 else 'MODERATE — generic questions present but not dominating' if total_generic_sel/total_all_sel > 0.2 else 'LOW — generic questions not the main issue'}
4. **Next most impactful change:**
   - If nonzero_rate < 0.25: contrastive/hard-negative selector training, task-type gating, or creative/script-specific bank expansion
   - If N/A rate > 3/sample: better question applicability gating or conditional questions
   - If small margins dominate: tiebreak adjudicator or weight calibration
   - If annotation noise suspected: clean individual_preference with entity-overlap filtering
"""

    with open(out_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP-LEVEL SUMMARY")
    print("=" * 70)
    print(summary)
    print(f"\nAll outputs written to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
