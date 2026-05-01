"""Run comparative checklist evaluation on dev_600.

Usage:
    python scripts/run_comparative_eval.py --question-source generated --aggregation both --output results/comparative/gen_both
    python scripts/run_comparative_eval.py --question-source hr-oracle --aggregation weighted --output results/comparative/hr_weighted
    python scripts/run_comparative_eval.py --question-source fullbank --aggregation count --output results/comparative/fullbank_count

Question sources:
    generated — per-sample generated checklists from data/generated_checklists/
    hr-oracle  — HR-oracle top-15 qids + weights from predictions.parquet + review parquet
    fullbank   — all 58 v4 bank questions

Aggregation:
    count     — simple count of A vs B answers
    weighted  — sum of question weights per side (uniform for generated/fullbank; existing for hr-oracle)
    both      — compute both metrics, write both
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT as BASE_DIR, CHECKLISTS_DIR, COMPARATIVE_QUESTIONS_CACHE
from utils import (
    load_judge_model,
    generate_batch,
    build_comparative_prompt,
    parse_comparative_output,
    comparative_parse_ok,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-source", required=True,
                        choices=["generated", "hr-oracle", "fullbank"],
                        help="Source of questions to evaluate on")
    parser.add_argument("--aggregation", required=True,
                        choices=["count", "weighted", "both"],
                        help="count=simple A/B count, weighted=sum question weights, both=compute both")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for metrics.json + predictions.parquet")
    parser.add_argument("--split", default="dev_600",
                        help="Split name (default: dev_600)")
    parser.add_argument("--backend", default="llamacpp",
                        choices=["llamacpp", "vllm"])
    parser.add_argument("--model-id", default="Qwen3.5-9B",
                        help="Judge model ID")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Not used by llamacpp backend (parallel handled server-side)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples for debugging (0=all)")
    return parser.parse_args()


# ── Question loading ──


def load_generated_questions(split):
    """Load per-sample generated checklists from data/generated_checklists/."""
    gen_dir = BASE_DIR / "data" / "generated_checklists"
    parquets = sorted(gen_dir.glob(f"{split}*.parquet"))
    if not parquets:
        parquets = sorted(gen_dir.glob("*.parquet"))
    if not parquets:
        print("Error: No generated checklist parquet found in", gen_dir)
        print("Regenerate via run_generator_infer.py or copy from server.")
        sys.exit(1)
    path = parquets[-1]
    print(f"Loading generated questions from: {path}")
    df = pd.read_parquet(path)
    if "questions" not in df.columns and "generated_checklist" in df.columns:
        df = df.copy()
        df["questions"] = df["generated_checklist"].apply(parse_generated_questions)
    if "questions" not in df.columns:
        print(f"Error: no questions found in {path}. Columns: {df.columns.tolist()}")
        sys.exit(1)
    if "prompt_id" not in df.columns and "sample_id" not in df.columns:
        print(f"Error: generated checklist file needs prompt_id or sample_id. Columns: {df.columns.tolist()}")
        sys.exit(1)
    return df


def parse_generated_questions(text):
    """Extract bullet questions from a canonical generated checklist string."""
    questions = []
    for line in str(text).splitlines():
        line = line.strip()
        if line.startswith("- "):
            q = line[2:].strip()
            if q:
                questions.append(q)
    return questions


def load_generated_eval_rows(split):
    """Load the row table that matches generated checklist ids."""
    with_reason_path = BASE_DIR / "data" / "with_reason" / f"{split}_reasoning.parquet"
    if with_reason_path.exists():
        df = pd.read_parquet(with_reason_path)
        if "swap_flag" in df.columns:
            df = df[df["swap_flag"] == False].reset_index(drop=True)
        return df
    split_path = BASE_DIR / "data" / "splits" / f"{split}.parquet"
    return pd.read_parquet(split_path)


def load_hr_oracle_questions():
    """Load HR-oracle top-15 qids and weights from predictions + review parquets."""
    pred_dir = BASE_DIR / "results" / "dynamic_test"
    pred_paths = list(pred_dir.glob("**/predictions.parquet"))
    hr_pred = [p for p in pred_paths if "hroracle" in str(p).lower()]
    if not hr_pred:
        print("Error: No HR-oracle predictions.parquet found.")
        sys.exit(1)
    pred_path = sorted(hr_pred)[-1]
    predictions = pd.read_parquet(pred_path)
    print(f"HR-oracle predictions: {len(predictions)} samples from {pred_path}")

    # Review parquet has selected_question_weights (for tie/error samples)
    review_path = BASE_DIR / "results" / "review" / "hroracle_weighted_tie_error_review_with_weights.parquet"
    weight_lookup = {}
    if review_path.exists():
        review_df = pd.read_parquet(review_path)
        for _, row in review_df.iterrows():
            qids = row.get("selected_qids")
            weights = row.get("selected_question_weights")
            if qids is not None and weights is not None:
                if isinstance(qids, np.ndarray):
                    qids = qids.tolist()
                if isinstance(weights, np.ndarray):
                    weights = weights.tolist()
                weight_lookup[row["prompt_id"]] = dict(zip(qids, weights))
        print(f"Weight lookup: {len(weight_lookup)} samples")
    else:
        print("Warning: No review parquet found. Using uniform weights.")
    return predictions, weight_lookup


def load_fullbank_questions():
    """Load all bank questions from v4_frozen bank_index.parquet."""
    bank_path = CHECKLISTS_DIR / "bank_index.parquet"
    if not bank_path.exists():
        print(f"Error: bank index not found at {bank_path}")
        sys.exit(1)
    df = pd.read_parquet(bank_path)
    print(f"Full bank: {len(df)} questions from {bank_path}")
    return df


# ── Comparative question text loading ──


def load_comparative_cache():
    """Load comparative question rewrites from cache JSON."""
    if not COMPARATIVE_QUESTIONS_CACHE.exists():
        print(f"Error: Comparative cache not found at {COMPARATIVE_QUESTIONS_CACHE}")
        print("Run scripts/rewrite_comparative_questions.py first.")
        sys.exit(1)
    with open(COMPARATIVE_QUESTIONS_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_comparative_text(original, cache):
    """Get comparative rewrite, falling back to naive transformation."""
    cached = cache.get(original)
    if cached:
        return cached
    # Naive fallback: prepend "Which response better..."
    lower = original.lower()
    if lower.startswith("does the response"):
        return original.replace("Does the response", "Which response", 1).replace(
            "does the response", "which response", 1
        ).rstrip("?") + "?"
    if lower.startswith("is the") or lower.startswith("are the"):
        return "Which response " + original[0].lower() + original[1:].rstrip("?") + "?"
    if lower.startswith("does") or lower.startswith("do"):
        return "Which response " + original[0].lower() + original[1:].rstrip("?") + "?"
    return f"Which response better satisfies: {original}"


# ── Scoring ──


def compute_simple_count(answers):
    """Count A vs B answers. Returns (score_a, score_b, winner)."""
    count_a = sum(1 for v in answers.values() if v == "A")
    count_b = sum(1 for v in answers.values() if v == "B")
    margin = count_a - count_b
    winner = "A" if margin > 0 else ("B" if margin < 0 else "Tie")
    return count_a, count_b, winner


def compute_weighted_score(answers, weights):
    """Sum weights per side. Returns (score_a, score_b, winner)."""
    score_a = sum(weights.get(qnum, 0) for qnum, label in answers.items() if label == "A")
    score_b = sum(weights.get(qnum, 0) for qnum, label in answers.items() if label == "B")
    winner = "A" if score_a > score_b else ("B" if score_b > score_a else "Tie")
    return score_a, score_b, winner


# ── Metrics ──


def compute_metrics(df):
    """Compute accuracy, macro_f1, tie_rate, effective_accuracy, parse_ok_rate."""
    total = len(df)
    if total == 0:
        return {"error": "empty predictions"}
    valid = int(df["parse_ok"].sum())
    correct = int((df["predicted_winner"] == df["winner"]).sum())
    ties = int((df["predicted_winner"] == "Tie").sum())
    wrong = total - correct - ties

    accuracy = correct / total
    tie_rate = ties / total
    effective_accuracy = correct / (total - ties) if (total - ties) > 0 else 0.0

    # Macro F1: precision/recall per class, averaged
    classes = ["A", "B", "Tie"]
    class_f1s = []
    for c in classes:
        tp = int(((df["predicted_winner"] == c) & (df["winner"] == c)).sum())
        fp = int(((df["predicted_winner"] == c) & (df["winner"] != c)).sum())
        fn = int(((df["predicted_winner"] != c) & (df["winner"] == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_f1s.append(f1)
    macro_f1 = sum(class_f1s) / len(class_f1s)

    avg_a = float(df["score_a"].mean()) if "score_a" in df.columns else 0.0
    avg_b = float(df["score_b"].mean()) if "score_b" in df.columns else 0.0
    avg_tie = float(df["n_ties"].mean()) if "n_ties" in df.columns else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "tie_rate": round(tie_rate, 4),
        "effective_accuracy": round(effective_accuracy, 4),
        "correct": correct,
        "ties": ties,
        "wrong": wrong,
        "n_valid": valid,
        "n_total": total,
        "parse_ok_rate": round(valid / total, 4),
        "avg_answers_per_sample": round(avg_a + avg_b + avg_tie, 2),
        "avg_a_count": round(avg_a, 2),
        "avg_b_count": round(avg_b, 2),
        "avg_tie_count": round(avg_tie, 2),
    }


# ── Evaluation loops ──


def evaluate_generated(args, judge):
    """Evaluate on per-sample generated checklists."""
    gen_df = load_generated_questions(args.split)
    cache = load_comparative_cache()

    split_df = load_generated_eval_rows(args.split)
    id_col = "prompt_id" if "prompt_id" in gen_df.columns else "sample_id"
    if id_col not in split_df.columns:
        print(f"Error: generated ids use {id_col}, but eval rows have columns: {split_df.columns.tolist()}")
        sys.exit(1)

    results = []
    limit = args.limit or len(split_df)
    for idx, row in split_df.iloc[:limit].iterrows():
        sample_key = row[id_col]

        # Find generated questions for this sample
        gen_row = gen_df[gen_df[id_col] == sample_key]
        if gen_row.empty:
            continue
        questions = gen_row.iloc[0].get("questions", [])
        if not isinstance(questions, list) or len(questions) == 0:
            continue

        comp_questions = [get_comparative_text(q, cache) for q in questions]
        prompt = build_comparative_prompt(
            row["context"], row["response_a"], row["response_b"], comp_questions
        )

        try:
            raw = generate_batch(judge, [[{"role": "user", "content": prompt}]],
                                 max_tokens=args.max_new_tokens)[0]
        except Exception as e:
            print(f"  [ERROR] judge call failed for {sample_key}: {e}")
            continue

        parsed = parse_comparative_output(raw, len(comp_questions))
        parse_ok = comparative_parse_ok(parsed, len(comp_questions))

        count_a, count_b, count_win = compute_simple_count(parsed)
        uniform = {i+1: 1.0/len(comp_questions) for i in range(len(comp_questions))}
        ws_a, ws_b, ws_win = compute_weighted_score(parsed, uniform)

        if args.aggregation in ("count", "both"):
            predicted = count_win
            score_a, score_b = count_a, count_b
        else:
            predicted = ws_win
            score_a, score_b = ws_a, ws_b

        results.append({
            "sample_id": row.get("sample_id", idx), "prompt_id": row.get("prompt_id", sample_key),
            "domain": row.get("domain", ""),
            "winner": row["winner"], "predicted_winner": predicted,
            "parse_ok": parse_ok, "answers_raw": raw,
            "answers_parsed": str(parsed),
            "score_a": score_a, "score_b": score_b,
            "weighted_score_a": ws_a, "weighted_score_b": ws_b,
            "n_questions": len(comp_questions),
            "n_ties": sum(1 for v in parsed.values() if v == "Tie"),
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{limit}] ok={parse_ok} win={predicted}")

    return pd.DataFrame(results)


def evaluate_hr_oracle(args, judge):
    """Evaluate on HR-oracle top-15 qids."""
    predictions, weight_lookup = load_hr_oracle_questions()
    cache = load_comparative_cache()
    bank_df = load_fullbank_questions()

    qid_to_text = dict(zip(bank_df["qid"], bank_df["question_text"]))

    split_path = BASE_DIR / "data" / "splits" / f"{args.split}.parquet"
    split_df = pd.read_parquet(split_path)

    results = []
    limit = args.limit or len(split_df)
    for idx, row in split_df.iloc[:limit].iterrows():
        prompt_id = row["prompt_id"]

        pred_row = predictions[predictions["prompt_id"] == prompt_id]
        if pred_row.empty:
            continue
        pred = pred_row.iloc[0]
        qids = pred.get("selected_qids", [])
        if isinstance(qids, np.ndarray):
            qids = qids.tolist()
        elif isinstance(qids, float) and np.isnan(qids):
            continue
        if not qids:
            continue

        texts = [qid_to_text.get(qid, "") for qid in qids if qid_to_text.get(qid)]
        comp_qs = [get_comparative_text(t, cache) for t in texts]
        if not comp_qs:
            continue

        prompt = build_comparative_prompt(
            row["context"], row["response_a"], row["response_b"], comp_qs
        )

        try:
            raw = generate_batch(judge, [[{"role": "user", "content": prompt}]],
                                 max_tokens=args.max_new_tokens)[0]
        except Exception as e:
            print(f"  [ERROR] judge call failed for {prompt_id}: {e}")
            continue

        parsed = parse_comparative_output(raw, len(comp_qs))
        parse_ok = comparative_parse_ok(parsed, len(comp_qs))

        count_a, count_b, count_win = compute_simple_count(parsed)

        # Weighted: use existing weights if available in lookup
        wmap = weight_lookup.get(prompt_id, {})
        qnum_weights = {}
        for i, qid in enumerate(qids):
            qnum_weights[i + 1] = wmap.get(qid, 1.0 / len(qids))
        ws_a, ws_b, ws_win = compute_weighted_score(parsed, qnum_weights)

        if args.aggregation in ("count", "both"):
            predicted = count_win
            score_a, score_b = count_a, count_b
        else:
            predicted = ws_win
            score_a, score_b = ws_a, ws_b

        results.append({
            "sample_id": idx, "prompt_id": prompt_id,
            "domain": row.get("domain", ""),
            "winner": row["winner"], "predicted_winner": predicted,
            "parse_ok": parse_ok, "answers_raw": raw,
            "answers_parsed": str(parsed),
            "score_a": score_a, "score_b": score_b,
            "weighted_score_a": ws_a, "weighted_score_b": ws_b,
            "n_questions": len(comp_qs),
            "n_ties": sum(1 for v in parsed.values() if v == "Tie"),
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{limit}] ok={parse_ok} win={predicted}")

    return pd.DataFrame(results)


def evaluate_fullbank(args, judge):
    """Evaluate on all 58 v4 bank questions."""
    bank_df = load_fullbank_questions()
    cache = load_comparative_cache()

    questions = bank_df["question_text"].tolist()
    comp_qs = [get_comparative_text(q, cache) for q in questions]
    n_q = len(comp_qs)
    uniform = {i+1: 1.0/n_q for i in range(n_q)}

    split_path = BASE_DIR / "data" / "splits" / f"{args.split}.parquet"
    split_df = pd.read_parquet(split_path)

    results = []
    limit = args.limit or len(split_df)
    for idx, row in split_df.iloc[:limit].iterrows():
        prompt = build_comparative_prompt(
            row["context"], row["response_a"], row["response_b"], comp_qs
        )

        try:
            raw = generate_batch(judge, [[{"role": "user", "content": prompt}]],
                                 max_tokens=args.max_new_tokens)[0]
        except Exception as e:
            print(f"  [ERROR] judge call failed for {row.get('prompt_id', '?')}: {e}")
            continue

        parsed = parse_comparative_output(raw, n_q)
        parse_ok = comparative_parse_ok(parsed, n_q)

        count_a, count_b, count_win = compute_simple_count(parsed)
        ws_a, ws_b, ws_win = compute_weighted_score(parsed, uniform)

        if args.aggregation in ("count", "both"):
            predicted = count_win
            score_a, score_b = count_a, count_b
        else:
            predicted = ws_win
            score_a, score_b = ws_a, ws_b

        results.append({
            "sample_id": idx, "prompt_id": row["prompt_id"],
            "domain": row.get("domain", ""),
            "winner": row["winner"], "predicted_winner": predicted,
            "parse_ok": parse_ok, "answers_raw": raw,
            "answers_parsed": str(parsed),
            "score_a": score_a, "score_b": score_b,
            "weighted_score_a": ws_a, "weighted_score_b": ws_b,
            "n_questions": n_q,
            "n_ties": sum(1 for v in parsed.values() if v == "Tie"),
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{limit}] ok={parse_ok} win={predicted}")

    return pd.DataFrame(results)


# ── Main ──


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading judge: {args.model_id} (backend={args.backend})")
    judge = load_judge_model(args.model_id, backend=args.backend)

    print(f"Eval: source={args.question_source}, agg={args.aggregation}")
    if args.question_source == "generated":
        df = evaluate_generated(args, judge)
    elif args.question_source == "hr-oracle":
        df = evaluate_hr_oracle(args, judge)
    elif args.question_source == "fullbank":
        df = evaluate_fullbank(args, judge)
    else:
        print(f"Error: unknown source: {args.question_source}")
        sys.exit(1)

    if df.empty:
        print("Error: no results produced")
        sys.exit(1)

    pred_path = args.output / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    print(f"Saved {len(df)} predictions to {pred_path}")

    metrics = compute_metrics(df)
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
