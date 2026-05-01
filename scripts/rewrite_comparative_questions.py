"""Rewrite pointwise checklist questions into comparative format via ChatGPT-5.5.

Usage:
    python scripts/rewrite_comparative_questions.py

Reads all unique pointwise questions from:
    1. Generated checklists: data/generated_checklists/*.parquet
    2. Bank questions: checklists/v4_frozen/bank_index.parquet

Output: data/comparative_questions.json  {original_text: comparative_text}
Skips already-cached questions on re-run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from config import COMPARATIVE_QUESTIONS_CACHE, PROJECT_ROOT as BASE_DIR

REWRITE_PROMPT = """Rewrite this checklist question into a comparative format that asks "Which response better satisfies..." instead of "Does the response..."

Original: Does the response include concrete scene actions beyond dialogue?
Comparative: Which response includes more concrete scene actions beyond dialogue?

Original: Does the response follow the prompt constraints?
Comparative: Which response follows the prompt constraints more closely?

Original: Does the response avoid stereotyping or offensive content?
Comparative: Which response better avoids stereotyping or offensive content?

Original: Is the code accurate and free of bugs?
Comparative: Which response provides more accurate and bug-free code?

Original: {question}
Comparative:"""


def load_unique_questions():
    """Collect all unique pointwise questions from all sources."""
    questions = set()

    # 1. Generated checklists (per-sample)
    gen_dir = BASE_DIR / "data" / "generated_checklists"
    if gen_dir.exists():
        for f in sorted(gen_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                if "questions" in df.columns:
                    for q_list in df["questions"]:
                        if isinstance(q_list, list):
                            for q in q_list:
                                if isinstance(q, str) and q.strip():
                                    questions.add(q.strip())
            except Exception as e:
                print(f"  [warn] skipping {f.name}: {e}")

    # 2. Bank questions (v4_frozen)
    bank_path = BASE_DIR / "checklists" / "v4_frozen" / "bank_index.parquet"
    if bank_path.exists():
        try:
            df = pd.read_parquet(bank_path)
            if "question_text" in df.columns:
                for q in df["question_text"]:
                    if isinstance(q, str) and q.strip():
                        questions.add(q.strip())
        except Exception as e:
            print(f"  [warn] skipping bank: {e}")

    return sorted(questions)


def load_cache():
    """Load existing cache, return empty dict if missing or corrupt."""
    if COMPARATIVE_QUESTIONS_CACHE.exists():
        try:
            with open(COMPARATIVE_QUESTIONS_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("  [warn] cache corrupt, starting fresh")
            return {}
    return {}


def save_cache(cache):
    COMPARATIVE_QUESTIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARATIVE_QUESTIONS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def rewrite_question(client, question, model="gpt-4.1-nano"):
    """Call ChatGPT-5.5 to rewrite a single question."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": REWRITE_PROMPT.format(question=question)},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    rewritten = response.choices[0].message.content.strip()
    # Strip leading "Comparative:" if model echoes it
    if rewritten.lower().startswith("comparative:"):
        rewritten = rewritten[len("Comparative:"):].strip()
    return rewritten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-nano",
                        help="OpenAI model for rewriting")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (default: env OPENAI_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Provide --api-key or set env var.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    questions = load_unique_questions()
    print(f"Found {len(questions)} unique questions to process.")

    cache = load_cache()
    to_rewrite = [q for q in questions if q not in cache]
    print(f"{len(cache)} cached, {len(to_rewrite)} to rewrite.")

    for i, q in enumerate(to_rewrite):
        try:
            rewritten = rewrite_question(client, q, model=args.model)
            cache[q] = rewritten
            print(f"  [{i+1}/{len(to_rewrite)}] OK: {q[:40]}... -> {rewritten[:50]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(to_rewrite)}] ERROR: {q[:40]}... -> {e}")

    save_cache(cache)
    print(f"Saved {len(cache)} rewrites to {COMPARATIVE_QUESTIONS_CACHE}")


if __name__ == "__main__":
    main()
