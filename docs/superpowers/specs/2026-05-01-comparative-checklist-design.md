# Comparative Checklist Evaluation — Design Spec

**Date:** 2026-05-01
**Status:** approved
**Goal:** Test whether reformatting pointwise checklist questions (yes/no per side) into comparative format (A/B/Tie for which response is better on each criterion) improves pairwise evaluation accuracy by reducing yes/yes collapse.

## 1. Motivation

Current pointwise format evaluates Response A and Response B independently: judge sees one response at a time, answers yes/no/N/A per criterion. Our error analysis found 82.3% of selected questions are generic (Task Adherence, Coverage Adequacy, etc.) and produce yes/yes for both responses — zero discrimination. 54.3% of error samples have literally zero nonzero-contributing questions.

Hypothesis: showing both responses simultaneously and asking "which is better on criterion X" forces the judge to notice differences it would otherwise overlook when evaluating each response in isolation.

## 2. Prompt Design

### 2.1 New comparative prompt template

```
<Task Overview>
You will be given a conversation between a user and an assistant, followed by
two candidate responses (Response A and Response B). For each quality criterion
below, decide which response better satisfies that criterion. Answer 'A', 'B',
or 'Tie' for each question.

<Instructions>
1. Read the conversation history and both responses carefully.
2. For each question, compare the two responses on that specific criterion.
3. Answer 'A' if Response A is better on this criterion.
4. Answer 'B' if Response B is better on this criterion.
5. Answer 'Tie' if both are equally good (or equally bad) on this criterion.
6. Do not provide explanations, rationale, notes, or extra text.
7. Output only the answer lines.

<Answer Format>
Q1: A
Q2: Tie
Q3: B
...

# Conversation History #
{context}

# Response A #
{response_a}

# Response B #
{response_b}

# Questions #
{checklist_text}

# Your Answer #
```

Defined as `CHECKEVAL_COMPARATIVE_PROMPT` in `src/utils.py`.

### 2.2 Question rewriting (ChatGPT-5.5)

Pointwise questions are rewritten via ChatGPT-5.5 with this prompt:

```
Rewrite this checklist question into a comparative format that asks
"Which response better satisfies..." instead of "Does the response..."

Original: Does the response include concrete scene actions beyond dialogue?
Comparative: Which response includes more concrete scene actions beyond dialogue?

Original: Does the response follow the prompt constraints?
Comparative: Which response follows the prompt constraints more closely?

Original: {question}
Comparative:
```

Edge case rules:
- "Does the response avoid X?" → "Which response better avoids X?"
- "Is the code accurate?" → "Which response provides more accurate code?"
- Conditional questions ("If the response contains code...") → "If applicable, which response..."
- NA concept is dropped in comparative format (always a comparison, no applicability gating)

Output cached to `data/comparative_questions.json`.

## 3. Experiment Matrix

All on dev_600 (600 samples), Qwen3.5-9B judge (llama.cpp Q4_K_M GGUF).

| # | Name | Question source | Format | Aggregation | Calls/sample |
|---|------|----------------|--------|-------------|---|
| C1 | gen_pointwise_weighted | Generated checklist | Pointwise | Weighted | 2 |
| C2 | gen_comparative_count | Generated checklist | Comparative | Simple count | 1 |
| C3 | gen_comparative_weighted | Generated checklist | Comparative | Weighted | 1 |
| C4 | hr_pointwise_weighted | HR-oracle top-15 k=15 | Pointwise | Weighted | 2 |
| C5 | hr_comparative_count | HR-oracle top-15 k=15 | Comparative | Simple count | 1 |
| C6 | hr_comparative_weighted | HR-oracle top-15 k=15 | Comparative | Weighted | 1 |
| C7 | fullbank_pointwise | Static v4 (all 58) | Pointwise | Weighted | 2 |
| C8 | fullbank_comparative_count | Static v4 (all 58) | Comparative | Simple count | 1 |

C1, C4, C7 reuse existing results. C2-C3, C5-C6, C8 are new.

**Question sources:**
- Generated checklists (C1-C3): existing `results/pipeline_judge_base_dev_600_4Bgen9Bjudge` generated questions
- HR-oracle top-15 (C4-C6): qids from HR-oracle weighted predictions.parquet, with existing weights
- Full bank (C7-C8): all 58 v4 bank questions

## 4. Scoring & Aggregation

### 4.1 Simple count
```
score_A = count(answers == "A")
score_B = count(answers == "B")
margin = score_A - score_B
if margin > 0 → winner = A
if margin < 0 → winner = B
if margin == 0 → Tie
```

### 4.2 Weighted
```
score_A = sum(weight[q] for q where answer[q] == "A")
score_B = sum(weight[q] for q where answer[q] == "B")
```

Weights: uniform (1/N) for generated checklists. Existing `selected_question_weights` for HR-oracle. Uniform for full bank.

### 4.3 Tie resolution
No `tie_delta` threshold in comparative mode by default. Comparative format should naturally reduce ties by forcing a decision per criterion. If overall tie rate remains high, test tie_delta as secondary diagnostic.

## 5. Parse Logic

Parse `Q\d+:\s*(A|B|Tie)` case-insensitive. Accept "Tie"/"TIE"/"tie". `parse_ok` = True if ≥ 80% of expected questions return valid label (A, B, or Tie). Unlike pointwise, no NA parsing needed.

Expected output per sample: list of `{"qnum": N, "answer": "A"|"B"|"Tie"}`.

## 6. Implementation

**Script:** `scripts/run_comparative_eval.py`

CLI:
```
python scripts/run_comparative_eval.py \
  --question-source {generated|hr-oracle|fullbank} \
  --split dev_600 \
  --aggregation {count|weighted|both} \
  --output results/comparative/<run_name>
```

Flow per sample:
1. Load question list (generated questions, or bank qids from predictions)
2. Look up comparative rewrite from cache (or rewrite on first run)
3. Build `CHECKEVAL_COMPARATIVE_PROMPT` with context + response_a + response_b + rewritten questions
4. Call judge via `load_judge_model` (llama.cpp HTTP backend, port 8080)
5. Parse A/B/Tie answers
6. Compute simple count and/or weighted scores
7. Determine winner

**New in `src/utils.py`:**
- `CHECKEVAL_COMPARATIVE_PROMPT` template string
- `build_comparative_prompt(context, response_a, response_b, questions)` — fills template
- `parse_comparative_output(raw_text, n_expected)` → `dict[int, str]`

**Reuse:**
- `load_judge_model` / `make_lora_handle` for backend dispatch
- `load_checklists` for bank questions
- Existing parquet readers for generated questions and predictions

**Question rewriting cache:**
- On first run, call ChatGPT-5.5 API to rewrite all unique questions
- Save `data/comparative_questions.json`: `{original_text: comparative_text}`
- On subsequent runs, read from cache
- Handles both bank questions and generated per-sample questions

## 7. Output

```
results/comparative/
├── gen_comparative_count/
│   ├── metrics.json
│   └── predictions.parquet
├── gen_comparative_weighted/
│   ├── metrics.json
│   └── predictions.parquet
├── hr_comparative_count/
│   ├── metrics.json
│   └── predictions.parquet
├── hr_comparative_weighted/
│   ├── metrics.json
│   └── predictions.parquet
├── fullbank_comparative_count/
│   ├── metrics.json
│   └── predictions.parquet
├── comparative_questions.json
└── comparison_summary.md
```

**metrics.json fields:** accuracy, macro_f1, tie_rate, effective_accuracy, n_valid, n_total, parse_ok_rate, avg_answers_per_sample, avg_a_count, avg_b_count, avg_tie_count.

**predictions.parquet:** sample_id, prompt_id, domain, winner (gold), predicted_winner, parse_ok, answers_raw, answers_parsed, score_a, score_b, n_questions.

**comparison_summary.md:** side-by-side table comparing all 8 conditions on accuracy, tie rate, effective accuracy, parse rate; per-question-source delta (comparative vs pointwise); A/B/Tie answer distribution; interpretation.

## 8. Success Criteria

Comparative format is better if vs pointwise counterpart (same questions):
- Higher effective accuracy
- Lower tie rate
- Parse reliability ≥ 95%

Core hypothesis: simultaneous side-by-side comparison + forced A/B choice reduces the yes/yes collapse that dominates current pointwise errors.
