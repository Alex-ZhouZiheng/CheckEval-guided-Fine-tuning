# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Repository Guidelines

## Project Structure & Module Organization

This repository contains CheckEval-guided preference evaluation and fine-tuning code. Core Python modules live in `src/`, with data preparation under `src/data_process/`, evaluation scripts under `src/evaluation/`, analysis utilities under `src/analysis/`, and training code under `src/train/`. Checklist banks are versioned in `checklists/` (`v4_frozen` is the current default). Data splits and generated training artifacts live in `data/`; evaluation outputs, metrics, and checkpoints belong in `results/`. Tests and smoke checks live in `tests/`.

## Build, Test, and Development Commands

Create or refresh the environment from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Run focused tests:

```bash
python -m pytest tests/test_review_app_helpers.py
```

Check syntax after script edits:

```bash
python -m py_compile src/config.py src/evaluation/run_checkeval_judge.py
```

Run a CheckEval baseline on the current bank:

```bash
python src/evaluation/run_checkeval_judge.py --eval-split dev_600 --checklists-dir checklists/v4_frozen
```

Build oracle labels from a frozen bank:

```bash
python src/data_process/build_oracle_labels.py --bank checklists/v4_frozen --split dev_600 --out data/oracle/dev_600_oracle_v4.parquet
```

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, type hints where useful, and small functions with explicit inputs. Prefer existing helpers in `src/utils.py` and `src/config.py` over duplicating parsing, prompt construction, or path logic. Keep file and artifact names descriptive, for example `*_v4.parquet`, `*_metrics.json`, and `*_sample.parquet`.

## Testing Guidelines

Add or update `pytest` tests for helper logic and parsing behavior. Name test files `test_*.py`. For pipeline scripts, at minimum run `py_compile` and a small `--max-samples` smoke run when dependencies and models are available.

## Commit & Pull Request Guidelines

The current history uses short messages such as `add frozen`; prefer clearer imperative commits, for example `update v4 checklist config` or `fix review helper bank path`. Pull requests should summarize the changed pipeline stage, list commands run, mention affected artifacts, and include key metric changes when evaluation behavior changes.

## Security & Configuration Tips

Do not commit `.env`, model weights, raw private data, or large generated outputs. Keep server-specific paths and tokens in environment variables or local command arguments.


<claude-mem-context>
# Memory Context

# [CheckEval-guided-Fine-tuning] recent context, 2026-04-30 8:14pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (19,869t read) | 1,165,877t work | 98% savings

### Apr 30, 2026
S2 BSc thesis research strategy: closing retrieval gap between learned selector and HR oracle for checklist-guided LLM judge (Apr 30, 11:37 AM)
S1 Caveman mode activation — user invoked caveman:caveman skill (Apr 30, 11:37 AM)
2 3:13p ⚖️ Bachelor Thesis Research Scope: Checklist-Guided LLM-as-a-Judge on HelpSteer3
S4 Initiate Lever 2.1 tiebreak second-pass on remote server using DeepSeek V3 Pro with thinking mode, concurrency 100 (Apr 30, 3:18 PM)
3 3:30p ⚖️ Research Priority Reoriented: Accuracy-Maximizing over Cost-Minimizing
4 " ⚖️ Tiebreak Second-Pass Experiment: DeepSeek V3 Pro + Thinking, Concurrency 100
S6 Monitoring DeepSeek V4 Pro tiebreak run — waiting on 850-call job to complete before computing accuracy (Apr 30, 3:30 PM)
S3 Accuracy-maximizing research strategy for BSc thesis: checklist-guided LLM judge with selector and tie handling (Apr 30, 3:30 PM)
5 3:47p 🔵 DeepSeek Thinking Mode Already Supported in run_dynamic_eval.py
6 " 🔵 No Parquet Raw Data in dynamic_test Results Dir — Only JSON Metrics
7 3:48p 🔵 Server State Confirmed: predictions.parquet Exists, DEEPSEEK_API_KEY Set, venvmerge Active
8 " 🔵 predictions.parquet Schema: 1073 Rows, predicted_winner Column for Tie Filtering
9 " 🔵 Tie Rate 39.7% (425/1073), Perfectly Balanced by True Winner — Confirms Genuine Ambiguity
10 3:49p 🔵 run_dynamic_eval.py Has --subset and --input-path Flags for Tie Subset Targeting
11 " 🔵 Complete CLI Interface for DeepSeek Tiebreak Run Identified
12 " 🔵 load_eval_pairs Requires context/response_a/response_b Columns — Tie Parquet Must Join with Split Data
13 3:50p 🔵 DeepSeek V4 Pro Already Used in Past Runs — Model Name and URL Confirmed
14 " 🟣 Tie Subset Extraction Script Written: scripts/extract_hr_oracle_tie_subset.py
15 3:51p 🟣 Tie Subset Extraction Succeeded: 425/425 Rows Matched — Confirms Predictions Cover Test Split
16 " 🔴 Picks Parquet Column Name Fixed: "qids" → "selected_qids"
17 3:52p 🔵 Original HR Oracle Weighted Run Used Local Judge — Config Parameters Confirmed for Tiebreak Replication
18 " 🚨 DEEPSEEK_API_KEY Exposed in Plaintext via SSH Command Output
19 3:53p 🟣 DeepSeek V4 Pro Tiebreak Run Launched: 425 Tie Samples, Thinking On, Concurrency 100
20 " 🔵 Tiebreak Run Active: 850 Total Judge Calls (425 samples × 2 sides), ~4h Estimated Runtime
21 " 🔵 Tiebreak Run Throughput: 2.7 it/s at Concurrency 100 — Total Runtime ~9 min, Not 4h
S7 Run rigor-reviewer Level 2 ARA Seal review on CheckEval-Guided Preference Evaluation research artifact (Apr 30, 3:53 PM)
S5 Lever 2.1 tiebreak second-pass: extract tie subset, launch DeepSeek V4 Pro + thinking run on server, monitor progress (Apr 30, 3:53 PM)
22 4:31p 🔵 Level 2.1 Tiebreak Second-Pass Analysis on 425 Tie Samples
28 " 🔵 Root Cause of 39.61% Tie Rate in Weighted HR-Oracle Run Identified
32 " 🔵 Tie Root Cause Fully Diagnosed: 300/425 Ties Have Exactly Zero Weighted Margin
35 " 🔵 Lever 2.1 DeepSeek Tiebreak Full Metrics Confirmed on Remote Server
36 " 🔵 Root Cause of 39.61% Tie Rate: 70.6% Are Exact-Zero Weighted Margin
37 " 🔵 Full Per-Question Contribution Audit Blocked: predictions.parquet Lacks labels_a/labels_b
38 " ✅ Lever 2.1 Tiebreak Analysis Section Added to Dynamic Eval Summary
23 4:32p 🔵 Lever 2.1 Tiebreak: Residual Abstention Is Bottleneck, Not A/B Discrimination
24 " ✅ Lever 2.1 Tiebreak Analysis Section Added to Dynamic Eval Summary
25 5:10p ✅ Lever 2.1 Tiebreak Results Documented in Evidence Table
26 " ⚖️ Next Experiment Priority: Forced Binary Tiebreak + Abstention Calibration
27 " 🔵 Tie Generation Mechanism in Weighted Pairwise Scoring
41 5:30p 🔵 Full per-question contribution audit reveals 69% of ties are all-zero-contribution
42 " 🔵 Tie-causing checklist questions are high-frequency same-direction agreement questions
43 " 🟣 Dynamic eval rerun with --save-raw-outputs flag for contribution audit
44 " 🔵 Scoring rule sweep: top-1 decisive question is best tie-resolver but still only 57.9% effective accuracy
29 " 🔵 70% of Tie Predictions Have Exact-Zero Margin, Not Threshold-Induced
30 " 🔵 Sparse HelpSteer3 Rationale Structurally Incompatible With Dense Checklist Averaging
31 " 🔵 predictions.parquet Does Not Save labels_a/labels_b By Default
45 5:34p 🔵 question_contributions.parquet uses 'weighted_contribution' not 'contribution' column
33 " 🔵 Margin-Level Tie Audit Completed on Remote Server
34 " ⚖️ Rerun HR-Oracle Weighted Eval with --save-raw-outputs for Full Contribution Audit
39 5:48p 🟣 HR-Oracle Weighted Dynamic Eval Rerun Completed with Raw Outputs Saved
40 " 🟣 Full Per-Question Contribution Audit Script Built for Tie Decomposition
46 6:56p 🟣 CheckEval-Guided Preference Evaluation ARA
47 " ✅ Level 2 ARA rigor-review report produced
S8 Research gap analysis and next-direction prioritization based on Level 2 review findings for CheckEval dynamic-evaluation ARA (Apr 30, 7:01 PM)
48 7:17p 🔵 Discriminativeness Reranker Experiment Completed: Positive Direction, Coarse Mechanism
49 " 🔵 Gentler Reranker Confirms Tradeoff: Less Harm But Lower Net Gain
50 " 🔵 294 All-Zero Tie Samples: Root Cause Diagnosis Initiated
51 7:23p 🔵 All-Zero 294 Tie Attribution Analysis Performed on Remote Server

Access 1166k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>