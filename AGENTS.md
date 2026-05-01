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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-01 2:26pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (18,766t read) | 490,114t work | 96% savings

### Apr 30, 2026
S13 Gated contrastive selector — full design spec written and presented for user approval before implementation (Apr 30, 10:12 PM)
S14 Gated/conditional contrastive selector implementation for CheckEval bi-encoder checklist selector — complete wiring of contrastive_gate_threshold and contrastive_gate_topk params through run_selector_train.py (Apr 30, 10:13 PM)
S15 Analyze ARA experiment results and identify next research steps for Checklist-Guided Fine-Tuning thesis (Apr 30, 10:37 PM)
### May 1, 2026
S16 Analyse current experiment results in ara/ folder and determine next steps for CheckEval-guided Preference Evaluation research (May 1, 11:09 AM)
S17 Analyse current experiment results in ara/ folder and determine next steps for CheckEval-guided Preference Evaluation research (May 1, 11:24 AM)
S20 Analyse current experiment results in ara/ folder and determine next steps for CheckEval-guided Preference Evaluation research (May 1, 11:25 AM)
S19 Analyse current experiment results in ara/ folder and determine next steps for CheckEval-guided Preference Evaluation research (May 1, 11:26 AM)
S18 Analyse current experiment results in ara/ folder and determine next steps for CheckEval-guided Preference Evaluation research (May 1, 11:26 AM)
S21 Benchmark DeepSeek V4 Pro as judge for HR top-15 evaluation on full test split (1073 samples), then distill into 9B model (May 1, 11:26 AM)
134 1:15p 🔵 DeepSeek V4 Pro HTTP Judge Runs: 100% Tie Rate, 0% Parse Rate
135 " 🔵 Review App Encoding Corruption from PowerShell Set-Content
136 " 🔵 Local Project Missing Three Key HR Oracle Input Files
146 1:27p 🔵 DeepSeek V4 Pro 5-sample smoke test produces no output after 604s timeout
138 " 🔵 DeepSeek V4 Pro Judge Successfully Parses with Thinking Mode Enabled
139 " ⚖️ Judge-Ceiling Experiment: DeepSeek V4 Pro on Full 1073-Sample Test Set
147 1:38p 🔵 DeepSeek V4 Pro smoke test succeeds with http-concurrency=100 in 131s
152 " 🔵 DeepSeek V4 Pro smoke test succeeds — 100% parse rate, 100% accuracy on 5 samples
144 " 🔵 DeepSeek V4 Pro smoke test times out after 604s with thinking-on
145 " 🔵 DeepSeek V4 Pro Judge Ceiling Experiment: Serial Throughput Too Slow
154 1:39p 🔵 Full DeepSeek V4 Pro HR-oracle experiment now feasible with concurrency=100
148 1:41p 🔵 No observable work completed yet
149 " 🔵 Target parquet file confirmed at 6.9MB
156 " 🔴 Added retry logic with exponential backoff to HTTP judge requests
158 1:42p 🔵 Retry logic recovers from transient HTTP failures but full experiment still crashes
150 " 🔵 Parquet schema inspected: 595 rows, 42 columns
151 " 🔵 Dataset composition: 71% ties, 20% wrong winners, 9% correct
153 " 🔵 97.5% of weight arrays contain NaN values; 21 weight parse failures
159 " 🔵 Primary session re-running exploration commands
160 " 🔵 Question bank index loaded: 58 questions with dimension/sub_aspect mapping
166 " 🟣 Comprehensive tie/error analysis script created and executed
167 " 🔴 Critical bug: classify_failure_modes reads wrong key names for computed stats
161 1:44p 🔵 build_oracle_labels.py has HTTP resume cache — run_dynamic_eval.py lacks it
155 " 🔵 Full DeepSeek V4 Pro experiment crashes after initial API burst — rate limit or OOM
162 1:45p 🔵 run_dynamic_eval.py uses fragile ex.map — build_oracle_labels.py has robust as_completed + resume cache
157 " ✅ No new primary session activity to record
163 " 🟣 Port HTTP resume cache from build_oracle_labels.py to run_dynamic_eval.py
176 1:48p 🟣 HTTP resume cache added to run_dynamic_eval.py
177 " 🔵 DeepSeek V4 Pro concurrency=100 hits 429 rate limit after ~53s
178 " 🔵 Cache writes survive process exit in detached process but not foreground
164 1:50p 🟣 HTTP Resume Cache Added to run_dynamic_eval.py
165 " ✅ _http_judge_generate Signature Updated with as_completed and Lock
171 1:54p 🔴 All 10 failure modes firing after _calc_ prefix bug fixes
179 1:57p 🔵 Background DeepSeek experiment at 591/2146 (27%) after 6.5 minutes
168 " 🔵 HTTP Resume Cache Not Writing Despite Successful DeepSeek API Responses
169 " 🔵 DeepSeek V4 Pro Thinking Latency: 3s to 56s Per Request at Concurrency=100
170 " ✅ DeepSeek Judge Run Relaunched as Hidden Background Process via Start-Process
180 1:58p 🔵 DeepSeek experiment at 1279/2146 (60%) after 15 minutes
172 2:00p 🔵 Zero annotator sign-disagreement across all 595 samples
173 " 🟣 All 9 failure mode case files and 9 analysis tables generated
174 " 🔵 FM1 case confirmed: all-yes-yes with 0.000 nonzero rate masks clear annotator preference
175 2:02p 🔵 Even task-specific questions show poor discrimination (lowest nonzero rates)
S22 Bug fix verification and output validation for HR-oracle tie/error analysis script (May 1, 2:02 PM)
181 2:04p 🔵 DeepSeek V4 Pro HR evaluation at 93% completion under sustained rate limiting
185 2:11p 🔵 DeepSeek V4 Pro HR top-15 evaluation completed — accuracy 0.805 vs Qwen3.5-9B baseline 0.816
187 2:18p 🔵 DeepSeek V4 Pro vs Qwen3.5-9B cross-comparison — DS rescues 229, regresses 134, net +95
188 2:23p ✅ HR oracle baseline matched test split exported for reuse
182 " 🔵 DeepSeek V4 Pro Judge Ceiling: Tie Rate Down, Valid Accuracy Slightly Down
183 " 🟣 HTTP Resume Cache Proven: 2146/2146 Prompts Survived 429 Rate Limits
184 " 🔵 DeepSeek V4 Pro Latency: ~1.3s Median at Concurrency=100, $0.75/1K Samples Estimated
186 2:24p ✅ Baseline predictions downloaded from remote server for DeepSeek cross-comparison

Access 490k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>