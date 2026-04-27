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
