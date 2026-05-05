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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-05 6:58pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (11,435t read) | 511,829t work | 98% savings

### May 5, 2026
S129 GPU memory optimization for autotuning process (May 5, 11:36 AM)
S130 Autotuner OOM fallback and CUDA graph capture completion (May 5, 11:40 AM)
S131 Performance optimization progress summary after autotuning and OOM fallback (May 5, 11:46 AM)
S132 Distillation Experiment Analysis and Next Steps (May 5, 11:48 AM)
S133 RL微调判官方法分析与GRPO实施计划 (May 5, 1:08 PM)
S134 Create GRPO fine-tuning components for the self-checklist judge, including a dataset preparation script, custom reward plugin, and training shell script, following existing project patterns for generator GRPO training. (May 5, 1:10 PM)
S135 Explain need for 0.15 format reward after SFT and clarify whether GRPO data should match SFT data (May 5, 1:18 PM)
S136 Investigate and fix mergekit breakage caused by removal of TRANSFORMERS_CACHE in new Transformers version (May 5, 4:04 PM)
S137 SSH connection timeout to remote host (May 5, 5:21 PM)
617 5:32p 🔵 Missing torch module detected in .vengrpo environment
618 5:35p 🔴 Remote server disk space exhausted during pip install
619 5:40p ✅ GRPO training environment packages installed on remote SeetaCloud server
620 5:41p ✅ Install flash-attn into .vengrpo environment
621 " 🔴 Missing immutables dependency for TRL GRPO trainer
622 5:42p ✅ Installed missing immutables package for mergekit compatibility
623 " 🔴 Pydantic schema generation error in TRL GRPO trainer import
624 5:43p 🔴 Persisting pydantic schema error after downgrading to pydantic 2.11.7
625 5:44p 🔵 Located Task class definition in mergekit source
626 5:45p 🔴 sed command failed while attempting to patch mergekit Task class
S138 Decision on whether to continue RL training from SFT checkpoint or start RL directly from base model (May 5, 5:48 PM)
627 5:49p 🔵 Verified remote environment Python and vLLM version
628 " ✅ Uploaded base model format hit-rate probe script to remote server
630 5:51p ✅ Re-uploaded base model format hit-rate probe script with main guard
631 5:56p 🔵 Investigate availability of “think” mode for Qwen 3.6 27B on HuggingFace
632 6:01p 🔵 Qwen3.5-4B format compliance probed with increased generation limits
633 6:18p 🔵 Syntax check passed for run_judge_grpo_swift.sh
634 " 🔵 Training session initiated
635 6:21p ✅ Update GRPO training defaults and AGENTS documentation
636 " ✅ Remote GRPO training defaults updated via git pull
637 6:22p 🔵 Remote training server GPU and tmux status checked
638 " ✅ Started GRPO training run in new tmux session
639 6:24p 🔵 Monitored GRPO training session logs and GPU usage via remote SSH
640 6:25p 🔵 User Environment Context Update
641 6:26p 🟣 Upgrade Transformers to 5.7.0 with Cache Shim
642 " 🔵 TRL Import Failure Due to Missing vllm_ascend
643 6:27p 🔵 TRL vLLM Ascend Availability Check
644 6:28p 🔴 Patch TRL import_utils for Transformers 5 Compatibility
645 6:29p 🟣 GRPO Training Session Restarted
646 6:35p 🔵 Permission error when reading SKILL.md
648 6:36p 🔵 Memory file search blocked by sandbox
647 " 🔵 Read GRPO RL training skill documentation
649 6:40p 🟣 Added Unsloth‑based GRPO training entrypoint
650 " 🟣 Added Unsloth-based GRPO training pipeline for self-checklist judge
651 6:42p 🟣 Implemented lightweight self-checklist parser in GRPO training script
652 6:43p 🔵 Syntax validation passed for GRPO training entrypoint
653 " 🔵 Key function and class definitions located in GRPO training script
654 6:45p 🔵 Repository configuration paths and tier sizes identified
655 6:47p 🔵 Self‑checklist evaluation script loaded
657 6:48p 🔵 Utility functions for evaluation loaded
656 " 🔵 Checking prior memory for GRPO eval conventions
658 " 🟣 Added self-checklist evaluation prompts to GRPO training script
659 6:50p 🟣 Implemented self‑checklist evaluation workflow in GRPO script
660 6:51p 🔵 Confirmed modified GRPO training script passes Python syntax check
661 6:55p 🔴 Resolved TRANSFORMERS_CACHE ImportError
662 " 🔵 rg command unavailable due to permission error
664 " 🔵 PowerShell file search blocked by sandbox
663 " 🔵 Missing TRANSFORMERS_CACHE symbol causes import failure in llm_blender/TRL stack
665 6:56p 🔵 Aborted attempt to read GRPO training script
666 6:57p 🔴 Added TRANSFORMERS_CACHE compatibility patch to GRPO trainer
667 6:58p 🔵 PowerShell Select-String blocked by sandbox

Access 512k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>