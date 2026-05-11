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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-10 7:50pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (9,881t read) | 514,703t work | 98% savings

### May 10, 2026
S262 Configure and run judge SFT on debug_5k_selfcheck using Qwen3.5-4B without unsloth, with custom environment variables and LoRA settings (May 10, 4:17 PM)
1423 4:50p 🟣 run_judge_sft supports multiple training configurations
1424 " 🔵 Available train split parquet files listed
1425 " 🔵 Key configuration paths and parameters identified in config.py
S263 Prepare final‑only SFT data and run Swift SFT training for zero‑shot judge (May 10, 4:51 PM)
1426 4:53p 🔵 Training shell scripts cataloged in src/train
1427 4:54p 🔵 PAIRWISE_TABLE defined in utils.py
1428 " 🔵 Zero-shot vanilla judge implementation in run_zeroshot.py
1429 " 🔵 Vanilla prompt builder in utils.py
1430 " 🔵 VANILLA_JUDGE_PROMPT definition and build_vanilla_prompt implementation
1431 4:55p 🔴 Added prepare_final_only_sft.py data processing script
S264 Launch and monitor judge model fine‑tuning with Qwen3.5‑4B and LoRA (May 10, 4:55 PM)
1432 4:56p 🔴 Repository state after adding prepare_final_only_sft.py
1446 5:00p 🔵 Missing scikit-learn dependency in Conda base environment
1447 5:01p 🔵 Existing unified test run script identified
1448 " 🟣 Data preparation script executed successfully with .venvmerge
1449 " 🔵 GPU resources and swift binary available on remote host
1445 " 🔵 Only base Conda environment present on remote host
1450 5:02p 🟣 Background fine‑tuning job for judge model launched
1451 " 🟣 Judge SFT training process started via swift
S265 Confirm background training continuation and log monitoring (May 10, 5:02 PM)
S266 Check GPU usage and training status for Swift SFT run (May 10, 5:03 PM)
1452 5:26p 🔵 Wandb communication timeout during training launch
1453 " ✅ Training relaunched with Wandb disabled and environment overrides
1454 " 🔵 Training command executed with extensive distributed launch options
1455 5:27p 🔵 Training process hierarchy established
1456 " 🟣 Training successfully running with GPU utilization
S267 Confirm training process is running after nohup detach (May 10, 5:27 PM)
S268 Adjust max_len and train QLoRA 4-bit model successfully (May 10, 5:27 PM)
1457 5:28p 🔴 Out‑of‑memory error caused training crash
1460 5:31p ✅ Restarted training with increased max sequence length
1462 " 🔴 Training aborted during initialization with MAX_LEN=9216
1458 5:32p 🔵 Token length distribution of training data
1463 " 🔴 Bitsandbytes CUDA dependency missing
1464 " 🔵 CUDA version mismatch identified
1465 " 🔵 CUDA 13 runtime libraries present in venv
1466 5:33p ✅ Restarted training with CUDA library path configured
1461 " 🔴 Training failed again with MAX_LEN=9216
1467 5:34p 🟣 Training started successfully with CUDA 13 libraries
1468 " 🟣 First training step completed
S269 检查 GPU 使用率异常并评估训练进度 (May 10, 5:34 PM)
1469 5:36p 🔵 Required max_seq_len identified for context handling
S270 Final-only SFT training for 4B model with 5k dataset - training binary classifier to select winner (A or B) from response pairs (May 10, 5:44 PM)
1470 7:22p ⚖️ Implement final-only SFT for 4‑bit model using 5k examples
1471 7:23p 🔵 Checkpoint directory and final log captured
1472 " 🔵 Final checkpoint directory contents listed
1473 7:24p 🔵 Contents of final checkpoint-594 adapter directory listed
1474 " 🔵 Zero-shot evaluation script lacks adapter handling
1475 " 🔵 Finetuned evaluation script includes adapter path argument
1476 " 🔵 Finetuned evaluation script loads LoRA adapter via vLLM
1477 7:25p 🔵 Finetuned evaluation launched in background
1478 7:41p 🔵 vLLM Model Loading and CUDA Graph Capture
1479 " 🔵 Evaluation Completed Successfully
1480 " 🔵 Finetuned Model Evaluation Metrics
S271 Display final evaluation results for finetuned 4B model (May 10, 7:42 PM)
1481 7:44p 🔵 Robustness Evaluation Script Arguments
1482 " 🔵 Robustness Evaluation Script Core Logic
1483 7:45p 🔵 Started Batch Evaluation and Robustness Computation
1485 7:49p 🔵 Robustness Evaluation Completed for finalonly 4B Model
1484 " ✅ Final Only SFT Setup for 4B Model

Access 515k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>