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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-04 4:21pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (12,642t read) | 346,731t work | 96% savings

### May 3, 2026
506 9:48p ✅ Removed debug raw-dump hook from prepare_self_checklist_sft.py
501 9:49p 🔵 User requests replacing prompt-forced Qwen3 think output with native think mode
503 " 🔴 Parser now uses post‑thinking content for all sections
508 10:25p 🔵 User requests enabling Qwen3 native think mode in prepare_self_checklist_sft.py
510 10:35p 🔵 Measured actual prompt token length distribution for Qwen3 think mode
509 10:36p 🔵 Investigating Qwen3 native think mode on HuggingFace
511 10:40p 🔵 Investigate native think mode support for Qwen3.6 27B on HuggingFace
### May 4, 2026
512 11:01a 🔵 nohup process ignoring input during shutdown monitoring
513 11:03a 🔵 Chat template and target output alignment verified for Qwen3.5-9B SFT
518 11:15a ✅ [Optimizing inference speed with vLLM backend and speculative decoding]
S92 Clarify vLLM max_model_len and output budget interaction for dev_600 evaluation (May 4, 11:18 AM)
519 11:19a 🔵 Measured dev_600 prompt token lengths for vLLM max_model_len sizing
S93 Parameter evaluation and optimization for LoRA SFT training of Qwen3.5-9B model (May 4, 11:21 AM)
520 11:43a 🟣 Enhanced parsing accuracy across domains
S94 Investigated Qwen3-27B native think mode support and implemented training configuration changes to enable thinking mode in SFT pipeline (May 4, 11:45 AM)
521 11:45a ⚖️ [**title**: Configured PEFT fine-tuning run]
522 " 🔵 [**title**: Retrieved PEFT fine-tuning documentation]
523 " 🔵 [**title**: Missing training data file]
524 11:46a 🔵 [**title**: No training data files found]
525 11:47a 🔵 [**title**: File search timed out]
526 11:52a 🔴 CUDA memory allocation optimized to prevent OOM errors
527 " 🔵 Code Structure and Configuration for Base Model Loading
528 11:53a ✅ Tokenizer truncation side changed to left to preserve structured answer
529 12:02p 🔵 Qwen3-27B Think Mode Investigation Requested
530 " 🔵 CUDA memory allocation failure during SFT training
S95 Investigated Qwen3-27B native think mode implementation (May 4, 12:02 PM)
S96 Memory optimization strategies for Qwen3-27B training with thinking mode enabled (May 4, 12:05 PM)
S97 Diagnose and resolve OOM during judge SFT training with Qwen3.5-9B using Unsloth and QLoRA (May 4, 12:05 PM)
532 12:36p 🔵 CUDA OOM Encountered During Qwen3.5-9B SFT Training With Unsloth
533 12:37p 🔵 Swift SFT Training Script for Qwen3.5-9B with OOM Mitigations
534 12:39p 🔴 CUDA OOM Error When Using Unsloth for QLoRA Training
535 12:40p 🔵 Search for Advanced Attention and CUDA Features Timed Out
536 12:42p 🔵 SSH Connection to AutoDL Instance Refused
537 12:43p ✅ Configured Qwen3.5-9B judge SFT training run with QLoRA parameters
S98 Resolve CUDA OOM with unsloth and ensure fast‑path attention libraries are installed (May 4, 12:43 PM)
538 " ⚖️ Inquired about PEFT fine-tuning parameter suitability for first-time judge SFT run
539 12:44p 🔵 SSH command to inspect remote Python packages failed due to PowerShell parsing error
540 " 🔵 Remote server environment lacks CUDA toolkit and proper flash-attn installation
541 12:58p 🔵 User requested to learn Unsloth fine-tuning for Qwen3.5 via official docs
542 12:59p 🔵 Extracted full Unsloth Qwen3.5 fine‑tuning code and configuration
543 1:01p 🔵 CUDA OOM Encountered When Training Qwen3.5-9B With Unsloth and QLoRA
544 1:10p 🔵 OOM Debugging - Unsloth Memory Issue Investigation
S99 Install causal_conv1d and flash-linear-attention to resolve OOM and enable fast-path attention (May 4, 1:12 PM)
S100 Download and install causal_conv1d wheel from GitHub releases to enable fast-path attention (May 4, 1:15 PM)
545 1:27p 🔵 Unsloth OOM Error with Qwen3.5-9B Training
546 3:19p 🔵 User inquired about suitability of PEFT fine-tuning parameters for first-time run
548 " 🔵 Remote server environment inspection revealed RTX 5090 GPU with CUDA 13
547 " 🔵 Missing pip module in .unsloth Python environment on remote server
549 3:22p 🔵 Remote flash_attn environment check failed due to malformed Python command
550 3:26p 🔵 User inquired about PEFT fine-tuning parameter suitability for first-time training
551 " 🟣 Installed flash‑attn wheel for Python 3.12 .venvmerge
552 3:28p ✅ Installed ABI-compatible flash-attn 2.8.3 on remote SeetaCloud training server
S101 User asked if provided PEFT fine-tuning command parameters are suitable for a first-time fine-tuning run of Qwen3.5-9B using judge SFT training data; subsequent work prepared the remote training server by installing a compatible flash-attn version (May 4, 3:30 PM)
553 3:32p 🔵 Unsloth 16bit LoRA configuration and performance impact discovered
554 3:42p 🔵 User inquired about suitability of PEFT fine-tuning parameters for first-time judge SFT run
555 3:44p 🔵 Remote GPU status and Python package versions inspected
556 3:57p 🔵 User inquired about OOM avoidance methods for dual-GPU setups
557 4:00p 🔵 [**PEFT Fine-Tuning Parameter Consultation for Qwen3.5-9B**]
558 4:11p 🔵 CUDA out‑of‑memory despite 64 GiB total GPU memory
559 4:16p 🟣 Add --deepspeed-zero3-no-offload flag for ZeRO-3 without CPU offload

Access 347k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>