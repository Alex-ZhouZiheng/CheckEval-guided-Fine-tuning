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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-06 11:58pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (10,384t read) | 416,725t work | 98% savings

### May 5, 2026
S138 Decision on whether to continue RL training from SFT checkpoint or start RL directly from base model (May 5, 5:28 PM)
S139 User requested to connect to a server to upload data using base; discussion covered adapting Unsloth Colab GRPO templates for text self-check judge GRPO, Colab free tier constraints, and data/training sync strategies (May 5, 5:48 PM)
S140 Run Swift training script with vLLM colocate settings (May 5, 7:07 PM)
S141 Implement in‑training self‑checklist evaluation callback for ms‑swift GRPO training (May 5, 7:46 PM)
S142 Patch transformers cache path for swift CLI via sitecustomize (May 5, 7:51 PM)
S143 Resolve KeyError 'input_ids' in Qwen2VLTemplate during GRPO training (May 5, 7:54 PM)
S144 Extract pure language model checkpoint from Qwen3.5-4B VL model (May 5, 8:00 PM)
S145 Adjust training configuration to reduce clipped_ratio and improve reward signal for vLLM colocate training (May 5, 8:03 PM)
S146 Reconfigure training to accommodate longer prompts and completions while managing GPU memory (May 5, 8:14 PM)
S147 Checkpoint retention policy for model training (May 5, 9:16 PM)
### May 6, 2026
772 1:36p 🔵 Syntax error in remote Python inspection command
771 " 🔵 Self‑check training data schema identified
773 1:37p 🔵 Remote dataset inspected: shape, schema, and stats
774 " 🔵 Server-side code references to parse failures and winner mismatches identified
775 1:38p 🔵 Self-check parser validates all rows successfully
777 1:39p 🟣 Cleaned SFT dataset written without drops
776 " 🔵 SFT Eval Comparison Search
778 1:40p 🔵 LaTeX Results Table Read
779 1:41p ✅ Update SFT Checkpoint Results in LaTeX Table
780 " ✅ Sync HTML Preview with Updated LaTeX Table
781 1:42p 🔵 Playwright Preview Artifacts Listing
782 3:38p 🟣 Trained self-checklist model on cleaned data checkpoint 265
783 3:41p ⚖️ Base checkpoint selected for GRPO training
784 3:49p ⚖️ Defined contrastive_intent and generic_question_penalty reward calculations
791 4:34p 🔵 Extracted launcher environment lines from run_judge_grpo_swift.sh
792 4:42p ⚖️ Initiate unit tests for new quality reward
793 4:46p ✅ Added line‑number collection step to reward‑validation plan
794 5:59p 🔵 Quality reward audit failed to meet parse_ok_rate threshold
795 6:02p 🔵 Search for rollout generation scripts
796 6:03p 🔵 User requested review of 2026-05-06-checklist-quality-proxy-reward.md plan
797 6:11p 🔵 Self-Checklist model evaluation completed with performance metrics
798 6:24p 🔵 DDP unused‑parameter flag warning and max‑tokens auto‑adjustment observed
799 6:28p ✅ Adjusted max_tokens for Swift trainer based on model length constraints
800 7:28p 🔵 Training process stalls after loading diversity encoder
801 9:04p 🔵 File search command failed due to Windows sandbox permission error
802 9:06p 🔵 PowerShell Get-Content command blocked by sandbox permissions
803 9:07p 🔵 Academic paper skill metadata read successfully after escalated permission
804 " 🔵 Read first 180 lines of thesis.tex from template zip with escalated permissions
806 9:08p 🔵 PDF text extraction command failed due to PowerShell here-doc syntax error
805 " 🟣 Retry PDF extraction with UTF-8 encoding
807 " 🔵 Retrieved final 80 lines of thesis.tex template from zip archive
808 9:10p 🟣 Created related work chapter and bibliography files
809 " 🔵 All citation keys in related_work.tex have corresponding BibTeX entries
810 11:03p 🔵 Windows Sandbox Process Creation Failure
811 " 🔵 File Listing Timeout
812 11:04p 🔵 SKILL.md Read Timeout
813 " 🔵 Academic-Paper Skill Definition Loaded
814 11:05p 🔵 Git Workspace State Revealed
815 11:06p 🔵 PowerShell JSON parsing error during metrics extraction
816 11:07p 🔵 Successful metrics extraction via PowerShell
817 11:08p 🔵 Read research narrative and evidence markdown files
818 11:39p 🔵 Timeout reading GRPO skill markdown file
819 11:40p 🔵 File search operation timed out during diagnosis
820 11:43p 🔵 Remote checkpoint directory inaccessible via SSH
821 11:52p 🔵 Reward selfcheck test suite passed
822 11:54p 🔵 Git status command failed due to permission error
823 " 🔵 Repository status shows pending changes and permission warning
824 11:55p 🔄 Adjusted thinking flag handling in run_judge_grpo_swift.sh
825 " 🔵 Git diff confirms training script thinking flag modifications
826 11:56p ✅ Sync GRPO eval thinking with training setting

Access 417k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>