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

# [CheckEval-guided-Fine-tuning] recent context, 2026-05-06 5:01pm GMT+2

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (10,012t read) | 335,079t work | 97% savings

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
736 10:25p 🔵 Playwright skill documentation read
737 10:26p 🟣 LaTeX tables converted to HTML preview
738 10:27p 🔵 Playwright browsers not installed
740 10:30p 🔵 Playwright Chromium install timed out
739 " 🔵 Playwright Chromium install times out, screenshot retry planned for local HTML preview
741 10:31p 🔵 Microsoft Edge not found
742 10:41p ✅ Git status after visualization generation
743 10:46p 🔵 Evaluation not triggered after checkpoint 50
744 10:50p 🔵 Self‑check evaluation callback code inspected
745 10:51p 🔵 Checkpoint‑50 location identified for manual evaluation
746 11:21p 🔵 GRPO training performance vs baseline
747 " 🔵 Search for self-checklist baseline evaluation artifacts
748 11:24p 🔵 SFT parse rate approaching 100%
749 11:31p 🔵 Inspected SFT training script and data‑prep files for hyperparameter definitions
750 11:41p 🔵 User inquired about tuning SFT v2 parameters based on training curves
### May 6, 2026
751 10:12a 🟣 User requested MTP support for run_zeroshot.py and accuracy calculation explanation
752 10:13a ✅ run_zeroshot.py has uncommitted modifications
753 10:14a 🟣 Added MTP speculative decoding flags to run_zeroshot.py
754 10:17a 🔵 enable_thinking flag usage across project
755 " 🔴 chat_template_kwargs integration in utils.py
757 10:18a 🟣 Added --enable-thinking and --max-new-tokens arguments
758 " 🟣 Integrated thinking mode arguments into run_vanilla_judge call
759 10:42a ✅ User ran two experiments and added results to table
760 " 🔵 Git status shows modified .gitignore and AGENTS.md on master branch
761 " 🟣 LaTeX table visualization pipeline implemented
762 " 🟣 Added new benchmark rows to paper tables
763 10:43a 🟣 Generated HTML preview for paper tables
764 10:51a 🟣 Self-checklist implementation confirmed
766 10:55a 🟣 reward1 Calculation Method Implemented
767 11:29a 🔵 Inspecting training code for thinking-mode handling
768 1:24p 🔵 Fine-tuned Qwen3.5-4B checkpoint ready for evaluation
769 1:35p 🔵 Permission error executing ripgrep on Windows sandbox
770 1:36p 🔵 Found training dataset paths on remote server
772 " 🔵 Syntax error in remote Python inspection command
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

Access 335k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>