Compressing inline — text provided directly, no file path to run script against.

# CLAUDE.md

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

## Project Overview

CheckEval-guided fine-tuning: two-model pipeline. **Checklist generator** produces structured yes/no eval questions. **Checklist-conditioned judge** answers to pick better response in pairwise LLM comparisons. HelpSteer3; trains Qwen3.5 via DPO, SFT, joint training.

## Environment Setup

```bash
pip install -r requirements.txt
# Required env vars in .env:
# HF_TOKEN=<huggingface token>
# OPENAI_API_KEY=<openai key>
```

Training: DeepSpeed ZeRO-2. Inference: vLLM + LoRA. Known-good for Qwen3.5-27B-AWQ-4bit: `max_model_len=16384, max_num_seqs=16, gpu_memory_utilization=0.90`.

## Data Pipeline (run in order)

```bash
python src/data_process/download_data.py
python src/data_process/prepare_data.py
python src/data_process/prepare_data_reasoning.py --split dev
python src/data_process/prepare_dpo_data.py --splits train dev

# Choose a tier: debug_5k | tier_10k | tier_20k
python src/data_process/prepare_generator_sft.py --tier tier_10k
python src/data_process/prepare_judge_sft.py --tier tier_10k --n-samples 1500
```

Data under `data/`, subdirs: `raw/`, `splits/`, `dpo/`, `with_reason/`, `generator_sft/`, `judge_sft/`, `checklist_sft/`, `generated_checklists/`. Parquet throughout.

## Training

```bash
# DPO judge
python src/train/run_dpo_train.py --tier tier_10k --no-wandb

# SFT generator
python src/train/run_generator_sft.py --tier tier_10k --epochs 2

# SFT judge (checklist-conditioned)
python src/train/run_judge_sft.py --tier tier_10k --lr 1e-5

# Joint DPO + checklist SFT
python src/train/run_joint_train.py --tier tier_10k --sft-lambda 0.1

# Merge LoRA adapter into base model
python src/train/merge_adapter.py --adapter-path <path> --output-path <path>
```

Hyperparams (`src/config.py`): LR=1e-6, epochs=1, batch_size=1, grad_accum=32, max_len=2048, LoRA rank=16/alpha=32, DPO beta=0.1. Models: Qwen3.5-9B (judge), Qwen3.5-4B (generator).

## Evaluation

```bash
# Zero-shot baseline (no training, no checklist) — bar to beat: 0.786 pairwise accuracy
python src/evaluation/run_zeroshot.py --eval-split dev

# Full CheckEval pipeline
python src/evaluation/run_generator_infer.py --adapter-path <path> --split dev_600
python src/evaluation/run_judge_eval.py --judge-adapter <path> --generated <parquet>

# Or orchestrated:
python src/evaluation/run_pipeline_eval.py --generator-adapter <path> --judge-adapter <path>

# Fine-tuned adapter eval
python src/evaluation/run_eval_finetuned.py --adapter-path <path> --split dev
```

## Architecture

```
src/
  config.py              # Central config: paths, model IDs, hyperparams, domain definitions
  utils.py               # vLLM loading, prompt building, batch generation, metrics (accuracy, F1)
  data_process/          # Numbered pipeline steps: download → prepare → DPO/SFT splits
  train/                 # DPO, SFT, joint training, adapter merging
  evaluation/            # Zero-shot baseline, generator/judge inference, pipeline orchestration, ablations
  analysis/              # Audit and comparison scripts (not in main pipeline)
  plugin/
    checkeval_reward.py  # CheckEval reward function for RL training
```

Scripts read paths + model IDs from `src/config.py`. Wandb project `"Thesis"` (skip via `--no-wandb`).

## Domains

Five eval domains in `config.py`: `correctness_completeness`, `clarity_communication`, `helpfulness_usefulness`, `coding_communication_conditional`, `relevance_instruction_following`.

## HelpSteer3 Data Structure

`individual_preference` **NOT** in results parquets — stripped by `prepare_data.py`. Lives only in:
- `data/raw/helpsteer3_train.parquet`
- `data/raw/helpsteer3_test.parquet`

Type: `numpy.ndarray` of dicts, ~3 annotators per example. Keys:
```
score       int   e.g. -2, -1, 0, 1, 2
reasoning   str   "@Response 1 is better because..."
feedback1   str   per-response feedback for Response A
feedback2   str   per-response feedback for Response B
```

Join key: `prompt_id` in results = `hashlib.sha256(context_text.encode()).hexdigest()[:16]`
where `context_text = "\n\n".join(f"[{turn['role']}]\n{turn['content']}" for turn in context)`

Review app (`src/evaluation/review_app.py`) rebuilds join at startup via `_build_individual_pref_lookup()` scanning raw parquets.

## Dynamic Instance-Adaptive Selector (thesis pivot)

Freeze v3 bank (61 Qs, 5 dims, 77.05% dev_600 baseline), train learned per-instance selector, cascade judge. Full plan: `Plan.md`.

```bash
# 1. Freeze bank + build flat qid index
python src/data_process/build_bank_index.py --bank checklists/v3 --out checklists/v3_frozen

# 2. Oracle labels (full 61 Qs over train, ~5-10 GPU-h on 5090)
python src/data_process/build_oracle_labels.py --bank checklists/v3_frozen --split train --tier tier_10k --out data/oracle/train_oracle_v3.parquet

# 3. Train bi-encoder selector (bge-m3 frozen + MLP head, listwise)
python src/train/run_selector_train.py --oracle data/oracle/train_oracle_v3.parquet --out results/checkpoints/selector_v1

# 4. Dynamic eval (policies: static_v3, random_k, domain_fixed_k, learned_topk, learned_topk_escalate, learned_topk_fallback)
python src/evaluation/run_dynamic_eval.py --bank checklists/v3_frozen --split dev_600 --policy learned_topk --k 20 --selector results/checkpoints/selector_v1 --out results/dynamic_dev_600/p3
```

## Baseline Prompt Variant (DO NOT SWAP)

v3 77.05% baseline uses **na-aware** `CHECKEVAL_POINTWISE_PROMPT` (via `build_checkeval_prompt` in `utils.py`), not `CHECKEVAL_POINTWISE_PROMPT_BINARY`. Metrics: `results/checkeval_pairwise_naaware_dev_600_v3_q9b_metrics.json`.

Canonical bank path post-freeze: `checklists/v3_frozen` (61 Qs). Pass via `--checklists-dir checklists/v3_frozen` to `run_checkeval_judge.py` / `run_ablation.py`.

Shared helper for per-qid subset prompts: `build_pointwise_prompt_from_qids(row, qids, qmeta, side)` in `utils.py`. Oracle, selector-infer, dynamic-eval all route through this to keep prompt parity.

## qid Ordering Source-of-Truth

Flat `qid` (1..N global) = `utils.load_checklists(bank_dir)` traversal order:
  - YAML files sorted by name (`*_filtered.yaml`)
  - `sub_aspects` in Python dict insertion order
  - Dedup questions **within each sub_aspect only** (not global)

`build_bank_index.py` asserts alignment against `load_checklists()` + cross-checks per-domain counts vs baseline metrics JSON. Do not pass `--skip-alignment-check` without understanding contract.

## dev_600 Holdout

`data/splits/dev_600.parquet` is selector eval holdout. `run_selector_train.py` checks `prompt_id` disjointness from oracle by default (`--holdout-splits`). Override with `--allow-holdout-leak` only for diagnostic runs.

## Encoding Gotcha

Some scripts carry UTF-8 BOM. Syntax check: `python -c "import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8-sig').read())" path.py`.

## Inference Backend

Default backend: **llama.cpp** (llama-server HTTP). vLLM kept as fallback. Toggle via `INFERENCE_BACKEND=vllm` env or `--backend vllm` CLI flag on eval scripts.

New eval scripts: call `load_judge_model(model_id, backend=args.backend, llamacpp_adapter_path=str(adapter)?, ...)` + `make_lora_handle(adapter_path, backend, name, lora_int_id)` from `utils.py`. Both dispatch internally; downstream code backend-agnostic.

Setup:
```bash
# Set $LLAMA_CPP_HOME to a built llama.cpp clone (with convert_hf_to_gguf.py + build/bin/llama-quantize)
export LLAMA_CPP_HOME=/path/to/llama.cpp

# Convert base model to Q4_K_M GGUF
python src/train/convert_to_gguf.py base \
    --hf-path models/Qwen3.5-9B \
    --out models/gguf/Qwen3.5-9B \
    --quant Q4_K_M

# Convert LoRA adapter to GGUF-LoRA
python src/train/convert_to_gguf.py lora \
    --adapter-path results/checkpoints/<run>/final_adapter \
    --base models/Qwen3.5-9B \
    --out models/gguf/adapters/<run>.gguf

# Start llama-server (port 8080)
bash scripts/start_llamacpp_server.sh \
    models/gguf/Qwen3.5-9B.Q4_K_M.gguf \
    models/gguf/adapters/<run>.gguf
```

llama-server `--ctx-size` is **total** context, split across `--parallel` slots → per-slot = ctx / parallel. `start_llamacpp_server.sh` accepts `CTX_PER_SLOT` (default 16384) and `PARALLEL` (default 4) env vars; pass per-slot value, script multiplies. Symptom of misconfiguration: `exceed_context_size_error, n_ctx=1024` on 5k-token prompts.

Eval scripts (`run_zeroshot.py`, `run_checkeval_judge.py`, `run_judge_eval.py`, `run_eval_finetuned.py`, `run_generator_infer.py`, `run_dynamic_eval.py`, `run_ablation.py`, `run_teacher_review.py`, `run_warmup_swap_eval.py`) accept `--backend llamacpp|vllm`. Default reads `cfg.INFERENCE_BACKEND`.

Server lifecycle: launch `start_llamacpp_server.sh` BEFORE running any eval. LoRA pinned at server launch (`--lora <gguf>`); switching adapters = restart server. Eval scripts do not auto-spawn server.

Backend toggle precedence: `--backend` CLI flag wins over `INFERENCE_BACKEND` env var; env wins over `cfg.INFERENCE_BACKEND` default.

Quant baseline: Q4_K_M (≈ AWQ-4bit). Q5_K_M ≈ 19 GB fits 5090 too. Training stays on HF/transformers (untouched).

## Bank Versions

| Path | Qs | Notes |
|---|---|---|
| `checklists/v3_frozen` | 61 | Baseline bank; produced 0.7705 dev_600 accuracy. Do not modify. |
| `checklists/v4_draft` | 58 | Editable source YAMLs for v4. |
| `checklists/v4_frozen` | 58 | Frozen v4 index. Use for new oracle runs. |

v4 changes vs v3: dropped 5 NA-cliff Format/Constraint Qs (>75% N/A), deduped Practical Usability (6→3) and Coverage (6→4), rewrote Structural Effectiveness as conditional, added Tone/Register Match (2), Ambiguity/Premise Handling (2), Temporal Awareness (1), Explicit Numeric Constraints (2).

v4 acceptance threshold (27B Q6 oracle on dev_600): `oracle_agreement_rate_valid >= 0.78` overall; general domain >= 0.74.

## Oracle Metrics & judge-mode Flag

`build_oracle_labels.py --judge-mode local` dispatches via `load_judge_model` → `cfg.INFERENCE_BACKEND` (default llamacpp HTTP, **not** in-process). For real in-process vLLM use `INFERENCE_BACKEND=vllm`. `--judge-mode http` uses script's own OpenAI client with `--http-concurrency` (typically faster).

Oracle reports both `oracle_agreement_rate_valid` (excl Tie/null, baseline-comparable) and `_total` (Tie counted as wrong). Compare `_valid` against `results/checkeval_pairwise_naaware_dev_600_v3_q9b_metrics.json` `accuracy` (baseline 0.7705 on n_valid=427). 9B Q4 GGUF reproduces baseline within ±1pp; 27B Q6 ≈ +5pp.

`--review-out` only dumps samples where true winner received 0 "yes" answers (judge missed winner entirely), not all wrong predictions. Schema differs from `*_sample.parquet` — `review_app.py` requires review parquet (has `_review_split` column).

Oracle parity flags (must match baseline): `--max-new-tokens 2048` (baseline used 2048, oracle default 1024 — truncation raises parse-fail rate) and `--judge-url http://127.0.0.1:8080/v1` (oracle default port 8000, llama-server default 8080).

Canonical oracle command (v4, 27B, http mode):
```bash
python src/data_process/build_oracle_labels.py \
    --bank checklists/v4_frozen --split dev_600 \
    --judge-mode http --judge-url http://127.0.0.1:8080/v1 \
    --judge-model <gguf-model-name> --judge-api-key EMPTY \
    --max-new-tokens 2048 --http-concurrency 4 --tie-delta 0.05 \
    --out data/oracle/dev_600_oracle_v4_27b.parquet \
    --review-out data/oracle/dev_600_oracle_v4_27b_review.parquet
```