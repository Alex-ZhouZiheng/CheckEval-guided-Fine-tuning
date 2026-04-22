# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CheckEval-guided fine-tuning: a two-model pipeline where a **checklist generator** produces structured yes/no evaluation questions, and a **checklist-conditioned judge** answers them to determine the better response in pairwise LLM comparisons. Built on the HelpSteer3 dataset; trains Qwen3.5 models via DPO, SFT, and joint training.

## Environment Setup

```bash
pip install -r requirements.txt
# Required env vars in .env:
# HF_TOKEN=<huggingface token>
# OPENAI_API_KEY=<openai key>
```

Training uses DeepSpeed ZeRO-2. Inference uses vLLM with LoRA support (known-good config for Qwen3.5-27B-AWQ-4bit: `max_model_len=16384, max_num_seqs=16, gpu_memory_utilization=0.90`).

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

Data lives under `data/` with subdirs: `raw/`, `splits/`, `dpo/`, `with_reason/`, `generator_sft/`, `judge_sft/`, `checklist_sft/`, `generated_checklists/`. Format is Parquet throughout.

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

Key hyperparameters (in `src/config.py`): LR=1e-6, epochs=1, batch_size=1, grad_accum=32, max_len=2048, LoRA rank=16/alpha=32, DPO beta=0.1. Models: Qwen3.5-9B (judge), Qwen3.5-4B (generator).

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

All scripts read paths and model IDs from `src/config.py`. Experiment tracking uses Wandb project `"Thesis"` (skip with `--no-wandb`).

## Domains

Five evaluation domains defined in `config.py`: `correctness_completeness`, `clarity_communication`, `helpfulness_usefulness`, `coding_communication_conditional`, `relevance_instruction_following`.
