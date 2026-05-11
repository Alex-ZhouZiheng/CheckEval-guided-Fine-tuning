#!/bin/bash
set -e
cd /root/autodl-tmp/Thesis
mkdir -p results/logs/selfcheck_ei_r1
.venvmerge/bin/python -m src.data_process.build_selfchk_ei_data   --input-parquet /root/autodl-tmp/Thesis/data/splits/train_tier_5k_clean.parquet   --adapter-path /root/autodl-tmp/Thesis/checkpoints/judge_sft_swift_Qwen3.5-4B_train_debug_5k_selfcheck_clean_r16_lr2e-5/v0-20260506-200134/checkpoint-265   --base-model /root/autodl-tmp/Thesis/models/Qwen3.5-4B   --backend vllm   --k 8   --temperature 0.8   --top-p 0.95   --max-new-tokens 1536   --enable-thinking   --skip-diversity   --min-reward 0.5   --min-pseudo-confidence 0.6   --min-swap-consistency 0.5   --max-model-len 9106   --max-num-seqs 64   --max-num-batched-tokens 16384   --batch-size 128   --output-path /root/autodl-tmp/Thesis/data/judge_sft/train_tier_5k_clean_selfcheck_ei_r1.parquet   --log-dir /root/autodl-tmp/Thesis/results/logs/selfcheck_ei_r1   --seed 42
