#!/usr/bin/env python3
"""
Thin orchestrator for end-to-end eval of the generator → judge pipeline.

Runs, in sequence:
  1. run_generator_infer.py   (writes data/generated_checklists/<split>.parquet)
  2. run_judge_eval.py        (reads that parquet, writes results/pipeline_*.*)

Each step is a separate subprocess — we avoid loading two models into the same
vLLM process which would fight for GPU memory and use different bases.

Usage:
    # Post-FT (both adapters)
    python run_pipeline_eval.py \\
        --generator-adapter results/checkpoints/generator_sft_.../final_adapter \\
        --judge-adapter     results/checkpoints/judge_sft_.../final_adapter \\
        --eval-split dev --subset dev_600

    # Pre-FT baseline (no adapters — base models end-to-end)
    python run_pipeline_eval.py --eval-split dev --subset dev_600
"""

from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-adapter", type=str, default=None,
                        help="Generator final_adapter. Omit for pre-FT baseline.")
    parser.add_argument("--judge-adapter", type=str, default=None,
                        help="Judge final_adapter. Omit for pre-FT baseline.")
    parser.add_argument("--generator-base", type=str,
                        default=str(cfg.GENERATOR_MODEL_ID))
    parser.add_argument("--judge-base", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--eval-split", type=str, default="dev")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tie-delta", type=float, default=0.0)
    parser.add_argument("--skip-generator", action="store_true",
                        help="Reuse existing generated_checklists parquet")
    parser.add_argument("--experiment-suffix",type=str,default=None)
    args = parser.parse_args()

    split_tag = args.subset or args.eval_split
    gen_fname = f"{split_tag}.parquet" if args.generator_adapter else f"{split_tag}_base.parquet"
    gen_out = cfg.GENERATED_CHECKLIST_DIR / gen_fname

    if not args.generator_adapter:
        log.info("No generator adapter — will run base model for Step 1")
    if not args.judge_adapter:
        log.info("No judge adapter — will run base model for Step 2")

    py = sys.executable
    src = Path(__file__).parent

    # Step 1: generator inference.
    if args.skip_generator and gen_out.exists():
        log.info("Reusing existing generated checklists at %s", gen_out)
    else:
        cmd = [
            py, str(src / "run_generator_infer.py"),
            "--base-model", args.generator_base,
            "--split", args.eval_split,
            "--batch-size", str(args.batch_size),
            "--output-path", str(gen_out),
        ]
        if args.generator_adapter:
            cmd += ["--adapter-path", args.generator_adapter]
        if args.subset:
            cmd += ["--subset", args.subset]
        if args.max_samples:
            cmd += ["--max-samples", str(args.max_samples)]
        log.info("Running generator step:\n  %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # Step 2: judge eval.
    cmd = [
        py, str(src / "run_judge_eval.py"),
        "--base-model", args.judge_base,
        "--generated", str(gen_out),
        "--eval-split", args.eval_split,
        "--batch-size", str(args.batch_size),
        "--tie-delta", str(args.tie_delta),
        "--experiment-suffix",args.experiment_suffix
    ]
    if args.judge_adapter:
        cmd += ["--judge-adapter", args.judge_adapter]
    if args.subset:
        cmd += ["--subset", args.subset]
    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]
    log.info("Running judge step:\n  %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    log.info("Pipeline eval finished. See results/ for metrics and predictions.")


if __name__ == "__main__":
    main()
