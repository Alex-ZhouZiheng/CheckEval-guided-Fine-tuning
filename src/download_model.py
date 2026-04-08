#!/usr/bin/env python3
"""
Pre-download model weights so that inference scripts do not block on I/O.

Usage
-----
    python download_model.py
    python download_model.py --cache-dir /data/hf
    python download_model.py --model-id Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import logging
import time

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download judge model weights")
    parser.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    model_id = args.model_id
    cache_dir = args.cache_dir

    log.info("Downloading model: %s", model_id)
    if cache_dir:
        log.info("  Cache dir: %s", cache_dir)

    t0 = time.time()

    log.info("Step 1/3: Downloading model snapshot...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )

if __name__ == "__main__":
    main()
