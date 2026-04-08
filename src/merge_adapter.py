#!/usr/bin/env python3
"""
Stage 1: Merge LoRA adapter into base model weights.

This script ONLY handles the merge step. It requires a newer version of
transformers that may be incompatible with vLLM. Run this in a separate
environment (e.g. a different conda env or venv) from the inference stage.

Usage:
    # Basic merge
    python merge_adapter.py \
        --adapter-path results/checkpoints/dpo_debug_5k_.../final_adapter

    # Custom base model and output path
    python merge_adapter.py \
        --adapter-path results/checkpoints/dpo_debug_5k_.../final_adapter \
        --base-model-id meta-llama/Llama-3.1-8B \
        --output-path /data/merged_models/my_judge

Environment:
    pip install torch transformers peft accelerate
    (use whatever transformers version the adapter requires)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelVL
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelVL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Default base model — override with --base-model-id if needed.
# We import config only if available; otherwise fall back to CLI arg.
try:
    import config as cfg
    DEFAULT_BASE_MODEL = cfg.JUDGE_MODEL_ID
except Exception:
    DEFAULT_BASE_MODEL = None


def merge_adapter(
    adapter_path: Path,
    output_path: Path,
    base_model_id: str,
) -> Path:
    """Merge LoRA adapter into base model weights on CPU."""

    try:
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception:
        processor = None
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    base_model = AutoModelVL.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    if processor is not None:
        processor.save_pretrained(str(output_path))

    import shutil
    base_dir = Path(base_model_id)

    for fname in [
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    ]:
        src = base_dir / fname
        if src.exists():
            shutil.copy(src, output_path / fname)
            log.info("Copied %s from base", fname)

    log.info("Merged model saved to", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to save the merged model (default: <adapter-path>/../merged)",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge even if output already exists",
    )
    args = parser.parse_args()

    if args.base_model_id is None:
        parser.error(
            "--base-model-id is required (config.py not found or JUDGE_MODEL_ID not set)"
        )

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        parser.error(f"Adapter path does not exist: {adapter_path}")

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = adapter_path.parent / "merged"

    if args.force and (output_path / "config.json").exists():
        log.info("--force specified, removing existing merged model at %s", output_path)
        import shutil
        shutil.rmtree(output_path)

    merge_adapter(adapter_path, output_path, args.base_model_id)
    log.info("Done. You can now run inference with:")
    log.info(
        "  python run_eval_finetuned.py --model-path %s --eval-mode vanilla",
        output_path,
    )


if __name__ == "__main__":
    main()