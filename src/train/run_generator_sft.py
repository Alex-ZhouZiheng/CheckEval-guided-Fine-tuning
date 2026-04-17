"""
SFT the checklist-generator model with LoRA.

Uses TRL's SFTTrainer on the conversational parquet produced by
``prepare_generator_sft.py``.  The target completion is the structured
checklist string; only assistant tokens contribute to the loss
(``assistant_only_loss=True``).

Usage:
    python run_generator_sft.py --tier debug_5k --no-wandb
    python run_generator_sft.py --tier tier_10k --lr 2e-5 --epochs 2
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import config as cfg

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_sft_dataset(tier: str) -> Dataset:
    path = cfg.GENERATOR_SFT_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_generator_sft.py --tier {tier} first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded %d SFT rows from %s", len(df), path)

    def _row_to_messages(row) -> list[dict[str, str]]:
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        else:
            msgs = list(msgs)
        msgs = list(msgs) + [{"role": "assistant", "content": row["target_output"]}]
        return msgs

    records = [{"messages": _row_to_messages(r)} for _, r in df.iterrows()]
    return Dataset.from_list(records)


def build_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base(model_id: str):
    log.info("Loading tokenizer and base model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        log.info("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        log.info("  Using SDPA")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="debug_5k",
                        choices=["debug_5k", "tier_10k", "tier_20k"])
    parser.add_argument("--model-id", type=str, default=str(cfg.GENERATOR_MODEL_ID))
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="SFT uses a higher LR than DPO (default 2e-5)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=cfg.LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=cfg.LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=cfg.LORA_DROPOUT)
    parser.add_argument("--max-length", type=int, default=cfg.SFT_MAX_LENGTH)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or f"generator_sft_{args.tier}_r{args.lora_rank}_lr{args.lr}"
    output_dir = cfg.CHECKPOINTS_DIR / run_name

    use_wandb = not args.no_wandb and _WANDB_AVAILABLE
    use_tb = not args.no_tensorboard
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT
        _wandb.init(
            project=cfg.WANDB_PROJECT,
            name=run_name,
            config={
                "tier": args.tier,
                "model_id": args.model_id,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "seed": cfg.SEED,
            },
            tags=["generator-sft", args.tier],
        )
    if use_tb:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)

    train_ds = load_sft_dataset(args.tier)
    model, tokenizer = load_base(args.model_id)
    lora_config = build_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)

    # assistant_only_loss=True requires {% generation %} in the chat template,
    # which Qwen3.5 doesn't have. Use DataCollatorForCompletionOnlyLM instead:
    # encode the assistant header as it appears in the full tokenized sequence.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    eff_bs = args.batch_size * args.grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    save_steps = max(50, steps_per_epoch // 2)

    report_to = []
    if use_wandb:
        report_to.append("wandb")
    if use_tb:
        report_to.append("tensorboard")
    if not report_to:
        report_to = ["none"]

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        # Data / loss
        max_length=args.max_length,
        packing=False,
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        # Logging
        logging_steps=10,
        report_to=report_to,
        seed=cfg.SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    log.info("Starting generator SFT: %d rows, %d steps", len(train_ds), total_steps)
    trainer.train()

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved generator adapter to %s", final_dir)

    meta = {
        "tier": args.tier,
        "model_id": args.model_id,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "warmup_steps": warmup_steps,
        "train_samples": len(train_ds),
        "seed": cfg.SEED,
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    if use_wandb:
        _wandb.summary.update({
            "train_samples": len(train_ds),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "output_dir": str(output_dir),
        })
        _wandb.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
