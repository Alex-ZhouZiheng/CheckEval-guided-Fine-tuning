"""
SFT the checklist-generator model with LoRA.

Uses TRL's SFTTrainer on the conversational parquet produced by
``prepare_generator_sft.py``. The assistant-only loss mask is handled by a
small custom ``_AssistantOnlyCollator`` because TRL 1.0's
``assistant_only_loss=True`` requires ``{% generation %}`` in the chat template
— Qwen3.5 lacks it, and ``get_training_chat_template`` returns a patched
template that still has no marker.

Other settings follow ms-swift's Qwen3.5 SFT recipe:
  - target_modules all-linear  (attention + MLP projections)
  - deepspeed zero2            (optional, via --deepspeed flag)

Usage:
    python run_generator_sft.py --tier debug_5k --no-wandb
    python run_generator_sft.py --tier tier_10k --lr 2e-5 --epochs 2 --deepspeed
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
from trl import SFTConfig, SFTTrainer

import config as cfg

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


class _AssistantOnlyCollator:
    """
    TRL 1.0's ``assistant_only_loss=True`` requires ``{% generation %}`` blocks
    in the chat template, which Qwen3.5 doesn't ship and which TRL's
    ``get_training_chat_template`` fails to inject (returns a template without
    the marker, mask is all-zero). This collator replaces that mechanism:
    it finds the last ``<|im_start|>assistant\\n`` in each sequence and masks
    every label before it.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0
        self.header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        log.info("AssistantOnlyCollator: header token ids = %s", self.header_ids)

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        # TRL 1.0's tokenize_fn doesn't always produce attention_mask; derive it if absent.
        attn_masks = [
            torch.tensor(f["attention_mask"], dtype=torch.long)
            if "attention_mask" in f
            else torch.ones_like(input_ids[i])
            for i, f in enumerate(features)
        ]
        max_len = max(t.size(0) for t in input_ids)

        batch_ids    = input_ids[0].new_full((len(features), max_len), self.pad_id)
        batch_attn   = input_ids[0].new_zeros(len(features), max_len)
        batch_labels = input_ids[0].new_full((len(features), max_len), -100)

        h, hl = self.header_ids, len(self.header_ids)
        for i, (ids, attn) in enumerate(zip(input_ids, attn_masks)):
            n = ids.size(0)
            batch_ids[i, :n]  = ids
            batch_attn[i, :n] = attn
            last_end = -1
            for j in range(n - hl + 1):
                if ids[j : j + hl].tolist() == h:
                    last_end = j + hl
            if last_end >= 0:
                batch_labels[i, last_end:n] = ids[last_end:n]

        return {"input_ids": batch_ids, "attention_mask": batch_attn, "labels": batch_labels}


def load_sft_dataset(
    tier: str,
    eval_split_ratio: float = 0.0,
) -> tuple[Dataset, Dataset | None]:
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
    full_ds = Dataset.from_list(records)

    if eval_split_ratio > 0:
        split = full_ds.train_test_split(
            test_size=eval_split_ratio, seed=cfg.SEED, shuffle=True
        )
        log.info("Split: %d train / %d eval", len(split["train"]), len(split["test"]))
        return split["train"], split["test"]

    return full_ds, None


def build_lora_config(
    rank: int, alpha: int, dropout: float, target_modules: str | list[str]
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        log.info("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        log.info("  Using SDPA")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ── Data ──
    parser.add_argument("--tier", type=str, default="debug_5k",
                        choices=["debug_5k", "tier_10k", "tier_20k"])
    parser.add_argument("--eval-split-ratio", type=float, default=0.01,
                        help="Fraction of train data held out for eval (0 = no eval)")
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    # ── Model ──
    parser.add_argument("--model-id", type=str, default=str(cfg.GENERATOR_MODEL_ID))
    # ── LoRA ──
    parser.add_argument("--lora-rank", type=int, default=cfg.LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=cfg.LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=cfg.LORA_DROPOUT)
    parser.add_argument("--target-modules", type=str, default="all-linear",
                        help="'all-linear' (ms-swift default) or comma list e.g. q_proj,v_proj")
    # ── Training ──
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=cfg.SFT_MAX_LENGTH)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    # ── DeepSpeed ──
    parser.add_argument("--deepspeed", action="store_true",
                        help="Enable DeepSpeed ZeRO-2 (cfg.DEEPSPEED_CONFIG)")
    # ── Logging ──
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    target_modules: str | list[str] = args.target_modules
    if "," in target_modules:
        target_modules = [m.strip() for m in target_modules.split(",")]

    run_name = args.run_name or (
        f"generator_sft_{args.tier}_r{args.lora_rank}_lr{args.lr}"
    )
    output_dir = cfg.CHECKPOINTS_DIR / run_name

    # ── Logging setup ──
    use_wandb = not args.no_wandb and _WANDB_AVAILABLE
    use_tb    = not args.no_tensorboard
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
                "target_modules": target_modules,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "deepspeed": args.deepspeed,
                "seed": cfg.SEED,
            },
            tags=["generator-sft", args.tier],
        )
    if use_tb:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)

    # ── Data ──
    train_ds, eval_ds = load_sft_dataset(
        args.tier, eval_split_ratio=args.eval_split_ratio,
    )

    # ── Model + LoRA ──
    model, tokenizer = load_base(args.model_id)
    lora_config  = build_lora_config(
        args.lora_rank, args.lora_alpha, args.lora_dropout, target_modules
    )
    data_collator = _AssistantOnlyCollator(tokenizer)

    # ── Step/schedule math ──
    world_size      = int(os.environ.get("WORLD_SIZE", "1"))
    eff_bs          = args.batch_size * args.grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * cfg.WARMUP_RATIO)
    save_steps      = max(50, steps_per_epoch // 2)
    eval_steps      = save_steps

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
        # ── Data / loss ──
        max_length=args.max_length,
        assistant_only_loss=False,      # handled by _AssistantOnlyCollator below
        packing=False,
        dataset_num_proc=args.dataset_num_proc,
        # ── Training ──
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        # ── Precision ──
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # ── DeepSpeed ──
        deepspeed=cfg.DEEPSPEED_CONFIG if args.deepspeed else None,
        # ── Saving ──
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        # ── Eval ──
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_steps if eval_ds is not None else None,
        # ── Logging ──
        logging_steps=5,
        report_to=report_to,
        seed=cfg.SEED,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    log.info("Starting generator SFT: %d train rows, %d steps", len(train_ds), total_steps)
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
        "target_modules": target_modules,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "warmup_steps": warmup_steps,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds else 0,
        "deepspeed": args.deepspeed,
        "seed": cfg.SEED,
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    if use_wandb:
        _wandb.summary.update({
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds else 0,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "output_dir": str(output_dir),
        })
        _wandb.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
