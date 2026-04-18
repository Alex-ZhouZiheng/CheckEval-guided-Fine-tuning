"""
SFT the checklist-conditioned judge with LoRA on (instruction+responses+checklist)
→ Yes/No-per-question targets produced by ``prepare_judge_sft.py``.

Usage:
    python run_judge_sft.py --tier debug_5k --no-wandb
    python run_judge_sft.py --tier tier_10k --lr 1e-5 --epochs 1
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
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Liger Kernel: fused RoPE / RMSNorm / SwiGLU saves ~6-10 GB on Qwen3 ──
def _apply_liger():
    if os.environ.get("DISABLE_LIGER", "0") == "1":
        return
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        log.warning("liger-kernel not installed; running without it (higher memory)")
        return

    kwargs = dict(rope=True, rms_norm=True, swiglu=True,
                  cross_entropy=False, fused_linear_cross_entropy=False)
    for fn_name in ("apply_liger_kernel_to_qwen3_5",
                    "apply_liger_kernel_to_qwen3",
                    "apply_liger_kernel_to_qwen2"):
        fn = getattr(lk, fn_name, None)
        if fn is not None:
            try:
                fn(**kwargs)
                log.info("Applied Liger Kernel via %s", fn_name)
                return
            except Exception as e:
                log.warning("%s failed: %s", fn_name, e)


_apply_liger()


class _AssistantOnlyCollator:
    """
    TRL 1.0's ``assistant_only_loss=True`` requires ``{% generation %}`` blocks
    in the chat template, which Qwen3.5 doesn't ship. This collator replaces
    that mechanism: it finds the last ``<|im_start|>assistant\\n`` in each
    sequence and masks every label before it.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0
        self.header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        log.info("AssistantOnlyCollator: header token ids = %s", self.header_ids)

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
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


def load_sft_dataset(tier: str) -> Dataset:
    path = cfg.JUDGE_SFT_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_judge_sft.py --tier {tier} first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded %d judge SFT rows from %s", len(df), path)

    def _to_messages(row) -> list[dict[str, str]]:
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        else:
            msgs = list(msgs)
        msgs = list(msgs) + [{"role": "assistant", "content": row["target_output"]}]
        return msgs

    return Dataset.from_list([{"messages": _to_messages(r)} for _, r in df.iterrows()])


def build_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base(model_id: str, qlora: bool = False):
    log.info("Loading tokenizer and base model: %s (qlora=%s)", model_id, qlora)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(trust_remote_code=True)
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs,
        )
        log.info("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs,
        )
        log.info("  Using SDPA")

    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="debug_5k")
    parser.add_argument("--model-id", type=str, default=str(cfg.JUDGE_MODEL_ID))
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=cfg.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--lora-rank", type=int, default=cfg.LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=cfg.LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=cfg.LORA_DROPOUT)
    parser.add_argument("--max-length", type=int, default=cfg.MAX_LENGTH)
    parser.add_argument("--qlora", action="store_true",
                        help="Load base model in 4-bit NF4 (QLoRA). Saves ~13GB on a 9B model.")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        choices=["adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or f"judge_sft_{args.tier}_r{args.lora_rank}_lr{args.lr}"
    output_dir = cfg.CHECKPOINTS_DIR / run_name

    use_wandb = not args.no_wandb
    use_tb = not args.no_tensorboard
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT
    if use_tb:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)

    train_ds = load_sft_dataset(args.tier)
    model, tokenizer = load_base(args.model_id, qlora=args.qlora)
    lora_config = build_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    data_collator = _AssistantOnlyCollator(tokenizer)

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
        max_length=args.max_length,
        assistant_only_loss=False,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        optim=args.optim,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
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

    log.info("Starting judge SFT: %d rows, %d steps", len(train_ds), total_steps)
    trainer.train()

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved judge adapter to %s", final_dir)

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
    log.info("Done.")


if __name__ == "__main__":
    main()
