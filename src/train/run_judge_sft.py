"""
SFT the checklist-conditioned judge with LoRA on (instruction+responses+checklist)
→ Yes/No-per-question targets produced by ``prepare_judge_sft.py``.

Usage:
    python run_judge_sft.py --tier debug_5k --no-wandb
    python run_judge_sft.py --tier tier_10k --lr 1e-5 --epochs 1
    python run_judge_sft.py --sft-path data/judge_sft/train_tier_10k_selfcheck.parquet
"""
from __future__ import annotations
from unsloth import FastLanguageModel
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def build_deepspeed_zero3_config() -> dict:
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }


def build_deepspeed_zero3_no_offload_config() -> dict:
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }


def init_torch_distributed_for_deepspeed() -> None:
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available; cannot use DeepSpeed.")
    if torch.distributed.is_initialized():
        return

    required = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DeepSpeed launch is missing torch distributed environment variables "
            f"{missing}. Launch with: python -m torch.distributed.run --standalone "
            "--nproc_per_node=<N> src/train/run_judge_sft.py ..."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    log.info(
        "Initialized torch.distributed for DeepSpeed: rank=%s/%s local_rank=%s",
        os.environ["RANK"], os.environ["WORLD_SIZE"], local_rank,
    )


# ── Liger Kernel: fused RoPE / RMSNorm / SwiGLU saves ~6-10 GB on Qwen3 ──
# Skipped when --use-unsloth (unsloth ships its own fused kernels; double-patch
# can break activation shapes). Set DISABLE_LIGER=1 to force-skip.
def _apply_liger():
    if os.environ.get("DISABLE_LIGER", "0") == "1":
        return
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        log.warning("liger-kernel not installed; running without it (higher memory)")
        return

    kwargs = dict(rope=True, rms_norm=True, swiglu=True,
                  cross_entropy=False, fused_linear_cross_entropy=True)
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


def load_sft_dataset(
    tier: str,
    sft_path: str | None = None,
    tokenizer=None,
    enable_thinking: bool = False,
) -> Dataset:
    if sft_path is not None:
        path = Path(sft_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Check --sft-path argument."
            )
    else:
        path = cfg.JUDGE_SFT_DIR / f"train_{tier}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_judge_sft.py --tier {tier} first."
            )
    df = pd.read_parquet(path)
    log.info("Loaded %d judge SFT rows from %s (enable_thinking=%s)",
             len(df), path, enable_thinking)

    if tokenizer is None:
        # Legacy path: hand TRL the messages column, let it auto-render via
        # the tokenizer's default chat template (no enable_thinking control).
        def _to_messages(row) -> list[dict[str, str]]:
            msgs = row["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            else:
                msgs = list(msgs)
            msgs = list(msgs) + [{"role": "assistant", "content": row["target_output"]}]
            return msgs

        return Dataset.from_list([{"messages": _to_messages(r)} for _, r in df.iterrows()])

    # Thinking-aware path: pre-render full text so we control enable_thinking.
    # Render user prefix with apply_chat_template(add_generation_prompt=True,
    # enable_thinking=...), then concatenate raw target_output (which already
    # contains <think>...</think>### Final\n... when teacher ran in thinking
    # mode) and the EOS token.
    eos = tokenizer.eos_token or ""

    def _to_text(row) -> str:
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        else:
            msgs = list(msgs)
        prefix = tokenizer.apply_chat_template(
            list(msgs),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prefix + row["target_output"] + eos

    return Dataset.from_list([{"text": _to_text(r)} for _, r in df.iterrows()])


def build_lora_config(rank: int, alpha: int, dropout: float,
                      target_modules: list[str] | None = None) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base(model_id: str, qlora: bool = False, device_map: str | None = None):
    log.info("Loading tokenizer and base model: %s (qlora=%s, device_map=%s)",
             model_id, qlora, device_map or "<none>")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.truncation_side = "left"

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
    if device_map is not None:
        kwargs["device_map"] = device_map

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


def load_base_unsloth(
    model_id: str, max_seq_length: int, qlora: bool,
    lora_rank: int, lora_alpha: int, lora_dropout: float,
    device_map: str | None = None,
    target_modules: list[str] | None = None,
):

    if qlora:
        log.warning(
            "QLoRA (4-bit) is NOT recommended for Qwen3.5 by unsloth docs — "
            "quantization differences larger than normal. Use bf16 LoRA instead "
            "(drop --qlora). If OOM persists, reduce --max-length."
        )
    log.info("Loading via unsloth: %s (qlora=%s, max_seq=%d, device_map=%s)",
             model_id, qlora, max_seq_length, device_map or "<none>")
    load_kwargs = {}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=qlora,
        dtype=torch.bfloat16 if not qlora else None,
        trust_remote_code=True,
        **load_kwargs,
    )
    # Newer unsloth returns a Processor wrapper for VL-capable repos. The text
    # tokenizer lives at processor.tokenizer; downstream code (collator, chat
    # template) needs the bare tokenizer interface (encode/decode/etc).
    if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
        log.info("  Extracting .tokenizer from %s wrapper", type(tokenizer).__name__)
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or cfg.LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # async offload to CPU RAM
        random_state=cfg.SEED,
        max_seq_length=max_seq_length,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="debug_5k")
    parser.add_argument("--sft-path", type=str, default=None,
                        help="When set, load SFT data from this parquet path instead of the "
                             "tier-based path (--tier is still used for the run name).")
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
                        help="Load base model in 4-bit NF4 (QLoRA). Note: unsloth docs "
                             "recommend AGAINST QLoRA for Qwen3.5 — use bf16 LoRA instead.")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        choices=["adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit", "adamw_8bit"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--use-unsloth", action="store_true",
                        help="Load via unsloth FastLanguageModel (fused kernels + async "
                             "activation offload). Allows 9B QLoRA @ 8192 ctx on 32GB. "
                             "Mutually exclusive with liger-kernel; we auto-disable liger.")
    parser.add_argument("--device-map", type=str, default=None,
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="Pass device_map to from_pretrained for single-process model "
                             "sharding, e.g. --device-map balanced. Do not combine with "
                             "torchrun/DDP or --deepspeed-zero3.")
    parser.add_argument("--deepspeed-zero3", action="store_true",
                        help="Use DeepSpeed ZeRO-3 with CPU parameter/optimizer offload. "
                             "Launch with torchrun/accelerate for multi-GPU sharding.")
    parser.add_argument("--deepspeed-zero3-no-offload", action="store_true",
                        help="Use DeepSpeed ZeRO-3 without CPU offload. This avoids "
                             "DeepSpeedCPUAdam CUDA extension builds when system CUDA "
                             "does not match torch CUDA.")
    parser.add_argument("--packing", action="store_true",
                        help="Concat short samples up to max_length for higher throughput. "
                             "Disables custom assistant-only collator (loss includes prompt).")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.WARMUP_RATIO)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear",
                        choices=["linear", "cosine", "constant", "constant_with_warmup",
                                 "cosine_with_restarts", "polynomial"])
    parser.add_argument("--target-modules", type=str, default=None,
                        help="Comma-separated module names overriding cfg.LORA_TARGET_MODULES, "
                             "e.g. 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'.")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Pre-render chat template with enable_thinking=True so the prefix "
                             "matches Qwen3 thinking-mode inference. Required when target_output "
                             "contains <think>...</think> blocks (self-checklist CoT data).")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.device_map is not None and args.deepspeed_zero3:
        parser.error("--device-map and --deepspeed-zero3 are mutually exclusive.")
    if args.device_map is not None and args.deepspeed_zero3_no_offload:
        parser.error("--device-map and --deepspeed-zero3-no-offload are mutually exclusive.")
    if args.deepspeed_zero3 and args.deepspeed_zero3_no_offload:
        parser.error("--deepspeed-zero3 and --deepspeed-zero3-no-offload are mutually exclusive.")
    if args.device_map is not None and world_size > 1:
        parser.error("--device-map is single-process model sharding; do not use it with torchrun/DDP.")
    if args.deepspeed_zero3 or args.deepspeed_zero3_no_offload:
        init_torch_distributed_for_deepspeed()

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

    # liger and unsloth ship overlapping kernel patches — running both crashes
    # at the first forward pass. Liger applies on import; gate it on the flag.
    if not args.use_unsloth:
        _apply_liger()
    else:
        log.info("--use-unsloth set: skipping liger-kernel (unsloth has its own).")
        if args.lora_dropout > 0:
            log.warning(
                "Unsloth requires dropout=0 for fast LoRA patching. "
                "lora_dropout=%.2f will prevent patching LoRA matrices → speed drop. "
                "Pass --lora-dropout 0 to fix.",
                args.lora_dropout,
            )

    # Tokenizer needed before dataset when --enable-thinking, so we pre-render
    # the chat template with the flag explicitly set.
    target_modules = (
        [m.strip() for m in args.target_modules.split(",") if m.strip()]
        if args.target_modules else None
    )
    if args.use_unsloth:
        model, tokenizer = load_base_unsloth(
            args.model_id, max_seq_length=args.max_length, qlora=args.qlora,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, device_map=args.device_map,
            target_modules=target_modules,
        )
    else:
        model, tokenizer = load_base(
            args.model_id, qlora=args.qlora, device_map=args.device_map,
        )
    train_ds = load_sft_dataset(
        args.tier, args.sft_path,
        tokenizer=tokenizer if args.enable_thinking else None,
        enable_thinking=args.enable_thinking,
    )
    # unsloth injected LoRA via get_peft_model already; passing peft_config to
    # SFTTrainer would double-wrap and break grad flow. HF path keeps PEFT in
    # the trainer call (TRL handles wrap).
    lora_config = None if args.use_unsloth else build_lora_config(
        args.lora_rank, args.lora_alpha, args.lora_dropout,
        target_modules=target_modules,
    )
    # Packing concats samples; our collator searches for assistant header which
    # only catches the LAST occurrence inside a packed sequence → would mask all
    # earlier samples. Disable custom collator and let TRL's default packing
    # collator run (loss covers prompt tokens too, acceptable for small SFT).
    data_collator = None if args.packing else _AssistantOnlyCollator(tokenizer)

    eff_bs = args.batch_size * args.grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
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
        packing=args.packing,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=True,
        # Unsloth's get_peft_model already wired async-offload checkpointing;
        # toggling it again from TRL re-wraps and disables the offload path.
        gradient_checkpointing=not args.use_unsloth,
        gradient_checkpointing_kwargs=(
            None if args.use_unsloth else {"use_reentrant": False}
        ),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=report_to,
        seed=cfg.SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        deepspeed=(
            build_deepspeed_zero3_config()
            if args.deepspeed_zero3
            else build_deepspeed_zero3_no_offload_config()
            if args.deepspeed_zero3_no_offload
            else None
        ),
        ddp_find_unused_parameters=False if world_size > 1 else None,
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
        "sft_path": args.sft_path,
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
        "enable_thinking": bool(args.enable_thinking),
        "use_unsloth": bool(args.use_unsloth),
        "device_map": args.device_map,
        "deepspeed_zero3": bool(args.deepspeed_zero3),
        "deepspeed_zero3_no_offload": bool(args.deepspeed_zero3_no_offload),
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Done.")


if __name__ == "__main__":
    main()
