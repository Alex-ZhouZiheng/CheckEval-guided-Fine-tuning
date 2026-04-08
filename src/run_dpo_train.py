"""
Usage:
1) Quick smoke test on a small debug split (recommended first run)
python run_dpo_train.py --tier debug_5k --no-wandb

2) Standard baseline run on the 10k split
python run_dpo_train.py --tier tier_10k

3) Larger run on the 20k split
python run_dpo_train.py --tier tier_20k

4) Full-data run with custom epochs and learning rate
python run_dpo_train.py --tier full --epochs 3 --lr 1e-6

5) Disable both WandB and TensorBoard logging
python run_dpo_train.py --tier tier_10k --no-wandb --no-tensorboard

6) Override effective batch size settings
python run_dpo_train.py --tier tier_10k --batch-size 1 --grad-accum 16

7) Change LoRA rank / alpha
python run_dpo_train.py --tier tier_10k --lora-rank 32 --lora-alpha 64

8) Use a different base model
python run_dpo_train.py --tier tier_10k --model-id /path/to/your/model

"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Reduce CUDA fragmentation for large intermittent allocations (DPO logits).
# Must be set BEFORE torch imports CUDA context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Patch TRL's DPOTrainer to skip ref-logp precompute if the columns already
# exist in the dataset. The stock implementation re-computes unconditionally,
# which defeats our on-disk cache.
_orig_precompute = DPOTrainer._precompute_ref_logps

def _patched_precompute(self, dataset, split, batch_size):
    cols = set(dataset.column_names)
    if {"ref_chosen_logps", "ref_rejected_logps"}.issubset(cols):
        log.info("Skipping ref-logp precompute for '%s' (columns already present)", split)
        return dataset
    return _orig_precompute(self, dataset, split, batch_size)

DPOTrainer._precompute_ref_logps = _patched_precompute

def _apply_liger():
    """Apply Liger kernels for Qwen3.5 (qwen3_5 model_type)."""
    if os.environ.get("DISABLE_LIGER", "0") == "1":
        log.info("DISABLE_LIGER=1 set; skipping Liger Kernel")
        return
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        log.warning("liger-kernel not installed; running without it")
        return

    kwargs = dict(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,            # DPO does not use standard CE loss
        fused_linear_cross_entropy=False,
    )
    # Try in order: qwen3_5 → qwen3 → qwen2 (Qwen3.5 reuses Qwen3/Qwen2 layers)
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
    log.warning("No compatible Liger patcher found for Qwen3.5; skipping")


_apply_liger()


# ────────────────────────── data loading ─────────────────────


def load_dpo_dataset(tier: str) -> tuple[Dataset, Dataset]:
    """Load DPO chat-template data as HuggingFace Datasets.

    Parameters
    ----------
    tier : str
        One of "debug_5k", "tier_10k", "tier_20k", "full".
        Maps to data/dpo/train_{tier}_chat.parquet (or train_chat.parquet).

    Returns
    -------
    train_ds, dev_ds : Dataset
    """
    if tier == "full":
        train_path = cfg.DPO_DIR / "train.parquet"
    else:
        train_path = cfg.DPO_DIR / f"train_{tier}.parquet"
    dev_path = cfg.DPO_DIR / "dev.parquet"

    for path in [train_path, dev_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python prepare_dpo_data.py --chat-template"
            )

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)

    # Parquet may serialize chat-template lists as strings; ensure they are lists
    for df in [train_df, dev_df]:
        for col in ["prompt", "chosen", "rejected"]:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(json.loads)

    # Keep only the columns TRL needs
    keep_cols = ["prompt", "chosen", "rejected"]
    train_ds = Dataset.from_pandas(train_df[keep_cols], preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df[keep_cols], preserve_index=False)

    # Disable Qwen3.5 thinking mode to avoid token-level prefix mismatch warnings
    thinking_off = [{"enable_thinking": False}]
    train_ds = train_ds.add_column("chat_template_kwargs", thinking_off * len(train_ds))
    dev_ds = dev_ds.add_column("chat_template_kwargs", thinking_off * len(dev_ds))

    log.info("Train: %d samples, Dev: %d samples", len(train_ds), len(dev_ds))
    return train_ds, dev_ds


# ────────────────────────── model setup ──────────────────────


def build_lora_config(
    rank: int,
    alpha: int = cfg.LORA_ALPHA,
    dropout: float = cfg.LORA_DROPOUT,
) -> LoraConfig:
    """Create LoRA config for Qwen architecture."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base_model(model_id: str = cfg.JUDGE_MODEL_ID):
    """Load base model and tokenizer for training."""
    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model: %s (bf16)", model_id)
    # Try flash_attention_2, fall back to sdpa
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
        log.info("  Using SDPA (Flash Attention not available)")

    # CRITICAL for PEFT + gradient_checkpointing:
    # Without this, gradient checkpointing silently fails to save activation
    # memory because LoRA's frozen base-model inputs have requires_grad=False,
    # so autograd does not re-enter the checkpointed region during backward.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        log.info("  Enabled input require grads (needed for PEFT + grad ckpt)")

    return model, tokenizer


# ────────────────────────── training args ────────────────────


def build_training_args(
    output_dir: Path,
    run_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup_ratio: float,
    eval_steps,
    beta: float,
    use_deepspeed: bool,
    use_wandb: bool,
    use_tensorboard: bool,
) -> DPOConfig:
    """Build TRL DPOConfig with all training parameters."""
    # Build report_to list
    report_to = []
    if use_wandb:
        report_to.append("wandb")
    if use_tensorboard:
        report_to.append("tensorboard")
    if not report_to:
        report_to = ["none"]

    tb_dir = str(cfg.TENSORBOARD_DIR / run_name) if use_tensorboard else None

    return DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        # DPO
        beta=beta,
        max_length=cfg.MAX_LENGTH,
        precompute_ref_log_probs=True,

        # Training
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        # Precision
        bf16=True,
        # Checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Saving
        save_strategy="steps",
        save_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=10,
        report_to=report_to,
        logging_dir=tb_dir,
        # DeepSpeed
        deepspeed=cfg.DEEPSPEED_CONFIG if use_deepspeed else None,
        # Misc
        seed=cfg.SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


# ────────────────────────── main ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tune Qwen judge with LoRA")
    parser.add_argument(
        "--tier",
        type=str,
        default="debug_5k",
        choices=["debug_5k", "tier_10k", "tier_20k", "full"],
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Power off the machine after training finishes (also on failure)",
    )
    args = parser.parse_args()

    # Resolve defaults from config
    lr = args.lr or cfg.LEARNING_RATE
    epochs = args.epochs or cfg.NUM_EPOCHS
    batch_size = args.batch_size or cfg.PER_DEVICE_BATCH_SIZE
    grad_accum = args.grad_accum or cfg.GRADIENT_ACCUMULATION_STEPS
    lora_rank = args.lora_rank or cfg.LORA_RANK
    lora_alpha = args.lora_alpha or cfg.LORA_ALPHA
    beta = args.beta or cfg.DPO_BETA
    run_name = args.run_name or f"dpo_{args.tier}_r{lora_rank}_b{beta}_lr{lr}"

    # WandB setup
    use_wandb = not args.no_wandb
    use_tensorboard = not args.no_tensorboard
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT

    log.info("=" * 50)
    log.info("DPO Training Configuration")
    log.info("=" * 50)
    log.info("  tier:       %s", args.tier)
    log.info("  model:      %s", args.model_id)
    log.info("  lora_rank:  %s  alpha: %s", lora_rank, lora_alpha or 2 * lora_rank)
    log.info("  lr:         %s", lr)
    log.info("  epochs:     %s", epochs)
    log.info("  batch_size: %s x %s (grad_accum)", batch_size, grad_accum)
    log.info("  beta:       %s", beta)
    log.info("  wandb:      %s", use_wandb)
    log.info("  tensorboard:%s", use_tensorboard)
    log.info("=" * 50)

    # 1. Load data (possibly from ref-logp cache)
    # Cache is keyed by tier + model name + max_length so different configs
    # don't collide. If present, DPOTrainer will see the `ref_chosen_logps` /
    # `ref_rejected_logps` columns and skip the (slow) precompute step.
    model_tag = Path(args.model_id).name
    cache_dir = cfg.DPO_DIR / "ref_cache" / f"{model_tag}_len{cfg.MAX_LENGTH}" / args.tier
    train_cache = cache_dir / "train"
    eval_cache = cache_dir / "eval"

    if train_cache.exists() and eval_cache.exists():
        from datasets import load_from_disk
        log.info("Loading cached ref log probs from %s", cache_dir)
        train_ds = load_from_disk(str(train_cache))
        dev_ds = load_from_disk(str(eval_cache))
        log.info("  train: %d  dev: %d  (ref-logp cache hit)",
                 len(train_ds), len(dev_ds))
    else:
        train_ds, dev_ds = load_dpo_dataset(args.tier)

    # 2. Load model + tokenizer
    model, tokenizer = load_base_model(args.model_id)

    # 3. LoRA config
    import math
    lora_config = build_lora_config(lora_rank, lora_alpha)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = batch_size * grad_accum * world_size
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
    eval_steps = max(1, steps_per_epoch // 4)

    # 4. Training args
    output_dir = cfg.CHECKPOINTS_DIR / run_name
    training_args = build_training_args(
        output_dir=output_dir,
        run_name=run_name,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        warmup_ratio=cfg.WARMUP_RATIO,
        beta=beta,
        eval_steps=eval_steps,
        use_deepspeed=False,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )

    # 5. DPOTrainer (pass peft_config so TRL handles LoRA wrapping)
    # Note: __init__ triggers ref-logp precompute when precompute_ref_log_probs
    # is True and the columns are missing; if we loaded from cache it is a no-op.
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 5b. Save ref-logp cache on first compute, so subsequent runs skip it.
    if not (train_cache.exists() and eval_cache.exists()):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("Saving ref-logp cache to %s", cache_dir)
            trainer.train_dataset.save_to_disk(str(train_cache))
            trainer.eval_dataset.save_to_disk(str(eval_cache))
            log.info("  Cached. Next run with same tier/model/max_length will skip precompute.")
        except Exception as e:
            log.warning("Failed to save ref-logp cache: %s", e)

    # 6. Train
    log.info("Starting training...")
    trainer.train()

    # 7. Save final adapter
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved adapter to %s", final_dir)

    # 8. Save training config for reproducibility
    train_meta = {
        "tier": args.tier,
        "model_id": args.model_id,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha or 2 * lora_rank,
        "lora_dropout": cfg.LORA_DROPOUT,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "beta": beta,
        "max_length": cfg.MAX_LENGTH,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "seed": cfg.SEED,
        "train_samples": len(train_ds),
        "dev_samples": len(dev_ds),
        "deepspeed": False,
    }
    meta_path = output_dir / "train_config.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2,default=str)
    log.info("Saved training config to %s", meta_path)

    log.info("Done.")


def _shutdown_machine():
    """Power off the host. Used by --shutdown."""
    import subprocess
    log.info("Shutting down machine in 60s ...")
    try:
        subprocess.Popen(["shutdown", "-h", "+1"])
    except FileNotFoundError:
        # Some images (e.g. autodl) only have /usr/bin/shutdown via different path
        try:
            subprocess.Popen(["/sbin/shutdown", "-h", "+1"])
        except Exception as e:
            log.error("Failed to schedule shutdown: %s", e)


if __name__ == "__main__":
    import sys
    shutdown_flag = "--shutdown" in sys.argv
    try:
        main()
    finally:
        if shutdown_flag:
            _shutdown_machine()
