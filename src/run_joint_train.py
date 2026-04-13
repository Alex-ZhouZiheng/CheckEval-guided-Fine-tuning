"""
Joint DPO + Checklist SFT training.

Loss = L_DPO + λ · L_checklist_SFT

The DPO component learns pairwise preference alignment.
The checklist SFT component teaches the model to produce accurate,
well-formatted checklist evaluations.

Usage:
    # Quick smoke test (synthetic SFT data)
    python run_joint_train.py --tier debug_5k --no-wandb

    # Standard run with teacher-generated SFT data
    python run_joint_train.py --tier tier_10k --sft-lambda 0.1

    # Sweep λ
    python run_joint_train.py --tier tier_10k --sft-lambda 0.5

    # Control experiment: different checklist source
    python run_joint_train.py --tier tier_10k \
        --sft-data ../data/checklist_sft_v2/train_tier_10k.parquet

    # Disable SFT loss (pure DPO baseline)
    python run_joint_train.py --tier tier_10k --sft-lambda 0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Reuse patches and helpers from run_dpo_train ───────────────

_orig_precompute = DPOTrainer._precompute_ref_logps


def _patched_precompute(self, dataset, split, batch_size):
    cols = set(dataset.column_names)
    if {"ref_chosen_logps", "ref_rejected_logps"}.issubset(cols):
        log.info("Skipping ref-logp precompute for '%s' (columns already present)", split)
        return dataset
    return _orig_precompute(self, dataset, split, batch_size)


DPOTrainer._precompute_ref_logps = _patched_precompute


def _apply_liger():
    if os.environ.get("DISABLE_LIGER", "0") == "1":
        return
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        log.warning("liger-kernel not installed; running without it")
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


# ────────────────────────── SFT collator ───────────────────────


class ChecklistSFTCollator:
    """Tokenize checklist (prompt, completion) pairs on the fly.

    Produces ``input_ids``, ``attention_mask``, and ``labels`` with the
    prompt portion masked (label = -100).
    """

    def __init__(self, tokenizer, max_length: int = cfg.SFT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Detect whether the chat template supports enable_thinking
        self._template_kwargs = {}
        try:
            test = tokenizer.apply_chat_template(
                [{"role": "user", "content": "t"}],
                tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            if isinstance(test, list) and all(isinstance(x, int) for x in test):
                self._template_kwargs = {"enable_thinking": False}
        except (TypeError, Exception):
            pass

    def _apply_and_tokenize(self, messages, *, add_generation_prompt: bool) -> list[int]:
        """Apply chat template and guarantee a list[int] of token ids."""
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            **self._template_kwargs,
        )
        # Fallback: some tokenizer versions return a string instead of ids
        if isinstance(result, str):
            result = self.tokenizer.encode(result, add_special_tokens=False)
        return list(result)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []

        for sample in batch:
            prompt_text = sample["prompt_text"]
            completion_text = sample["completion_text"]

            # Full conversation tokens
            full_messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": completion_text},
            ]
            full_ids = self._apply_and_tokenize(
                full_messages, add_generation_prompt=False,
            )

            # Prompt-only tokens (including generation prompt / assistant header)
            prompt_messages = [{"role": "user", "content": prompt_text}]
            prompt_ids = self._apply_and_tokenize(
                prompt_messages, add_generation_prompt=True,
            )

            n_prompt = len(prompt_ids)

            # Truncate
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            # Labels: mask prompt tokens
            labels = [-100] * min(n_prompt, len(full_ids))
            if len(full_ids) > n_prompt:
                labels += full_ids[n_prompt:]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded_ids = []
        padded_labels = []
        padded_mask = []
        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ────────────────────────── Joint Trainer ──────────────────────


class JointDPOSFTTrainer(DPOTrainer):
    """DPO trainer with an auxiliary checklist SFT loss.

    Total loss = L_DPO + sft_lambda * L_SFT

    The SFT dataloader cycles independently of the DPO dataloader.
    SFT loss is only added during training (skipped during eval).
    """

    def __init__(
        self,
        *args,
        sft_dataset: Dataset | None = None,
        sft_lambda: float = 0.1,
        sft_collator: ChecklistSFTCollator | None = None,
        sft_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_lambda = sft_lambda
        self._sft_step_losses: list[float] = []
        self._dpo_step_losses: list[float] = []

        if sft_dataset is not None and sft_lambda > 0:
            self.sft_dataloader = DataLoader(
                sft_dataset,
                batch_size=sft_batch_size,
                shuffle=True,
                collate_fn=sft_collator,
                drop_last=True,
                num_workers=2,
                pin_memory=True,
            )
            self._sft_iter = iter(self.sft_dataloader)
            log.info("SFT dataloader: %d samples, batch_size=%d, λ=%.4f",
                     len(sft_dataset), sft_batch_size, sft_lambda)
        else:
            self.sft_dataloader = None
            self._sft_iter = None
            if sft_lambda > 0:
                log.warning("sft_lambda=%.4f but no sft_dataset provided; "
                            "training is pure DPO", sft_lambda)

    def _get_sft_batch(self) -> dict[str, torch.Tensor]:
        """Get next SFT batch, cycling the dataloader when exhausted."""
        try:
            return next(self._sft_iter)
        except StopIteration:
            self._sft_iter = iter(self.sft_dataloader)
            return next(self._sft_iter)

    def _compute_sft_loss(self, model, sft_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass on checklist SFT batch and return CE loss."""
        sft_batch = {k: v.to(model.device) for k, v in sft_batch.items()}
        outputs = model(
            input_ids=sft_batch["input_ids"],
            attention_mask=sft_batch["attention_mask"],
        )
        logits = outputs.logits

        # Shift for causal LM loss: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sft_batch["labels"][:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Combined DPO + SFT loss."""
        # DPO loss (from parent class)
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            dpo_loss, outputs = result
        else:
            dpo_loss = result

        # Add SFT loss only during training and when dataloader is available
        if model.training and self.sft_dataloader is not None and self.sft_lambda > 0:
            sft_batch = self._get_sft_batch()
            sft_loss = self._compute_sft_loss(model, sft_batch)
            total_loss = dpo_loss + self.sft_lambda * sft_loss

            # Track for logging
            self._dpo_step_losses.append(dpo_loss.detach().item())
            self._sft_step_losses.append(sft_loss.detach().item())
        else:
            total_loss = dpo_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject SFT/DPO component losses into the standard log dict."""
        if self._dpo_step_losses:
            logs["dpo_loss"] = sum(self._dpo_step_losses) / len(self._dpo_step_losses)
            self._dpo_step_losses.clear()
        if self._sft_step_losses:
            logs["sft_loss"] = sum(self._sft_step_losses) / len(self._sft_step_losses)
            self._sft_step_losses.clear()
        if self.sft_lambda > 0:
            logs["sft_lambda"] = self.sft_lambda
        super().log(logs, *args, **kwargs)


# ────────────────────────── data loading ───────────────────────


def load_dpo_dataset(tier: str) -> tuple[Dataset, Dataset]:
    """Load DPO data (identical logic to run_dpo_train.py)."""
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

    for df in [train_df, dev_df]:
        for col in ["prompt", "chosen", "rejected"]:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(json.loads)

    keep_cols = ["prompt", "chosen", "rejected"]
    train_ds = Dataset.from_pandas(train_df[keep_cols], preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df[keep_cols], preserve_index=False)

    thinking_off = [{"enable_thinking": False}]
    train_ds = train_ds.add_column("chat_template_kwargs", thinking_off * len(train_ds))
    dev_ds = dev_ds.add_column("chat_template_kwargs", thinking_off * len(dev_ds))

    log.info("DPO  — Train: %d,  Dev: %d", len(train_ds), len(dev_ds))
    return train_ds, dev_ds


def load_sft_dataset(sft_path: str | Path) -> Dataset:
    """Load checklist SFT data from parquet."""
    sft_path = Path(sft_path)
    if not sft_path.exists():
        raise FileNotFoundError(
            f"{sft_path} not found. Run: python prepare_checklist_sft.py first."
        )

    sft_df = pd.read_parquet(sft_path)

    # Filter to valid parses only
    if "parse_valid" in sft_df.columns:
        before = len(sft_df)
        sft_df = sft_df[sft_df["parse_valid"]].reset_index(drop=True)
        if len(sft_df) < before:
            log.info("SFT filter: %d → %d valid samples", before, len(sft_df))

    keep_cols = ["prompt_text", "completion_text"]
    sft_ds = Dataset.from_pandas(sft_df[keep_cols], preserve_index=False)
    log.info("SFT  — %d samples from %s", len(sft_ds), sft_path.name)
    return sft_ds


# ────────────────────────── model setup ────────────────────────


def build_lora_config(rank: int, alpha: int, dropout: float = cfg.LORA_DROPOUT) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base_model(model_id: str = cfg.JUDGE_MODEL_ID):
    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model: %s (bf16)", model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
        log.info("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        )
        log.info("  Using SDPA")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


# ────────────────────────── training args ──────────────────────


def build_training_args(
    output_dir: Path,
    run_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup_steps: int,
    eval_steps: int,
    beta: float,
    use_wandb: bool,
    use_tensorboard: bool,
) -> DPOConfig:
    report_to = []
    if use_wandb:
        report_to.append("wandb")
    if use_tensorboard:
        report_to.append("tensorboard")
    if not report_to:
        report_to = ["none"]

    if use_tensorboard:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)

    return DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        beta=beta,
        max_length=cfg.MAX_LENGTH,
        precompute_ref_log_probs=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=report_to,
        seed=cfg.SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


# ────────────────────────── main ───────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Joint DPO + Checklist SFT fine-tuning"
    )
    # Data
    parser.add_argument("--tier", type=str, default="debug_5k",
                        choices=["debug_5k", "tier_10k", "tier_20k", "full"])
    parser.add_argument(
        "--sft-data", type=str, default=None,
        help="Path to checklist SFT parquet. Default: auto-detect from tier.",
    )

    # Joint training
    parser.add_argument("--sft-lambda", type=float, default=None,
                        help=f"Checklist SFT loss weight (default: {cfg.JOINT_LAMBDA})")
    parser.add_argument("--sft-batch-size", type=int, default=None,
                        help="SFT micro-batch size (default: same as DPO batch-size)")

    # Standard DPO args
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
    parser.add_argument("--shutdown", action="store_true",
                        help="Power off machine after training")
    args = parser.parse_args()

    # Resolve defaults
    lr = args.lr or cfg.LEARNING_RATE
    epochs = args.epochs or cfg.NUM_EPOCHS
    batch_size = args.batch_size or cfg.PER_DEVICE_BATCH_SIZE
    grad_accum = args.grad_accum or cfg.GRADIENT_ACCUMULATION_STEPS
    lora_rank = args.lora_rank or cfg.LORA_RANK
    lora_alpha = args.lora_alpha or cfg.LORA_ALPHA
    beta = args.beta or cfg.DPO_BETA
    sft_lambda = args.sft_lambda if args.sft_lambda is not None else cfg.JOINT_LAMBDA
    sft_batch_size = args.sft_batch_size or batch_size

    run_name = args.run_name or (
        f"joint_dpo_{args.tier}_r{lora_rank}_b{beta}_lr{lr}_lam{sft_lambda}"
    )

    # WandB
    use_wandb = not args.no_wandb
    use_tensorboard = not args.no_tensorboard
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT

    log.info("=" * 60)
    log.info("Joint DPO + Checklist SFT Training")
    log.info("=" * 60)
    log.info("  tier:        %s", args.tier)
    log.info("  model:       %s", args.model_id)
    log.info("  lora_rank:   %s  alpha: %s", lora_rank, lora_alpha)
    log.info("  lr:          %s", lr)
    log.info("  epochs:      %s", epochs)
    log.info("  batch_size:  %s x %s (grad_accum)", batch_size, grad_accum)
    log.info("  beta:        %s", beta)
    log.info("  sft_lambda:  %s", sft_lambda)
    log.info("  sft_batch:   %s", sft_batch_size)
    log.info("=" * 60)

    # ── 1. Load DPO data ──
    model_tag = Path(args.model_id).name
    cache_dir = cfg.DPO_DIR / "ref_cache" / f"{model_tag}_len{cfg.MAX_LENGTH}" / args.tier
    train_cache = cache_dir / "train"
    eval_cache = cache_dir / "eval"

    if train_cache.exists() and eval_cache.exists():
        from datasets import load_from_disk
        log.info("Loading cached ref log probs from %s", cache_dir)
        train_ds = load_from_disk(str(train_cache))
        dev_ds = load_from_disk(str(eval_cache))
        log.info("  train: %d  dev: %d (ref-logp cache hit)", len(train_ds), len(dev_ds))
    else:
        train_ds, dev_ds = load_dpo_dataset(args.tier)

    # ── 2. Load SFT data ──
    sft_ds = None
    if sft_lambda > 0:
        if args.sft_data:
            sft_path = Path(args.sft_data)
        else:
            # Auto-detect: try teacher first, then synthetic
            sft_path = cfg.CHECKLIST_SFT_DIR / (
                f"train_{args.tier}.parquet" if args.tier != "full"
                else "train.parquet"
            )
            if not sft_path.exists():
                sft_path_syn = sft_path.with_name(
                    sft_path.stem.replace(args.tier, f"{args.tier}_synthetic") + ".parquet"
                )
                if sft_path_syn.exists():
                    sft_path = sft_path_syn

        sft_ds = load_sft_dataset(sft_path)

    # ── 3. Load model + tokenizer ──
    model, tokenizer = load_base_model(args.model_id)

    # ── 4. LoRA config ──
    lora_config = build_lora_config(lora_rank, lora_alpha)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = batch_size * grad_accum * world_size
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
    eval_steps = max(1, steps_per_epoch // 4)

    # ── 5. Training args ──
    output_dir = cfg.CHECKPOINTS_DIR / run_name
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    training_args = build_training_args(
        output_dir=output_dir,
        run_name=run_name,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        warmup_steps=warmup_steps,
        beta=beta,
        eval_steps=eval_steps,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )

    # ── 6. SFT collator ──
    sft_collator = ChecklistSFTCollator(tokenizer, max_length=cfg.SFT_MAX_LENGTH)

    # ── 7. Build joint trainer ──
    trainer = JointDPOSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        sft_dataset=sft_ds,
        sft_lambda=sft_lambda,
        sft_collator=sft_collator,
        sft_batch_size=sft_batch_size,
    )

    # Save ref-logp cache
    if not (train_cache.exists() and eval_cache.exists()):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("Saving ref-logp cache to %s", cache_dir)
            trainer.train_dataset.save_to_disk(str(train_cache))
            trainer.eval_dataset.save_to_disk(str(eval_cache))
        except Exception as e:
            log.warning("Failed to save ref-logp cache: %s", e)

    # ── 8. Train ──
    log.info("Starting joint training...")
    trainer.train()

    # ── 9. Save ──
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved adapter to %s", final_dir)

    # Training config
    train_meta = {
        "mode": "joint_dpo_sft",
        "tier": args.tier,
        "model_id": str(args.model_id),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": cfg.LORA_DROPOUT,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": effective_batch_size,
        "beta": beta,
        "sft_lambda": sft_lambda,
        "sft_batch_size": sft_batch_size,
        "sft_data": str(args.sft_data or sft_path) if sft_lambda > 0 else None,
        "sft_samples": len(sft_ds) if sft_ds else 0,
        "max_length": cfg.MAX_LENGTH,
        "sft_max_length": cfg.SFT_MAX_LENGTH,
        "warmup_steps": warmup_steps,
        "seed": cfg.SEED,
        "train_samples": len(train_ds),
        "dev_samples": len(dev_ds),
    }
    meta_path = output_dir / "train_config.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, default=str)
    log.info("Saved training config to %s", meta_path)

    log.info("Done.")


def _shutdown_machine():
    import subprocess
    log.info("Shutting down machine in 60s ...")
    try:
        subprocess.Popen(["shutdown", "-h", "+1"])
    except FileNotFoundError:
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
