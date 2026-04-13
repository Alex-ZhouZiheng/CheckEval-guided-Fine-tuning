"""
Joint DPO + Checklist SFT training.

Loss = L_DPO + λ · L_checklist_SFT

The DPO component learns pairwise preference alignment.
The checklist SFT component teaches the model to produce accurate,
well-formatted checklist evaluations.

Usage:
    # Quick smoke test
    python run_joint_train.py --tier debug_5k --no-wandb

    # Standard run with teacher-generated SFT data
    python run_joint_train.py --tier tier_10k --sft-lambda 0.1

    # Sweep λ
    python run_joint_train.py --tier tier_10k --sft-lambda 0.5

    # Custom checklist source
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
import re
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

# ── Monkey-patch: skip ref-logp precompute when cache columns exist ──

_orig_precompute = DPOTrainer._precompute_ref_logps


def _patched_precompute(self, dataset, split, batch_size):
    if {"ref_chosen_logps", "ref_rejected_logps"}.issubset(dataset.column_names):
        log.info("Skipping ref-logp precompute for '%s' (columns already present)", split)
        return dataset
    return _orig_precompute(self, dataset, split, batch_size)


DPOTrainer._precompute_ref_logps = _patched_precompute


# ── Liger Kernel (optional fused ops) ────────────────────────────


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


# ────────────────────────── SFT Collator ─────────────────────────

# Regex for stripping Qwen3 thinking blocks.
_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_RE_THINK_TRAILING = re.compile(r"<think>\s*$")


class ChecklistSFTCollator:
    """Tokenize checklist (prompt, completion) pairs on the fly.

    Produces ``input_ids``, ``attention_mask``, and ``labels`` with the
    prompt portion masked (``label = -100``).

    Tokenization strategy — prompt and completion are encoded *separately*
    then concatenated.  This avoids fragile "find-the-common-prefix" logic
    and lets us truncate the prompt while preserving completion tokens.
    """

    MIN_COMPLETION_TOKENS = 16  # keep at least this many after truncation

    def __init__(self, tokenizer, max_length: int = cfg.SFT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove Qwen3 ``<think>`` blocks (closed and trailing-unclosed)."""
        text = _RE_THINK_CLOSED.sub("", text)
        text = _RE_THINK_TRAILING.sub("", text)
        return text

    def _apply_and_tokenize(self, messages: list[dict], *,
                            add_generation_prompt: bool) -> list[int]:
        """Render chat template to text, strip ``<think>``, then encode."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        text = self._strip_think(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ── main entry point ─────────────────────────────────────────

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        completion_lengths: list[int] = []

        eos_id = self.tokenizer.eos_token_id

        for sample in batch:
            # Tokenize prompt (with assistant header) and completion separately
            prompt_ids = self._apply_and_tokenize(
                [{"role": "user", "content": sample["prompt_text"]}],
                add_generation_prompt=True,
            )
            comp_ids = self.tokenizer.encode(
                sample["completion_text"], add_special_tokens=False,
            ) + [eos_id]

            n_prompt, n_comp = len(prompt_ids), len(comp_ids)
            completion_lengths.append(n_comp)

            # Truncate: prompt first, then completion; always keep
            # at least MIN_COMPLETION_TOKENS of the completion.
            if n_prompt + n_comp > self.max_length:
                max_prompt = self.max_length - min(n_comp, self.max_length - 1)
                max_prompt = max(max_prompt, 1)
                if n_prompt > max_prompt:
                    prompt_ids = prompt_ids[:max_prompt]
                    n_prompt = max_prompt
                remaining = self.max_length - n_prompt
                if n_comp > remaining:
                    comp_ids = comp_ids[:remaining]
                    n_comp = remaining

            # Skip samples where no completion tokens survive
            if n_comp < self.MIN_COMPLETION_TOKENS:
                continue

            input_ids_list.append(prompt_ids + comp_ids)
            labels_list.append([-100] * n_prompt + comp_ids)

        if not input_ids_list:
            preview = ", ".join(str(n) for n in completion_lengths[:5]) or "none"
            raise ValueError(
                "ChecklistSFTCollator filtered out every sample in the batch. "
                f"batch_size={len(batch)}, min_completion_tokens={self.MIN_COMPLETION_TOKENS}, "
                f"completion_token_lengths=[{preview}]. "
                "Increase completion_text length, lower MIN_COMPLETION_TOKENS, "
                "or inspect truncation."
            )

        # Pad batch to uniform length
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or eos_id

        padded_ids, padded_labels, padded_mask = [], [], []
        for ids, labs in zip(input_ids_list, labels_list):
            pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad)
            padded_labels.append(labs + [-100] * pad)
            padded_mask.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ────────────────────────── Joint Trainer ────────────────────────


class JointDPOSFTTrainer(DPOTrainer):
    """DPO trainer with an auxiliary checklist SFT loss.

    ``total_loss = L_DPO + sft_lambda * L_SFT``

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
        self._dpo_step_losses: list[float] = []
        self._sft_step_losses: list[float] = []

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

    # ── SFT loss computation ─────────────────────────────────────

    def _get_sft_batch(self) -> dict[str, torch.Tensor]:
        """Get next SFT batch, cycling the dataloader when exhausted."""
        try:
            return next(self._sft_iter)
        except StopIteration:
            self._sft_iter = iter(self.sft_dataloader)
            return next(self._sft_iter)

    def _compute_sft_loss(self, model, sft_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass on a checklist SFT batch → cross-entropy loss."""
        sft_batch = {k: v.to(model.device) for k, v in sft_batch.items()}
        logits = model(
            input_ids=sft_batch["input_ids"],
            attention_mask=sft_batch["attention_mask"],
        ).logits

        # Shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sft_batch["labels"][:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        # Guard: all-masked labels → NaN; treat as zero loss
        if torch.isnan(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

    # ── Combined loss ────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Combined DPO + SFT loss."""
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            dpo_loss, outputs = result
        else:
            dpo_loss = result

        if model.training and self.sft_dataloader is not None and self.sft_lambda > 0:
            sft_loss = self._compute_sft_loss(model, self._get_sft_batch())
            total_loss = dpo_loss + self.sft_lambda * sft_loss
            self._dpo_step_losses.append(dpo_loss.detach().item())
            self._sft_step_losses.append(sft_loss.detach().item())
        else:
            total_loss = dpo_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject SFT / DPO component losses into the standard log dict."""
        if self._dpo_step_losses:
            logs["dpo_loss"] = sum(self._dpo_step_losses) / len(self._dpo_step_losses)
            self._dpo_step_losses.clear()
        if self._sft_step_losses:
            logs["sft_loss"] = sum(self._sft_step_losses) / len(self._sft_step_losses)
            self._sft_step_losses.clear()
        if self.sft_lambda > 0:
            logs["sft_lambda"] = self.sft_lambda
        super().log(logs, *args, **kwargs)


# ────────────────────────── Data Loading ─────────────────────────


def load_dpo_dataset(tier: str) -> tuple[Dataset, Dataset]:
    """Load DPO preference pairs from parquet files."""
    train_path = cfg.DPO_DIR / ("train.parquet" if tier == "full"
                                else f"train_{tier}.parquet")
    dev_path = cfg.DPO_DIR / "dev.parquet"

    for p in (train_path, dev_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run: python prepare_dpo_data.py --chat-template")

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)

    for df in (train_df, dev_df):
        for col in ("prompt", "chosen", "rejected"):
            if df[col].dtype == object and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(json.loads)

    keep_cols = ["prompt", "chosen", "rejected"]
    train_ds = Dataset.from_pandas(train_df[keep_cols], preserve_index=False)
    dev_ds = Dataset.from_pandas(dev_df[keep_cols], preserve_index=False)

    # Disable Qwen3 thinking for DPO tokenization
    thinking_off = [{"enable_thinking": False}]
    train_ds = train_ds.add_column("chat_template_kwargs", thinking_off * len(train_ds))
    dev_ds = dev_ds.add_column("chat_template_kwargs", thinking_off * len(dev_ds))

    log.info("DPO  — Train: %d,  Dev: %d", len(train_ds), len(dev_ds))
    return train_ds, dev_ds


def load_sft_dataset(sft_path: str | Path) -> Dataset:
    """Load checklist SFT data from a parquet file."""
    sft_path = Path(sft_path)
    if not sft_path.exists():
        raise FileNotFoundError(
            f"{sft_path} not found. Run: python prepare_checklist_sft.py first.")

    sft_df = pd.read_parquet(sft_path)

    if "parse_valid" in sft_df.columns:
        before = len(sft_df)
        sft_df = sft_df[sft_df["parse_valid"]].reset_index(drop=True)
        if len(sft_df) < before:
            log.info("SFT filter: %d → %d valid samples", before, len(sft_df))

    sft_ds = Dataset.from_pandas(
        sft_df[["prompt_text", "completion_text"]], preserve_index=False)
    log.info("SFT  — %d samples from %s", len(sft_ds), sft_path.name)
    return sft_ds


# ────────────────────────── Model Setup ──────────────────────────


def build_lora_config(rank: int, alpha: int,
                      dropout: float = cfg.LORA_DROPOUT) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_base_model(model_id: str = cfg.JUDGE_MODEL_ID):
    """Load tokenizer and model in bf16 with best available attention."""
    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model: %s (bf16)", model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2")
        log.info("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa")
        log.info("  Using SDPA")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


# ────────────────────────── Training Args ────────────────────────


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
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)
    if not report_to:
        report_to = ["none"]

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
        warmup_steps=warmup_steps,
        bf16=True,
        # Checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Evaluation & saving
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
        # Misc
        seed=cfg.SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


# ────────────────────────── CLI & Main ───────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Joint DPO + Checklist SFT fine-tuning")

    # Data
    parser.add_argument("--tier", type=str, default="debug_5k",
                        choices=["debug_5k", "tier_10k", "tier_20k", "full"])
    parser.add_argument("--sft-data", type=str, default=None,
                        help="Path to checklist SFT parquet (default: auto-detect)")

    # Joint training
    parser.add_argument("--sft-lambda", type=float, default=None,
                        help=f"Checklist SFT loss weight (default: {cfg.JOINT_LAMBDA})")
    parser.add_argument("--sft-batch-size", type=int, default=None,
                        help="SFT micro-batch size (default: same as --batch-size)")

    # Standard DPO hyper-parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=cfg.JUDGE_MODEL_ID)

    # Logging & infra
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--shutdown", action="store_true",
                        help="Power off machine after training")

    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Resolve defaults ─────────────────────────────────────────
    lr         = args.lr or cfg.LEARNING_RATE
    epochs     = args.epochs or cfg.NUM_EPOCHS
    batch_size = args.batch_size or cfg.PER_DEVICE_BATCH_SIZE
    grad_accum = args.grad_accum or cfg.GRADIENT_ACCUMULATION_STEPS
    lora_rank  = args.lora_rank or cfg.LORA_RANK
    lora_alpha = args.lora_alpha or cfg.LORA_ALPHA
    beta       = args.beta or cfg.DPO_BETA
    sft_lambda = args.sft_lambda if args.sft_lambda is not None else cfg.JOINT_LAMBDA
    sft_batch_size = args.sft_batch_size or batch_size

    run_name = args.run_name or (
        f"joint_dpo_{args.tier}_r{lora_rank}_b{beta}_lr{lr}_lam{sft_lambda}")

    use_wandb = not args.no_wandb
    use_tensorboard = not args.no_tensorboard
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT

    log.info("=" * 60)
    log.info("Joint DPO + Checklist SFT Training")
    log.info("=" * 60)
    log.info("  tier=%s  model=%s", args.tier, args.model_id)
    log.info("  lora r=%d α=%d  lr=%s  epochs=%d", lora_rank, lora_alpha, lr, epochs)
    log.info("  batch=%d × %d (grad_accum)  β=%s", batch_size, grad_accum, beta)
    log.info("  sft_λ=%s  sft_batch=%d", sft_lambda, sft_batch_size)
    log.info("=" * 60)

    # ── 1. Load DPO data (with ref-logp cache) ──────────────────
    model_tag = Path(args.model_id).name
    cache_dir = cfg.DPO_DIR / "ref_cache" / f"{model_tag}_len{cfg.MAX_LENGTH}" / args.tier
    train_cache, eval_cache = cache_dir / "train", cache_dir / "eval"

    if train_cache.exists() and eval_cache.exists():
        from datasets import load_from_disk
        log.info("Loading cached ref log probs from %s", cache_dir)
        train_ds = load_from_disk(str(train_cache))
        dev_ds = load_from_disk(str(eval_cache))
        log.info("  train: %d  dev: %d (ref-logp cache hit)", len(train_ds), len(dev_ds))
    else:
        train_ds, dev_ds = load_dpo_dataset(args.tier)

    # ── 2. Load SFT data ────────────────────────────────────────
    sft_ds = None
    if sft_lambda > 0:
        if args.sft_data:
            sft_path = Path(args.sft_data)
        else:
            sft_path = cfg.CHECKLIST_SFT_DIR / (
                f"train_{args.tier}.parquet" if args.tier != "full"
                else "train.parquet")
            if not sft_path.exists():
                sft_path_syn = sft_path.with_name(
                    sft_path.stem.replace(args.tier, f"{args.tier}_synthetic") + ".parquet")
                if sft_path_syn.exists():
                    sft_path = sft_path_syn
        sft_ds = load_sft_dataset(sft_path)

    # ── 3. Load model + tokenizer ────────────────────────────────
    model, tokenizer = load_base_model(args.model_id)

    # ── 4. LoRA + scheduling ─────────────────────────────────────
    lora_config = build_lora_config(lora_rank, lora_alpha)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = batch_size * grad_accum * world_size
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
    eval_steps = max(1, steps_per_epoch // 4)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

    # ── 5. Training args ─────────────────────────────────────────
    output_dir = cfg.CHECKPOINTS_DIR / run_name
    training_args = build_training_args(
        output_dir=output_dir, run_name=run_name,
        lr=lr, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        warmup_steps=warmup_steps, beta=beta, eval_steps=eval_steps,
        use_wandb=use_wandb, use_tensorboard=use_tensorboard,
    )

    # ── 6. Build trainer ─────────────────────────────────────────
    sft_collator = ChecklistSFTCollator(tokenizer, max_length=cfg.SFT_MAX_LENGTH)
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

    # Save ref-logp cache for future runs
    if not (train_cache.exists() and eval_cache.exists()):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("Saving ref-logp cache to %s", cache_dir)
            trainer.train_dataset.save_to_disk(str(train_cache))
            trainer.eval_dataset.save_to_disk(str(eval_cache))
        except Exception as e:
            log.warning("Failed to save ref-logp cache: %s", e)

    # ── 7. Train ─────────────────────────────────────────────────
    log.info("Starting joint training...")
    trainer.train()

    # ── 8. Save adapter + metadata ───────────────────────────────
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved adapter to %s", final_dir)

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
    for cmd in (["shutdown", "-h", "+1"], ["/sbin/shutdown", "-h", "+1"]):
        try:
            subprocess.Popen(cmd)
            return
        except (FileNotFoundError, Exception):
            continue
    log.error("Failed to schedule shutdown")


if __name__ == "__main__":
    import sys
    shutdown_flag = "--shutdown" in sys.argv
    try:
        main()
    finally:
        if shutdown_flag:
            _shutdown_machine()
