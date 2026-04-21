"""
Pairwise preference warm-up for the judge.

Goal:
    Teach the (already checklist-SFT'd) judge to answer "which response is better"
    in the simplest form: target = single token "A" or "B" after VANILLA_JUDGE_PROMPT.
    This is a warm-up before CheckEval-reward GRPO / DPO — the model needs a
    working pairwise prior first.

Data:
    HelpSteer3 pairwise train splits (data/splits/train_{tier}.parquet).
    Ties (winner == "tie") are dropped. Dev split is data/splits/dev.parquet.

Training:
    SFT with masked-prompt loss (AssistantOnlyCollator). Optionally continues
    from a previously trained LoRA adapter via --resume-adapter.

Early stopping signals (checked each eval step):
    * pairwise accuracy > --acc-threshold        (beat prompt-only baseline)
    * macro-F1 > --f1-threshold                  (no class collapse)
    * |P(A) - 0.5| < --balance-tol               (not over-biased to one side)
    All three must hold for N consecutive evals (--patience) to stop.

Usage:
    # single GPU
    python run_judge_pairwise_warmup.py --tier tier_10k

    # 2-GPU DDP (recommended on the 2x-GPU server)
    torchrun --nproc_per_node=2 run_judge_pairwise_warmup.py --tier tier_10k

    # resume from a prior SFT adapter
    torchrun --nproc_per_node=2 run_judge_pairwise_warmup.py --tier tier_10k \\
        --resume-adapter results/checkpoints/judge_sft_tier_10k_r16_lr1e-05/final_adapter
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

import config as cfg
from utils import VANILLA_JUDGE_PROMPT
from train.run_judge_sft import _AssistantOnlyCollator, _apply_liger

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_IS_MAIN = _LOCAL_RANK == 0
if not _IS_MAIN:
    log.setLevel(logging.WARNING)

_apply_liger()


# ───────────────────────── adapter loading ──────────────
def _load_adapter_with_patched_targets(
    model, adapter_path: str, is_trainable: bool = True
):
    """Load a LoRA adapter, overriding target_modules if the saved config
    contains a stale / wrong regex (e.g. from a VLM checkpoint that has a
    'model.language_model' prefix that doesn't exist in a pure-LM base)."""
    import json, shutil, tempfile

    cfg_file = Path(adapter_path) / "adapter_config.json"
    with open(cfg_file, encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    saved_targets = adapter_cfg.get("target_modules", [])
    needs_patch = isinstance(saved_targets, str) or (
        isinstance(saved_targets, list)
        and any("language_model" in str(t) for t in saved_targets)
    )

    if needs_patch:
        log.info(
            "Patching adapter target_modules: %s  →  %s",
            saved_targets, cfg.LORA_TARGET_MODULES,
        )
        adapter_cfg["target_modules"] = list(cfg.LORA_TARGET_MODULES)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copytree(adapter_path, tmpdir, dirs_exist_ok=True)
            with open(Path(tmpdir) / "adapter_config.json", "w", encoding="utf-8") as f:
                json.dump(adapter_cfg, f, indent=2)
            return PeftModel.from_pretrained(model, tmpdir, is_trainable=is_trainable)
    else:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


# ───────────────────────── data ─────────────────────────
def _build_messages(row: pd.Series, swap: bool) -> list[dict]:
    """Build chat messages. If swap=True, swap A/B to reduce position bias."""
    if swap:
        ra, rb = row["response_b"], row["response_a"]
        winner = "A" if row["winner"] == "B" else "B"
    else:
        ra, rb = row["response_a"], row["response_b"]
        winner = row["winner"]
    user = VANILLA_JUDGE_PROMPT.format(
        context=row["context"], response_a=ra, response_b=rb,
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": winner},
    ], winner


def load_pairwise_split(tier: str, split: str, augment_swap: bool, seed: int) -> Dataset:
    if split == "dev":
        path = cfg.SPLITS_DIR / "dev.parquet"
    else:
        path = cfg.SPLITS_DIR / f"train_{tier}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_parquet(path)
    n0 = len(df)
    df = df[df["winner"].isin(["A", "B"])].reset_index(drop=True)
    log.info("[%s] %d rows (%d dropped as tie)", split, len(df), n0 - len(df))

    rng = random.Random(seed)
    records = []
    for _, row in df.iterrows():
        msgs, winner = _build_messages(row, swap=False)
        records.append({"messages": msgs, "label": winner})
        if augment_swap and split != "dev":
            msgs_s, winner_s = _build_messages(row, swap=True)
            records.append({"messages": msgs_s, "label": winner_s})

    if split != "dev":
        rng.shuffle(records)
    labels = [r["label"] for r in records]
    log.info("[%s] A=%d B=%d", split, labels.count("A"), labels.count("B"))
    return Dataset.from_list(records)


# ───────────────────────── eval ─────────────────────────
class PairwiseEvalCallback(TrainerCallback):
    """Evaluate A/B accuracy on dev via teacher-forced next-token logits.

    Uses only the logits over tokens 'A' and 'B' at the assistant-header position,
    which is both fast and matches what the model will emit at inference.
    """

    def __init__(
        self,
        tokenizer,
        dev_ds: Dataset,
        max_length: int,
        max_eval_samples: int,
        acc_threshold: float,
        f1_threshold: float,
        balance_tol: float,
        patience: int,
    ):
        self.tok = tokenizer
        self.dev_ds = dev_ds.select(range(min(max_eval_samples, len(dev_ds))))
        self.max_length = max_length
        self.acc_th = acc_threshold
        self.f1_th = f1_threshold
        self.balance_tol = balance_tol
        self.patience = patience
        self._hits = 0

        self.id_A = tokenizer.encode("A", add_special_tokens=False)[0]
        self.id_B = tokenizer.encode("B", add_special_tokens=False)[0]
        log.info("PairwiseEval: token_id A=%d B=%d  dev_n=%d",
                 self.id_A, self.id_B, len(self.dev_ds))

        self._prompts: list[torch.Tensor] = []
        self._labels: list[int] = []
        for ex in self.dev_ds:
            msgs = ex["messages"][:-1]  # drop assistant
            ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                chat_template_kwargs={"enable_thinking": False},
            )[0]
            if ids.numel() > max_length:
                ids = ids[-max_length:]
            self._prompts.append(ids)
            self._labels.append(0 if ex["label"] == "A" else 1)

    @torch.no_grad()
    def _eval(self, model) -> dict:
        was_training = model.training
        model.eval()
        preds, probs_A = [], []
        device = next(model.parameters()).device
        for ids in self._prompts:
            ids = ids.to(device).unsqueeze(0)
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[0, -1, [self.id_A, self.id_B]]
            p = torch.softmax(logits.float(), dim=-1)
            probs_A.append(p[0].item())
            preds.append(0 if p[0] >= p[1] else 1)
        if was_training:
            model.train()

        y = np.array(self._labels)
        yhat = np.array(preds)
        acc = float((yhat == y).mean())
        # macro-F1 over {A,B}
        f1s = []
        for c in (0, 1):
            tp = int(((yhat == c) & (y == c)).sum())
            fp = int(((yhat == c) & (y != c)).sum())
            fn = int(((yhat != c) & (y == c)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        macro_f1 = float(np.mean(f1s))
        frac_A = float((yhat == 0).mean())
        mean_p_A = float(np.mean(probs_A))
        return {
            "pairwise_acc": acc,
            "macro_f1": macro_f1,
            "frac_pred_A": frac_A,
            "mean_prob_A": mean_p_A,
        }

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        # All ranks run the same forward on the same dev prompts so DDP's
        # forward-pass collectives stay in sync; only rank 0 logs/decides.
        m = self._eval(model)
        if _IS_MAIN:
            log.info(
                "[warmup-eval step=%d] acc=%.4f  macro_f1=%.4f  frac_pred_A=%.3f  p(A)=%.3f",
                state.global_step, m["pairwise_acc"], m["macro_f1"],
                m["frac_pred_A"], m["mean_prob_A"],
            )
            if state.log_history is not None:
                state.log_history.append({"step": state.global_step, **{f"eval_{k}": v for k, v in m.items()}})

        ok = (
            m["pairwise_acc"] > self.acc_th
            and m["macro_f1"] > self.f1_th
            and abs(m["frac_pred_A"] - 0.5) < self.balance_tol
        )
        self._hits = self._hits + 1 if ok else 0
        if self._hits >= self.patience:
            if _IS_MAIN:
                log.info("Early-stop: thresholds held for %d consecutive evals.", self._hits)
            control.should_training_stop = True
        return control


# ───────────────────────── model ─────────────────────────
def load_base(model_id: str, qlora: bool):
    log.info("Loading base: %s (qlora=%s)", model_id, qlora)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True)
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs,
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs,
        )

    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tok


# ───────────────────────── main ─────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", type=str, default="tier_10k")
    p.add_argument("--model-id", type=str, default=str(cfg.JUDGE_MODEL_ID))
    p.add_argument("--resume-adapter", type=str, default=None,
                   help="Path to a previously trained LoRA adapter to continue from.")
    p.add_argument("--merge-resume", action="store_true",
                   help="Merge --resume-adapter into the base before attaching a fresh LoRA. "
                        "Default: keep it as a trainable adapter (no merge).")

    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=cfg.GRADIENT_ACCUMULATION_STEPS)
    p.add_argument("--lora-rank", type=int, default=cfg.LORA_RANK)
    p.add_argument("--lora-alpha", type=int, default=cfg.LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=cfg.LORA_DROPOUT)
    p.add_argument("--max-length", type=int, default=cfg.MAX_LENGTH)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--optim", type=str, default="adamw_torch")

    p.add_argument("--augment-swap", action="store_true", default=True,
                   help="Duplicate each pair with A/B swapped (debias position). Default: on.")
    p.add_argument("--no-augment-swap", dest="augment_swap", action="store_false")

    # eval / early-stop
    p.add_argument("--eval-every", type=int, default=100, help="Eval every N optimizer steps.")
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--acc-threshold", type=float, default=0.70,
                   help="Prompt-only baseline on dev is ~0.66-0.70; set above yours.")
    p.add_argument("--f1-threshold", type=float, default=0.65)
    p.add_argument("--balance-tol", type=float, default=0.12,
                   help="|frac(pred=A) - 0.5| must be below this.")
    p.add_argument("--patience", type=int, default=2)

    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args()

    run_name = args.run_name or f"judge_warmup_{args.tier}_r{args.lora_rank}_lr{args.lr}"
    output_dir = cfg.CHECKPOINTS_DIR / run_name

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT
    if not args.no_tensorboard:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.TENSORBOARD_DIR / run_name)

    # data
    train_ds = load_pairwise_split(args.tier, "train", args.augment_swap, cfg.SEED)
    dev_ds = load_pairwise_split(args.tier, "dev", False, cfg.SEED)

    # model
    model, tok = load_base(args.model_id, qlora=args.qlora)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=cfg.LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )

    if args.resume_adapter:
        log.info("Resuming from adapter: %s (merge=%s)", args.resume_adapter, args.merge_resume)
        model = _load_adapter_with_patched_targets(
            model, args.resume_adapter, is_trainable=not args.merge_resume,
        )
        if args.merge_resume:
            model = model.merge_and_unload()
            log.info("  Merged prior adapter into base; fresh LoRA will be attached by SFTTrainer.")
            pass_peft = peft_config
        else:
            log.info("  Continuing training on the existing LoRA adapter (no fresh attach).")
            pass_peft = None
    else:
        pass_peft = peft_config

    collator = _AssistantOnlyCollator(tok)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    eff_bs = args.batch_size * args.grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

    report_to = []
    if not args.no_wandb:
        report_to.append("wandb")
    if not args.no_tensorboard:
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
        save_steps=max(args.eval_every, steps_per_epoch // 4),
        save_total_limit=2,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        per_device_eval_batch_size=1,
        report_to=report_to,
        seed=cfg.SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if _WORLD_SIZE > 1 else None,
    )

    # NOTE: SFTTrainer's built-in eval_loss is cheap but does not report accuracy;
    # the real signal comes from PairwiseEvalCallback below. Give SFTTrainer a tiny
    # eval slice so eval_strategy=steps fires the callback without being slow.
    eval_slice = dev_ds.select(range(min(32, len(dev_ds))))

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_slice,
        processing_class=tok,
        peft_config=pass_peft,
        data_collator=collator,
    )
    trainer.add_callback(PairwiseEvalCallback(
        tokenizer=tok,
        dev_ds=dev_ds,
        max_length=args.max_length,
        max_eval_samples=args.max_eval_samples,
        acc_threshold=args.acc_threshold,
        f1_threshold=args.f1_threshold,
        balance_tol=args.balance_tol,
        patience=args.patience,
    ))

    log.info("Starting pairwise warm-up: %d train rows, %d dev rows, %d total steps",
             len(train_ds), len(dev_ds), total_steps)
    trainer.train()

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    if _IS_MAIN:
        tok.save_pretrained(str(final_dir))

    if not _IS_MAIN:
        return

    meta = {
        "tier": args.tier, "model_id": args.model_id,
        "resume_adapter": args.resume_adapter, "merge_resume": args.merge_resume,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "augment_swap": args.augment_swap,
        "acc_threshold": args.acc_threshold, "f1_threshold": args.f1_threshold,
        "balance_tol": args.balance_tol, "patience": args.patience,
        "train_samples": len(train_ds), "dev_samples": len(dev_ds),
        "seed": cfg.SEED,
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Saved adapter to %s", final_dir)


if __name__ == "__main__":
    main()
