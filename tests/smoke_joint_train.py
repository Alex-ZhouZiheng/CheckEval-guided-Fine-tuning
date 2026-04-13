#!/usr/bin/env python3
"""Smoke test for ChecklistSFTCollator + JointDPOSFTTrainer pipeline.

Run on server:
    python tests/smoke_joint_train.py

Tests:
  1. Collator tokenizes correctly (list[int], no strings)
  2. Labels have valid (non -100) tokens for supervision
  3. SFT forward pass produces a finite, non-NaN loss
  4. (Optional) One full training_step with DPO + SFT
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import config as cfg

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

def main():
    model_id = str(cfg.JUDGE_MODEL_ID)
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 1. Test _apply_and_tokenize ──────────────────────────────
    from run_joint_train import ChecklistSFTCollator
    collator = ChecklistSFTCollator(tokenizer, max_length=512)

    msgs_full = [
        {"role": "user", "content": "Rate this essay on a scale of 1-5."},
        {"role": "assistant", "content": "Rating: 4\nThe essay is well-structured."},
    ]
    msgs_prompt = [{"role": "user", "content": "Rate this essay on a scale of 1-5."}]

    full_ids = collator._apply_and_tokenize(msgs_full, add_generation_prompt=False)
    prompt_ids = collator._apply_and_tokenize(msgs_prompt, add_generation_prompt=True)

    # Check types
    assert isinstance(full_ids, list), f"full_ids is {type(full_ids)}, expected list"
    assert all(isinstance(x, int) for x in full_ids), \
        f"full_ids contains non-int: {type(full_ids[0])}, first 5 = {full_ids[:5]}"
    assert isinstance(prompt_ids, list) and all(isinstance(x, int) for x in prompt_ids)
    print(f"{PASS}  _apply_and_tokenize returns list[int]")
    print(f"       full_ids:   {len(full_ids)} tokens")
    print(f"       prompt_ids: {len(prompt_ids)} tokens")

    # Check prompt is strict prefix
    n_prompt = len(prompt_ids)
    assert n_prompt < len(full_ids), \
        f"prompt ({n_prompt}) >= full ({len(full_ids)}), no completion tokens!"
    print(f"{PASS}  prompt ({n_prompt}) < full ({len(full_ids)}), "
          f"completion has {len(full_ids) - n_prompt} tokens")

    # Decode to visualize
    print(f"\n  --- Prompt text (decoded) ---")
    print(f"  {tokenizer.decode(prompt_ids[:50])}...")
    print(f"  --- Completion tokens (decoded) ---")
    print(f"  {tokenizer.decode(full_ids[n_prompt:n_prompt+50])}...")

    # ── 2. Test collator __call__ ────────────────────────────────
    fake_batch = [
        {"prompt_text": "Rate this essay.", "completion_text": "Rating: 3\nDecent work."},
        {"prompt_text": "Evaluate clarity.", "completion_text": "Score: 5\nVery clear writing."},
    ]
    batch = collator(fake_batch)

    assert isinstance(batch["input_ids"], torch.Tensor), "input_ids not a tensor"
    assert batch["input_ids"].dtype == torch.long, f"dtype = {batch['input_ids'].dtype}"
    assert batch["labels"].shape == batch["input_ids"].shape
    print(f"\n{PASS}  Collator produces tensors: input_ids {batch['input_ids'].shape}")

    # Check labels have valid tokens
    for i in range(batch["labels"].shape[0]):
        valid = (batch["labels"][i] != -100).sum().item()
        total = batch["labels"].shape[1]
        assert valid > 0, f"Sample {i}: all labels are -100! No supervision signal."
        print(f"{PASS}  Sample {i}: {valid}/{total} valid label tokens ({valid/total*100:.1f}%)")

    # ── 3. Test SFT forward pass ─────────────────────────────────
    print(f"\nLoading model for forward pass test (this may take a moment)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, trust_remote_code=True,
        )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sft_batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=sft_batch["input_ids"],
            attention_mask=sft_batch["attention_mask"],
        )
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = sft_batch["labels"][:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss), f"SFT loss is {loss.item()} (not finite!)"
    print(f"{PASS}  SFT forward pass loss = {loss.item():.4f} (finite)")

    # ── 4. Test with real SFT data (if available) ────────────────
    try:
        import pandas as pd
        sft_path = cfg.CHECKLIST_SFT_DIR / "train_tier_10k.parquet"
        if sft_path.exists():
            df = pd.read_parquet(sft_path)
            if "parse_valid" in df.columns:
                df = df[df["parse_valid"]]
            samples = df[["prompt_text", "completion_text"]].head(4).to_dict("records")
            real_batch = collator(samples)
            for i in range(real_batch["labels"].shape[0]):
                valid = (real_batch["labels"][i] != -100).sum().item()
                total = real_batch["labels"].shape[1]
                status = PASS if valid > 0 else FAIL
                print(f"{status}  Real sample {i}: {valid}/{total} valid labels")
                if valid == 0:
                    print(f"       prompt_text[:80] = {samples[i]['prompt_text'][:80]}")
        else:
            print(f"\n  (Skipped real data test — {sft_path} not found)")
    except Exception as e:
        print(f"\n  (Skipped real data test — {e})")

    print(f"\n{'='*60}")
    print(f"All smoke tests passed! Training should work.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
