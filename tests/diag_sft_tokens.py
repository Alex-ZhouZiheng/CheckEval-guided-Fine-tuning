#!/usr/bin/env python3
"""Diagnose SFT tokenization: why are all labels -100 for real data?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import re
import pandas as pd
from transformers import AutoTokenizer
import config as cfg

tokenizer = AutoTokenizer.from_pretrained(
    str(cfg.JUDGE_MODEL_ID), trust_remote_code=True, padding_side="right",
)

def apply_and_tokenize(messages, *, add_generation_prompt):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    text_stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    ids = tokenizer.encode(text_stripped, add_special_tokens=False)
    return text, text_stripped, ids


sft_path = cfg.CHECKLIST_SFT_DIR / "train_tier_10k.parquet"
df = pd.read_parquet(sft_path)
if "parse_valid" in df.columns:
    df = df[df["parse_valid"]]

print(f"Loaded {len(df)} samples\n")
print(f"SFT_MAX_LENGTH = {cfg.SFT_MAX_LENGTH}\n")

for i in range(min(3, len(df))):
    row = df.iloc[i]
    prompt_text = row["prompt_text"]
    completion_text = row["completion_text"]

    print(f"{'='*70}")
    print(f"Sample {i}")
    print(f"  prompt_text length (chars):     {len(prompt_text)}")
    print(f"  completion_text length (chars):  {len(completion_text)}")
    print(f"  completion_text[:200]: {completion_text[:200]}")
    print()

    full_msgs = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]
    prompt_msgs = [{"role": "user", "content": prompt_text}]

    full_text, full_stripped, full_ids = apply_and_tokenize(full_msgs, add_generation_prompt=False)
    prompt_text_t, prompt_stripped, prompt_ids = apply_and_tokenize(prompt_msgs, add_generation_prompt=True)

    print(f"  full_text    has <think>: {'<think>' in full_text}")
    print(f"  full_stripped has <think>: {'<think>' in full_stripped}")
    print(f"  prompt_text  has <think>: {'<think>' in prompt_text_t}")
    print(f"  prompt_stripped has <think>: {'<think>' in prompt_stripped}")
    print()
    print(f"  full_ids    length: {len(full_ids)}")
    print(f"  prompt_ids  length: {len(prompt_ids)}")
    print(f"  completion tokens:  {len(full_ids) - len(prompt_ids)}")
    print()

    # Check prefix match
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a == b:
            common += 1
        else:
            break
    print(f"  common prefix length: {common}")
    if common < len(prompt_ids) and common < len(full_ids):
        print(f"  divergence at position {common}:")
        print(f"    prompt_ids[{common}] = {prompt_ids[common]} -> '{tokenizer.decode([prompt_ids[common]])}'")
        print(f"    full_ids[{common}]   = {full_ids[common]} -> '{tokenizer.decode([full_ids[common]])}'")

    # Show tail of prompt_stripped and head of full_stripped around boundary
    print(f"\n  prompt_stripped tail (last 200 chars):")
    print(f"    ...{repr(prompt_stripped[-200:])}")
    print(f"\n  full_stripped tail of prompt part + start of completion:")
    # Find where assistant content starts
    assistant_marker = "<|im_start|>assistant"
    idx = full_stripped.rfind(assistant_marker)
    if idx >= 0:
        snippet = full_stripped[idx:idx+300]
        print(f"    {repr(snippet)}")
    print()
