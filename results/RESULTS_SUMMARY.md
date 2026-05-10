# Results Summary

Compiled from `results/unified_test_20260509/` (test split, n=1073) and `results/robustness/` (dev_600 perturbation sweep). All entries have `parse_rate ≥ 0.99` unless noted. Results with truncation-induced low parse rates are listed at the bottom and excluded from the main tables.

Adapter / checkpoint training data legend:

| Adapter / checkpoint | Base | Method | Training data |
|---|---|---|---|
| `judge_warmup_tier_5k_r16_lr5e-06/final_adapter` | Qwen3.5-9B | DPO warmup (vanilla pairwise judge) | HelpSteer3 DPO pairs, `tier_5k` |
| `joint_dpo_tier_10k_r16_b0.1_lr1e-06_lam1.0/final_adapter` | Qwen3.5-9B | Joint DPO + checklist SFT | HelpSteer3 DPO `tier_10k` + checklist SFT (`λ=1.0`) |
| `judge_sft_swift_..._selfcheck_clean_r16_lr2e-5/checkpoint-265` | Qwen3.5-4B | Self-checklist SFT | `judge_sft` `debug_5k` (clean variant, ~1.5k filtered samples) |
| `judge_sft_swift_..._selfcheck_r16_lr3e-5/checkpoint-200` | Qwen3.5-4B | Self-checklist SFT | `judge_sft` `debug_5k` (unfiltered) |
| `judge_sft_swift_..._selfcheck_r8_lr1e-4/checkpoint-558` | Qwen3.5-4B | Self-checklist SFT | `judge_sft` `debug_5k` (unfiltered, longer run) |
| Pipeline comparative checklist `base` | Qwen3.5-4B (gen) + Qwen3.5-4B (judge) | CheckEval pipeline, no fine-tuning | — |
| Vanilla judge `base` | Qwen3.5-9B | Zero-shot pairwise | — |

---

## Table 1 — Test set (unified_test_20260509, n_total = 1073)

| # | Method | Base | Adapter / source | Real acc | Valid acc | Parse | n_valid |
|---|---|---|---|---:|---:|---:|---:|
| 1 | Self-checklist SFT (ckpt-265, clean) | Qwen3.5-4B | judge_sft debug_5k (clean) | 0.7521 | 0.7620 | 0.9925 | 1059 |
| 2 | Pipeline comparative checklist | Qwen3.5-4B + 4B | base (no FT) | 0.7297 | 0.7304 | 0.9991 | 1072 |
| 3 | Fine-tuned vanilla adapter (DPO) | Qwen3.5-9B | judge_warmup tier_5k | 0.7260 | 0.7267 | 0.9991 | 1072 |
| 4 | Vanilla judge (zero-shot) | Qwen3.5-9B | — | 0.7158 | 0.7164 | 0.9991 | 1072 |
| 5 | Fine-tuned CheckEval adapter (Joint DPO+SFT) | Qwen3.5-9B | joint_dpo tier_10k λ=1.0 | 0.6729 | 0.7315 | 0.9981 | 987 |

`Real acc = Valid acc × Parse` (unparseable counted as wrong). Sorted by real accuracy.

---

## Table 2 — Robustness sweep (dev_600 with perturbations)

Original dev_600 accuracy (real, n=600), reproduced from `results/robustness/SUMMARY.md`:

| Judge | Base | Adapter / source | dev_600 acc |
|---|---|---|---:|
| Vanilla judge | Qwen3.5-9B | — | 0.7467 |
| Fine-tuned vanilla (DPO) | Qwen3.5-9B | judge_warmup tier_5k | 0.7383 |
| Pipeline comparative checklist | Qwen3.5-4B + 4B | base | 0.7333 |
| Self-checklist SFT (ckpt-265, clean) | Qwen3.5-4B | judge_sft debug_5k (clean) | 0.7383 |

Perturbation results (overlapping subset across orig vs. pert prompts):

| Judge | Mode | n | acc_orig | acc_pert | Δacc | Invariance |
|---|---|---:|---:|---:|---:|---:|
| Vanilla | swap | 608 | 0.7484 | 0.7138 | +0.0345 | 0.6678 |
| Vanilla | verbose | 608 | 0.7484 | 0.7829 | −0.0345 | 0.8109 |
| Vanilla | format | 601 | 0.7454 | 0.7537 | −0.0083 | 0.8386 |
| FT vanilla (DPO) | swap | 608 | 0.7418 | 0.7270 | +0.0148 | 0.6678 |
| FT vanilla (DPO) | verbose | 608 | 0.7418 | 0.7829 | −0.0411 | 0.8339 |
| FT vanilla (DPO) | format | 601 | 0.7404 | 0.7587 | −0.0183 | 0.8319 |
| Pipeline cmp checklist | swap | 608 | 0.7336 | 0.7237 | +0.0099 | 0.7862 |
| Pipeline cmp checklist | verbose | 608 | 0.7336 | 0.7500 | −0.0164 | 0.7895 |
| Pipeline cmp checklist | format | 601 | 0.7321 | 0.7338 | −0.0017 | 0.8386 |
| Self-checklist (ckpt-265) | swap | 608 | 0.7385 | 0.7385 | +0.0000 | 0.7451 |
| Self-checklist (ckpt-265) | verbose | 608 | 0.7385 | 0.8224 | −0.0839 | 0.7862 |
| Self-checklist (ckpt-265) | format | 601 | 0.7371 | 0.7205 | +0.0166 | 0.8037 |

Δacc = acc_orig − acc_pert (positive = drop under perturbation). Invariance = fraction of items where prediction unchanged between original and perturbed prompt.

---

## Excluded due to truncation (low parse rate, max_new_tokens insufficient)

These rows are kept here for completeness but are NOT comparable to Table 1 — large fraction of generations were cut off before the judge produced a parseable verdict.

| Method | Base | Adapter / source | Real acc | Valid acc | Parse | n_valid | n_total |
|---|---|---|---:|---:|---:|---:|---:|
| Self-checklist SFT (ckpt-558, 4B SFT, r=8 lr=1e-4) | Qwen3.5-4B | judge_sft debug_5k (unfiltered) | 0.7139 | 0.7897 | 0.9049 | 970 | 1073 |
| Self-checklist SFT (ckpt-200, r=16 lr=3e-5) | Qwen3.5-4B | judge_sft debug_5k (unfiltered) | 0.6980 | 0.7876 | 0.8863 | 951 | 1073 |

The "Valid acc" of these two runs (≈0.79) is on the parseable subset only and overstates true ability because the model failed to terminate within `max_new_tokens=2048` on ≈10–11% of test items. Compare the matched ckpt-265 (clean) run, which used the same flag and reached parse_rate 0.9925.

---

## Source files

- Test set: `results/unified_test_20260509/main_table.md`, `main_table_with_paths.json`, `run_all.sh`
- Robustness: `results/robustness/SUMMARY.md`, `preds_index.tsv`, individual `*_{format,swap,verbose}.json`
- Per-row metrics JSONs: `results/{vanilla_judge,finetuned_vanilla_final_adapter,finetuned_checkeval_final_adapter,pipeline_judge_cmp_base,selfchecklist_checkpoint-265,...}_*_metrics.json`
