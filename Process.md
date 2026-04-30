# Process: Experiments and Research Directions

Last updated: 2026-04-30

This document summarizes the experiment routes currently visible in this repository, with emphasis on artifacts under `results/`, `data/`, `checklists/`, and `wandb/`. The numbers below are copied from local result files, so they should be read as an inventory of explored directions rather than a single perfectly controlled leaderboard.

Important caveats:

- Runs are not all directly comparable. They differ in checklist bank version, model size, split, scoring rule, `tie_delta`, NA policy, and whether reported accuracy excludes ties.
- In the CheckEval metric JSONs, `parse_rate` usually means the fraction of examples that produce an A/B decision after ties and unparseable outputs are excluded. It is decision coverage, not always checklist-answer parser success. When present, `checklist_parse_rate` is the checklist parser success rate.
- `accuracy` is usually computed over `n_valid`, so higher accuracy with many ties can mean a more selective judge rather than better end-to-end coverage.

## 1. Static Judge Baselines

### Vanilla Pairwise Judge

The plain pairwise judge is the non-checklist baseline.

| Artifact | Split | Model | n_valid / n_total | Accuracy | Parse rate | Notes |
|---|---:|---|---:|---:|---:|---|
| `results/vanilla_judge_test_metrics.json` | test | Qwen3.5-9B | 1072 / 1073 | 0.7183 | 0.9991 | Strong parser coverage but A-position bias around 0.603. |
| `results/finetuned_vanilla_final_adapter_test_2026-04-22_metrics.json` | test | Qwen3.5-9B + vanilla adapter | 1072 / 1073 | 0.7229 | 0.9991 | Slightly above vanilla base, still not close to best checklist route. |
| `results/finetuned_vanilla_dpo_tier_10k_r64_b0.1_lr5e-06_test_metrics.json` | test-like, 1950 rows | merged DPO model | 1950 / 1950 | 0.6477 | 1.0000 | Underperformed; large A bias at 0.611. |

Conclusion: plain DPO or vanilla preference training did not produce the main thesis gain. It remains a baseline, but the task mismatch is visible: the model is trained as an answer preference model while evaluation asks it to behave as a judge.

### Original Static CheckEval Judge

Early full-bank CheckEval used all checklist questions and then scored pairwise.

| Artifact | Split | Bank size | n_valid / n_total | Accuracy | Parse rate | Checklist parse |
|---|---:|---:|---:|---:|---:|---:|
| `results/checkeval_judge_dev_600_metrics.json` | dev_600 | 194 | 521 / 600 | 0.6891 | 0.8683 | 0.9283 |
| `results/checkeval_judge_test_metrics.json` | test | 194 | 668 / 1098 | 0.7036 | 0.6084 | 1.0000 |

Conclusion: the first full-bank checklist judge was too broad and tie-heavy. It did not beat the vanilla judge on test unless the scoring was changed.

## 2. Pairwise NA-Aware Scoring

The strongest route so far is pairwise NA-aware scoring: compare each response on checklist questions, skip or neutralize N/A appropriately, and allow ties when the score margin is too small.

| Artifact | Split | Model | Bank / run label | n_valid / n_total | Accuracy | Parse rate | Tie count | Notes |
|---|---:|---|---|---:|---:|---:|---:|---|
| `results/checkeval_pairwise_naaware_dev_600_metrics.json` | dev_600 | Qwen3.5-9B | older pairwise NA-aware | 406 / 600 | 0.7562 | 0.6767 | 194 | First clear lift over static full-bank scoring. |
| `results/checkeval_pairwise_naaware_dev_6009B_filtered_metrics.json` | dev_600 | Qwen3.5-9B | filtered / 104 Q | 425 / 600 | 0.7600 | 0.7083 | 174 | Better coverage and slightly better accuracy than earlier pairwise run. |
| `results/checkeval_pairwise_naaware_dev_600_v1_q9b_metrics.json` | dev_600 | Qwen3.5-9B | v1 | 418 / 600 | 0.7560 | 0.6967 | 182 | Similar to filtered baseline. |
| `results/checkeval_pairwise_naaware_dev_600_v3_q9b_metrics.json` | dev_600 | Qwen3.5-9B | v3 | 427 / 600 | 0.7705 | 0.7117 | 165 | Best 9B dev run among v1/v3 artifacts. |
| `results/checkeval_pairwise_naaware_dev_600v2_metrics.json` | dev_600 | Qwen3.5-9B | v2 | 411 / 600 | 0.7762 | 0.6850 | 188 | High dev accuracy, but later notes flag possible dev overfit. |
| `results/checkeval_pairwise_naaware_dev_60027B_metrics.json` | dev_600 | Qwen3.5-27B-AWQ | 104 Q | 446 / 600 | 0.7915 | 0.7433 | 154 | Strongest static dev result in `results/`. |
| `results/checkeval_pairwise_naaware_test_metrics.json` | test | Qwen3.5-9B | 104 Q | 807 / 1098 | 0.7856 | 0.7350 | 291 | Strongest local test result for static CheckEval. |
| `results/checkeval_pairwise_naaware_testv2_metrics.json` | test | Qwen3.5-9B | v2 | 798 / 1098 | 0.7719 | 0.7268 | 299 | v2 is worse than the 104-Q test run despite high dev result. |

Conclusion: pairwise NA-aware scoring is the best validated direction. The test-set result `0.7856` over 807 valid examples is the main positive result to preserve. The 27B dev run suggests judge strength helps, but it has not been mirrored as a full test artifact here.

## 3. Checklist Bank Cleaning and v4 Bank Direction

The bank evolved from broad/filtered versions toward smaller, higher-signal banks.

### v2 Cleanup

Artifacts:

- `results/bank_cleanup/meta.json`
- `results/bank_cleanup/before_after.md`
- `results/bank_cleanup/teacher_audit.md`

The cleanup pass started from 194 questions and used teacher-label statistics to flag low-signal or saturated questions. Only 2 items were hard-dropped in the recorded metadata, but the after-bank dimension counts were reduced substantially:

| Dimension | Before | After | Hard dropped | Comment |
|---|---:|---:|---:|---|
| clarity_and_communication | 44 | 31 | 0 | Reduction mainly from filtering / consolidation. |
| coding_communication_conditional | 40 | 20 | 0 | Large reduction. |
| correctness_and_completeness | 44 | 26 | 0 | Large reduction. |
| helpfulness_and_usefulness | 48 | 25 | 2 | Only hard drops recorded here. |

### v3 and v4

`checklists/v4_frozen/SELECTION_RATIONALE.md` records the latest bank-design rationale. v4 was derived from dev_600 oracle analysis on `v3_frozen` with a 9B Q4 judge. The diagnosis found five concrete failure modes:

- Missing tone/style/persona coverage.
- Missing ambiguity and false-premise handling.
- Weak numeric constraint handling.
- Conflict between depth and brevity questions.
- Structural-formatting bias that rewarded surface organization.

v4 edits:

- Deleted 5 questions.
- Deduplicated 7 net questions.
- Rewrote 2 structural-effectiveness questions to be conditional.
- Added 8 questions covering tone/register, ambiguity/premise handling, temporal awareness, and explicit numeric constraints.
- Raw size changed from 63 to 58 questions, with expected flattened qid count around 57-58 after deduplication.

Conclusion: the current research direction is not "make the checklist longer"; it is to make the bank more discriminative, less redundant, and less biased toward surface formatting. v4 is the current default bank for future oracle/selector work.

## 4. Ablation Runs

Ablations tested whether dropping instruction-following style dimensions hurt or helped.

| Artifact | n_valid / n_total | Accuracy | Parse rate | Tie count | Notes |
|---|---:|---:|---:|---:|---|
| `results/ablation/ablation_drop_instruction_following_dev_600_metrics.json` | 418 / 600 | 0.7632 | 0.6967 | 181 | Dropped one instruction-following dimension. |
| `results/ablation/ablation_drop_instruction_following_stem_dev_600_metrics.json` | 417 / 600 | 0.7674 | 0.6950 | 182 | Dropped instruction-following plus STEM-related stem. |

Conclusion: these ablations did not clearly beat v3/v2 static runs or the 27B run. They are useful evidence that some instruction-following questions may be redundant, but not a decisive replacement for careful bank editing.

## 5. Dynamic Checklist Selection

The dynamic route trains or uses a selector to choose a small subset of checklist questions per sample, then runs the judge only on selected questions.

### Selector Inference

Artifacts:

- `results/selector/v4_9b_5k_e2_dev600_top10.meta.json`
- `results/dynamic_eval/sweep_v4_9b_5k_e2_dev600/selector/dev_600_top15.meta.json`

Observed selector speed:

- top-10 selector: 600 dev samples in 10.02 s, about 59.88 samples/s.
- top-15 selector: 600 dev samples in 9.94 s, about 60.37 samples/s.

This confirms the selector itself is not the bottleneck. The cost sits in judge inference.

### Dynamic Eval Sweep

Primary sweep artifact:

- `results/dynamic_eval/sweep_v4_9b_5k_e2_dev600/sweep_summary.csv`

Representative dynamic results:

| k | tie_delta | n_valid / n_total | Valid accuracy | Effective accuracy | Tie rate | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 15 | 0.00 | 486 / 600 | 0.7654 | 0.6200 | 0.1900 | Highest coverage among top-k dynamic runs. |
| 15 | 0.05 | 478 / 600 | 0.7720 | 0.6150 | 0.2033 | Slightly higher valid accuracy, slightly lower coverage. |
| 15 | 0.08 | 406 / 600 | 0.8005 | 0.5417 | 0.3233 | Accuracy rises because the judge abstains more. |
| 15 | 0.10 | 397 / 600 | 0.8035 | 0.5317 | 0.3383 | Best valid accuracy, but too selective. |
| 12 | 0.05 | 460 / 600 | 0.7761 | 0.5950 | 0.2333 | Middle ground. |
| 10 | 0.05 | 449 / 600 | 0.7773 | 0.5817 | 0.2517 | More compact but less coverage. |
| 8 | 0.08 | 440 / 600 | 0.7750 | 0.5683 | 0.2667 | No clear win over larger k. |

Conclusion: dynamic selection is fast and promising as a cost-control method, but current accuracy does not beat the best static pairwise NA-aware result. The most defensible dynamic setting is top-15 with low or zero `tie_delta` if coverage matters; high `tie_delta` produces attractive valid accuracy by abstaining too often.

### v4 Human-Relevance Selector Dynamic Eval

On 2026-04-30, the selector checkpoint
`results/checkpoints/selector_v4bank_v4hr_noOracleFilter/best` was evaluated on `dev_600`
with `checklists/v4_frozen` and `compare_checklists_pairwise` scoring. Selector picks came
from `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_dev600_k20_picks.parquet`.

| Artifact | Policy | k / avg_k | tie_delta | n_valid / n_total | Valid accuracy | Effective accuracy | Tie rate | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k10_pairwise/metrics.json` | learned_topk | 10 / 10.00 | 0.05 | 398 / 600 | 0.7940 | 0.5267 | 0.3367 | Highest learned valid accuracy, but many ties. |
| `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k10_td0.0_pairwise/metrics.json` | learned_topk | 10 / 10.00 | 0.00 | 417 / 600 | 0.7794 | 0.5417 | 0.3050 | Lower abstention, lower valid accuracy than `td=0.05`. |
| `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_pairwise/metrics.json` | learned_topk | 15 / 15.00 | 0.05 | 439 / 600 | 0.7790 | 0.5700 | 0.2683 | Best learned fixed-k tradeoff by effective accuracy. |
| `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k20_td005_pairwise/metrics.json` | learned_topk | 20 / 20.00 | 0.05 | 405 / 600 | 0.7556 | 0.5100 | 0.3250 | Adding more selected questions degraded both accuracy and coverage. |
| `results/dynamic_dev_600/selector_v4bank_v4hr_noOracleFilter_best_k10_escalate_td005_pairwise/metrics.json` | learned_topk_escalate | 10 / 13.93 | 0.05 | 444 / 600 | 0.7680 | 0.5683 | 0.2600 | Nearly matches k15 effective accuracy with slightly fewer questions on average. |
| `results/dynamic_dev_600/random_k15_v4_td005_pairwise/metrics.json` | random_k | 15 / 15.00 | 0.05 | 411 / 600 | 0.7299 | 0.5000 | 0.3150 | Confirms learned selector ranking is useful at the same question budget. |
| `results/dynamic_dev_600/static_v4_td005_pairwise/metrics.json` | static_v3 | all / 49.22 | 0.05 | 414 / 600 | 0.8043 | 0.5550 | 0.3100 | Best valid accuracy, but much higher question cost and lower effective accuracy than learned k15. |

Interpretation:

- The current selector should not be trained further before more evaluation analysis. It already beats random selection at the same `k=15` budget by about 4.9 valid-accuracy points and 7.0 effective-accuracy points.
- `k=20` is not a free improvement. It likely adds lower-signal or conflicting questions that increase noise.
- For reporting a compact dynamic setting, use learned `k=15, tie_delta=0.05`: it has the best effective accuracy among fixed learned top-k runs.
- For a cost-aware adaptive setting, use `learned_topk_escalate` from `k=10`: it reaches almost the same effective accuracy as learned k15 while asking 13.93 questions on average.
- Static full-bank v4 remains the high valid-accuracy reference, but it asks about 49 questions on average. Dynamic selection is therefore best framed as a cost-control method that preserves or improves effective coverage, not as a strict valid-accuracy win over full-bank static judging.

### Test Human-Relevance Oracle and Importance Weighting

On 2026-04-30, the test split was used to separate two questions:

1. whether human-relevance qids are a useful selection target;
2. whether selected qids should receive context-conditioned importance weights.

Human-relevance oracle qids came from
`data/oracle/test_human_relevance_v4_qwen36_27b_nvfp4.parquet`; the ranked picks were saved to
`results/dynamic_test/human_relevance_oracle_test_v4_qwen36_27b_nvfp4_ranked_picks.parquet`.
Question-importance weights were generated with `models/qwen3.6-27B-nvfp4` from context plus the selected
questions only, then normalized with softmax.

| Artifact | Setting | tie_delta | n_valid / n_total | Valid accuracy | Effective accuracy | Tie rate | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `results/dynamic_test/static_v4_td005_pairwise/metrics.json` | static full v4 | 0.05 | 750 / 1073 | 0.7840 | 0.5480 | 0.3010 | Full-bank reference, avg_k 49.13. |
| `results/dynamic_test/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_pairwise/metrics.json` | learned k15 uniform | 0.05 | 759 / 1073 | 0.7563 | 0.5349 | 0.2926 | Current learned selector baseline. |
| `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_pairwise/metrics.json` | HR-oracle k15 uniform | 0.05 | 765 / 1073 | 0.7935 | 0.5657 | 0.2870 | Shows human-relevance qids are a strong target. |
| `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise/metrics_td0_rescored.json` | HR-oracle k15 weighted | 0.00 | 773 / 1073 | 0.7943 | 0.5722 | 0.2796 | Best weighted HR-oracle point from threshold sweep. |
| `results/dynamic_test/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_weighted_pairwise/metrics_td0_rescored.json` | learned k15 weighted | 0.00 | 777 / 1073 | 0.7645 | 0.5536 | 0.2759 | Weighted learned selector improves effective accuracy over learned uniform. |
| `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise/metrics.json` | HR-oracle k15 weighted | 0.05 | 648 / 1073 | 0.8164 | 0.4930 | 0.3961 | Valid accuracy rises, but abstention is too high. |
| `results/dynamic_test/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_weighted_pairwise/metrics.json` | learned k15 weighted | 0.05 | 642 / 1073 | 0.7850 | 0.4697 | 0.4017 | Same over-abstention pattern. |

Weight artifacts:

- `results/dynamic_test/question_weights/hr_oracle_k15_qwen36_27b_weights.parquet`
  - parse_ok_rate 0.9664, avg_missing_qids 0.512.
- `results/dynamic_test/question_weights/learned_k15_qwen36_27b_weights.parquet`
  - parse_ok_rate 0.9907, avg_missing_qids 0.141.
- Threshold sweeps were saved as `threshold_sweep.json` in each weighted result directory.

Interpretation:

- Human-relevance qids are validated as a useful selection target: HR-oracle k15 beats static full v4 while asking far fewer questions.
- Context-conditioned importance weighting is mildly positive only after threshold calibration. With `tie_delta=0`, HR-oracle weighted improves effective accuracy from 0.5657 to 0.5722, and learned weighted improves from 0.5349 to 0.5536.
- `tie_delta=0.05` is not appropriate for the weighted pairwise margin because weights shrink many margins toward zero, creating too many ties.
- Next selector work should focus on closing the retrieval gap to HR-oracle. Weighting can be kept as an ablation or secondary method, but not yet as the main claim unless calibrated thresholds are part of the method.

## 6. Stronger Judge and HTTP Judge Tests

DeepSeek V4 Pro was tested through the HTTP judge path, including reasoning-mode handling.

Artifacts:

- `results/dynamic_eval/deepseek_v4_pro_reasoning_learned_k15_debug_raw_fields/metrics.json`
- `results/dynamic_eval/deepseek_v4_pro_reasoning_static_debug_raw/metrics.json`
- `results/dynamic_eval/deepseek_v4_pro_reasoning_learned_k15_dev600_n10/metrics.json`
- `results/dynamic_eval/deepseek_v4_pro_static_dev600_n50/metrics.json`
- `results/remote_inspect/deepseek_v4_pro_reasoning_learned_k15_dev600_n10/metrics.json`

Observed local/remote probes:

| Artifact | n_total | Parse ok | Accuracy | Tie rate | Notes |
|---|---:|---:|---:|---:|---|
| `deepseek_v4_pro_reasoning_learned_k15_debug_raw_fields` | 2 | 1.0000 | 1.0000 | 0.0000 | Raw-field/debug path succeeded. |
| `deepseek_v4_pro_reasoning_static_debug_raw` | 2 | 1.0000 | 0.5000 | 0.0000 | Static debug path parsed. |
| `deepseek_v4_pro_reasoning_learned_k15_dev600_n10` | 2 recorded | 0.0000 | n/a | 1.0000 | Reasoning/dev probe collapsed to ties in recorded metrics. |
| `deepseek_v4_pro_static_dev600_n50` | 50 | 0.0000 | n/a | 1.0000 | Static n=50 run recorded all ties / parse failure. |

Conclusion: the HTTP judge integration and raw reasoning capture were explored, but the recorded larger probes are not yet reliable evaluation results. They are useful for debugging the judge-vs-selector bottleneck, not yet for claiming DeepSeek improves the pipeline.

## 7. OpenAI / ChatGPT Teacher Comparison

Artifacts:

- `results/teacher_comparison/comparison_dev_metrics.json`
- `results/teacher_comparison/comparison_dev_600_metrics.json`

Small code-only dev comparison:

| Teacher | n_total | n_evaluated | Ties | Accuracy excl. ties | Accuracy incl. ties |
|---|---:|---:|---:|---:|---:|
| gpt-5.4-mini | 50 | 42 | 8 | 0.6905 | 0.5800 |
| gpt-4o-mini | 50 | 19 | 31 | 0.6842 | 0.2600 |

Full dev_600 comparison:

| Teacher | n_total | n_evaluated | Ties | Accuracy excl. ties | Accuracy incl. ties | Time |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4-mini | 600 | 470 | 130 | 0.7447 | 0.5833 | 3000.6 s |

Conclusion: gpt-5.4-mini has clean parse behavior and reasonable excluding-ties accuracy, but it remains tie-heavy and lower than the best local pairwise NA-aware CheckEval result.

## 8. Teacher Review / Human Reasoning Review Artifacts

Teacher-review and review-app artifacts were built to inspect failures, not only aggregate metrics.

Artifacts:

- `results/teacher_review_dev_n100_Qwen3.5-9B_naskip_td0.05_metrics.json`
- `results/teacher_review_dev_n200_Qwen3.5-9B_naskip_td0.05_metrics.json`
- `results/review/dynamic_learned_topk_k15_td0_dev600_full_raw.parquet`
- `results/review/dynamic_learned_topk_k15_td0_dev600_full_raw.meta.json`
- `results/oracle_review_analysis/row_summary.csv`
- `results/oracle_review_analysis/differing_questions.csv`
- `results/oracle_review_analysis/question_diff_stats.csv`

Teacher review metrics:

| Artifact | n_valid / n_total | Accuracy | Parse rate | Tie count | Notes |
|---|---:|---:|---:|---:|---|
| `teacher_review_dev_n100...` | 78 / 100 | 0.8333 | 0.7800 | 22 | Small sample, code-only in per-domain output. |
| `teacher_review_dev_n200...` | 146 / 200 | 0.7671 | 0.7300 | 54 | More representative but lower. |

Full dynamic review export:

- `results/review/dynamic_learned_topk_k15_td0_dev600_full_raw.meta.json` contains 600 rows.
- It joins human reasoning for all 600 rows from `helpsteer3_train.parquet`.
- It marks 114 wrong, 114 tie, and 372 correct cases.
- It preserves raw judge outputs, parsed outputs, and selected checklist questions for review.

Conclusion: review tooling became a necessary part of the research loop. Aggregate metrics alone were not enough to diagnose bank gaps, dynamic selector picks, or why human preferences differed from the checklist judgement.

## 9. Generated Checklist Pipeline

The generated-checklist route asks a generator model to produce per-sample checklists, then a judge model scores responses using those generated checklists.

Artifacts:

- `results/pipeline_judge_base_dev_600_2026-04-18_metrics.json`
- `results/pipeline_judge_base_dev_600_4Bgen9Bjudge_metrics.json`
- `results/pipeline_judge_base_dev_600_9Bgen4Bjudge_metrics.json`
- `results/pipeline_judge_checkpoint-282_dev_600_2026-04-19_metrics.json`
- `results/pipeline_judge_checkpoint-282_dev_600_2026-04-21_metrics.json`

| Artifact | Generator / judge setup | n_valid / n_total | Accuracy | Parse rate | Tie count |
|---|---|---:|---:|---:|---:|
| `pipeline_judge_base_dev_600_2026-04-18` | generated checklist + 9B judge | 486 / 600 | 0.7675 | 1.0000 | 114 |
| `pipeline_judge_base_dev_600_4Bgen9Bjudge` | 4B generator, 9B judge | 491 / 600 | 0.7699 | 1.0000 | 109 |
| `pipeline_judge_base_dev_600_9Bgen4Bjudge` | 9B generator, 4B judge | 491 / 600 | 0.7475 | 1.0000 | 109 |
| `pipeline_judge_checkpoint-282_dev_600_2026-04-19` | generator checkpoint 282 | 495 / 600 | 0.7657 | 1.0000 | 105 |
| `pipeline_judge_checkpoint-282_dev_600_2026-04-21` | generated source checkpoint-100 | 492 / 600 | 0.7622 | 1.0000 | 108 |

Conclusion: generated checklists work technically and reach the mid/high 0.76 range on dev_600, but they do not yet outperform the best static pairwise NA-aware bank. The stronger direction is improving generated checklist quality, not simply adding more generated questions.

## 10. Oracle Label Generation

Oracle labels are used to train the selector and to analyze bank usefulness. The current artifacts live mainly under `data/oracle/`.

Key artifacts:

- `data/oracle/train_10k_oracle_v4_9b.meta.json`
- `data/oracle/train_10k_oracle_v4_9b_flash_thinking8192_merged.meta.json`
- `data/oracle/train_10k_oracle_v4_flash.meta.json`
- `data/oracle/train_10k_oracle_v4_flash_then_thinking8192.meta.json`

Main 9B v4 oracle run:

- `n_samples`: 9551
- `n_valid`: 8897
- `n_tie`: 548
- `n_unparseable`: 106
- `parse_ok_rate`: 0.9889
- `avg_questions_per_sample`: 49.11
- `oracle_agreement_rate_valid`: 0.6903
- `oracle_agreement_rate_total`: 0.6503
- Runtime: 6856.5 s
- Throughput: 1.393 samples/s

Merged flash + thinking rerun:

- Started from `train_10k_oracle_v4_9b.parquet`.
- Overwrote 1970 touched samples using `train_10k_oracle_v4_flash_then_thinking8192.parquet`.
- Improved valid oracle agreement from 0.6903 to 0.8018.
- Improved total oracle agreement from 0.6503 to 0.7745.
- Final counts: `n_valid=9173`, `n_tie=323`, `n_unparseable=55`.

DeepSeek / HTTP probes:

- Thinking-off probe of 20 rows: valid agreement 0.3158.
- Thinking-on probe of 20 rows: valid agreement 0.6316 but much slower.
- Several probe2 thinking runs were unparseable until raw reasoning/content handling was debugged.

Conclusion: oracle generation is central but fragile. Reasoning-enabled reruns can greatly improve agreement on hard subsets, but HTTP judge outputs need careful parsing and resumability. For `train_tier_10k`, the workload is effectively pairwise A/B, so it behaves like about 20k judgement prompts.

## 11. Checklist-SFT / OpenAI Batch Data

Artifacts:

- `data/checklist_sft/train_tier_10k_teacher_openai_batch.meta.json`
- `data/checklist_sft/batch_jobs/gpt-5.4-mini/tier_10k/batch_status.json`
- `data/checklist_sft/batch_jobs/gpt-5.4-mini/tier_10k/shard_index.json`

Current observed state:

- Target workload: `tier_10k`, 20,000 requests.
- Teacher backend: OpenAI Batch.
- Model: `gpt-5.4-mini`.
- Shards: 320.
- `batch_status.json` records status as `running`.
- `shard_index.json` records 320 shards, 20,000 requests, and a migrated target estimated-token budget.
- The older aggregate meta file reports `n_valid=0`, `parse_rate=0.0`, and `n_missing_outputs=20000`, so it should not be treated as usable training data.

Conclusion: checklist-SFT data generation is in progress / partially staged. Existing final meta is not a valid dataset; future joint/SFT runs must check shard completion and parsed-valid row count before training.

## 12. Fine-Tuning and Joint Training

Training logs are visible under `wandb/` and test metrics under `results/`.

### DPO Runs

Representative W&B summaries:

| Run | Output dir | Final train loss | Eval loss | Eval reward accuracy | Notes |
|---|---|---:|---:|---:|---|
| `run-20260408_195815-zkbh7sof` | `dpo_tier_10k_r16_b0.1_lr1e-06` | 0.6845 | 0.6883 | 0.5324 | Completed 1 epoch. |
| `run-20260410_175344-5z82zpia` | `0.2` | 0.6572 | 0.6467 | 0.6050 | Completed 3 epochs, stronger internal reward metrics. |

But the downstream vanilla DPO merged model test result was weak:

- `results/finetuned_vanilla_dpo_tier_10k_r64_b0.1_lr5e-06_test_metrics.json`
- Accuracy 0.6477 over 1950 valid rows.

Conclusion: DPO improves internal reward metrics but does not transfer cleanly to evaluator behavior.

### Joint DPO + Checklist SFT

Representative W&B summaries:

| Run | Output dir | Train loss | DPO loss | SFT loss | Eval reward accuracy | Notes |
|---|---|---:|---:|---:|---:|---|
| `run-20260414_020340-iy3eyaqf` | `joint_dpo_tier_10k_r16_b0.1_lr1e-06_lam1.0` | 0.0000 | 0.6930 | NaN | 0.4837 | Broken SFT loss path. |
| `run-20260414_055016-ansonjvq` | same | 1.5401 | 0.7059 | 0.7706 | 0.4778 | Completed but reward accuracy under DPO baseline. |

Downstream CheckEval adapter evaluation:

| Artifact | n_valid / n_total | Accuracy | Parse rate | Notes |
|---|---:|---:|---:|---|
| `results/finetuned_checkeval_final_adapter_test_2026-04-14_metrics.json` | 0 / 1098 | n/a | 0.0000 | All predictions unparseable. |
| `results/finetuned_checkeval_final_adapter_test_2026-04-15_metrics.json` | 1053 / 1098 | 0.7331 | 1.0000 | Fixed parsing, but still below best static CheckEval. |

Conclusion: joint training is not dead, but the current artifacts do not show a win. The main blockers are data validity and objective alignment, especially the broken or unfinished checklist-SFT teacher data.

## 13. GRPO / Reward Audit

The GRPO route was evaluated through an offline reward audit before committing to a larger RL story.

Artifacts:

- `results/reward_audit/smoke_test/summary_metrics.json`
- `results/reward_audit/smoke_test/perturbation_metrics.json`
- `results/reward_audit/smoke_test/replay_pool_meta.json`
- `results/reward_audit/smoke_test/score_meta.json`

Summary metrics:

- `n_primary_rows`: 80
- `parse_rate`: 0.9875
- `nan_rate`: 0.0
- `exception_rate`: 0.0
- `best_of_k_accuracy`: 0.9000
- `reward_selected_candidate_accuracy`: 0.9000
- mean pairwise ranking accuracy: 0.6807
- zero-variance group rate: 0.15
- reward-length correlation: -0.0967
- reward-question-count correlation: -0.0677

Perturbation metrics:

| Perturbation | Count | Parse rate | Mean reward delta | Notes |
|---|---:|---:|---:|---|
| content_damage | 80 | 1.0000 | 0.0894 | Worrying: content damage can increase reward on average. |
| format_only | 80 | 0.9875 | -0.0378 | Mostly neutral/slightly negative. |
| length_only | 80 | 0.9875 | -0.0376 | Mostly neutral/slightly negative. |

Conclusion: the reward pipeline runs and has no NaN/exception issue in the smoke test, but content-damage perturbations are not penalized reliably. This weakens the case for scaling GRPO before the reward is made more content-sensitive.

## 14. Current Best Evidence

The clearest positive findings so far:

1. Pairwise NA-aware CheckEval with the 104-question bank is the best validated route.
   - Test: `results/checkeval_pairwise_naaware_test_metrics.json`
   - Accuracy: 0.7856 over 807 valid examples.
   - Coverage/parse rate: 0.7350, with 291 ties.

2. A stronger judge helps static CheckEval on dev.
   - Dev: `results/checkeval_pairwise_naaware_dev_60027B_metrics.json`
   - Accuracy: 0.7915 over 446 valid examples.
   - This suggests model quality is a real bottleneck, but it still needs a comparable test run.

3. Dynamic selection is computationally viable but not yet a quality win.
   - Selector speed is about 60 samples/s on dev_600.
   - top-15 dynamic selection at `tie_delta=0` reaches 0.7654 valid accuracy and 0.8100 coverage.
   - higher `tie_delta` reaches 0.8035 valid accuracy only by raising tie rate to 0.3383.

4. Generated checklists are viable but currently plateau around 0.76-0.77 valid accuracy on dev_600.
   - Best generated-checklist artifact: `pipeline_judge_base_dev_600_4Bgen9Bjudge`, accuracy 0.7699 over 491 valid examples.

5. Fine-tuning routes have not yet beaten the static checklist method.
   - Vanilla DPO merged test: 0.6477.
   - Joint DPO+SFT final adapter after parser fix: 0.7331.
   - Both are below static pairwise NA-aware CheckEval on test.

## 15. Research Directions Explored

### Direction A: Static Checklist-Guided Judging

Status: strongest validated direction.

What was tried:

- Full 194-question bank.
- Filtered / 104-question bank.
- v1/v2/v3 comparisons.
- v4 design from oracle error analysis.
- Pairwise NA-aware scoring.
- 9B and 27B judge sizes.

What we learned:

- Static checklist quality matters more than raw question count.
- NA-aware pairwise scoring is critical.
- v2 may overfit dev; test favors the 104-question bank over v2.
- 27B improves dev performance and should be evaluated on a controlled test setup.

### Direction B: Adaptive / Dynamic Checklist Selection

Status: promising for efficiency, not yet best for accuracy.

What was tried:

- Learned selector from `selector_v4_9b_5k_no_test_cleanrank_e2`.
- top-k values 8, 10, 12, 15.
- tie_delta sweep 0, 0.03, 0.05, 0.08, 0.10.
- Review export with selected qids and raw judge outputs.

What we learned:

- The selector is fast enough.
- top-15 is the most defensible quality/coverage point.
- high `tie_delta` inflates valid accuracy by abstaining.
- Current selector quality is not clearly above a well-curated static bank.

### Direction C: Stronger External Judges

Status: integration/debugging route, not yet validated result route.

What was tried:

- DeepSeek V4 Pro HTTP judge.
- Reasoning mode and raw reasoning-content capture.
- static and learned top-k probes.
- OpenAI/ChatGPT teacher comparisons.

What we learned:

- Reasoning-enabled HTTP outputs require preserving more than `message.content`.
- DeepSeek probes in current metrics often collapsed to ties or parse failures outside tiny debug runs.
- gpt-5.4-mini is parse-stable but tie-heavy and not obviously superior to the local CheckEval route.

### Direction D: Generated Checklists

Status: technically working, needs quality improvement.

What was tried:

- generated checklist per sample.
- 4B generator with 9B judge.
- 9B generator with 4B judge.
- generator checkpoints.

What we learned:

- 4B generator + 9B judge slightly beat the base generated-checklist setup.
- 4B judge hurt quality.
- Generator checkpoints did not show a clear improvement over base generation.

### Direction E: Fine-Tuning Judges

Status: current artifacts do not validate it as the main path.

What was tried:

- vanilla DPO.
- joint DPO + checklist SFT.
- final-adapter evaluation in vanilla and CheckEval modes.
- W&B/TensorBoard-style diagnosis of losses.

What we learned:

- DPO reward accuracy can improve without improving downstream judge accuracy.
- Joint training was blocked by SFT data validity and at least one NaN SFT-loss run.
- The best joint adapter test result is still below static CheckEval.

### Direction F: GRPO / RL

Status: should stay secondary until reward quality is fixed.

What was tried:

- replay pool scoring.
- best-of-k analysis.
- perturbation audit.

What we learned:

- The reward is parse-stable in the smoke test.
- It can select high-reward candidates, but content-damage perturbations are not reliably punished.
- Scaling RL before fixing reward sensitivity would be risky.

## 16. Suggested Next Order

Based only on current artifacts, the cleanest next steps are:

1. Treat `results/checkeval_pairwise_naaware_test_metrics.json` as the current main test baseline until the v4 dev findings are mirrored on test.
2. Carry the controlled v4 comparison to test: static full-bank v4, learned `k=15`, and learned `k10_escalate`, all with `compare_checklists_pairwise` and `tie_delta=0.05`.
3. Use learned `k=15` as the fixed-budget dynamic branch and `k10_escalate` as the adaptive-cost branch; do not keep increasing `k` without inspecting which added questions create noise.
4. Stop selector training for now. The current selector already beats random selection at the same `k=15` budget; the next bottleneck is scoring/noisy-question analysis and test transfer.
5. Run the 27B pairwise NA-aware setup on test or a fixed random probe to check whether the dev gain transfers.
6. Do not resume joint training from auto-detected checklist-SFT data until `data/checklist_sft/...` reports nonzero parsed-valid rows.
7. Keep GRPO exploratory until content-damage perturbations lower reward consistently.
