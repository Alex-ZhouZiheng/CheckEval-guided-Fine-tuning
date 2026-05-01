# Dynamic Evaluation Summary, Test Split

Source artifacts are local metrics files under `results/dynamic_test/`.

| Method | Avg. Q | Valid | Accuracy | Macro-F1 | Tie rate | Effective accuracy | HR recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static full bank | 49.13 | 750 | 78.40 | 78.40 | 30.10 | 54.80 | -- |
| Learned selector, top-15 | 15.00 | 759 | 75.63 | 75.61 | 29.26 | 53.49 | -- |
| Human-relevance oracle, top-15 | 15.00 | 765 | 79.35 | 79.31 | 28.70 | 56.57 | 93.22 |
| Learned selector + weights, td=0.05 | 15.00 | 642 | 78.50 | 78.48 | 40.17 | 46.97 | 72.17 |
| Human-relevance oracle + weights, td=0.05 | 15.00 | 648 | 81.64 | 81.63 | 39.61 | 49.30 | 93.22 |
| Learned selector + weights, td=0 diagnostic | 15.00 | 777 | 76.45 | 76.44 | 27.59 | 55.36 | 72.17 |
| Human-relevance oracle + weights, td=0 diagnostic | 15.00 | 773 | 79.43 | 79.43 | 27.96 | 57.22 | 93.22 |

Interpretation: human-relevance oracle selection is a strong upper bound; learned selector retrieval is the main gap; weighted aggregation needs dev-calibrated tie thresholds before it can be used as a primary test claim.

## Lever 2.1 Tiebreak Second-Pass

Source artifacts:

- `results/dynamic_test/deepseek_v4_thinking_tiebreak_hroracle_weighted/metrics.json`
- `results/dynamic_test/deepseek_v4_thinking_tiebreak_hroracle_weighted/predictions.parquet`
- `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise/predictions.parquet`

This run re-evaluates the 425 samples that the HR-oracle + weighted `td=0.05` full run originally predicted as `Tie`.

| Scope | Total | Valid | Predicted Tie | Accuracy | Tie rate | Effective accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Tiebreak subset | 425 | 242 | 183 | 72.31 | 43.06 | 41.18 |
| Full run before tiebreak | 1073 | 648 | 425 | 81.64 | 39.61 | 49.30 |
| Full run after tiebreak merge | 1073 | 890 | 183 | 79.10 | 17.05 | 65.61 |

Tiebreak subset accounting:

- Gold labels are balanced: `B=213`, `A=212`.
- Predictions: `Tie=183`, `B=129`, `A=113`.
- Correct non-tie decisions: `175`.
- Wrong non-tie decisions: `67`.
- Remaining abstentions: `183`.
- Valid decision accuracy: `175 / 242 = 72.31%`.
- Effective subset accuracy, counting remaining `Tie` predictions as wrong: `175 / 425 = 41.18%`.
- Majority baseline on this subset is `213 / 425 = 50.12%`.

Confusion matrix on the tiebreak subset:

| Gold winner | Pred A | Pred B | Pred Tie |
|---|---:|---:|---:|
| A | 81 | 35 | 96 |
| B | 32 | 94 | 87 |

Merged full-run effect:

- Original prediction distribution: `Tie=425`, `B=327`, `A=321`.
- Merged prediction distribution: `B=456`, `A=434`, `Tie=183`.
- Original correct count: `529 / 1073`.
- Merged correct count: `704 / 1073`.
- Absolute effective accuracy gain: `+16.31 pp`.

Accuracy ceiling analysis:

- If only the original 425 `Tie` predictions are repaired and the 119 original non-tie errors remain fixed, the perfect-tiebreak ceiling is `(529 + 425) / 1073 = 88.91%`.
- Current merged effective accuracy is `704 / 1073 = 65.61%`.
- Remaining gap to perfect tiebreak is `250 / 1073 = 23.30 pp`.
- If the second pass made no abstentions while keeping its current valid-decision accuracy of `72.31%`, the expected full effective accuracy would be `(529 + 0.7231 * 425) / 1073 = 77.94%`.

Interpretation: Lever 2.1 is a strong improvement over the weighted HR-oracle baseline because it reduces the full-run tie rate from `39.61%` to `17.05%` and adds `+16.31 pp` effective accuracy. Its main bottleneck is not A/B discrimination once it commits, but excessive residual abstention: `43.06%` of the tiebreak subset is still predicted as `Tie`.

## Selected-Question Discriminativeness Audit

Source artifact:

- Remote report: `results/analysis/tie_contribution_audit_hroracle_weighted_raw/hypothesis_verification_selected_questions.md`

Hypothesis: the HR-oracle selector chooses questions that are human-relevant but often not contrastive enough for the paired responses, so weighted aggregation collapses to `Tie`.

Key evidence:

- HR-oracle weighted raw rerun: `1073` samples, `424` ties, `649` non-ties.
- Tie rows have only `3.39%` nonzero selected-question contributions, versus `51.02%` for non-tie rows.
- Tie rows put only `0.94%` mean weight mass on nonzero contributions, versus `54.72%` for non-tie rows.
- Tie rows have `95.61%` same-label question pairs (`yes/yes`, `no/no`, or `na/na`), versus `46.26%` for non-tie rows.
- `294 / 424 = 69.3%` tie samples have zero nonzero selected-question contributions.

Top selected questions inside tie rows are broad relevance/task/coverage checks with very low nonzero rates:

| QID | Selected in ties | Nonzero rate | Sub-aspect |
|---:|---:|---:|---|
| 52 | 401 | 1.25% | Task Adherence |
| 26 | 396 | 6.82% | Factual Accuracy |
| 47 | 392 | 0.51% | Relevance to User |
| 22 | 392 | 1.02% | Coverage Adequacy |
| 3 | 388 | 1.80% | Explanation Sufficiency |
| 50 | 384 | 0.00% | Relevance to User |
| 48 | 376 | 0.00% | Relevance to User |

Conclusion: the hypothesis is strongly supported. The main failure is not parse errors, NA handling, or exact cancellation; it is that the selected questions are too often relevant-but-nondiscriminative for the actual A/B contrast. This supports a selector/reranker objective that combines human relevance with empirical discriminativeness and reduces redundant broad relevance/task-adherence questions.

## Discriminativeness Reranker Experiment

Source artifacts:

- Script: `src/evaluation/rerank_selector_picks.py`
- Remote reranked picks: `results/dynamic_test/human_relevance_oracle_test_v4_qwen36_27b_nvfp4_ranked_picks_discriminative_rerank_beta1_cap2.parquet`
- Remote reranked weights: `results/dynamic_test/question_weights/hr_oracle_k15_qwen36_27b_weights_discriminative_rerank_beta1_cap2.parquet`
- Remote eval: `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise_discriminative_rerank_beta1_cap2`

Setup:

- Exploratory prior source: `results/analysis/tie_contribution_audit_hroracle_weighted_raw/question_contributions.parquet`.
- Reranker score: original rank-decay multiplied by qid-level empirical decisiveness.
- Parameters: `k=15`, `rank_alpha=0.5`, `prior_beta=1.0`, `epsilon=0.02`, `min_support=20`, `subaspect_cap=2`.
- Warning: this run uses a test-derived contribution prior, so it is an exploratory mechanism check, not a clean held-out test claim.

Reranker summary:

- Samples changed at top-k: `1073 / 1073`.
- Mean top-15 overlap with original selection: `10.44 / 15`.
- Mean qid decisiveness prior before rerank: `0.1084`.
- Mean qid decisiveness prior after rerank: `0.1277`.
- Recomputed weights parse rate: `99.91%`.

| Run | Valid | Tie | Accuracy | Macro-F1 | Tie rate | Effective accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 649 | 424 | 81.36 | 81.36 | 39.52 | 49.21 |
| + discriminativeness reranker | 671 | 402 | 81.07 | 81.06 | 37.47 | 50.70 |

Net effect versus raw rerun:

- Effective accuracy gain: `+1.49 pp`.
- Tie-rate reduction: `-2.05 pp`.
- Correct count gain: `528 -> 544`, net `+16`.
- Prediction changes: `261 / 1073`.
- Old Tie outcomes: `298` remained Tie, `87` became correct A/B, `39` became wrong A/B.
- Old non-Tie correct samples harmed: `84`.
- Old wrong/Tie samples fixed: `100`.

QID movement confirms the intervention but also shows it is too blunt:

- Large decreases: q44 Practical Usability `1005 -> 101`, q48 Relevance to User `1042 -> 358`, q54 Task Adherence `774 -> 128`, q50 Relevance to User `1040 -> 694`.
- Large increases: q23 Coverage Adequacy `36 -> 1062`, q46 Practical Usability `68 -> 897`, q45 Practical Usability `70 -> 827`, q9 Tone/Register `129 -> 668`.

Interpretation: adding discriminativeness helps the exact failure mode we identified, but the current global prior is too coarse. It reduces abstention and improves effective accuracy slightly, yet it damages some previously correct non-tie decisions by over-penalizing useful broad questions and over-promoting globally discriminative questions without sample-specific relevance calibration. The next clean version should calibrate the prior on train/dev and use a gentler mixture or a conditional reranker that only activates when the selected set is predicted to be low-discriminativeness.

### Gentler Reranker Check

Source artifacts:

- Remote reranked picks: `results/dynamic_test/human_relevance_oracle_test_v4_qwen36_27b_nvfp4_ranked_picks_discriminative_rerank_beta05_nocap.parquet`
- Remote reranked weights: `results/dynamic_test/question_weights/hr_oracle_k15_qwen36_27b_weights_discriminative_rerank_beta05_nocap.parquet`
- Remote eval: `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise_discriminative_rerank_beta05_nocap`

Setup:

- Same exploratory test-derived prior as above.
- Gentler parameters: `prior_beta=0.5`, `subaspect_cap=0`, with `k=15`, `rank_alpha=0.5`, `epsilon=0.02`, `min_support=20`.

Reranker shift:

- Mean top-15 overlap with original selection: `12.44 / 15`, versus `10.44 / 15` for beta1+cap2.
- Mean qid decisiveness prior after rerank: `0.1206`, versus `0.1277` for beta1+cap2.
- Weight parse rate: `99.81%`.

| Run | Valid | Tie | Accuracy | Macro-F1 | Tie rate | Effective accuracy | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 649 | 424 | 81.36 | 81.36 | 39.52 | 49.21 | 528 |
| Reranker beta1+cap2 | 671 | 402 | 81.07 | 81.06 | 37.47 | 50.70 | 544 |
| Gentler beta0.5+no-cap | 661 | 412 | 80.64 | 80.63 | 38.40 | 49.67 | 533 |

Sample-level effect versus raw rerun:

| Run | Changed predictions | Old correct harmed | Old wrong/Tie fixed | Old Tie -> correct | Old Tie -> wrong non-Tie |
|---|---:|---:|---:|---:|---:|
| Reranker beta1+cap2 | 261 | 84 | 100 | 87 | 39 |
| Gentler beta0.5+no-cap | 195 | 62 | 67 | 57 | 38 |

Interpretation: the gentler variant verifies the expected tradeoff. It reduces the side effect on previously correct non-tie decisions (`84 -> 62` harmed) and changes fewer predictions (`261 -> 195`), but it also fixes fewer wrong/Tie samples (`100 -> 67`) and ends with a weaker net gain (`+5` correct over baseline, versus `+16` for beta1+cap2). The mechanism is smoother but still not ideal: q48/q54/q50 are no longer severely suppressed, yet q44 remains heavily penalized (`1005 -> 164`) and q23 is still promoted almost globally (`36 -> 1044`). This argues against a pure global-prior reranker as the final direction. The next candidate should be conditional: preserve the original HR-oracle top-k by default, and only apply discriminativeness reranking when the selected set has low predicted nonzero mass or excessive same-subaspect redundancy.

### All-Zero 294 Attribution

Source artifact:

- Remote report: `results/analysis/tie_contribution_audit_hroracle_weighted_raw/all_zero_294_attribution_report.md`

Question: among the `294` all-zero selected-question ties, are they caused by wrong checklist selection or by judge-model errors?

Evidence:

- All-zero subset: `294` samples, domain distribution `general=190`, `code=62`, `stem=42`; gold labels are balanced, `A=149`, `B=145`.
- Selected top-15 question rows: `3902`.
- Selected top-15 label-pair distribution: `yes/yes=3069`, `no/no=624`, `na/na=181`, `no/na=17`, `na/no=11`.
- Same-label rate: `99.28%`; `na/na` rate: only `4.64%`.
- Human-relevance coverage is high: selected top-15 covers mean `94.16%` of qids with `h>0`, and `97.88%` of qids with `h>=0.5`.

Existing static full-bank v4 pairwise results on the same 294 samples:

- Prediction distribution: `Tie=164`, `A=67`, `B=63`.
- Non-tie decisions: `130 / 294`.
- Correct: `82 / 294 = 27.89%`.
- Valid accuracy among non-ties: `82 / 130 = 63.08%`.

DeepSeek V4 thinking tiebreak results on overlapping samples:

- Covered: `284 / 294`.
- Prediction distribution: `Tie=124`, `B=81`, `A=79`, missing `10`.
- Correct, counting missing as wrong: `118 / 294 = 40.14%`.
- Valid accuracy among non-ties: `118 / 160 = 73.75%`.

Proxy attribution buckets:

- `82` samples: static full-bank is correct. These are strong evidence that the original selected top-15 omitted discriminative checklist evidence.
- `48` samples: static full-bank makes a non-tie prediction but is wrong. These point to misleading checklist evidence, judge labeling error, or scoring/aggregation error.
- `71` samples: static full-bank is still Tie, but DeepSeek tiebreak is correct. These suggest the current checklist bank or checklist aggregation is too weak for the human reason, even when all bank questions are available; direct pairwise reasoning can see a signal.
- `75` samples: static full-bank is Tie and DeepSeek is Tie or missing. These are unresolved/hard cases.
- `18` samples: static full-bank is Tie but DeepSeek tiebreak is wrong. These are likely difficult or misleading cases.

Interpretation: this is not mainly a simple retrieval miss under the existing human-relevance labels, because selected top-15 already covers most human-relevant qids. It is mostly a discriminativeness failure: the selected questions are human-relevant but too broad, so the judge assigns the same label to A and B. For at least `82 / 294`, better checklist selection from the existing bank would help. For another `71 / 294`, even full-bank checklist aggregation remains too weak while direct tiebreak can recover the winner, so this points toward sharper contrastive checklist questions or a conditional tiebreak fallback. A smaller but real bucket (`48 + 18`) reflects misleading judge/scoring behavior rather than pure selection.

### Selector Training With Gold-Aligned Contrastive Signal

Source artifacts:

- Training script mode: `src/train/run_selector_train.py --target-mode human_contrastive`
- Remote checkpoints:
  - `results/checkpoints/selector_v4_human_contrastive_b1`
  - `results/checkpoints/selector_v4_human_contrastive_b03`
- Remote evals:
  - `results/dynamic_test/selector_v4_human_contrastive_b1_k15_td005_weighted_pairwise`
  - `results/dynamic_test/selector_v4_human_contrastive_b03_k15_td005_weighted_pairwise`

Training target:

`rank_target = human_relevance + contrastive_bonus * gold_aligned_discriminative_signal`

where the contrastive signal is positive only when the full-bank oracle row has nonzero `pair_contrib` in the gold-winner direction: `winner=A and pair_contrib>0`, or `winner=B and pair_contrib<0`.

Setup:

- Train oracle: `data/oracle/train_10k_oracle_v4_9b_flash_thinking8192_merged.parquet`
- Human relevance: `data/oracle/train_tier_10k_human_relevance_v3.parquet`
- Encoder: `models/bge-m3`
- Train samples: `9551`; questions: `58`
- Human-relevance rows applied: `63134`
- Gold-aligned discriminative rows in train target: `100821`
- Skipped parse-fail oracle rows: `2602`
- Test policy: learned top-15, weighted pairwise, `tie_delta=0.05`, Qwen3.5-9B local judge.

Overall test results:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 528 | 49.21 | 649 | 81.36 | 424 | 39.52 | 121 |
| Contrastive selector b1.0 | 496 | 46.23 | 609 | 81.44 | 464 | 43.24 | 113 |
| Contrastive selector b0.3 | 493 | 45.95 | 602 | 81.89 | 471 | 43.90 | 109 |
| Static full-bank | 588 | 54.80 | 750 | 78.40 | 323 | 30.10 | 162 |

All-zero 294 subset:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 0 | 0.00 | 0 | 0.00 | 294 | 100.00 | 0 |
| Contrastive selector b1.0 | 60 | 20.41 | 92 | 65.22 | 202 | 68.71 | 32 |
| Contrastive selector b0.3 | 66 | 22.45 | 90 | 73.33 | 204 | 69.39 | 24 |
| Static full-bank | 82 | 27.89 | 130 | 63.08 | 164 | 55.78 | 48 |

Fullbank-cracked 82 subset:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 0 | 0.00 | 0 | 0.00 | 82 | 100.00 | 0 |
| Contrastive selector b1.0 | 33 | 40.24 | 42 | 78.57 | 40 | 48.78 | 9 |
| Contrastive selector b0.3 | 36 | 43.90 | 41 | 87.80 | 41 | 50.00 | 5 |
| Static full-bank | 82 | 100.00 | 82 | 100.00 | 0 | 0.00 | 0 |

Selected-question contribution check:

| Slice | Run | Nonzero sample rate | Mean nonzero count | Gold-signal sample rate | Mean gold-signal count |
|---|---|---:|---:|---:|---:|
| All-zero 294 | HR-oracle baseline | 5.46 | 0.109 | 4.44 | 0.099 |
| All-zero 294 | Contrastive b1.0 | 54.42 | 2.020 | 36.39 | 1.279 |
| All-zero 294 | Contrastive b0.3 | 54.42 | 2.044 | 35.37 | 1.316 |
| Fullbank-cracked 82 | HR-oracle baseline | 8.54 | 0.232 | 8.54 | 0.232 |
| Fullbank-cracked 82 | Contrastive b1.0 | 75.61 | 3.085 | 60.98 | 2.671 |
| Fullbank-cracked 82 | Contrastive b0.3 | 70.73 | 3.122 | 58.54 | 2.744 |

Overall transition from HR-oracle baseline:

- Contrastive b1.0: old correct -> correct/tie/wrong = `381/123/24`; old tie -> correct/tie/wrong = `95/287/42`; old wrong -> correct/tie/wrong = `20/54/47`.
- Contrastive b0.3: old correct -> correct/tie/wrong = `372/131/25`; old tie -> correct/tie/wrong = `103/286/35`; old wrong -> correct/tie/wrong = `18/54/49`.

Interpretation: the central hypothesis is supported on the targeted failure mode. Adding gold-aligned discriminative supervision makes the selector choose questions with real A/B signal: on all-zero 294, nonzero-signal sample rate jumps from `5.46%` to `54.42%`, and on the fullbank-cracked 82 slice it jumps from `8.54%` to about `71-76%`. This recovers `33-36 / 82` of the previously all-tie cracked cases. However, the current unconditional training objective hurts the overall system: it increases tie rate and loses `32-35` correct predictions net versus the HR-oracle baseline. The mechanism works, but it needs gating or a better objective, not a global replacement for human-relevance selection.

### Gated Contrastive Selector Training

Source artifacts:

- Local commit pushed for server transfer: `4093d30 add gated contrastive selector target`
- Server merge commit: `d688f72 Merge remote-tracking branch 'origin/master'`
- Training script mode: `src/train/run_selector_train.py --target-mode human_contrastive --contrastive-gate-threshold {1,2} --contrastive-gate-topk 15`
- Remote checkpoints:
  - `results/checkpoints/selector_v4_human_contrastive_gated_t1_b03`
  - `results/checkpoints/selector_v4_human_contrastive_gated_t1_b1`
  - `results/checkpoints/selector_v4_human_contrastive_gated_t2_b01`
  - `results/checkpoints/selector_v4_human_contrastive_gated_t2_b03`
- Remote evals:
  - `results/dynamic_test/selector_v4_human_contrastive_gated_t1_b03_k15_td005_weighted_pairwise`
  - `results/dynamic_test/selector_v4_human_contrastive_gated_t1_b1_k15_td005_weighted_pairwise`
  - `results/dynamic_test/selector_v4_human_contrastive_gated_t2_b01_k15_td005_weighted_pairwise`
  - `results/dynamic_test/selector_v4_human_contrastive_gated_t2_b03_k15_td005_weighted_pairwise`

Training target:

`rank_target = human_relevance + gate(sample) * contrastive_bonus * gold_aligned_discriminative_signal`

where `gate(sample)` fires when the HR-ranked top-15 questions have `gold_signal_count < threshold`. Threshold 1 is the conservative "zero gold-aligned signal" gate; threshold 2 is the more permissive "fewer than two gold signals" gate.

Setup and training diagnostics:

- Train oracle: `data/oracle/train_10k_oracle_v4_9b_flash_thinking8192_merged.parquet`
- Human relevance: `data/oracle/train_tier_10k_human_relevance_v3.parquet`
- Encoder: `models/bge-m3`
- Train samples: `9551`; questions: `58`
- Gate-open samples: `2726 / 9551` for threshold 1; `4236 / 9551` for threshold 2
- Human-relevance rows applied: `63134`
- Gold-aligned discriminative rows in full target: `100821`
- Gold-aligned discriminative rows inside gated samples: `4798` for threshold 1; `11058` for threshold 2
- Skipped parse-fail oracle rows: `2602`
- Test policy: learned top-15, weighted pairwise, `tie_delta=0.05`, Qwen3.5-9B local judge.
- Weight parse quality: threshold 1 runs had `parse_ok_rate=0.9981`, `avg_missing_qids=0.029`; threshold 2 runs had `parse_ok_rate=0.9991`, `avg_missing_qids=0.018`.

Overall test results:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 528 | 49.21 | 649 | 81.36 | 424 | 39.52 | 121 |
| Ungated contrastive b0.3 | 493 | 45.95 | 602 | 81.89 | 471 | 43.90 | 109 |
| Ungated contrastive b1.0 | 496 | 46.23 | 609 | 81.44 | 464 | 43.24 | 113 |
| Gated contrastive t1 b0.3 | 497 | 46.32 | 613 | 81.08 | 460 | 42.87 | 116 |
| Gated contrastive t1 b1.0 | 490 | 45.67 | 612 | 80.07 | 461 | 42.96 | 122 |
| Gated contrastive t2 b0.1 | 494 | 46.04 | 618 | 79.94 | 455 | 42.40 | 124 |
| Gated contrastive t2 b0.3 | 494 | 46.04 | 618 | 79.94 | 455 | 42.40 | 124 |
| Static full-bank | 588 | 54.80 | 750 | 78.40 | 323 | 30.10 | 162 |

All-zero 294 subset:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 0 | 0.00 | 0 | 0.00 | 294 | 100.00 | 0 |
| Ungated contrastive b0.3 | 66 | 22.45 | 90 | 73.33 | 204 | 69.39 | 24 |
| Ungated contrastive b1.0 | 60 | 20.41 | 92 | 65.22 | 202 | 68.71 | 32 |
| Gated contrastive t1 b0.3 | 57 | 19.39 | 84 | 67.86 | 210 | 71.43 | 27 |
| Gated contrastive t1 b1.0 | 58 | 19.73 | 87 | 66.67 | 207 | 70.41 | 29 |
| Gated contrastive t2 b0.1 | 65 | 22.11 | 91 | 71.43 | 203 | 69.05 | 26 |
| Gated contrastive t2 b0.3 | 65 | 22.11 | 91 | 71.43 | 203 | 69.05 | 26 |
| Static full-bank | 82 | 27.89 | 130 | 63.08 | 164 | 55.78 | 48 |

Fullbank-cracked 82 subset:

| Run | Correct | Effective acc | Valid | Valid acc | Tie | Tie rate | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| HR-oracle weighted raw rerun | 0 | 0.00 | 0 | 0.00 | 82 | 100.00 | 0 |
| Ungated contrastive b0.3 | 36 | 43.90 | 41 | 87.80 | 41 | 50.00 | 5 |
| Ungated contrastive b1.0 | 33 | 40.24 | 42 | 78.57 | 40 | 48.78 | 9 |
| Gated contrastive t1 b0.3 | 32 | 39.02 | 35 | 91.43 | 47 | 57.32 | 3 |
| Gated contrastive t1 b1.0 | 33 | 40.24 | 39 | 84.62 | 43 | 52.44 | 6 |
| Gated contrastive t2 b0.1 | 35 | 42.68 | 40 | 87.50 | 42 | 51.22 | 5 |
| Gated contrastive t2 b0.3 | 35 | 42.68 | 40 | 87.50 | 42 | 51.22 | 5 |
| Static full-bank | 82 | 100.00 | 82 | 100.00 | 0 | 0.00 | 0 |

Overall transition from HR-oracle baseline:

- Ungated b0.3: old correct -> correct/tie/wrong = `372/131/25`; old tie -> correct/tie/wrong = `103/286/35`; old wrong -> correct/tie/wrong = `18/54/49`.
- Ungated b1.0: old correct -> correct/tie/wrong = `381/123/24`; old tie -> correct/tie/wrong = `95/287/42`; old wrong -> correct/tie/wrong = `20/54/47`.
- Gated t1 b0.3: old correct -> correct/tie/wrong = `379/116/33`; old tie -> correct/tie/wrong = `93/293/38`; old wrong -> correct/tie/wrong = `25/51/45`.
- Gated t1 b1.0: old correct -> correct/tie/wrong = `371/125/32`; old tie -> correct/tie/wrong = `96/288/40`; old wrong -> correct/tie/wrong = `23/48/50`.
- Gated t2 b0.1: old correct -> correct/tie/wrong = `370/122/36`; old tie -> correct/tie/wrong = `102/283/39`; old wrong -> correct/tie/wrong = `22/50/49`.
- Gated t2 b0.3: old correct -> correct/tie/wrong = `370/122/36`; old tie -> correct/tie/wrong = `102/283/39`; old wrong -> correct/tie/wrong = `22/50/49`.

Interpretation: among the gated training runs, the best overall balance is still the conservative `threshold=1, bonus=0.3` point: it reaches `497` correct, reduces old-correct-to-tie damage relative to ungated b0.3 (`131 -> 116`), and only gives up some all-zero recovery (`66 -> 57`). The more permissive threshold 2 recovers almost as many all-zero samples as ungated b0.3 (`65` vs `66`), but it increases wrong predictions (`124`) and drops overall correct to `494`. The `threshold=2, bonus=0.1` and `threshold=2, bonus=0.3` runs are prediction-identical, suggesting the threshold changed the selected question ranking while the 0.1-0.3 bonus magnitude did not materially affect the learned top-k. Overall, train-time gated contrastive selection does not beat the HR-oracle baseline (`497` best vs `528`), so it should not be pursued as the main selector objective. It remains useful as evidence that discriminative questions exist in the bank and can recover targeted all-zero failures; the next serious direction should be a dev-calibrated conditional reranker or a tiebreak/fallback, not more train-time gate sweeps.
