# Experiments

## E01: Test dynamic evaluation with static, learned, oracle, and weighted question policies
- **Date**: 2026-04-30
- **Split**: test, 1,073 examples
- **Judge**: local Qwen3.5-9B dynamic evaluator
- **Bank**: frozen v4 checklist bank
- **Artifacts**:
  - `results/dynamic_test/static_v4_td005_pairwise/metrics.json`
  - `results/dynamic_test/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_pairwise/metrics.json`
  - `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_pairwise/metrics.json`
  - `results/dynamic_test/selector_v4bank_v4hr_noOracleFilter_best_k15_td005_weighted_pairwise/metrics.json`
  - `results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise/metrics.json`
  - `results/paper_table_dynamic_eval_test.tex`
- **Summary**: Human-relevance oracle top-15 reached 56.57 effective accuracy versus 54.80 for static full-bank evaluation while using 15.0 versus 49.1 average questions. Learned top-15 reached 53.49 effective accuracy, below static and below the human-relevance oracle.

## E02: Selector training with sampled-negative ListMLE
- **Date**: 2026-04-29
- **Split**: train selector validation
- **Artifacts**:
  - `results/checkpoints/selector_v4_pure_human_sampledneg`
  - `results/checkpoints/selector_v4bank_v4hr_noOracleFilter/best`
- **Summary**: Moving from full-active pure-human ListMLE to positive plus sampled-negative ListMLE directly targeted human-relevance retrieval and produced much stronger human-relevance metrics than the earlier pure-human run.
