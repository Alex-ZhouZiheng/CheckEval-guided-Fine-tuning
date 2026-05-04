# Tie Contribution Audit: HR-oracle weighted run

Run: results/dynamic_test/human_relevance_oracle_qwen36_27b_k15_td005_weighted_pairwise_raw_for_audit

## Summary

- n_total: 1073
- n_tie: 424
- parse_fail_recomputed: 5
- max_margin_abs_error: 4.440892098500626e-16
- tie_categories: {'all_zero_contributions': 294, 'sparse_correct_signal_diluted': 90, 'sparse_wrong_signal_or_conflict': 35, 'exact_cancellation': 5}
- tie_top1_matches_gold_rate: 0.2169811320754717
- tie_top3_matches_gold_rate: 0.21226415094339623
- tie_mass_matches_gold_rate: 0.2099056603773585
- tie_nonzero_count_dist: {'0': 294, '1': 87, '2': 31, '3': 8, '4': 3, '5': 1}
- tie_zero_weight_mass_mean: 0.9905845488332119
- tie_na_weight_mass_mean: 0.03440813525067354

## Scoring Rule Sweep

| rule             |   n_tie |   tie_rate |   accuracy_valid |   effective_acc |
|:-----------------|--------:|-----------:|-----------------:|----------------:|
| current_td005    |     424 |   0.395154 |         0.813559 |        0.492078 |
| recomputed_td005 |     424 |   0.395154 |         0.813559 |        0.492078 |
| avg_td0          |     299 |   0.278658 |         0.797158 |        0.575023 |
| avg_td0.001      |     304 |   0.283318 |         0.797139 |        0.571295 |
| avg_td0.005      |     353 |   0.328984 |         0.804167 |        0.539609 |
| avg_td0.01       |     370 |   0.344828 |         0.803698 |        0.526561 |
| avg_td0.02       |     382 |   0.356011 |         0.806078 |        0.519105 |
| avg_td0.03       |     413 |   0.384902 |         0.809091 |        0.49767  |
| avg_td0.05       |     424 |   0.395154 |         0.813559 |        0.492078 |
| top1_abs         |     294 |   0.273998 |         0.79846  |        0.579683 |
| top3_abs_sum     |     297 |   0.276794 |         0.796392 |        0.575955 |
| pos_vs_neg_mass  |     299 |   0.278658 |         0.797158 |        0.575023 |

## Tie Categories

| category                        |   count |
|:--------------------------------|--------:|
| all_zero_contributions          |     294 |
| sparse_correct_signal_diluted   |      90 |
| sparse_wrong_signal_or_conflict |      35 |
| exact_cancellation              |       5 |
