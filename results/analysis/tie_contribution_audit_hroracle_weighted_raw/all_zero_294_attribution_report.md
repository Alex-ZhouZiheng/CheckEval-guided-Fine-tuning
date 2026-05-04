# All-zero 294 Attribution Report

## Scope
- n_all_zero: 294
- domain_dist: {'general': 190, 'code': 62, 'stem': 42}
- winner_dist: {'A': 149, 'B': 145}

## Selected top-15 checklist labels
- selected_question_rows: 3902
- pair_dist: {'yes/yes': 3069, 'no/no': 624, 'na/na': 181, 'no/na': 17, 'na/no': 11}
- same_label_rate: 0.9928
- na_pair_rate: 0.0464

## Static full-bank on same samples
- pred_dist: {'Tie': 164, 'A': 67, 'B': 63}
- correct: 82/294 = 0.2789
- valid: 130, valid_acc: 0.6308
- tie_rate: 0.5578

## DeepSeek tiebreak on overlapping samples
- covered: 284/294
- pred_dist: {'Tie': 124, 'B': 81, 'A': 79, nan: 10}
- correct_counting_missing_wrong: 118/294 = 0.4014
- valid: 160, valid_acc: 0.7375
- tie_rate_on_covered: 0.4366

## Cross table
tiebreak_status  correct  missing  tie  wrong
static_status                                
correct               31        5   35     11
tie                   71        4   71     18
wrong                 16        1   18     13

## Proxy attribution buckets
- selection_missed_discriminative_fullbank_correct: 82
- fullbank_has_signal_but_misleading_or_judge_error: 48
- fullbank_tie_but_tiebreak_correct: 71
- still_unresolved_static_tie_tiebreak_tie_or_missing: 75
- static_tie_tiebreak_wrong: 18

## Human relevance coverage
- samples_with_h_gt_0: 283/283
- mean_n_h_gt_0: 6.6749
- mean_h_gt_0_recall_in_selected_top15: 0.9416
- samples_with_h_ge_0p5: 223/283
- mean_h_ge_0p5_recall_in_selected_top15: 0.9788
- all_h_gt_0_qids_selected: 228/283

## Interpretation
- The top-15 selector usually covers human-relevance qids, so this is not mainly a retrieval miss under the current human-relevance labels.
- Static full-bank breaks many all-zero ties, which means the selected top-15 omitted discriminative checklist evidence for a substantial subset.
- DeepSeek tiebreak can resolve many cases even when checklist aggregation ties, suggesting some remaining cases need direct pairwise/tiebreak reasoning or sharper checklist questions.
