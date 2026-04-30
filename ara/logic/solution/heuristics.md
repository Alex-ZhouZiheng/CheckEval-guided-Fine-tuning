# Heuristics

## H01: Use sampled negatives for sparse human-relevance ListMLE
- **Rationale**: Human-relevance labels are sparse; treating every unmentioned active question as a negative creates many arbitrary zero-target ties. Sampling a small number of negatives per positive preserves the "positive above negative" signal without forcing a full ordering of the tail.
- **Provenance**: user
- **Sensitivity**: medium
- **Code ref**: [`src/train/run_selector_train.py`]

## H02: Calibrate tie thresholds separately for weighted aggregation
- **Rationale**: Softmax-normalized importance weights compress pairwise score margins. Reusing the unweighted tie threshold can turn many close decisions into ties, improving non-tie accuracy while lowering effective accuracy.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/evaluation/run_dynamic_eval.py`]

## H03: Keep importance weights context-only
- **Rationale**: Question-importance prompts should see the context and selected questions, not responses A/B, so the weighting model estimates criterion importance rather than acting as a hidden judge.
- **Provenance**: user
- **Sensitivity**: medium
- **Code ref**: [`src/evaluation/build_question_importance_weights.py`]
