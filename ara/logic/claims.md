# Claims

## C01: Human-relevance question selection is a useful upper bound
- **Statement**: Selecting the top human-relevance questions can outperform the static full-bank baseline while asking substantially fewer questions.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: On the same held-out test split, a human-relevance oracle top-k selector fails to match or exceed the static full-bank effective accuracy, or requires a comparable number of questions.
- **Proof**: [E01]
- **Dependencies**: []
- **Tags**: selector, human-relevance, dynamic-eval

## C02: The current learned selector is primarily limited by human-relevance retrieval
- **Statement**: The learned selector underperforms the human-relevance oracle at the same top-k budget, indicating that retrieval of human-relevant qids is a major bottleneck.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The learned selector reaches the human-relevance oracle's human-recall and dynamic-eval accuracy at the same question budget without changing the judge or aggregation method.
- **Proof**: [E01]
- **Dependencies**: [C01]
- **Tags**: selector, retrieval, bottleneck

## C03: Context-conditioned importance weights need tie-threshold calibration
- **Statement**: Softmax-normalized question-importance weights can improve weighted effective accuracy after margin calibration, but the default tie threshold of 0.05 over-abstains.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A dev-calibrated threshold fails to reduce the weighted tie rate or fails to improve effective accuracy compared with unweighted scoring.
- **Proof**: [E01]
- **Dependencies**: [C01]
- **Tags**: weighting, calibration, tie-delta

## C04: Full-active pure-human ListMLE is noisy under sparse human labels
- **Statement**: Training ListMLE over the entire active question pool with sparse human relevance labels creates many zero-target ties that dilute the positive signal.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: Full-active pure-human ListMLE matches or exceeds sampled-negative ListMLE on human-relevance retrieval metrics under the same selector architecture.
- **Proof**: [E02]
- **Dependencies**: []
- **Tags**: selector-training, listmle, negative-sampling
