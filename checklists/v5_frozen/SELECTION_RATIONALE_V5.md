# v5 Discriminative Checklist Bank Rationale

Goal: reduce pointwise-checklist ties by adding high-discrimination questions that are likely to produce A/B-different yes/no answers in HelpSteer3 pairwise evaluation.

Main change from v4: v4 mostly covered broad relevance, task adherence, coverage, and explanation sufficiency. Error analysis showed these were selected too often and produced yes/yes or no/no evidence. v5 keeps the same YAML schema but adds task-specific, constraint-specific, and context-specific sub-aspects.

Added emphasis:
- exact word/count/entity/format/prohibited-content constraints
- creative/script/roleplay scene quality and character continuity
- coding exact functionality, requested libraries/backends, runnable completeness, debugging specificity
- multi-turn current request tracking and source-grounded transformations
- factual specificity and hallucination control
- ambiguity, infeasible-request, and limitation handling

Recommended selector constraint when using this bank:
- cap generic Task Adherence/Relevance/Coverage/Explanation questions
- prefer Exact Constraint Compliance, Artifact Format Fidelity, Creative Scene and Dialogue Quality, Exact Functional Fit, Conversation Context Accuracy, and Factual Specificity on matching task types
