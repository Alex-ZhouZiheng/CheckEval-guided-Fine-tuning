# Thesis Wiki Purpose

## Core Research Goal

Build and evaluate checklist-guided LLM judges for pairwise preference evaluation, with emphasis on whether decomposed criteria improve reliability, interpretability, and efficiency compared with direct pairwise judging.

## Active Thesis Framing

The current thesis is shifting from "a learned selector alone solves dynamic CheckEval" toward a broader decomposed-judging story:

- Static checklist banks and pairwise NA-aware scoring provide strong validated baselines.
- Dynamic checklist selection is useful for reducing question cost, but learned selectors must close the gap to human-relevance oracle selection.
- Self-generated checklist traces and checklist-behavior distillation may be the strongest practical recipe because they preserve decomposition while improving parse and tie behavior.
- Criteria-rationale alignment against HelpSteer3 human reasons is the most important missing analysis for a publishable contribution.

## Key Questions

1. Does criterion decomposition improve pairwise judging accuracy, robustness, or interpretability relative to a direct judge?
2. When should the judge use a static bank, a learned selector over a bank, or self-generated per-instance criteria?
3. Can a smaller student model preserve a larger teacher's question-asking behavior through SFT?
4. Do selected or generated checklist questions cover the same decisive considerations that human annotators mention?
5. Which failures come from bank coverage, selector retrieval, binary checklist aggregation, tie thresholds, or the judge model itself?

## Current Scope

- Dataset: HelpSteer3 pairwise preferences, especially `dev_600` and `test`.
- Main banks: `checklists/v4_frozen` and relevant older v1/v2/v3 comparisons.
- Main evidence: `Process.md`, `ara/`, result metrics under `results/`, design specs under `docs/superpowers/`, and review analyses under `results/review/`.
- Out of scope for the current wiki seed: external web literature review and full paper drafting.

## Success Criteria For The Wiki

- A new session can read `index.md` and `wiki/overview.md` and recover the current thesis map in minutes.
- Every major claim points to a source page or original artifact path.
- New experiment results can be filed without overwriting older context.
- Contradictions, caveats, and non-comparable metrics are explicit rather than buried.

