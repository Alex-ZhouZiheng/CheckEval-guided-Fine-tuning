# Thesis Wiki Index

> Topic: Checklist-Guided Fine-Tuning and Decomposed LLM Judging  
> Initialized: 2026-05-10  
> Entry point: [[overview]]

## Synthesis

- [[Current Thesis Map]] - the current integrated story, evidence chain, and near-term research path.
- [[Open Questions and Next Ingests]] - gaps, missing artifacts, and suggested next sources to process.

## Topics

- [[Checklist-Guided LLM Judge]] - umbrella concept for decomposed pairwise evaluation.
- [[Static CheckEval and NA-Aware Scoring]] - validated static checklist baseline and pairwise scoring logic.
- [[Dynamic Checklist Selection]] - learned top-k and oracle top-k selection as cost-control methods.
- [[Human-Relevance Selector]] - selector training target, retrieval gap, and HR-oracle upper bound.
- [[Checklist Bank Evolution]] - v1/v2/v3/v4 bank changes and the rationale for v4.
- [[Tie Calibration and Error Modes]] - tie-rate, weighted margins, generic-question failures, and threshold calibration.
- [[Self-Checklist Distillation]] - self-generated criteria and SFT of question-asking behavior.

## Entities

- [[HelpSteer3]] - dataset, splits, human rationales, and role in alignment analysis.
- [[Qwen3.5 Family]] - model roles across judge, selector-labeling, teacher, and student runs.

## Sources

- [[Process Experiments and Research Directions]] - source summary for `Process.md`.
- [[ARA Problem Claims Experiments]] - source summary for `ara/logic/*`.
- [[v4 Bank Selection Rationale]] - source summary for `checklists/v4_frozen/SELECTION_RATIONALE.md`.
- [[Unified Test Robustness and EMNLP Design]] - source summary for current test table, robustness sweep, and decomposed-judging design.

## Raw Source Inventory

- `raw/local/2026-05-10-initial-thesis-corpus.md` - immutable inventory for this seed ingest.

## Maintenance Notes

- Prefer updating existing topic pages over creating near-duplicates.
- Treat `Process.md` as a high-value inventory, but verify against raw metric JSONs before making final numerical claims.
- Keep `log.md` append-only.

