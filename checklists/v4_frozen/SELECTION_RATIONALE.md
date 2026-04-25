# v4 Bank Selection Rationale

Derived from dev_600 oracle analysis on v3_frozen with 9B Q4 judge (n=600, n_valid=431, agree_valid=0.7680 vs baseline 0.7705).

## Diagnosis Summary

Five failure modes identified in 100 wrong predictions:

1. **Bank gap - Tone/Style/Persona**: role-play and creative writing prompts had `differ_count=0` because no Qs target voice/persona continuity.
2. **Bank gap - Ambiguity handling**: prompts with under-specification or false premises had no checklist coverage.
3. **Bank gap - Numeric constraints**: explicit word-count / format requests slipped through because Format/Constraint Qs were 76-88% N/A and too abstract.
4. **Bank conflict - Depth vs Brevity**: helpfulness Depth Qs and clarity Information Efficiency Qs fired in opposite directions on the same instance, canceling signal.
5. **Bank bias - Structural Effectiveness**: Qs 7-8 rewarded surface formatting (bullets, headings) regardless of content correctness, favoring verbose responses.

## v4 Edits

### Deletes (5 Qs)
- Information Efficiency Q3 (boilerplate/disclaimers) - duplicate of Q1.
- Format and Constraint Compliance Q1-Q3 (qids 59-61) - 76-88% N/A rate; replaced by Explicit Numeric Constraints sub-aspect.
- Source-Grounded Transformation Q2 - exact duplicate of Contextual Precision Q2.

### Dedups (-7 Qs net)
- Coverage Adequacy 6 -> 4 (removed Q2 "essential details qualifiers" and Q5 "level of detail sufficient" - both paraphrase Q1).
- Practical Usability 6 -> 3 (removed Q1, Q3, Q4 - all variants of "concrete enough to act").
- Depth of Guidance 4 -> 3 (removed "explain consequences" - covered by trade-offs Q).
- Relevance to User 5 -> 4 (removed "avoid digressions" - duplicate of "avoid background space").

### Restructures (2 Qs rewritten)
- Structural Effectiveness Q1-Q2 made conditional on response length / section count, so simple short responses correctly answer N/A instead of accidentally rewarding superficially formatted verbose content.

### Adds (8 Qs across 4 new sub-aspects)
- Tone and Register Match (clarity_and_communication) - 2 Qs covering tone alignment and persona/role-play continuity.
- Ambiguity and Premise Handling (correctness_and_completeness) - 2 Qs covering under-specification and false premises.
- Temporal Awareness (correctness_and_completeness) - 1 Q on time-sensitive facts and knowledge cutoff.
- Explicit Numeric Constraints (relevance_instruction_following) - 2 Qs replacing the deleted Format/Constraint Compliance Qs with sharper triggers.

## Size

| Dimension | v3 | v4 | Change |
|---|---|---|---|
| clarity_and_communication | 8 | 9 | +1 |
| correctness_and_completeness | 14 | 15 | +1 |
| helpfulness_and_usefulness | 19 | 14 | -5 |
| relevance_instruction_following | 10 | 8 | -2 |
| coding_communication_conditional | 12 | 12 | 0 |
| **TOTAL raw** | **63** | **58** | **-5** |

After cross-sub_aspect dedup at load time, expected flat qid count ~57-58 (v3 was 61 after dedup).

## Validation Plan

1. Run `build_bank_index.py --bank checklists/v4_draft --out checklists/v4_frozen`.
2. Run dev_600 oracle with 27B Q6 judge (~30 min on 5090).
3. Acceptance threshold: `oracle_agreement_rate_valid >= 0.78` overall and `general` domain >= 0.74 (vs v3 9B baseline 0.7227 on general).
4. If passes, run on `train tier_10k` for selector training labels.
