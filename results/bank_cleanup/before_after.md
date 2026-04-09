# Checklist Bank Cleanup — Before / After

| dimension | items_before | items_after | dropped | avg_na_before | avg_na_after | dim_acc_before |
|---|---:|---:|---:|---:|---:|---:|
| clarity_and_communication | 44 | 31 | 0 | 0.0436 | 0.0436 | 0.5417 |
| coding_communication_conditional | 40 | 20 | 0 | 0.2123 | 0.2123 | 0.6364 |
| correctness_and_completeness | 44 | 26 | 0 | 0.1809 | 0.1809 | 0.6050 |
| helpfulness_and_usefulness | 48 | 25 | 2 | 0.1105 | 0.1170 | 0.5920 |

Note: `dim_acc_before` is read from the existing
`*_dimension_diagnostics.json` if present. Re-run `run_checkeval_judge.py`
with `--checklists-dir checklists/v2` to measure the *after* dim_accuracy.

Drop reason legend:
- **saturated_yes / saturated_no** (hard drop): `yes_rate_nonNA` above `--yes-high` or below `--yes-low` with ≥ `sat_min_n` non-NA answers.
- **low_signal** (advisory drop): `|agree_rate - 0.5| < --signal-threshold` with ≥ `--signal-min-n` effective comparisons.
- **rarely_applicable** (tag only): `na_rate > --rare-na-threshold`. Does NOT trigger drop — NA means 'not applicable', which is legitimate.
- **tiny_effective_n** (tag only): too few non-NA answers to judge. Does NOT trigger drop.