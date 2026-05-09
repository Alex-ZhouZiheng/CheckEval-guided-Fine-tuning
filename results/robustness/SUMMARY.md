# Robustness sweep -- dev_600 (4 judges x 3 perturbations)

Original dev_600 accuracy (real, n=600):
- vanilla: 0.7467
- ft_vanilla: 0.7383
- pipeline_cmp: 0.7333
- selfcheck_265: 0.7383

| Judge | Mode | n | acc_orig | acc_pert | acc_drop | invariance |
|---|---|---:|---:|---:|---:|---:|
| vanilla | swap | 608 | 0.7484 | 0.7138 | +0.0345 | 0.6678 |
| vanilla | verbose | 608 | 0.7484 | 0.7829 | -0.0345 | 0.8109 |
| vanilla | format | 601 | 0.7454 | 0.7537 | -0.0083 | 0.8386 |
| ft_vanilla | swap | 608 | 0.7418 | 0.7270 | +0.0148 | 0.6678 |
| ft_vanilla | verbose | 608 | 0.7418 | 0.7829 | -0.0411 | 0.8339 |
| ft_vanilla | format | 601 | 0.7404 | 0.7587 | -0.0183 | 0.8319 |
| pipeline_cmp | swap | 608 | 0.7336 | 0.7237 | +0.0099 | 0.7862 |
| pipeline_cmp | verbose | 608 | 0.7336 | 0.7500 | -0.0164 | 0.7895 |
| pipeline_cmp | format | 601 | 0.7321 | 0.7338 | -0.0017 | 0.8386 |
| selfcheck_265 | swap | 608 | 0.7385 | 0.7385 | +0.0000 | 0.7451 |
| selfcheck_265 | verbose | 608 | 0.7385 | 0.8224 | -0.0839 | 0.7862 |
| selfcheck_265 | format | 601 | 0.7371 | 0.7205 | +0.0166 | 0.8037 |