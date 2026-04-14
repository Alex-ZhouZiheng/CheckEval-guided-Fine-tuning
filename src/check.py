# check.py
import numpy as np
import pandas as pd
from utils import parse_checkeval_output, compare_checklists_pairwise, load_checklists

df = pd.read_parquet(
    "/root/autodl-tmp/Thesis/results/finetuned_checkeval_final_adapter_test_2026-04-14_predictions.parquet"
)
checklists, _ = load_checklists()
N = sum(len(q) for q in checklists.values())

def _pred_and_margin(r):
    pa = parse_checkeval_output(r["raw_output_a"], expected_n=N)
    pb = parse_checkeval_output(r["raw_output_b"], expected_n=N)
    cmp = compare_checklists_pairwise(pa, pb, expected_n=N, tie_delta=0.05)
    if cmp is None:
        return pd.Series({"pw": None, "margin": np.nan})
    return pd.Series({"pw": cmp["winner"], "margin": cmp["margin"]})

df[["pw", "margin"]] = df.apply(_pred_and_margin, axis=1)

print("=== predicted winner dist ===")
print(df["pw"].value_counts(dropna=False))
print("\n=== gold winner dist ===")
print(df["winner"].value_counts())
print("\nraw A == raw B:", (df["raw_output_a"] == df["raw_output_b"]).mean())
print("\n=== per-domain acc ===")
for d, g in df.groupby("domain"):
    print(f"{d:10s} n={len(g):4d}  acc={(g['pw'] == g['winner']).mean():.3f}")
print("\n=== margin stats ===")
print(df["margin"].describe())
print("zero-margin rate:", (df["margin"].abs() < 1e-6).mean())
print("\noverall acc:", (df["pw"] == df["winner"]).mean())
