"""查看 parquet 文件内容的脚本。

用法:
    python view_parquet.py <parquet文件路径>
    python view_parquet.py vanilla_judge_test_predictions.parquet
"""

import sys
import pandas as pd

if len(sys.argv) < 2:
    print("用法: python view_parquet.py <parquet文件路径>")
    sys.exit(1)

path = sys.argv[1]
df = pd.read_parquet(path)

print(f"文件: {path}")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print(f"\n=== 列名和类型 ===")
print(df.dtypes)
print(f"\n=== 前5行 ===")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)
print(df.head())
print(f"\n=== 基本统计 ===")
print(df.describe(include="all"))
