# -*- coding: utf-8 -*-
"""查看 RoG-webqsp test parquet 的字段结构。"""
import glob
import json

import pandas as pd

files = sorted(glob.glob("/root/autodl-tmp/GooseRAG/RoG_ext/webqsp/data/test-*.parquet"))
print("files:", files)
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print("rows:", len(df))
print("columns:", list(df.columns))
for c in df.columns:
    v = df.iloc[0][c]
    s = str(v)
    print(f"\n--- {c} (type={type(v).__name__}) ---")
    print(s[:600])
