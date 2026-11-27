# -*- coding: utf-8 -*-
"""临时: 检查 RoG parquet 单行结构与三元组格式。"""
import pandas as pd

P = r"E:\GooseRAG\data\rog_hf\webqsp\train-00000-of-00002-d810a36ed97bc2cc.parquet"

df = pd.read_parquet(P)
print("columns:", list(df.columns))
print("rows:", len(df))
r = df.iloc[0]
print("id:", r["id"])
print("question:", r["question"])
print("q_entity:", list(r["q_entity"]))
print("answer:", list(r["answer"]))
print("a_entity:", list(r["a_entity"]))
g = list(r["graph"])
print("graph len:", len(g))
print("first 5 graph items:", g[:5])
for t in g[:3]:
    print("  type:", type(t), "value:", t)
