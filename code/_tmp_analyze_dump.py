# -*- coding: utf-8 -*-
"""临时: 分析端到端 dump 的排序错误与搜索错误样例。"""
import json
from collections import Counter

P = r"E:\GooseRAG\data\rog_qa\webqsp_rwkv7_val_dump.json"
rows = json.load(open(P, encoding="utf-8"))

bins = Counter()
for r in rows:
    g = r["gold_rank"]
    if g is None:
        bins["miss"] += 1
    elif g == 0:
        bins["rank1"] += 1
    elif g < 5:
        bins["rank2-5"] += 1
    elif g < 20:
        bins["rank6-20"] += 1
    else:
        bins["rank21+"] += 1
print("rank bins:", dict(bins), "n=", len(rows))

print("\n== gold 在候选但排名靠后 ==")
shown = 0
for r in rows:
    if r["gold_rank"] is not None and r["gold_rank"] > 0 and shown < 8:
        shown += 1
        print(f"\nQ: {r['question']}")
        print("gold:", r["gold"], "rank:", r["gold_rank"], "n_cand:", r["n_cand"])
        print("top8:", [e for e, _ in r["pred"][:8]])

print("\n== gold 完全没搜到 ==")
shown = 0
for r in rows:
    if r["gold_rank"] is None and shown < 5:
        shown += 1
        print(f"\nQ: {r['question']}")
        print("gold:", r["gold"], "n_cand:", r["n_cand"])
        print("top8:", [e for e, _ in r["pred"][:8]])
