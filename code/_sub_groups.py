# -*- coding: utf-8 -*-
"""按比例抽样 path_groups（P1-3 训练数据规模曲线）。"""
import json
import random
import sys

src, frac, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
random.Random(42).shuffle(rows)
k = max(1, int(round(len(rows) * frac)))
with open(out, "w", encoding="utf-8") as f:
    for r in rows[:k]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{src} -> {out}: {len(rows)} * {frac} = {k} groups")
