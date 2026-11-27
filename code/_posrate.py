# -*- coding: utf-8 -*-
"""统计各数据集 mined 候选的正例密度（gold 候选 / 全部候选）。"""
import json

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
for name in ("mined_metaqa_train_s5k.jsonl", "mined_webqsp_train_vghn.jsonl",
             "mined_cwq_train.jsonl"):
    n = p = q = 0
    try:
        f = open(f"{D}/{name}", encoding="utf-8")
    except FileNotFoundError:
        print(name, "missing")
        continue
    for l in f:
        if not l.strip():
            continue
        r = json.loads(l)
        g = set(r.get("gold") or [])
        c = r.get("cand") or []
        q += 1
        n += len(c)
        p += sum(1 for x in c if x[0] in g)
    print(f"{name}: q={q} cands={n} pos={p} rate={p/max(n,1):.4f}")
