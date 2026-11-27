# -*- coding: utf-8 -*-
"""WebQSP 按路径长度（1/2 跳）分解，plain vs path（同池，CPU）。"""
import json
import sys

import numpy as np

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import load_cache  # noqa: E402

mined = {}
for line in open(f"{D}/mined_webqsp_test_vghn.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("cand"):
        mined[r["id"]] = (set(r.get("gold") or []), r["cand"])

c = load_cache(f"{D}/setrank_cache_vp_test.pt")
pathq = {q: (set(c["gold"][i]), c["ents"][i],
             np.asarray(c["feat"][i], dtype=np.float64))
         for i, q in enumerate(c["ids"])}

agg = {}
for q, (gold, cand) in mined.items():
    if q not in pathq:
        continue
    pg, ents, f = pathq[q]
    gold = gold or pg
    if not gold:
        continue
    rels = [x[3] for x in cand]
    ws = np.array([x[1] for x in cand], dtype=np.float64)
    ps = np.array([x[2] for x in cand], dtype=np.float64)
    es = [x[0] for x in cand]
    # 跳数口径：金答案候选的路径长度（不在池内则记 miss）
    gi = next((k for k, e in enumerate(es) if e in gold), None)
    key = len(rels[gi]) if gi is not None else 0
    a = agg.setdefault(key, {"n": 0, "plain": 0, "path": 0, "path_plain": 0})
    a["n"] += 1
    a["plain"] += 1 if es[int(np.argmax(0.5 * ws + 2.0 * ps))] in gold else 0
    a["path"] += 1 if ents[int(np.argmax(0.5 * f[:, 0] + 2.0 * f[:, 1]))] in gold else 0
    a["path_plain"] += 1 if ents[int(np.argmax(f[:, 1]))] in gold else 0

res = {}
for k in sorted(agg):
    a = agg[k]
    label = f"{k}hop" if k else "gold_not_in_pool"
    res[label] = {kk: (a[kk] / a["n"] if kk != "n" else a[kk]) for kk in a}
    print(label, {kk: round(v, 4) for kk, v in res[label].items()}, flush=True)

json.dump(res, open(f"{D}/p16_webqsp_hop.json", "w"), indent=1)
