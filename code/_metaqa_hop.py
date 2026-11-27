# -*- coding: utf-8 -*-
"""P1-4 MetaQA 按跳数分解（1/2/3-hop），基于已有 plain/path dump。

跳数由子图 BFS 得到（q_entity → 任一金答案 的最短有向路径长度）。
"""
import json
from collections import deque

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def bfs_hop(edges, starts, golds, max_hops=3):
    g = {}
    for h, r, t in edges:
        g.setdefault(h, []).append(t)
    seen = set(starts)
    q = deque((s, 0) for s in starts)
    while q:
        e, d = q.popleft()
        if d > 0 and e in golds:
            return d
        if d >= max_hops:
            continue
        for t in g.get(e, []):
            if t not in seen:
                seen.add(t)
                q.append((t, d + 1))
    return None


def hits(path):
    out = {}
    for r in json.load(open(path, encoding="utf-8")):
        top = r["pred"][0]
        if isinstance(top, (list, tuple)):
            top = top[0]
        out[r["id"]] = 1 if r["pred"] and top in set(r["gold"]) else 0
    return out


rows = [json.loads(l) for l in open(f"{D}/metaqa/test_s2000.jsonl", encoding="utf-8")]
plain = hits(f"{D}/dump_metaqa_s2000_plain.json")
path = hits(f"{D}/dump_metaqa_s2000_path.json")

agg = {}
for r in rows:
    gold = set(r.get("a_entity") or r.get("answers") or [])
    h = bfs_hop(r["edges"], r["q_entity"], gold)
    key = h if h else 0
    a = agg.setdefault(key, {"n": 0, "plain": 0, "path": 0})
    a["n"] += 1
    a["plain"] += plain.get(r["id"], 0)
    a["path"] += path.get(r["id"], 0)

res = {}
for k in sorted(agg):
    a = agg[k]
    label = f"{k}hop" if k else "unreached"
    res[label] = {"n": a["n"], "plain_h1": a["plain"] / a["n"],
                  "path_h1": a["path"] / a["n"],
                  "delta_pt": (a["path"] - a["plain"]) / a["n"] * 100}
    print(label, res[label], flush=True)

tot = {k: sum(v[k] for v in agg.values()) for k in ("n", "plain", "path")}
print("ALL", tot, "plain_h1=", tot["plain"] / tot["n"], "path_h1=", tot["path"] / tot["n"])
json.dump(res, open(f"{D}/p14_metaqa_hop.json", "w"), indent=1)
