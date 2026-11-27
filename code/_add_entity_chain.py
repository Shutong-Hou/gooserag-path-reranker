# -*- coding: utf-8 -*-
"""为 path_groups 补上「实体链」，供含中间实体的路径呈现（P1-1b）使用。

实体链由子图 DFS 还原：从 q_entity 出发按关系序列走到该候选。
"""
import json

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def build_graph(edges):
    g = {}
    for h, r, t in edges:
        g.setdefault(h, {}).setdefault(r, []).append(t)
    return g


def find_chain(g, starts, rels, target):
    def dfs(node, i, path):
        if i == len(rels):
            return path if node == target else None
        for nxt in g.get(node, {}).get(rels[i], []):
            r = dfs(nxt, i + 1, path + [nxt])
            if r:
                return r
        return None

    for s in starts:
        r = dfs(s, 0, [s])
        if r:
            return r
    return None


rows = {}
for line in open(f"{D}/webqsp/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows[r["id"]] = r
print("train rows", len(rows), flush=True)

miss = kept = 0
maxchars = 0
out = open(f"{D}/path_groups_webqsp_train_ent.jsonl", "w", encoding="utf-8")
for line in open(f"{D}/mined_webqsp_train_vghn.jsonl", encoding="utf-8"):
    m = json.loads(line)
    if not m.get("cand"):
        continue
    r = rows.get(m["id"])
    if not r:
        continue
    g = build_graph(r["edges"])
    starts = [e for e in r["q_entity"] if e in g]
    gold = set(m.get("gold") or [])
    cands = []
    for c in m["cand"]:
        e, rels = c[0], c[3]
        ch = find_chain(g, starts, rels, e)
        if ch is None:
            miss += 1
            continue
        cands.append({"entity": e, "label": int(e in gold), "rels": rels, "ents": ch})
        maxchars = max(maxchars, len(" ".join(ch)))
    if cands:
        kept += 1
        out.write(json.dumps({"question": m["question"], "cands": cands},
                             ensure_ascii=False) + "\n")
out.close()
print(f"groups={kept} cands_missing_chain={miss} max_chain_chars={maxchars}", flush=True)
