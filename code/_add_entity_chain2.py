# -*- coding: utf-8 -*-
"""给 path_groups 原样补上实体链（严格保持原组/原候选集合不变）。

只对 path_groups_webqsp_train.jsonl 里已有的 (question, cands[entity, label, rels])
追加 "ents" 字段；不增删组或候选（除找不到实体链的候选），
保证与 rel/short/natural 三组严格同数据。
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


graph_of = {}
for line in open(f"{D}/webqsp/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    graph_of[r["question"]] = (build_graph(r["edges"]), list(r["q_entity"]))

groups = [json.loads(l) for l in open(f"{D}/path_groups_webqsp_train.jsonl",
                                     encoding="utf-8") if l.strip()]
kept = dropped = missing = 0
out = open(f"{D}/path_groups_webqsp_train_ent.jsonl", "w", encoding="utf-8")
for g in groups:
    q = g["question"]
    if q not in graph_of:
        dropped += 1
        continue
    gr, starts = graph_of[q]
    starts = [e for e in starts if e in gr]
    cands = []
    for c in g["cands"]:
        ch = find_chain(gr, starts, c.get("rels", ()), c["entity"])
        if ch is None:
            missing += 1
            continue
        cands.append({"entity": c["entity"], "label": c["label"],
                      "rels": c.get("rels", ()), "ents": ch})
    if cands:
        kept += 1
        out.write(json.dumps({"question": q, "cands": cands}, ensure_ascii=False) + "\n")
out.close()
print(f"groups_in={len(groups)} groups_out={kept} question_missing={dropped} cand_missing_chain={missing}",
      flush=True)
