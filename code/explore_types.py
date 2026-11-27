# -*- coding: utf-8 -*-
"""探针：WebQSP 子图里实体类型（notable_types）的覆盖率与分布。"""
import collections
import json

P = "/root/autodl-tmp/GooseRAG/data/rog_qa/webqsp/test.jsonl"

n = 0
ans_total = ans_typed = 0
cand_total = cand_typed = 0
type_cnt = collections.Counter()
qtype_ans_types = collections.defaultdict(collections.Counter)


def qtype(q):
    q = q.lower()
    if q.startswith("who") or "whose" in q:
        return "who"
    if q.startswith("when") or "what year" in q or "what date" in q:
        return "when"
    if q.startswith("where"):
        return "where"
    if "how many" in q or "how much" in q:
        return "howmany"
    return "other"


for line in open(P, encoding="utf-8"):
    d = json.loads(line)
    n += 1
    types = {}
    for s, r, o in d["edges"]:
        if r == "common.topic.notable_types":
            types.setdefault(s, set()).add(o)
    qt = qtype(d["question"])
    for e in d.get("a_entity") or []:
        ans_total += 1
        if e in types:
            ans_typed += 1
            for t in types[e]:
                type_cnt[t] += 1
                qtype_ans_types[qt][t] += 1
    for e in (d.get("a_entity") or []) + (d.get("q_entity") or []):
        cand_total += 1
        if e in types:
            cand_typed += 1

print(f"rows={n}")
print(f"answer_typed={ans_typed}/{ans_total} = {ans_typed/max(ans_total,1):.3f}")
print(f"entity_incl_topic_typed={cand_typed}/{cand_total} = {cand_typed/max(cand_total,1):.3f}")
print("\ntop types (answers):")
for t, c in type_cnt.most_common(20):
    print(f"  {c:5d}  {t}")
print("\nby question type:")
for qt, cc in qtype_ans_types.items():
    print(f"  [{qt}] " + ", ".join(f"{t}={c}" for t, c in cc.most_common(5)))
