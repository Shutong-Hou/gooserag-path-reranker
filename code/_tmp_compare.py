import json

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
bud = {}
for line in open(f"{D}/budget_webqsp_test_s42.jsonl", encoding="utf-8"):
    r = json.loads(line)
    bud[r["id"]] = r
mine = {}
for line in open(f"{D}/mined_webqsp_test_s42.jsonl", encoding="utf-8"):
    r = json.loads(line)
    mine[r["id"]] = r


def rank_of(r, key):
    c = r["cand"]
    if key == "fused":
        sc = [0.5 * x[1] + 2.0 * x[2] for x in c]
    else:
        sc = [x[2] for x in c]
    order = sorted(range(len(c)), key=lambda i: -sc[i])
    gold = set(r["gold"])
    for j, i in enumerate(order):
        if c[i][0] in gold:
            return j
    return None


agree = dis = 0
examples = []
n_m = n_b = 0
h_m = h_b = 0
for qid, m in mine.items():
    n_m += 1
    gr = rank_of(m, "fused")
    if gr == 0:
        h_m += 1
    b = bud.get(qid)
    if b is None:
        continue
    n_b += 1
    b0 = b["gold_rank"].get("200")
    if b0 == 0:
        h_b += 1
    if (gr == 0) == (b0 == 0):
        agree += 1
    else:
        dis += 1
        if len(examples) < 5:
            examples.append((qid, gr, b0, m["gold"][:2],
                             [c[0] for c in m["cand"][:3]]))
print("mine n", n_m, "hits1", round(h_m / max(n_m, 1), 4))
print("bud n", n_b, "hits1@200", round(h_b / max(n_b, 1), 4))
print("agree", agree, "disagree", dis)
for e in examples:
    print(e)
