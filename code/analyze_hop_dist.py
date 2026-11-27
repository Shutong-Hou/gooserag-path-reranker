# -*- coding: utf-8 -*-
"""统计 gold 在子图中的最短跳数分布（全关系 / 合法动作关系）。

用于解释 CWQ validation 与 test 的难度差，并决定 max_hops。
用法:
  python analyze_hop_dist.py --dataset cwq --splits validation,test --max-hops 5
"""
import argparse
import json
import os
from collections import Counter, defaultdict

DATA = r"E:\GooseRAG\data"
ROG_QA = os.path.join(DATA, "rog_qa")
GENERIC_PREFIX = ("common.topic.", "freebase.valuenotation.", "common.webpage.",
                  "common.document.", "type.object.")


def is_generic(rel):
    return rel.startswith(GENERIC_PREFIX)


def build_graph(edges):
    out = defaultdict(lambda: defaultdict(list))
    for h, r, t in edges:
        out[h][r].append(t)
    return out


def bfs_dist(starts, gold, out, max_hops, relset=None, ban_generic=False):
    cur = set(starts)
    seen = set(cur)
    for hop in range(1, max_hops + 1):
        nxt = set()
        for e in cur:
            for r, tails in out.get(e, {}).items():
                if relset is not None and r not in relset:
                    continue
                if ban_generic and is_generic(r):
                    continue
                nxt.update(tails)
        nxt -= seen
        if nxt & gold:
            return hop
        if not nxt:
            return None
        seen |= nxt
        cur = nxt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cwq")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--max-hops", type=int, default=5)
    args = ap.parse_args()

    rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                           encoding="utf-8"))
    relset = set(rels)
    for sp in args.splits.split(","):
        rows = [json.loads(l) for l in
                open(os.path.join(ROG_QA, args.dataset, f"{sp}.jsonl"), encoding="utf-8")]
        all_d, legal_d = Counter(), Counter()
        n = n_all = n_legal = 0
        for r in rows:
            out = build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            if not starts or not gold:
                continue
            n += 1
            d = bfs_dist(starts, gold, out, args.max_hops)
            if d is None:
                all_d["inf"] += 1
            else:
                all_d[d] += 1
                n_all += 1
            dl = bfs_dist(starts, gold, out, args.max_hops,
                          relset=relset, ban_generic=True)
            if dl is None:
                legal_d["inf"] += 1
            else:
                legal_d[dl] += 1
                n_legal += 1
        print(json.dumps({
            "dataset": args.dataset, "split": sp, "n": n,
            "all_relation_dist": dict(sorted(all_d.items(), key=lambda x: str(x[0]))),
            "all_relation_reachable": n_all / max(n, 1),
            "legal_relation_dist": dict(sorted(legal_d.items(), key=lambda x: str(x[0]))),
            "legal_relation_reachable": n_legal / max(n, 1),
            "max_hops": args.max_hops,
        }, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
