# -*- coding: utf-8 -*-
"""逐题剂量-反应：候选池里金答案个数 vs path 增益（把"密度"从数据集级降到题级）。

WebQSP 直接用 mined + setrank cache（CPU）。
"""
import json
import sys

import numpy as np

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import load_cache  # noqa: E402


def pair_stats(rows):
    n = len(rows)
    if n == 0:
        return None
    plain = np.array([r[0] for r in rows])
    path = np.array([r[1] for r in rows])
    return {"n": n, "plain_h1": float(plain.mean()), "path_h1": float(path.mean()),
            "delta_pt": float((path.mean() - plain.mean()) * 100)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default=f"{D}/mined_webqsp_test_vghn.jsonl")
    ap.add_argument("--cache", default=f"{D}/setrank_cache_vp_test.pt")
    ap.add_argument("--tag", default="webqsp")
    a = ap.parse_args()

    mined = {}
    for line in open(a.mined, encoding="utf-8"):
        r = json.loads(line)
        if r.get("cand"):
            mined[r["id"]] = (set(r.get("gold") or []), r["cand"])

    c = load_cache(a.cache)
    pathq = {q: (set(c["gold"][i]), c["ents"][i],
                 np.asarray(c["feat"][i], dtype=np.float64))
             for i, q in enumerate(c["ids"])}

    buckets = {}
    for q, (gold, cand) in mined.items():
        if q not in pathq:
            continue
        pg, ents, f = pathq[q]
        gold = gold or pg
        if not gold or not cand:
            continue
        es = [x[0] for x in cand]
        ws = np.array([x[1] for x in cand], dtype=np.float64)
        ps = np.array([x[2] for x in cand], dtype=np.float64)
        n_gold = sum(1 for e in es if e in gold)
        k = n_gold if n_gold < 4 else 4
        plain_hit = 1 if es[int(np.argmax(0.5 * ws + 2.0 * ps))] in gold else 0
        path_hit = 1 if ents[int(np.argmax(0.5 * f[:, 0] + 2.0 * f[:, 1]))] in gold else 0
        buckets.setdefault(k, []).append((plain_hit, path_hit))

    out = {}
    for k in sorted(buckets):
        label = {0: "0", 1: "1", 2: "2", 3: "3", 4: ">=4"}[k]
        out[label] = pair_stats(buckets[k])
        print(f"gold_in_pool={label}: {out[label]}", flush=True)

    json.dump(out, open(f"{D}/p17_doseresponse_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
