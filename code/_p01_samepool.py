# -*- coding: utf-8 -*-
"""P0-1（WebQSP）：同一候选池上 plain vs path 判定器的逐题配对显著性。

数据来源（与已发表数字完全同源）：
  - `mined_webqsp_test_vghn.jsonl`：每题候选 + walker 分(c[1]) + plain 判定器分(c[2])
  - `setrank_cache_vp_test.pt`         ：同一批候选 + walker 分 + path 判定器分(feat[:,1])
唯一变量 = 判定器（plain prompt vs path prompt）。CPU 运行。
"""
import json
import math
import sys

import numpy as np

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import load_cache  # noqa: E402


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pair(a, b, na, nb):
    b_, c_ = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    lo, hi = boot_ci((b - a).astype(np.float64))
    return {"A": na, "B": nb, "A_h1": float(a.mean()), "B_h1": float(b.mean()),
            "delta_pt": float((b.mean() - a.mean()) * 100),
            "b_Aonly": b_, "c_Bonly": c_, "mcnemar_p": mcnemar(b_, c_),
            "boot_ci_pt": [lo * 100, hi * 100]}


def main():
    mined = {}
    for line in open(f"{D}/mined_webqsp_test_vghn.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("cand"):
            mined[r["id"]] = (set(r.get("gold") or []), r["cand"])

    c = load_cache(f"{D}/setrank_cache_vp_test.pt")
    pathq = {}
    for i, qid in enumerate(c["ids"]):
        pathq[qid] = (set(c["gold"][i]), c["ents"][i],
                      np.asarray(c["feat"][i], dtype=np.float64))

    ids = [q for q in mined if q in pathq]
    plain_h = {k: [] for k in ("argmax", "fused")}
    path_h = {k: [] for k in ("argmax", "fused")}
    walk_h = []
    for q in ids:
        gold, cand = mined[q]
        pg, ents, f = pathq[q]
        gold = gold or pg
        if not gold or not cand:
            continue
        ws = np.array([x[1] for x in cand], dtype=np.float64)
        ps = np.array([x[2] for x in cand], dtype=np.float64)
        es = [x[0] for x in cand]
        plain_h["argmax"].append(1 if es[int(np.argmax(ps))] in gold else 0)
        plain_h["fused"].append(1 if es[int(np.argmax(0.5 * ws + 2.0 * ps))] in gold else 0)
        walk_h.append(1 if es[int(np.argmax(ws))] in gold else 0)
        path_h["argmax"].append(1 if ents[int(np.argmax(f[:, 1]))] in gold else 0)
        path_h["fused"].append(1 if ents[int(np.argmax(0.5 * f[:, 0] + 2.0 * f[:, 1]))] in gold else 0)

    n = len(walk_h)
    walk = np.array(walk_h, dtype=np.int8)
    res = {"dataset": "webqsp", "split": "test", "n_paired": n,
           "walker_h1": float(walk.mean())}
    for k in ("argmax", "fused"):
        res[f"plain_{k}_vs_path_{k}"] = pair(
            np.array(plain_h[k], dtype=np.int8), np.array(path_h[k], dtype=np.int8),
            f"plain_{k}", f"path_{k}")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{D}/p01_stats_webqsp_samepool.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
