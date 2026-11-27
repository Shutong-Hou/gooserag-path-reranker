# -*- coding: utf-8 -*-
"""P0-1 显著性：两份 --dump 逐题结果配对比较（McNemar 精确检验 + bootstrap 95% CI）。

用法:
  python _stats_pair.py A.json B.json   # A=baseline, B=ours
dump 格式: [{"id":..., "gold":[...], "pred":[...]}, ...]
"""
import json
import math
import sys

import numpy as np

np.random.seed(0)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hits(path):
    d = {r["id"]: r for r in json.load(open(path, encoding="utf-8"))}
    out = {}
    for k, r in d.items():
        gold = set(r["gold"])
        pred = r["pred"]
        if not pred:
            out[k] = 0
            continue
        top = pred[0]
        if isinstance(top, (list, tuple)):   # dump 里 pred 元素是 [entity, score]
            top = top[0]
        out[k] = 1 if top in gold else 0
    return out


def main():
    a = hits(sys.argv[1])
    b = hits(sys.argv[2])
    ids = sorted(set(a) & set(b))
    ha = np.array([a[i] for i in ids], dtype=np.int8)
    hb = np.array([b[i] for i in ids], dtype=np.int8)
    n = len(ids)
    bc = int(((ha == 1) & (hb == 0)).sum())   # baseline 对 / ours 错
    cb = int(((ha == 0) & (hb == 1)).sum())   # baseline 错 / ours 对
    diff = (hb - ha).astype(np.float64)
    lo, hi = boot_ci(diff)
    res = {
        "n_paired": n,
        "A_file": sys.argv[1], "B_file": sys.argv[2],
        "A_h1": float(ha.mean()), "B_h1": float(hb.mean()),
        "delta_pt": float((hb.mean() - ha.mean()) * 100),
        "b_Aonly": bc, "c_Bonly": cb,
        "mcnemar_p": mcnemar(bc, cb),
        "boot_ci_pt": [lo * 100, hi * 100],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
