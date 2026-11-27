# -*- coding: utf-8 -*-
"""Paired comparison for two per-question prediction files (JSONL).

Each line: {"id":..., "gold":[...], "pred": "entity" | ["entity"] | null, "hit": 0/1}
(hit is ignored and recomputed from pred/gold.)

Usage:
  python _stats_pair_preds.py A.jsonl B.jsonl      # A = baseline, B = ours
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
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            gold = set(r.get("gold") or [])
            pred = r.get("pred")
            if isinstance(pred, (list, tuple)):
                pred = pred[0] if pred else None
            out[r["id"]] = 0 if pred is None else int(pred in gold)
    return out


def main():
    a = hits(sys.argv[1])
    b = hits(sys.argv[2])
    ids = sorted(set(a) & set(b))
    ha = np.array([a[i] for i in ids], dtype=np.int8)
    hb = np.array([b[i] for i in ids], dtype=np.int8)
    bc = int(((ha == 1) & (hb == 0)).sum())
    cb = int(((ha == 0) & (hb == 1)).sum())
    diff = (hb - ha).astype(np.float64)
    lo, hi = boot_ci(diff)
    res = {"n_paired": len(ids), "A_file": sys.argv[1], "B_file": sys.argv[2],
           "A_h1": float(ha.mean()), "B_h1": float(hb.mean()),
           "delta_pt": float((hb.mean() - ha.mean()) * 100),
           "b_Aonly": bc, "c_Bonly": cb, "mcnemar_p": mcnemar(bc, cb),
           "boot_ci_pt": [lo * 100, hi * 100]}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w", encoding="utf-8") as w:
            json.dump(res, w, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
