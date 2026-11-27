# -*- coding: utf-8 -*-
"""逐题配对显著性：McNemar 精确检验 + 配对 bootstrap 置信区间。

输入：eval_pace.py --dump 产生的 json（每题含 hit / base_hit）。
"""
import json
import random
import sys
from math import comb


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main(path, n_boot=10000, seed=0):
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    h = sum(r["hit"] for r in rows)
    bh = sum(r["base_hit"] for r in rows)
    b = sum(1 for r in rows if r["hit"] and not r["base_hit"])  # 基线错、我们对
    c = sum(1 for r in rows if r["base_hit"] and not r["hit"])  # 基线对、我们错
    d = [r["hit"] - r["base_hit"] for r in rows]

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]

    out = {
        "n": n,
        "ours_h1": h / n,
        "base_h1": bh / n,
        "delta_pt": (h - bh) / n * 100,
        "ours_right_base_wrong": b,
        "base_right_ours_wrong": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "bootstrap_95ci_pt": [lo * 100, hi * 100],
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
