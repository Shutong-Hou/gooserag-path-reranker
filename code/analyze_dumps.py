# -*- coding: utf-8 -*-
"""对逐题 dump 做配对显著性检验与错误分解。

用法:
  python analyze_dumps.py <dump_a.json> <dump_b.json>
a 为对照(pointwise)，b 为 Ours(listwise)。
"""
import json
import random
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return {r["id"]: r for r in d}


def hit1(r):
    p = r["pred"]
    if not p:
        return 0
    return int(p[0][0] in set(r["gold"]))


def hitk(r, k):
    s = set(r["gold"])
    return int(any(e in s for e, _ in r["pred"][:k]))


def qtype(q):
    w = q.strip().lower().split()
    if not w:
        return "other"
    return w[0] if w[0] in {"what", "which", "who", "where", "when", "how", "whose", "why"} else "other"


def main(pa, pb):
    A, B = load(pa), load(pb)
    ids = sorted(set(A) & set(B))
    print(f"aligned={len(ids)} (A={len(A)}, B={len(B)})")
    ha = [hit1(A[i]) for i in ids]
    hb = [hit1(B[i]) for i in ids]
    print(f"A hit1={sum(ha)/len(ha):.4f}  B hit1={sum(hb)/len(hb):.4f}  diff={(sum(hb)-sum(ha))/len(ha)*100:+.2f}pt")
    for k in (5, 10, 20, 100):
        aa = sum(hitk(A[i], k) for i in ids) / len(ids)
        bb = sum(hitk(B[i], k) for i in ids) / len(ids)
        print(f"top{k}: A={aa:.4f} B={bb:.4f} diff={(bb-aa)*100:+.2f}pt")

    b = sum(1 for x, y in zip(ha, hb) if y == 1 and x == 0)
    c = sum(1 for x, y in zip(ha, hb) if y == 0 and x == 1)
    n = b + c
    # McNemar 精确二项检验(双侧)
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)) if n else 1.0
    print(f"McNemar: B-only={b}  A-only={c}  p={p:.4g}")

    # 配对 bootstrap: 95% CI
    rng = random.Random(0)
    diffs = []
    for _ in range(10000):
        idx = [rng.randrange(len(ids)) for _ in ids]
        d = sum(hb[j] - ha[j] for j in idx) / len(idx) * 100
        diffs.append(d)
    diffs.sort()
    lo, hi = diffs[250], diffs[9750]
    print(f"bootstrap diff 95% CI = [{lo:+.2f}, {hi:+.2f}] pt")

    # 按问题类型 / gold 数量分解
    for label, keyf in (("qtype", lambda r: qtype(r["question"])),
                        ("n_gold", lambda r: str(min(len(r["gold"]), 4)))):
        agg = defaultdict(lambda: [0, 0, 0])
        for i in ids:
            k = keyf(A[i])
            agg[k][0] += 1
            agg[k][1] += hit1(A[i])
            agg[k][2] += hit1(B[i])
        print(f"-- {label} --")
        for k, (nn, x, y) in sorted(agg.items()):
            print(f"  {k:8s} n={nn:4d} A={x/nn:.3f} B={y/nn:.3f} d={(y-x)/nn*100:+.1f}pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
