# -*- coding: utf-8 -*-
"""把生成式对数似然与判定器分数融合，离线评测 Hits@1/5。"""
import argparse
import json


def z(xs):
    if len(xs) < 2:
        return [0.0] * len(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 + 1e-9
    return [(x - m) / sd for x in xs]


def evaluate(rows, gen, lam, key="sum"):
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for r in rows:
        cand = r["cand"]
        gold = set(r["gold"])
        g = gen.get(r["id"], {})
        fused = [0.5 * c[1] + 2.0 * c[2] for c in cand]
        zf = z(fused)
        gv = [g.get(c[0], {}).get(key) for c in cand]
        if lam > 0:
            present = [x for x in gv if x is not None]
            if present:
                mm = sum(present) / len(present)
                sd = (sum((x - mm) ** 2 for x in present) / len(present)) ** 0.5 + 1e-9
                gv = [((x - mm) / sd if x is not None else -3.0) for x in gv]
            else:
                gv = [0.0] * len(cand)
            sc = [a + lam * b for a, b in zip(zf, gv)]
        else:
            sc = zf
        order = sorted(range(len(cand)), key=lambda i: -sc[i])
        n += 1
        for k in hits:
            if set(cand[i][0] for i in order[:k]) & gold:
                hits[k] += 1
    return {k: v / max(n, 1) for k, v in hits.items()}, n


def gen_only(rows, gen, key="sum"):
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for r in rows:
        g = gen.get(r["id"], {})
        if not g:
            continue
        items = sorted(g.items(), key=lambda kv: -kv[1][key])
        n += 1
        gold = set(r["gold"])
        for k in hits:
            if set(e for e, _ in items[:k]) & gold:
                hits[k] += 1
    return {k: v / max(n, 1) for k, v in hits.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--gen", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.mined, encoding="utf-8")]
    gen = {}
    for line in open(args.gen, encoding="utf-8"):
        r = json.loads(line)
        gen[r["id"]] = {e: {"sum": s, "mean": m} for e, s, m in r["gen"]}

    out = {"n": len(rows)}
    base, _ = evaluate(rows, gen, 0.0)
    out["base_fused"] = base
    out["gen_only_sum"] = gen_only(rows, gen, "sum")[0]
    out["gen_only_mean"] = gen_only(rows, gen, "mean")[0]
    for lam in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
        out[f"blend_sum_{lam}"] = evaluate(rows, gen, lam, "sum")[0]
        out[f"blend_mean_{lam}"] = evaluate(rows, gen, lam, "mean")[0]
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
