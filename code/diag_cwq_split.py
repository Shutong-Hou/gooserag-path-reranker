# -*- coding: utf-8 -*-
"""诊断 CWQ val800 与 test 的难度差异：只用同一判定器分数，看两个 split 的 H@1 差多少。"""
import json
import sys

import torch

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def stat2(name):
    d = torch.load(f"{D}/setrank_cache_cwq_{name}.pt", map_location="cpu", weights_only=False)
    golds, feats, ents = d["gold"], d["feat"], d["ents"]
    n = len(golds)
    h_v = h_w = h_f = in_c = 0
    n_cand = n_gold = 0
    for i in range(n):
        gs = set(golds[i] if isinstance(golds[i], (list, tuple)) else [golds[i]])
        f = feats[i]
        if isinstance(f, torch.Tensor):
            f = f.tolist()
        e = ents[i]
        if isinstance(e, torch.Tensor):
            e = e.tolist()
        if not f or not e:
            continue
        n_cand += len(f)
        n_gold += len(gs)
        vi = max(range(len(f)), key=lambda j: f[j][1])
        wi = max(range(len(f)), key=lambda j: f[j][0])
        fi = max(range(len(f)), key=lambda j: 0.5 * f[j][0] + 2.0 * f[j][1])
        h_v += int(e[vi] in gs)
        h_w += int(e[wi] in gs)
        h_f += int(e[fi] in gs)
        in_c += int(any(x in gs for x in e))
    return {
        "n": n,
        "verifier_top1": h_v / n,
        "walker_top1": h_w / n,
        "fused_top1": h_f / n,
        "in_cand": in_c / n,
        "avg_n_cand": n_cand / n,
        "avg_n_gold": n_gold / n,
    }


out = {}
for s in ["val800", "test"]:
    try:
        out[s] = stat2(s)
    except Exception as ex:
        out[s] = {"error": repr(ex)}
print(json.dumps(out, ensure_ascii=False, indent=1))
