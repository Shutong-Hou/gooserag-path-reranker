# -*- coding: utf-8 -*-
"""同一候选池上的四路排序对比（WebQSP test, n=1615）：
  walker / verifier / fused(0.5w+2v) / SetRank(5 seed)
唯一变量是排序函数，候选池与特征完全相同。
"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import SetRank, load_cache

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
c = load_cache(f"{D}/setrank_cache_vp_test.pt")
n = len(c["ids"])
GOLD = [set(g) for g in c["gold"]]

hits = {k: {} for k in ("walker", "verifier", "fused")}
for k in hits:
    for K in (1, 5, 10):
        hits[k][K] = np.zeros(n, dtype=np.int8)
for i in range(n):
    ents = c["ents"][i]
    f = np.asarray(c["feat"][i], dtype=np.float64)
    g = GOLD[i]
    for k, key in ((f[:, 0], "walker"), (f[:, 1], "verifier"),
                   (0.5 * f[:, 0] + 2.0 * f[:, 1], "fused")):
        order = np.argsort(-k)
        for K in (1, 5, 10):
            hits[key][K][i] = any(ents[j] in g for j in order[:K])

H32 = [torch.tensor(x, dtype=torch.float32) for x in c["h"]]
F32 = [torch.tensor(x, dtype=torch.float32) for x in c["feat"]]
sr_hits = []
for s in (42, 43, 44, 45, 46):
    m = SetRank()
    m.load_state_dict(torch.load(f"{D}/setrank_vp_d256_s{s}.pt", map_location="cpu",
                                 weights_only=False))
    m.eval()
    v = {K: np.zeros(n, dtype=np.int8) for K in (1, 5, 10)}
    with torch.no_grad():
        for i in range(n):
            order = m(H32[i], F32[i]).argsort(descending=True).tolist()
            for K in (1, 5, 10):
                v[K][i] = any(c["ents"][i][j] in GOLD[i] for j in order[:K])
    sr_hits.append(v)
    print(f"SetRank s{s}: h1={v[1].mean():.4f} h5={v[5].mean():.4f} h10={v[10].mean():.4f}",
          flush=True)

res = {f"{k}_h{K}": float(hits[k][K].mean()) for k in hits for K in (1, 5, 10)}
res["setrank_h1_mean"] = float(np.mean([v[1].mean() for v in sr_hits]))
res["setrank_h5_mean"] = float(np.mean([v[5].mean() for v in sr_hits]))
res["setrank_h10_mean"] = float(np.mean([v[10].mean() for v in sr_hits]))
res["setrank_h1_std"] = float(np.std([v[1].mean() for v in sr_hits], ddof=1))
res["n"] = n
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f"{D}/pool_table_test.json", "w"), indent=1)
