# -*- coding: utf-8 -*-
"""配对检验：微调主干（最终系统） vs 冻结判定器（单次打分 / +SetRank）。"""
import json
import math
import sys

import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import SetRank, load_cache

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
d = json.load(open(f"{D}/eval_joint_s42_128.json"))
ours = {r["id"]: r["s1_hit"] for r in d["recs"]}
c = load_cache(f"{D}/setrank_cache_vp_test.pt")
idx = {q: i for i, q in enumerate(c["ids"])}
m = SetRank()
m.load_state_dict(torch.load(f"{D}/setrank_vp_d256_s42.pt", map_location="cpu",
                             weights_only=False))
m.eval()

frozen, setrank = {}, {}
with torch.no_grad():
    for q, i in idx.items():
        ents, f = c["ents"][i], c["feat"][i]
        gold = set(c["gold"][i])
        kf = max(range(len(ents)), key=lambda k: f[k][1])
        frozen[q] = int(ents[kf] in gold)
        ks = int(m(torch.tensor(c["h"][i], dtype=torch.float32),
                   torch.tensor(f, dtype=torch.float32)).argmax())
        setrank[q] = int(ents[ks] in gold)


def report(name, a, b):
    n = len(a)
    bd = int(((b == 1) & (a == 0)).sum())
    cd = int(((b == 0) & (a == 1)).sum())
    mm = bd + cd
    kk = min(bd, cd)
    p = min(1.0, 2 * sum(math.comb(mm, i) for i in range(kk + 1)) / 2 ** mm) if mm else 1.0
    diff = (a - b).astype(float)
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(0, n, (10000, n))].mean(axis=1)
    print(f"{name}: ours={a.mean():.4f} base={b.mean():.4f} d={diff.mean():+.4f} "
          f"b={bd} c={cd} p={p:.2e} CI=[{np.percentile(bs,2.5):+.4f},"
          f"{np.percentile(bs,97.5):+.4f}]", flush=True)


ids = [q for q in ours if q in frozen]
a = np.array([ours[q] for q in ids])
report("vs frozen verifier", a, np.array([frozen[q] for q in ids]))
report("vs frozen verifier+SetRank", a, np.array([setrank[q] for q in ids]))
