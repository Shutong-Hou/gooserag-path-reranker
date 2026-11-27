# -*- coding: utf-8 -*-
"""最终系统（微调主干）vs 冻结判定器+SetRank 的逐题配对检验。"""
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

base = {}
with torch.no_grad():
    for q, i in idx.items():
        k = int(m(torch.tensor(c["h"][i], dtype=torch.float32),
                  torch.tensor(c["feat"][i], dtype=torch.float32)).argmax())
        base[q] = int(c["ents"][i][k] in set(c["gold"][i]))

ids = [q for q in ours if q in base]
a = np.array([ours[q] for q in ids])
b_ = np.array([base[q] for q in ids])
print(f"n={len(ids)} ours={a.mean():.4f} setrank={b_.mean():.4f}", flush=True)
b = int(((b_ == 1) & (a == 0)).sum())
cc = int(((b_ == 0) & (a == 1)).sum())
m_ = b + cc
k = min(b, cc)
p = min(1.0, 2 * sum(math.comb(m_, i) for i in range(k + 1)) / 2 ** m_) if m_ else 1.0
print(f"McNemar b={b} c={cc} p={p:.3e}", flush=True)
rng = np.random.default_rng(0)
diff = (a - b_).astype(float)
bs = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
print(f"delta={diff.mean():+.4f} CI=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]",
      flush=True)
