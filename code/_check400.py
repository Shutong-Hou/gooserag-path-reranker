# -*- coding: utf-8 -*-
"""一致性检查：cache 前 400 题的数字是否复现 E11 的端到端 400 题结果（0.5264 -> 0.5945）。"""
import torch

torch.set_num_threads(8)
import sys
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import SetRank, load_cache

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
c = load_cache(f"{D}/setrank_cache_vp_test.pt")
m = SetRank()
m.load_state_dict(torch.load(f"{D}/setrank_vp_d256_s42.pt", map_location="cpu",
                             weights_only=False))
m.eval()

for N in (400, len(c["ids"])):
    base = our = 0
    with torch.no_grad():
        for i in range(N):
            gold = set(c["gold"][i])
            ents = c["ents"][i]
            f = c["feat"][i]
            kb = max(range(len(ents)), key=lambda k: f[k][1])
            sc = m(torch.tensor(c["h"][i], dtype=torch.float32),
                   torch.tensor(f, dtype=torch.float32))
            ko = int(sc.argmax())
            base += ents[kb] in gold
            our += ents[ko] in gold
    print(f"N={N} base={base/N:.4f} ours={our/N:.4f} delta={(our-base)/N:+.4f}", flush=True)
