# -*- coding: utf-8 -*-
"""核对：e2e dump 的 397 题 vs cache 同题（路径判定器单次 vs SetRank）是否一致。"""
import json
import sys

import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import SetRank, load_cache

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
dump = json.load(open(f"{D}/pace_e2e_test400.jsonl"))
c = load_cache(f"{D}/setrank_cache_vp_test.pt")
idx = {qid: i for i, qid in enumerate(c["ids"])}
m = SetRank()
m.load_state_dict(torch.load(f"{D}/setrank_vp_d256_s42.pt", map_location="cpu",
                             weights_only=False))
m.eval()

n = miss = 0
bp = bo = bh = 0
agree = 0
with torch.no_grad():
    for rec in dump:
        i = idx.get(rec["id"])
        if i is None:
            miss += 1
            continue
        n += 1
        gold = set(c["gold"][i])
        ents = c["ents"][i]
        f = c["feat"][i]
        kb = max(range(len(ents)), key=lambda k: f[k][1])
        sc = m(torch.tensor(c["h"][i], dtype=torch.float32),
               torch.tensor(f, dtype=torch.float32))
        ko = int(sc.argmax())
        bp += ents[kb] in gold
        bo += ents[ko] in gold
        bh += rec["hit"]
        agree += int((ents[ko] in gold) == rec["hit"])
print(f"matched={n} missing_in_cache={miss}", flush=True)
print(f"cache base(path verifier)={bp/n:.4f}  cache SetRank={bo/n:.4f}", flush=True)
print(f"e2e dump base(plain verifier)={sum(r['base_hit'] for r in dump)/len(dump):.4f} "
      f"e2e dump full={bh/n:.4f}", flush=True)
print(f"per-question agreement(cache SetRank vs e2e full)={agree/n:.4f}", flush=True)
