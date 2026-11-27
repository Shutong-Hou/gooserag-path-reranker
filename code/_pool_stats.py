# -*- coding: utf-8 -*-
"""候选池规模与一致性检查：cache 池 vs 端到端 verifier-top 200。"""
import json
import statistics

import torch

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
for split in ("train", "validation", "test"):
    c = torch.load(f"{D}/setrank_cache_vp_{split}.pt", map_location="cpu", weights_only=False)
    n = [len(e) for e in c["ents"]]
    print(f"cache {split}: nq={len(n)} avg={statistics.mean(n):.1f} "
          f"med={statistics.median(n):.0f} max={max(n)} gt200={sum(1 for x in n if x > 200)}",
          flush=True)

rows = [json.loads(l) for l in open(f"{D}/mined_webqsp_test_vghn.jsonl", encoding="utf-8")]
n = [len(r["cand"]) for r in rows if r["cand"]]
print(f"mined test: nq={len(n)} avg={statistics.mean(n):.1f} med={statistics.median(n):.0f} "
      f"max={max(n)} empty={sum(1 for r in rows if not r['cand'])}", flush=True)
