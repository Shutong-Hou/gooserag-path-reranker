# -*- coding: utf-8 -*-
"""P0-4b 单题延迟口径：同一 joint 判定器在不同候选数 / batch 下的打分耗时。

用途：把「插入成本 +98ms（15.7 候选）」与「全系统 486ms（200 候选，batch=1）」
放进同一个测量口径，避免两套数字互相矛盾。
"""
import json
import time

import torch

import eval_e2e as E

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
CKPT = f"{D}/verifier_ft_webqsp_joint_s42.pt"

rows = [json.loads(l) for l in open(f"{D}/mined_webqsp_test_vghn.jsonl",
                                    encoding="utf-8")]
rows = [r for r in rows if r.get("cand")][:200]

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True,
                                    local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
fb = torch.load(CKPT, map_location="cpu", weights_only=False)
ft.lm.load_state_dict(fb["lm_state"])
ft.head.load_state_dict(fb["head_state"])
ft.to("cuda").eval()

res = {}
for K in (16, 32, 200):
    texts_all = [[E.prompt_verifier_path(r["question"], c[3], c[0])
                  for c in r["cand"][:K]] for r in rows]
    for bs in (1, 32):
        for i in range(3):
            E.score_ft(ft, tok, texts_all[i], "cuda", bs, 160)
        torch.cuda.synchronize()
        t0 = time.time()
        for t in texts_all:
            E.score_ft(ft, tok, t, "cuda", bs, 160)
        torch.cuda.synchronize()
        dt = time.time() - t0
        res[f"K{K}_bs{bs}_ms"] = dt / len(rows) * 1000
        print(f"K={K} bs={bs}: {dt/len(rows)*1000:.1f} ms/query", flush=True)

res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
print(json.dumps(res, indent=1))
json.dump(res, open(f"{D}/p04_latency.json", "w"), indent=1)
