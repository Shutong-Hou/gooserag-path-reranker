# -*- coding: utf-8 -*-
"""测「插入我们的重排层」的边际成本：只对既有候选打分需要多久。

对照 RoG 的 15.7 候选/题 与 我们系统的 200 候选/题。
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

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True,
                                    local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
fb = torch.load(CKPT, map_location="cpu", weights_only=False)
ft.lm.load_state_dict(fb["lm_state"])
ft.head.load_state_dict(fb["head_state"])
ft.to("cuda").eval()

for K in (16, 200):
    texts_all = [[E.prompt_verifier_path(r["question"], c[3], c[0])
                  for c in r["cand"][:K]] for r in rows]
    for i in range(5):
        E.score_ft(ft, tok, texts_all[i], "cuda", 32, 160)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in texts_all:
        E.score_ft(ft, tok, t, "cuda", 32, 160)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"K={K}: {dt/len(rows)*1000:.1f} ms/query "
          f"({sum(len(t) for t in texts_all)/dt:.1f} texts/s)", flush=True)
