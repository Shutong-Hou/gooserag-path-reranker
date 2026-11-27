# -*- coding: utf-8 -*-
"""在同一份候选池上评测任意 backbone 的判定器（P0-2 同参数量对照）。

用法:
  python eval_backbone_verifier.py --backing gpt2 --ckpt verifier_ft_webqsp_gpt2_path.pt \
      --mined mined_webqsp_test_vghn.jsonl
输出: H@1 / H@5 / in_cand（候选池完全一致，只有判定器不同）
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402
from train_verifier_ft import FTVerifier  # noqa: E402

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backing", required=True, choices=["rwkv7", "gpt2", "pythia"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mined", default=f"{D}/mined_webqsp_test_vghn.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    path = E.MODEL_NAMES[args.backing]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = FTVerifier(path, "double")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m.lm.load_state_dict(ck["lm_state"])
    m.head.load_state_dict(ck["head_state"])
    m.to("cuda").eval()

    rows = [json.loads(l) for l in open(args.mined, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    hit1 = hit5 = inc = n = 0
    with torch.no_grad():
        for r in rows:
            gold = set(r.get("gold") or [])
            cand = r.get("cand") or []
            if not gold:
                continue
            n += 1
            if not cand:
                continue
            if any(c[0] in gold for c in cand):
                inc += 1
            texts = [E.prompt_verifier_path(r["question"], c[3], c[0]) for c in cand]
            vs = E.score_ft(m, tok, texts, "cuda", args.batch, args.max_len)
            order = sorted(range(len(vs)), key=lambda k: -vs[k])
            hit1 += int(cand[order[0]][0] in gold)
            hit5 += int(any(cand[k][0] in gold for k in order[:5]))
    res = {"backing": args.backing, "ckpt": os.path.basename(args.ckpt), "n": n,
           "h1": hit1 / max(n, 1), "h5": hit5 / max(n, 1),
           "in_cand": inc / max(n, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    out = f"{D}/p02_{args.backing}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
