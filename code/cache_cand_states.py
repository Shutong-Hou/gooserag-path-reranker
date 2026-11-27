# -*- coding: utf-8 -*-
"""SetRank 第一步：缓存判定器主干对每个候选的隐状态。

输出 .pt: {"ids":[...], "ents":[[...]], "h":[tensor(n_i,768) fp16, ...],
          "feat":[[[walker,verifier,path_len,overlap,n_same_path],...]],
          "gold":[[...]]}
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--path", action="store_true",
                    help="用路径感知 prompt 编码（需与 path-aware verifier 配套）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.mined, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    ft = E.EvalFTVerifier(E.MODEL_NAMES[args.backing], feat_norm="double")
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to(device).eval()
    tok = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES[args.backing],
                                        trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"rows={len(rows)} device={device}", flush=True)

    out = {"ids": [], "ents": [], "h": [], "feat": [], "gold": []}
    for i, r in enumerate(rows):
        cand = r["cand"]
        if not cand:
            continue
        if args.path:
            texts = [E.prompt_verifier_path(r["question"], c[3], c[0]) for c in cand]
        else:
            texts = [E.prompt_verifier(r["question"], c[0]) for c in cand]
        h = E.encode_texts(ft.lm, tok, texts, device, args.backing,
                           batch_size=args.batch, max_len=args.max_len,
                           feat_norm="double").half()
        if args.path:
            vs = E.score_ft(ft, tok, texts, device, args.batch, args.max_len)
        else:
            vs = [c[2] for c in cand]
        q = toks(r["question"])
        paths = [tuple(c[3]) for c in cand]
        cnt = {}
        for p in paths:
            cnt[p] = cnt.get(p, 0) + 1
        feats = []
        for c, p, v in zip(cand, paths, vs):
            et = toks(c[0])
            feats.append([float(c[1]), float(v), float(len(p)),
                          len(q & et) / max(len(et), 1), float(cnt[p])])
        out["ids"].append(r["id"])
        out["ents"].append([c[0] for c in cand])
        out["h"].append(h)
        out["feat"].append(feats)
        out["gold"].append(sorted(r["gold"]))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    torch.save(out, args.out)
    print(f"saved {args.out} n={len(out['ids'])}", flush=True)


if __name__ == "__main__":
    main()
