# -*- coding: utf-8 -*-
"""评测已联合训练的 checkpoint：可指定推理 max_len，并逐题 dump（供配对检验）。"""
import argparse
import json

import torch

import train_joint as J
from train_verifier_ft import FTVerifier
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(J.MODELS["rwkv7"], trust_remote_code=True,
                                       local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    ft = FTVerifier(J.MODELS["rwkv7"], "double")
    model = J.Joint(ft.lm, ft.head).to(dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.lm.load_state_dict(ck["lm"])
    model.s1.load_state_dict(ck["s1"])
    model.s2.load_state_dict(ck["s2"])
    model.eval()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    recs = []
    h1 = h5 = hs1 = h5s1 = 0
    n = 0
    with torch.no_grad():
        for r in rows:
            gold = set(r.get("gold") or [])
            cand = list(r.get("cand") or [])
            if not cand:
                recs.append({"id": r["id"], "n": 0, "s1_hit": 0, "s2_hit": 0,
                             "s1_rank": None, "s2_rank": None, "gold": sorted(gold)})
                continue
            ids, mask = J.make_inputs(tok, r["question"], cand, pad_id, args.max_len)
            feats = torch.tensor(J.make_feats(tok, r["question"], cand),
                                 dtype=torch.float32, device=dev)
            v, s2 = model(ids.to(dev), mask.to(dev), feats)
            o1 = v.argsort(descending=True).tolist()
            o2 = s2.argsort(descending=True).tolist()
            n += 1
            r1 = next((j for j, k in enumerate(o1) if cand[k][0] in gold), None)
            r2 = next((j for j, k in enumerate(o2) if cand[k][0] in gold), None)
            h1 += r2 == 0
            h5 += r2 is not None and r2 < 5
            hs1 += r1 == 0
            h5s1 += r1 is not None and r1 < 5
            recs.append({"id": r["id"], "n": len(cand),
                         "s1_hit": int(r1 == 0), "s2_hit": int(r2 == 0),
                         "s1_rank": r1, "s2_rank": r2, "gold": sorted(gold)})

    res = {"ckpt": args.ckpt, "max_len": args.max_len, "n": n,
           "s2_h1": h1 / max(n, 1), "s2_h5": h5 / max(n, 1),
           "s1_h1": hs1 / max(n, 1), "s1_h5": h5s1 / max(n, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.out:
        json.dump({"res": res, "recs": recs}, open(args.out, "w"), ensure_ascii=False)
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
