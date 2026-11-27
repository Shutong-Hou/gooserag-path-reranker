# -*- coding: utf-8 -*-
"""按候选集规模分桶评估关系头（单候选样本单独列出）。"""
import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn

DATA = r"E:\GooseRAG\data"
OUT = os.path.join(DATA, "rog_qa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--head", required=True)
    args = ap.parse_args()
    rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                          encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"cond_{args.dataset}_test.jsonl"),
                                        encoding="utf-8")]
    f = torch.load(os.path.join(OUT, f"cond_feat_{args.dataset}_rwkv7_test.pt"),
                   map_location="cpu", weights_only=False)["f"]
    blob = torch.load(args.head, map_location="cpu", weights_only=False)
    head = nn.Linear(blob["hidden"], len(blob["rels"]))
    head.load_state_dict(blob["rel_state"])
    head.eval()

    buckets = defaultdict(lambda: [0, 0, 0])  # n, hit1, hit5
    with torch.no_grad():
        logits = head(f)
    for i, r in enumerate(rows):
        c = [rel2id[x] for x in r["cands"] if x in rel2id]
        t = rel2id.get(r["target"])
        if not c or t is None or t not in c:
            continue
        sz = len(c)
        name = "1" if sz == 1 else ("2-4" if sz <= 4 else ("5-19" if sz <= 19 else ">=20"))
        g = logits[i, torch.tensor(c)]
        order = g.argsort(descending=True)
        b = buckets[name]
        b[0] += 1
        b[1] += int(c[order[0].item()] == t)
        b[2] += int(any(c[j] == t for j in order[:min(5, sz)].tolist()))
    print(f"{args.head}")
    for k in ("1", "2-4", "5-19", ">=20"):
        if k in buckets:
            n, h1, h5 = buckets[k]
            print(f"  cand={k:5s} n={n:5d} top1={h1/n:.4f} top5={h5/n:.4f}")


if __name__ == "__main__":
    main()
