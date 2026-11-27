# -*- coding: utf-8 -*-
"""创新 C 的备选方案：训练无关的“关系名语义打分”冷启动探针。

对每道题（prefix=()），用 LM 状态与关系名状态的余弦相似度给全部关系排序，
看在留出关系（训练中从未出现）上的 top1/top5 是否显著高于随机。
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402


def name_text(rel):
    return rel.replace(".", " ").replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagged", required=True, help="tag_held_questions.py 的输出")
    ap.add_argument("--rels", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rels = json.load(open(args.rels, encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}
    held = set(json.load(open(args.held, encoding="utf-8")))
    rows = [json.loads(l) for l in open(args.tagged, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    model, tok, _ = E.load_model(args.backing)
    model.to(device)

    Xr = E.encode_texts(model, tok, [name_text(r) for r in rels], device,
                        args.backing, 64, 64, feat_norm="double").float()
    Xr = Xr / (Xr.norm(dim=1, keepdim=True) + 1e-9)

    def eval_subset(sub, tag):
        if not sub:
            print(f"{tag}: n=0")
            return
        qs = [E.prompt(r["question"], ()) for r in sub]
        Q = E.encode_texts(model, tok, qs, device, args.backing, 64, 320,
                           feat_norm="double").float()
        Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
        sim = Q @ Xr.T
        t1 = t5 = t10 = 0
        for r, row in zip(sub, sim):
            tgt = rel2id.get(r["gold_hop1"])
            if tgt is None:
                continue
            order = row.topk(10).indices.tolist()
            t1 += int(order[0] == tgt)
            t5 += int(tgt in order[:5])
            t10 += int(tgt in order[:10])
        n = len(sub)
        print(json.dumps({"subset": tag, "n": n, "top1": t1 / n, "top5": t5 / n,
                          "top10": t10 / n}, ensure_ascii=False), flush=True)

    hop1 = [r for r in rows if r["gold_hop1"]]
    eval_subset([r for r in hop1 if r["held"]], "held_hop1")
    eval_subset([r for r in hop1 if not r["held"]], "seen_hop1")
    eval_subset(hop1, "all_hop1")

    # 随机基线
    k = len(rels)
    print(json.dumps({"random_top1": 1 / k, "random_top5": 5 / k,
                      "n_rels": k}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
