# -*- coding: utf-8 -*-
"""训练实体条件化关系头（问题 + 当前实体 + 前缀 -> 下一跳关系）。"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g1_probe"))
from train_probe import train_head  # noqa: E402

DATA = r"E:\GooseRAG\data"
OUT = os.path.join(DATA, "rog_qa")


def topk_acc(head, feat, labels, device, k=5):
    head.eval()
    with torch.no_grad():
        logits = head(feat.to(device))
    kk = min(k, logits.shape[1])
    return (logits.topk(kk, dim=1).indices == labels.to(device)[:, None]).any(1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cwq", choices=["webqsp", "cwq"])
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                          encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}
    blobs = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(OUT, f"cond_feat_{args.dataset}_{args.backing}_{sp}.pt")
        blobs[sp] = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{sp}: f={tuple(blobs[sp]['f'].shape)}", flush=True)
    y = {sp: torch.tensor([rel2id[t] for t in blobs[sp]["labels"]]) for sp in blobs}

    head = train_head(blobs["train"]["f"], y["train"], blobs["val"]["f"], y["val"],
                      len(rels), "cpu", blobs["train"]["f"].shape[1],
                      epochs=args.epochs, seed=args.seed)
    t1 = topk_acc(head, blobs["test"]["f"], y["test"], "cpu", 1)
    t5 = topk_acc(head, blobs["test"]["f"], y["test"], "cpu", 5)
    print(f"cond head test top1={t1:.4f} top5={t5:.4f}", flush=True)
    out_path = os.path.join(OUT, f"cond_{args.dataset}_{args.backing}_heads.pt")
    torch.save({"rel_state": head.state_dict(), "rels": rels,
                "hidden": blobs["train"]["f"].shape[1],
                "backing": args.backing, "dataset": args.dataset,
                "test_top1": t1, "test_top5": t5}, out_path)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
