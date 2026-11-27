# -*- coding: utf-8 -*-
"""从已缓存的 G1 特征训练关系头并保存（供端到端评测加载）。

说明: 特征缓存 feat_{backing}_{dataset}/{split}.pt 自带 f/y/h,
      与当前 JSONL 行号不一定对齐, 但 f/y 本身是配对的, 直接用于训练即可。
      (G1a 的 0.869/0.905 结果就是在这批缓存上得到的。)

输出:
  E:\\GooseRAG\\data\\rog_qa\\{dataset}_{backing}_heads.pt
    {rel_state, rels, hidden, backing, dataset, test_top1, test_top5}
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g1_probe"))
from train_probe import train_head  # noqa: E402

DATA = r"E:\GooseRAG\data"
OUT = r"E:\GooseRAG\data\rog_qa"


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
    blobs = {sp: torch.load(os.path.join(DATA, f"feat_{args.backing}_{args.dataset}", f"{sp}.pt"),
                            map_location="cpu", weights_only=False)
             for sp in ("train", "val", "test")}
    for sp, b in blobs.items():
        print(f"{sp}: f={tuple(b['f'].shape)} y_max={int(b['y'].max())} "
              f"classes={len(rels)}", flush=True)
        assert int(b["y"].max()) < len(rels), f"{sp}: label >= n_classes"

    head = train_head(blobs["train"]["f"], blobs["train"]["y"].long(),
                      blobs["val"]["f"], blobs["val"]["y"].long(),
                      len(rels), "cpu", blobs["train"]["f"].shape[1],
                      epochs=args.epochs, seed=args.seed)
    t1 = topk_acc(head, blobs["test"]["f"], blobs["test"]["y"].long(), "cpu", 1)
    t5 = topk_acc(head, blobs["test"]["f"], blobs["test"]["y"].long(), "cpu", 5)
    print(f"test top1={t1:.4f} top5={t5:.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"{args.dataset}_{args.backing}_heads.pt")
    torch.save({
        "rel_state": head.state_dict(), "rels": rels,
        "hidden": blobs["train"]["f"].shape[1],
        "backing": args.backing, "dataset": args.dataset,
        "test_top1": t1, "test_top5": t5,
    }, out_path)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
