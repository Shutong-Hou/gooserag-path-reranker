# -*- coding: utf-8 -*-
"""训练答案判定头: (问题, 候选实体) -> 是否最终答案。"""
import argparse
import os

import torch
import torch.nn as nn

DATA = r"E:\GooseRAG\data\rog_qa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-file", default="")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    feat_path = args.feat_file or os.path.join(
        DATA, f"verifier_feat_{args.dataset}_{args.backing}.pt")
    blob = torch.load(feat_path, map_location="cpu", weights_only=False)
    f = blob["f"]
    y = torch.tensor([int(x) for x in blob["labels"]])
    n = len(f)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_tr = int(n * 0.9)
    tr, va = perm[:n_tr], perm[n_tr:]

    head = nn.Linear(f.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for ep in range(args.epochs):
        head.train()
        idx = tr[torch.randperm(len(tr))]
        for j in range(0, len(idx), 2048):
            b = idx[j:j + 2048]
            loss = lossf(head(f[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(f[va]).argmax(1)
            acc = (pred == y[va]).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"verifier val_acc={best_acc:.4f} n={n}", flush=True)
    out_path = args.out_file or os.path.join(DATA, f"verifier_{args.dataset}_{args.backing}.pt")
    torch.save({"state": head.state_dict(), "hidden": f.shape[1],
                "val_acc": best_acc, "dataset": args.dataset}, out_path)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
