# -*- coding: utf-8 -*-
"""创新 B：判定器引导的关系头难负例训练（VGHN）。

损失 = CE(正例) + lam * BCE(难负例关系 logit -> 0)
特征口径与线上一致: encode_texts(feat_norm="double", max_len=320)
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402


def read_pairs(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r["text"] for r in rows], [r["rel"] for r in rows]


def val_topk(head, feats, labels, device, k):
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(device))
    kk = min(k, logits.shape[1])
    return (logits.topk(kk, dim=1).indices == labels.to(device)[:, None]).any(1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--init", required=True, help="初始关系头 ckpt")
    ap.add_argument("--val-g1", required=True)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(args.init, map_location="cpu", weights_only=False)
    rels = blob["rels"]
    rel2id = {r: i for i, r in enumerate(rels)}

    ptext, prel = read_pairs(args.pos)
    ntext, nrel = read_pairs(args.neg)
    print(f"pos={len(ptext)} neg={len(ntext)} rels={len(rels)}", flush=True)

    vrows = [json.loads(l) for l in open(args.val_g1, encoding="utf-8")]
    vtext = [E.prompt(r["question"], tuple(r["prefix"])) for r in vrows]
    vlab = torch.tensor([rel2id[r["target"]] for r in vrows if r["target"] in rel2id])

    model, tok, _ = E.load_model(args.backing)
    model.to(device)

    def enc(texts):
        return E.encode_texts(model, tok, texts, device, args.backing,
                              batch_size=64, max_len=320, feat_norm="double")

    fpos = enc(ptext).to(device)
    fneg = enc(ntext).to(device) if ntext else None
    fval = enc(vtext)
    if fval.shape[0] != vlab.shape[0]:
        keep = [i for i, r in enumerate(vrows) if r["target"] in rel2id]
        fval = fval[keep]
    ypos = torch.tensor([rel2id[r] for r in prel]).to(device)
    yneg = torch.tensor([rel2id[r] for r in nrel]).to(device)

    head = nn.Linear(blob["hidden"], len(rels))
    head.load_state_dict(blob["rel_state"])
    head.to(device)
    print(f"init val top1={val_topk(head, fval, vlab, device, 1):.4f} "
          f"top5={val_topk(head, fval, vlab, device, 5):.4f}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    n = len(ptext)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        nidx = torch.randperm(len(ntext), device=device) if len(ntext) else None
        tot_loss = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = ce(head(fpos[idx]), ypos[idx].to(device))
            if nidx is not None and i < len(nidx):
                j = nidx[i:i + args.batch]
                lg = head(fneg[j]).gather(1, yneg[j].to(device)[:, None]).squeeze(1)
                loss = loss + args.lam * bce(lg, torch.zeros_like(lg))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1} loss={tot_loss/max(nb,1):.4f} "
                  f"val_top1={val_topk(head, fval, vlab, device, 1):.4f} "
                  f"val_top5={val_topk(head, fval, vlab, device, 5):.4f}", flush=True)

    t1 = val_topk(head, fval, vlab, device, 1)
    t5 = val_topk(head, fval, vlab, device, 5)
    torch.save({
        "rel_state": head.state_dict(), "rels": rels, "hidden": blob["hidden"],
        "backing": args.backing, "val_top1": t1, "val_top5": t5,
        "vghn": {"lam": args.lam, "n_pos": len(ptext), "n_neg": len(ntext),
                 "init": os.path.basename(args.init)},
    }, args.out)
    print(f"saved -> {args.out} val_top1={t1:.4f} val_top5={t5:.4f}", flush=True)


if __name__ == "__main__":
    main()
