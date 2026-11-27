# -*- coding: utf-8 -*-
"""候选约束训练: 只在当前节点的真实合法关系集 cands 上做 softmax。

训练目标与推理时的合法动作掩码完全一致, 避免闭集分类的捷径。
输入:
  E:\\GooseRAG\\data\\rog_qa\\cond_{dataset}_{train,val,test}.jsonl  (含 cands)
  E:\\GooseRAG\\data\\rog_qa\\cond_feat_{dataset}_rwkv7_{split}.pt
输出:
  E:\\GooseRAG\\data\\rog_qa\\cond_masked_{dataset}_rwkv7_heads.pt
"""
import argparse
import json
import os

import torch
import torch.nn as nn

DATA = r"E:\GooseRAG\data"
OUT = os.path.join(DATA, "rog_qa")


def load_split(dataset, split):
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"cond_{dataset}_{split}.jsonl"),
                                        encoding="utf-8")]
    blob = torch.load(os.path.join(OUT, f"cond_feat_{dataset}_rwkv7_{split}.pt"),
                      map_location="cpu", weights_only=False)
    assert len(rows) == len(blob["f"]), (split, len(rows), len(blob["f"]))
    return rows, blob["f"]


def make_batch(rows, idx, rel2id):
    cands, ys = [], []
    for i in idx:
        c = [rel2id[x] for x in rows[i]["cands"] if x in rel2id]
        t = rel2id.get(rows[i]["target"])
        if t is None or t not in c:
            continue
        cands.append(c)
        ys.append(c.index(t))
    L = max(len(c) for c in cands)
    cm = torch.full((len(cands), L), -1, dtype=torch.long)
    for b, c in enumerate(cands):
        cm[b, :len(c)] = torch.tensor(c)
    return cm, torch.tensor(ys, dtype=torch.long)


def masked_logits(head, feat, cand):
    logits = head(feat)
    g = logits.gather(1, cand.clamp(min=0))
    return g.masked_fill(cand < 0, -1e9)


def evaluate(head, rows, f, rel2id, device, k=5):
    head.eval()
    hit1 = hit5 = tot = 0
    with torch.no_grad():
        for j in range(0, len(rows), 512):
            idx = list(range(j, min(j + 512, len(rows))))
            cm, ys = make_batch(rows, idx, rel2id)
            if not len(ys):
                continue
            g = masked_logits(head, f[idx].to(device), cm.to(device)).cpu()
            kk = min(k, g.shape[1])
            pred = g.topk(kk, dim=1).indices
            hit1 += (pred[:, 0] == ys).sum().item()
            hit5 += (pred == ys[:, None]).any(1).sum().item()
            tot += len(ys)
    return hit1 / max(tot, 1), hit5 / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cwq", choices=["webqsp", "cwq"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                          encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}
    train_rows, train_f = load_split(args.dataset, "train")
    val_rows, val_f = load_split(args.dataset, "val")
    test_rows, test_f = load_split(args.dataset, "test")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    head = nn.Linear(train_f.shape[1], len(rels)).to(device)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None

    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(train_rows))
        for j in range(0, len(perm), args.batch):
            idx = perm[j:j + args.batch].tolist()
            cm, ys = make_batch(train_rows, idx, rel2id)
            if not len(ys):
                continue
            # 特征按同一 idx 取; 被跳过的样本不会进入 cm, 需重建有效特征索引
            kept = [i for i in idx if rel2id.get(train_rows[i]["target"]) in
                    [rel2id[x] for x in train_rows[i]["cands"] if x in rel2id]]
            feat = train_f[kept].to(device)
            loss = lossf(masked_logits(head, feat, cm.to(device)), ys.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        v1, v5, _ = evaluate(head, val_rows, val_f, rel2id, device)
        if v1 > best_val:
            best_val = v1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} val_top1={v1:.4f} val_top5={v5:.4f}", flush=True)

    head.load_state_dict(best_state)
    t1, t5, n = evaluate(head, test_rows, test_f, rel2id, device)
    print(f"test top1={t1:.4f} top5={t5:.4f} n={n}", flush=True)
    out = os.path.join(OUT, f"cond_masked_{args.dataset}_rwkv7_heads.pt")
    torch.save({"rel_state": head.state_dict(), "rels": rels,
                "hidden": train_f.shape[1], "dataset": args.dataset,
                "backing": "rwkv7", "test_top1": t1, "test_top5": t5}, out)
    print("saved ->", out)


if __name__ == "__main__":
    main()
