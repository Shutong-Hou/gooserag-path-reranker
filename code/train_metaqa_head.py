# -*- coding: utf-8 -*-
"""Train a relation head for MetaQA (RoG format) and save eval-compatible ckpt."""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import encode_texts, load_model  # noqa: E402

DATA = os.environ.get("GOOSERAG_ROGQA", r"E:\GooseRAG\data\rog_qa")
META = os.path.join(DATA, "metaqa")


def make_text(s):
    prefix = " -> ".join(s["prefix"]) if s["prefix"] else "(none)"
    return f"Question: {s['question']}\nKnown path: {prefix}\nNext relation:"


def load_split(name, rel2id):
    rows = [json.loads(l) for l in open(os.path.join(META, name), encoding="utf-8")]
    texts = [make_text(r) for r in rows]
    ys = torch.tensor([rel2id[r["target"]] for r in rows], dtype=torch.long)
    return texts, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rels = json.load(open(os.path.join(META, "relations.json"), encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}
    print(f"classes={len(rels)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_model(args.backing)
    model.to(device)

    splits = {}
    for name, fn in [("train", "g1_train.jsonl"), ("val", "g1_val.jsonl"), ("test", "g1_test.jsonl")]:
        texts, ys = load_split(fn, rel2id)
        print(f"{name}: n={len(texts)}", flush=True)
        splits[name] = (encode_texts(model, tok, texts, device, args.backing, args.batch,
                                     max_len=320, feat_norm="double"), ys)

    hidden = splits["train"][0].shape[1]
    head = nn.Linear(hidden, len(rels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def acc(split):
        head.eval()
        with torch.no_grad():
            logits = head(splits[split][0].to(device))
            y = splits[split][1].to(device)
            top1 = (logits.argmax(1) == y).float().mean().item()
            k = min(5, logits.shape[1])
            top5 = (logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean().item()
        return top1, top5

    x, y = splits["train"][0], splits["train"][1]
    best = -1.0
    best_state = None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(x))
        for j in range(0, len(perm), 4096):
            idx = perm[j:j + 4096]
            logits = head(x[idx].to(device))
            loss = lossf(logits, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        t1, t5 = acc("val")
        if t1 > best:
            best = t1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} loss={loss.item():.4f} val_top1={t1:.4f} val_top5={t5:.4f}", flush=True)
    head.load_state_dict(best_state)
    t1, t5 = acc("test")
    print(f"test top1={t1:.4f} top5={t5:.4f}", flush=True)

    out = os.path.join(DATA, f"metaqa_{args.backing}_heads.pt")
    torch.save({"rel_state": best_state, "rels": rels, "hidden": hidden,
                "backing": args.backing, "dataset": "metaqa",
                "test_top1": t1, "test_top5": t5}, out)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
