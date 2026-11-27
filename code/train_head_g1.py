# -*- coding: utf-8 -*-
"""创新 C：按 g1 文本训练关系头（支持留出关系后的重训）。

输入 g1 jsonl 需有 text 字段（make_heldout_split.py 已生成）。
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base, "g1_probe"))
sys.path.insert(0, os.path.join(os.path.dirname(_base), "g1_probe"))
import eval_e2e as E  # noqa: E402
from train_probe import train_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", required=True)
    ap.add_argument("--g1-val", required=True)
    ap.add_argument("--rels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rels = json.load(open(args.rels, encoding="utf-8"))
    rel2id = {r: i for i, r in enumerate(rels)}

    tr = [json.loads(l) for l in open(args.g1, encoding="utf-8")]
    va = [json.loads(l) for l in open(args.g1_val, encoding="utf-8")]
    tr = [r for r in tr if r["target"] in rel2id]
    va = [r for r in va if r["target"] in rel2id]
    print(f"train={len(tr)} val={len(va)} rels={len(rels)}", flush=True)

    model, tok, _ = E.load_model(args.backing)
    model.to(device)

    def enc(texts):
        return E.encode_texts(model, tok, texts, device, args.backing,
                              batch_size=args.batch, max_len=320, feat_norm="double")

    f = enc([r["text"] for r in tr])
    y = torch.tensor([rel2id[r["target"]] for r in tr])
    fv = enc([r["text"] for r in va])
    yv = torch.tensor([rel2id[r["target"]] for r in va])

    head = train_head(f, y, fv, yv, len(rels), device, f.shape[1],
                      epochs=args.epochs, seed=args.seed)

    def topk(feat, labels, k):
        head.eval()
        with torch.no_grad():
            logits = head(feat.to(device))
        return (logits.topk(k, dim=1).indices == labels.to(device)[:, None]).any(1).float().mean().item()

    t1, t5 = topk(fv, yv, 1), topk(fv, yv, 5)
    torch.save({
        "rel_state": {k: v.cpu() for k, v in head.state_dict().items()},
        "rels": rels, "hidden": f.shape[1], "backing": args.backing,
        "val_top1": t1, "val_top5": t5,
        "trained_on": os.path.basename(args.g1),
    }, args.out)
    print(f"saved -> {args.out} val_top1={t1:.4f} val_top5={t5:.4f}", flush=True)


if __name__ == "__main__":
    main()
