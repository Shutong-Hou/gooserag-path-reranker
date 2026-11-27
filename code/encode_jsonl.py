# -*- coding: utf-8 -*-
"""通用 JSONL 编码器: 读 text 字段 -> RWKV-7/GPT-2 最后位置特征 + 标签。

用法:
  python encode_jsonl.py --data cond_webqsp_train.jsonl --out cond_feat_webqsp_train.pt \
      --backing rwkv7 --label-key target
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import encode_texts, load_model  # noqa: E402

DATA = r"E:\GooseRAG\data\rog_qa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2"])
    ap.add_argument("--label-key", default="target")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = args.data if os.path.isabs(args.data) else os.path.join(DATA, args.data)
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows={len(rows)} backing={args.backing}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, name = load_model(args.backing)
    model.to(device)
    feats = encode_texts(model, tok, [r["text"] for r in rows], device,
                         args.backing, args.batch, args.max_len)
    labels = [r.get(args.label_key) for r in rows]
    torch.save({"f": feats, "labels": labels, "backing": args.backing}, args.out)
    print(f"saved {args.out} f={tuple(feats.shape)}", flush=True)


if __name__ == "__main__":
    main()
