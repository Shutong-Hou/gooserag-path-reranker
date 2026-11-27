# -*- coding: utf-8 -*-
"""MetaQA 分层抽样：按固定步长取子集，避免只拿到前 N 题（数据按 hop/类型成块排列）。"""
import json
import os

D = "/root/autodl-tmp/GooseRAG/data/rog_qa/metaqa"


def subset(split, k):
    src = os.path.join(D, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    stride = max(1, len(rows) // k)
    sub = rows[::stride][:k]
    out = os.path.join(D, f"{split}_s{k}.jsonl")
    with open(out, "w", encoding="utf-8") as w:
        for r in sub:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} -> {len(sub)} (stride {stride}) -> {out}", flush=True)


if __name__ == "__main__":
    subset("test", 2000)
