# -*- coding: utf-8 -*-
"""构建实体条件化关系归纳数据: 问题 + 当前实体 + 关系前缀 -> 下一跳关系。

输入:
  E:\\GooseRAG\\data\\g1sg_{dataset}_{split}.jsonl
    {question, prefix, target, n_hop_prefix, path_len, cur_ent, cands}
输出:
  E:\\GooseRAG\\data\\rog_qa\\cond_{dataset}_{split}.jsonl
    {text, target, n_hop_prefix, path_len, cur_ent}

动机: G1a 的关系头只看 question+prefix, 同一跳对不同当前实体给出相同分布;
在真实子图上必须按当前实体区分合法关系, 因此把 cur_ent 放进输入。
"""
import argparse
import json
import os

DATA = r"E:\GooseRAG\data"
OUT = os.path.join(DATA, "rog_qa")


def make_text(q, cur, prefix):
    p = " -> ".join(prefix) if prefix else "(none)"
    return f"Question: {q}\nCurrent entity: {cur}\nKnown path: {p}\nNext relation:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cwq", choices=["webqsp", "cwq"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "val", "test"):
        src = os.path.join(DATA, f"g1sg_{args.dataset}_{split}.jsonl")
        dst = os.path.join(OUT, f"cond_{args.dataset}_{split}.jsonl")
        n = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as w:
            for line in f:
                r = json.loads(line)
                row = {
                    "text": make_text(r["question"], r["cur_ent"], r["prefix"]),
                    "target": r["target"],
                    "n_hop_prefix": r["n_hop_prefix"],
                    "path_len": r["path_len"],
                    "cur_ent": r["cur_ent"],
                    "cands": r["cands"],
                }
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
