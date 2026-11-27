# -*- coding: utf-8 -*-
"""困难负例判定头数据: 用 train-0 真实子图 BFS 2 跳的候选做负例。

输入: E:\\GooseRAG\\data\\rog_qa\\webqsp\\train.jsonl
输出: E:\\GooseRAG\\data\\rog_qa\\verifier_webqsp_hard.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import build_graph, is_generic  # noqa: E402

DATA = r"E:\GooseRAG\data"
ROG = os.path.join(DATA, "rog_qa")


def make_text(q, e):
    return f"Question: {q}\nCandidate: {e}\nIs this the final answer?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--neg-per-q", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rel2id = set(json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                                encoding="utf-8")))
    rows = [json.loads(l) for l in open(os.path.join(ROG, args.dataset, "train.jsonl"),
                                        encoding="utf-8")]
    n_pos = n_neg = n_q = 0
    with open(os.path.join(ROG, f"verifier_{args.dataset}_hard.jsonl"), "w",
              encoding="utf-8") as w:
        for r in rows:
            out = build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            if not starts or not gold:
                continue
            cand = set()
            cur = set(starts)
            for _ in range(args.max_hops):
                nxt = set()
                for e in cur:
                    for rel, tails in out.get(e, {}).items():
                        if rel in rel2id and not is_generic(rel):
                            nxt.update(tails)
                nxt -= set(starts)
                cand |= nxt
                if len(cand) > args.cap:
                    cand = set(sorted(cand)[:args.cap])
                cur = set(sorted(nxt)[:args.cap])
                if not cur:
                    break
            pos = sorted(gold & cand)
            neg = sorted(cand - gold)
            if not pos or not neg:
                continue
            n_q += 1
            rng.shuffle(neg)
            for e in pos:
                w.write(json.dumps({"text": make_text(r["question"], e), "label": 1},
                                   ensure_ascii=False) + "\n")
                n_pos += 1
            for e in neg[:args.neg_per_q]:
                w.write(json.dumps({"text": make_text(r["question"], e), "label": 0},
                                   ensure_ascii=False) + "\n")
                n_neg += 1
    print(f"questions={n_q} pos={n_pos} neg={n_neg} total={n_pos+n_neg}", flush=True)


if __name__ == "__main__":
    main()
