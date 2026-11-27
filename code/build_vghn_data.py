# -*- coding: utf-8 -*-
"""创新 B：把“判定器认为很像答案的错误实体”转成关系头的难负例。

输入:
  --mined  mine_train_candidates.py 的输出
  --g1     g1_{dataset}_train.jsonl（原始正例，防止遗忘）
输出:
  --out-pos  正例 {"text","rel"}
  --out-neg  难负例 {"text","rel"}

正例: gold 实体路径上的 (prefix, next_rel)
负例: 判定器分数处于该题前 hard-frac、且不是 gold 的实体路径上的 (prefix, next_rel)
      若同一 (prefix, rel) 已是该题正例则跳过
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import prompt  # noqa: E402


def pairs_of(path, max_hops):
    return [(tuple(path[:j]), path[j]) for j in range(min(len(path), max_hops))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--g1", default="")
    ap.add_argument("--rels", required=True)
    ap.add_argument("--out-pos", required=True)
    ap.add_argument("--out-neg", required=True)
    ap.add_argument("--hard-frac", type=float, default=0.2)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--min-cand", type=int, default=10)
    ap.add_argument("--keep-g1-pos", action="store_true", default=True)
    args = ap.parse_args()

    relset = set(json.load(open(args.rels, encoding="utf-8")))
    pos, neg = {}, {}
    n_q = n_q_pos = n_q_neg = 0
    with open(args.mined, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_q += 1
            gold = set(r["gold"])
            cand = [c for c in r["cand"] if len(c[3]) > 0]
            if len(cand) < args.min_cand:
                continue
            vs = sorted(c[2] for c in cand)
            thr = vs[max(0, int((1.0 - args.hard_frac) * len(vs)) - 1)]

            q_pos = set()
            for e, _s, _v, path in cand:
                if e in gold:
                    q_pos.update(pairs_of(path, args.max_hops))
            if q_pos:
                n_q_pos += 1
            q_neg = set()
            for e, _s, v, path in cand:
                if e in gold or v < thr:
                    continue
                q_neg.update(pairs_of(path, args.max_hops))
            q_neg -= q_pos
            if q_neg:
                n_q_neg += 1

            for prefix, rel in q_pos:
                if rel in relset:
                    pos[(prompt(r["question"], prefix), rel)] = 1
            for prefix, rel in q_neg:
                if rel in relset:
                    neg[(prompt(r["question"], prefix), rel)] = 1

    n_g1 = 0
    if args.g1 and args.keep_g1_pos:
        with open(args.g1, encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                rel = g["target"]
                if rel not in relset:
                    continue
                pos[(prompt(g["question"], tuple(g["prefix"])), rel)] = 1
                n_g1 += 1

    with open(args.out_pos, "w", encoding="utf-8") as f:
        for (text, rel) in pos:
            f.write(json.dumps({"text": text, "rel": rel}, ensure_ascii=False) + "\n")
    with open(args.out_neg, "w", encoding="utf-8") as f:
        for (text, rel) in neg:
            f.write(json.dumps({"text": text, "rel": rel}, ensure_ascii=False) + "\n")

    print(json.dumps({
        "n_questions": n_q, "q_with_pos": n_q_pos, "q_with_neg": n_q_neg,
        "n_pos": len(pos), "n_neg": len(neg), "n_g1_added": n_g1,
        "hard_frac": args.hard_frac,
    }, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
