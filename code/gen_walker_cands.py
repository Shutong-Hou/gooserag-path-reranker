# -*- coding: utf-8 -*-
"""生成 walker 候选（供 listwise 判定器训练/分析）。

只读 train split；生成前先做 val/test 归一化问题去泄漏检查。
输出:
  E:\\GooseRAG\\data\\rog_qa\\walker_cands_{dataset}_{split}.jsonl
  E:\\GooseRAG\\data\\rog_qa\\walker_cands_{dataset}_{split}_stats.json

用法:
  python gen_walker_cands.py --dataset webqsp --split train --topk 50
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import (  # noqa: E402
    ROG_QA, build_graph, load_heads, load_model, run_beam,
)


def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--topk-rel", type=int, default=10)
    ap.add_argument("--max-children", type=int, default=64)
    ap.add_argument("--max-hops", type=int, default=0,
                    help="0 表示按数据集默认: webqsp=2, cwq=3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="按源行号跨步采样, 1=不采样")
    ap.add_argument("--out", default="")
    ap.add_argument("--stats-out", default="")
    args = ap.parse_args()
    if args.max_hops <= 0:
        args.max_hops = 2 if args.dataset == "webqsp" else 3

    src = os.path.join(ROG_QA, args.dataset, f"{args.split}.jsonl")

    # 去泄漏：train 问题不得与 val/test 归一化后重合
    eval_q = set()
    for sp in ("validation", "test"):
        p = os.path.join(ROG_QA, args.dataset, f"{sp}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                q = json.loads(line).get("question")
                if q:
                    eval_q.add(norm_q(q))
    leaked_count = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel_head, rels = load_heads(
        os.path.join(ROG_QA, f"{args.dataset}_{args.backing}_heads.pt"), device)
    model, tok, _ = load_model(args.backing)
    model.to(device)
    rel2id = {r: i for i, r in enumerate(rels)}
    print(f"loaded {args.dataset}/{args.backing}, classes={len(rels)}, "
          f"src={src}, device={device}", flush=True)

    gen_args = SimpleNamespace(
        policy="model", cond=False, search="entity",
        beam=args.beam, topk_rel=args.topk_rel, max_children=args.max_children,
        max_hops=args.max_hops, length_mode="all", score_norm="mean",
        ban_generic=True, generic_mode="all", batch=64, backing=args.backing,
    )

    out_path = args.out or os.path.join(ROG_QA, f"walker_cands_{args.dataset}_{args.split}.jsonl")
    n_skip = n_group = n_pos = 0
    cand_counts = []
    gold_ranks = []
    stats_rows = 0
    n_seen = 0
    with open(out_path, "w", encoding="utf-8") as w, open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_seen += 1
            if args.limit and n_seen > args.limit:
                break
            if args.stride > 1 and (n_seen - 1) % args.stride != 0:
                continue
            r = json.loads(line)
            if norm_q(r["question"]) in eval_q:
                leaked_count += 1
                continue
            out = build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            if not starts or not gold:
                n_skip += 1
                continue
            ranked, beams, path_of = run_beam(
                gen_args, r["question"], starts, out, rel2id,
                model, tok, rel_head, None, device)
            if not ranked:
                n_skip += 1
                continue
            cands = ranked[:args.topk]
            # 若 top-K 无正例，补入 rank 最高的 gold，保证每组至少一个正例
            if not any(e in gold for e, _ in cands):
                for e, s in ranked:
                    if e in gold:
                        cands = cands + [(e, s)]
                        break
            group = []
            for e, s in cands:
                group.append({
                    "entity": e, "score": float(s),
                    "label": int(e in gold),
                    "path": list(path_of.get(e, ())),
                })
            if not any(x["label"] for x in group):
                n_skip += 1
                continue
            gold_rank = None
            for j, (e, _) in enumerate(ranked):
                if e in gold:
                    gold_rank = j
                    break
            rec = {
                "id": r["id"], "question": r["question"],
                "gold": sorted(gold), "q_entity": r["q_entity"],
                "cands": group, "n_cand": len(ranked),
                "gold_rank": gold_rank,
                "gold_in_topk": bool(gold_rank is not None and gold_rank < args.topk),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_group += 1
            if gold_rank is not None and gold_rank < args.topk:
                n_pos += 1
            cand_counts.append(len(group))
            if gold_rank is not None:
                gold_ranks.append(gold_rank)
            stats_rows += 1
            if n_seen % 200 == 0:
                print(f"  seen={n_seen} groups={n_group} "
                      f"gold_in_topk={n_pos/max(n_group,1):.4f}", flush=True)

    stats = {
        "dataset": args.dataset, "split": args.split, "backing": args.backing,
        "n_rows": n_seen, "n_groups": n_group, "n_skipped": n_skip,
        "topk": args.topk, "beam": args.beam, "topk_rel": args.topk_rel,
        "max_children": args.max_children, "max_hops": args.max_hops,
        "leaked_train_questions_removed": leaked_count,
        "gold_in_topk_rate": n_pos / max(n_group, 1),
        "avg_group_size": sum(cand_counts) / max(len(cand_counts), 1),
        "gold_rank_mean": (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None,
        "gold_rank_median": (sorted(gold_ranks)[len(gold_ranks) // 2] if gold_ranks else None),
        "out": out_path,
    }
    stats_path = args.stats_out or os.path.join(
        ROG_QA, f"walker_cands_{args.dataset}_{args.split}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    print("saved ->", out_path, "and", stats_path)


if __name__ == "__main__":
    main()
