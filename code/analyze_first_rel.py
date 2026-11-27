# -*- coding: utf-8 -*-
"""首跳诊断: 对每题找出能到达 gold 的首跳关系, 看模型 top-K / 词面匹配 top-K 是否覆盖。

这决定端到端瓶颈是"关系头选错"还是"后续 beam 丢路径"。
"""
import argparse
import json
import os
import re
from collections import defaultdict

import torch

DATA = r"E:\GooseRAG\data"
ROG = os.path.join(DATA, "rog_qa")
CODE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, CODE)
from eval_e2e import (build_graph, encode_texts, is_generic, load_heads,  # noqa: E402
                      load_model, prompt)

STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was", "were",
        "what", "who", "where", "when", "which", "how", "many", "much", "does", "did",
        "do", "and", "or", "that", "this", "with", "by", "at", "from", "as"}


def lex_tokens(rel):
    s = rel.replace(".", " ").replace("_", " ").lower()
    return {w for w in re.findall(r"[a-z]+", s) if w not in STOP}


def reach(out, nodes, gold, rel2id, hops):
    cur = set(nodes)
    for _ in range(hops):
        nxt = set()
        for e in cur:
            for r, tails in out.get(e, {}).items():
                if r in rel2id:
                    nxt.update(tails)
        if not nxt:
            return False
        if nxt & gold:
            return True
        cur = nxt
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel_head, rels = load_heads(os.path.join(ROG, f"{args.dataset}_rwkv7_heads.pt"), device)
    rel2id = {r: i for i, r in enumerate(rels)}
    model, tok, _ = load_model("rwkv7")
    model.to(device)

    rows = [json.loads(l) for l in open(os.path.join(ROG, args.dataset, f"{args.split}.jsonl"),
                                        encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    n = 0
    has_gold_rel = 0
    lm_hit = {1: 0, 3: 0, 5: 0, 10: 0}
    lex_hit = {1: 0, 3: 0, 5: 0, 10: 0}
    union_hit = {1: 0, 3: 0, 5: 0, 10: 0}
    lm_rank_sum = 0

    for r in rows:
        out = build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        gold = set(r["a_entity"]) or set(r["answers"])
        if not starts or not gold:
            continue
        legal = set()
        for e in starts:
            for rel in out.get(e, {}):
                if rel in rel2id and not is_generic(rel):
                    legal.add(rel)
        if not legal:
            continue
        legal = sorted(legal)
        # 能到达 gold 的首跳关系
        gold_rels = set()
        for rel in legal:
            tails = set()
            for e in starts:
                tails.update(out[e].get(rel, ()))
            if not tails:
                continue
            if (tails & gold) or reach(out, tails, gold, rel2id, args.max_hops - 1):
                gold_rels.add(rel)
        if not gold_rels:
            continue
        n += 1
        has_gold_rel += 1

        # LM 排名
        feat = encode_texts(model, tok, [prompt(r["question"], ())], device, "rwkv7", 1)
        idx = torch.tensor([rel2id[x] for x in legal])
        logits = rel_head(feat.to(device)).cpu()[0, idx]
        lm_order = [legal[j] for j in torch.argsort(logits, descending=True).tolist()]
        # 词面排名
        qs = {w for w in re.findall(r"[a-z]+", r["question"].lower()) if w not in STOP}
        lex_scores = []
        for rel in legal:
            rt = lex_tokens(rel)
            lex_scores.append((len(qs & rt) / max(len(rt), 1), rel))
        lex_order = [x for _, x in sorted(lex_scores, key=lambda z: -z[0])]
        union_order = list(dict.fromkeys(lm_order[:args.topk] + lex_order[:args.topk]))

        for k in (1, 3, 5, 10):
            lm_hit[k] += int(any(x in gold_rels for x in lm_order[:k]))
            lex_hit[k] += int(any(x in gold_rels for x in lex_order[:k]))
            union_hit[k] += int(any(x in gold_rels for x in union_order[:k]))
        best = min(lm_order.index(x) + 1 for x in gold_rels)
        lm_rank_sum += best

    print(json.dumps({
        "dataset": args.dataset, "split": args.split, "n_with_gold_rel": n,
        "lm_recall": {k: v / n for k, v in lm_hit.items()},
        "lex_recall": {k: v / n for k, v in lex_hit.items()},
        "union_recall": {k: v / n for k, v in union_hit.items()},
        "lm_best_rank_mean": lm_rank_sum / max(n, 1),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
