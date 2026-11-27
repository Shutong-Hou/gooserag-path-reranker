# -*- coding: utf-8 -*-
"""把 MetaQA (kb.txt + qa_{split}.txt) 转成 RoG 每题子图格式。

输出:
  {out}/{split}.jsonl  {id, question, q_entity, a_entity, edges}
用法:
  python prepare_metaqa_rog.py --raw E:\\GooseRAG\\data\\metaqa --out E:\\GooseRAG\\data\\rog_qa\\metaqa --limit 500
"""
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict, deque

TOPIC_RE = re.compile(r"\[(.+?)\]")


def build_kb(path):
    out = defaultdict(lambda: defaultdict(list))
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            h, r, t = parts
            out[h][r + "_fwd"].append(t)
            out[t][r + "_bwd"].append(h)
            n += 1
    return out, n


def subgraph(out, topic, max_hops, max_children, max_edges):
    edges = []
    seen = set()
    nodes_seen = {topic}
    frontier = [topic]
    for _ in range(max_hops):
        nxt = []
        for e in frontier:
            for r, tails in out.get(e, {}).items():
                for t in tails[:max_children]:
                    key = (e, r, t)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append([e, r, t])
                    if len(edges) >= max_edges:
                        return edges, nodes_seen
                    if t not in nodes_seen:
                        nodes_seen.add(t)
                        nxt.append(t)
        frontier = nxt
        if not frontier:
            break
    return edges, nodes_seen


def shortest_chain(out, topic, answers, max_hops):
    """BFS 找 topic->任一答案 的最短关系链（用 fwd/bwd 关系名）。"""
    goal = set(answers)
    prev = {topic: None}
    frontier = deque([topic])
    depth = {topic: 0}
    while frontier:
        e = frontier.popleft()
        if depth[e] >= max_hops:
            continue
        for r, tails in out.get(e, {}).items():
            for t in tails:
                if t in prev:
                    continue
                prev[t] = (e, r)
                depth[t] = depth[e] + 1
                if t in goal:
                    seq, node = [], t
                    while prev[node] is not None:
                        p, rr = prev[node]
                        seq.append(rr)
                        node = p
                    seq.reverse()
                    return seq
                frontier.append(t)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"E:\GooseRAG\data\metaqa")
    ap.add_argument("--out", default=r"E:\GooseRAG\data\rog_qa\metaqa")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--max-children", type=int, default=64)
    ap.add_argument("--max-edges", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="按行跨步抽样，避免只取到 1-hop 段")
    ap.add_argument("--dump-train", action="store_true", help="训练划分也写出 QA 格式 train.jsonl")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--train-q", type=int, default=15000, help="训练问题上限（生成判定器对）")
    ap.add_argument("--neg-per-q", type=int, default=8)
    ap.add_argument("--g1-limit", type=int, default=150000, help="关系头训练样本上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    t0 = time.time()
    kb, n_triples = build_kb(os.path.join(args.raw, "kb.txt"))
    rels = sorted({r for h in kb for r in kb[h]})
    print(f"kb triples={n_triples} entities={len(kb)} rels={len(rels)} ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    split_map = {"train": "qa_train.txt", "dev": "qa_dev.txt", "test": "qa_test.txt"}
    n_g1 = 0
    g1_train = open(os.path.join(args.out, "g1_train.jsonl"), "w", encoding="utf-8")
    ver = open(os.path.join(args.out, "verifier_train.jsonl"), "w", encoding="utf-8")

    for sp in args.splits.split(","):
        src = os.path.join(args.raw, split_map[sp])
        dst = os.path.join(args.out, "validation.jsonl" if sp == "dev" else f"{sp}.jsonl")
        n_ok = n_skip = n_edge = 0
        n_gold_reach = n_chain = 0
        q_limit = args.train_q if sp == "train" else 0
        g1_out = g1_train if sp == "train" else open(
            os.path.join(args.out, f"g1_{'val' if sp == 'dev' else sp}.jsonl"), "w", encoding="utf-8")
        w = open(dst, "w", encoding="utf-8") if (sp != "train" or args.dump_train) else None
        with open(src, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.stride > 1 and i % args.stride != 0:
                    continue
                if args.limit and i >= args.limit:
                    break
                if q_limit and n_ok >= q_limit:
                    break
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                q_raw, ans_raw = line.split("\t", 1)
                m = TOPIC_RE.search(q_raw)
                if not m:
                    n_skip += 1
                    continue
                topic = m.group(1)
                if topic not in kb:
                    n_skip += 1
                    continue
                answers = [a.strip() for a in ans_raw.split("|") if a.strip()]
                if not answers:
                    n_skip += 1
                    continue
                edges, nodes_seen = subgraph(kb, topic, args.max_hops,
                                             args.max_children, args.max_edges)
                if not edges:
                    n_skip += 1
                    continue
                n_ok += 1
                n_edge += len(edges)
                q = TOPIC_RE.sub(lambda mm: mm.group(1), q_raw)
                if any(a in nodes_seen for a in answers):
                    n_gold_reach += 1
                if w is not None:
                    w.write(json.dumps({"id": f"{sp}_{i}", "question": q,
                                        "q_entity": [topic], "a_entity": answers,
                                        "edges": edges}, ensure_ascii=False) + "\n")
                # 关系头训练样本：最短答案链的逐跳 (prefix -> relation)
                seq = shortest_chain(kb, topic, answers, args.max_hops)
                if seq:
                    n_chain += 1
                    if sp != "train" or n_g1 < args.g1_limit:
                        for j, rel in enumerate(seq):
                            if sp == "train" and n_g1 >= args.g1_limit:
                                break
                            g1_out.write(json.dumps({
                                "question": q, "prefix": seq[:j], "target": rel,
                                "n_hop_prefix": j, "path_len": len(seq)},
                                ensure_ascii=False) + "\n")
                            n_g1 += 1
                if sp == "train":
                    pos = set(answers)
                    pool = [e for e in nodes_seen if e not in pos]
                    rng.shuffle(pool)
                    pairs = [(a, 1) for a in answers] + [(e, 0) for e in pool[:args.neg_per_q]]
                    for e, y in pairs:
                        ver.write(json.dumps({
                            "text": f"Question: {q}\nCandidate: {e}\nIs this the final answer?",
                            "label": y, "question": q, "entity": e},
                            ensure_ascii=False) + "\n")
        if w is not None:
            w.close()
        if sp != "train":
            g1_out.close()
        with open(os.path.join(args.out, f"{sp}_stats.json"), "w", encoding="utf-8") as sw:
            json.dump({"n_ok": n_ok, "n_skip": n_skip, "avg_edges": n_edge / max(n_ok, 1),
                       "gold_reach": n_gold_reach / max(n_ok, 1),
                       "chain_ok": n_chain / max(n_ok, 1),
                       "max_hops": args.max_hops, "max_children": args.max_children,
                       "max_edges": args.max_edges}, sw, indent=1)
        print(f"{sp}: ok={n_ok} skip={n_skip} avg_edges={n_edge/max(n_ok,1):.0f} "
              f"gold_reach={n_gold_reach/max(n_ok,1):.4f} chain={n_chain/max(n_ok,1):.4f} "
              f"g1_samples={n_g1} ({time.time()-t0:.1f}s)", flush=True)

    g1_train.close()
    ver.close()
    with open(os.path.join(args.out, "relations.json"), "w", encoding="utf-8") as w:
        json.dump(rels, w, ensure_ascii=False)
    print("rels:", rels)


if __name__ == "__main__":
    main()
