# -*- coding: utf-8 -*-
"""解析 RoG 官方 parquet（每题 Freebase 子图）为实体级 KGQA 数据。

输入:
  E:\\GooseRAG\\data\\rog_hf\\{dataset}\\*.parquet
字段 (RoG): id, question, answer[], q_entity[], a_entity[], graph[[]]
  graph 的每个元素是三元组 (head, relation, tail), 可能是 list 或 "h r t" 字符串。

输出:
  E:\\GooseRAG\\data\\rog_qa\\{dataset}\\{split}.jsonl
    每行 {id, question, answers, q_entity, a_entity, edges: [[h,r,t], ...]}
  E:\\GooseRAG\\data\\rog_qa\\{dataset}\\stats.json

用法:
  python prepare_rog.py --dataset webqsp --inspect
  python prepare_rog.py --dataset webqsp --split test
"""
import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd

BASE = r"E:\GooseRAG\data\rog_hf"
OUT = r"E:\GooseRAG\data\rog_qa"


def norm_triple(t):
    """统一三元组表示; 失败返回 None。
    parquet 里是 numpy.ndarray, 也兼容 list/tuple 与 "h r t" 字符串。
    """
    if isinstance(t, str):
        parts = t.split()
        if len(parts) >= 3:
            return [parts[0], parts[1], " ".join(parts[2:])]
        return None
    try:
        if len(t) == 3:
            return [str(t[0]), str(t[1]), str(t[2])]
    except TypeError:
        pass
    return None


def parse_split(dataset, split):
    files = sorted(glob.glob(os.path.join(BASE, dataset, f"{split}-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet for {dataset}/{split}: {files}")
    rows = []
    bad_graph = 0
    for f in files:
        df = pd.read_parquet(f)
        for rec in df.itertuples(index=False):
            g = getattr(rec, "graph", [])
            edges = []
            for t in g:
                e = norm_triple(t)
                if e is None:
                    bad_graph += 1
                else:
                    edges.append(e)
            rows.append({
                "id": str(getattr(rec, "id")),
                "question": str(getattr(rec, "question")),
                "answers": [str(x) for x in getattr(rec, "answer")],
                "q_entity": [str(x) for x in getattr(rec, "q_entity")],
                "a_entity": [str(x) for x in getattr(rec, "a_entity")],
                "edges": edges,
            })
    return rows, bad_graph


def stats(rows, dataset, split):
    n_edges = [len(r["edges"]) for r in rows]
    out_deg = Counter()
    rels = Counter()
    q_in = 0
    a_in_graph = 0
    for r in rows:
        heads = Counter()
        nodes = set()
        for h, rel, t in r["edges"]:
            heads[h] += 1
            nodes.add(h); nodes.add(t)
            rels[rel] += 1
        for h, c in heads.items():
            out_deg[c] += 1
        if any(q in nodes for q in r["q_entity"]):
            q_in += 1
        if any(a in nodes for a in r["answers"]):
            a_in_graph += 1
    return {
        "dataset": dataset, "split": split, "n": len(rows),
        "edges_total": sum(n_edges),
        "edges_mean": sum(n_edges) / max(len(n_edges), 1),
        "edges_max": max(n_edges) if n_edges else 0,
        "questions_with_graph_q_entity": q_in,
        "questions_with_any_answer_in_graph": a_in_graph,
        "relations_total": len(rels),
        "out_degree_hist": dict(sorted(out_deg.items())[:20]),
        "top_relations": rels.most_common(20),
    }


def parse_split_stream(dataset, split, out_path):
    """逐分片解析并流式写 JSONL，避免 CWQ train 27k 题子图占满内存。"""
    files = sorted(glob.glob(os.path.join(BASE, dataset, f"{split}-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet for {dataset}/{split}: {files}")
    rels = Counter()
    out_deg = Counter()
    n = edges_total = edges_max = q_in = a_in = bad = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for f in files:
            df = pd.read_parquet(f)
            for rec in df.itertuples(index=False):
                g = getattr(rec, "graph", [])
                edges = []
                for t in g:
                    e = norm_triple(t)
                    if e is None:
                        bad += 1
                    else:
                        edges.append(e)
                row = {
                    "id": str(getattr(rec, "id")),
                    "question": str(getattr(rec, "question")),
                    "answers": [str(x) for x in getattr(rec, "answer")],
                    "q_entity": [str(x) for x in getattr(rec, "q_entity")],
                    "a_entity": [str(x) for x in getattr(rec, "a_entity")],
                    "edges": edges,
                }
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                edges_total += len(edges)
                edges_max = max(edges_max, len(edges))
                heads = Counter()
                nodes = set()
                for h, rel, t in edges:
                    heads[h] += 1
                    nodes.add(h)
                    nodes.add(t)
                    rels[rel] += 1
                for h, c in heads.items():
                    out_deg[c] += 1
                if any(q in nodes for q in row["q_entity"]):
                    q_in += 1
                if any(a in nodes for a in row["answers"]):
                    a_in += 1
            print(f"  {f}: rows={n}", flush=True)
    st = {
        "dataset": dataset, "split": split, "n": n,
        "edges_total": edges_total,
        "edges_mean": edges_total / max(n, 1),
        "edges_max": edges_max,
        "questions_with_graph_q_entity": q_in,
        "questions_with_any_answer_in_graph": a_in,
        "relations_total": len(rels),
        "out_degree_hist": dict(sorted(out_deg.items())[:20]),
        "top_relations": rels.most_common(20),
    }
    return st, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="逐分片流式解析（CWQ train 用，避免内存爆）")
    args = ap.parse_args()

    if args.inspect:
        for split in args.splits.split(","):
            rows, bad = parse_split(args.dataset, split)
            print(json.dumps(stats(rows, args.dataset, split), ensure_ascii=False, indent=1))
            print("sample edges:", rows[0]["edges"][:3], "bad:", bad)
        return

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "stats.json")
    all_stats = {}
    if os.path.exists(stats_path):
        all_stats = json.load(open(stats_path, encoding="utf-8"))

    if args.stream:
        for split in args.splits.split(","):
            out_path = os.path.join(out_dir, f"{split}.jsonl")
            st, bad = parse_split_stream(args.dataset, split, out_path)
            st["bad_triples"] = bad
            all_stats[split] = st
            print(json.dumps(st, ensure_ascii=False, indent=1), flush=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=1)
        return

    for split in args.splits.split(","):
        rows, bad = parse_split(args.dataset, split)
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = stats(rows, args.dataset, split)
        st["bad_triples"] = bad
        all_stats[split] = st
        print(json.dumps(st, ensure_ascii=False, indent=1), flush=True)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
