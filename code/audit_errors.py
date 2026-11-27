# -*- coding: utf-8 -*-
"""P0-1 错误分解 + 合法答案审计。

输入 eval_e2e.py --dump 的逐题预测（pred / gold_rank / n_cand），结合每题子图：
  correct / rerank_error / search_miss
  search_miss 细分: not_in_graph / closed_set_unreachable / search_budget_miss
  rerank_error 细分: rank 桶 + 是否在 rerank 池（默认 top-200）
  启发式“可能的合法替代答案”: pred 与 gold 共享 (head, relation)，且 head 合法可达

用法:
  python audit_errors.py --dataset webqsp --split test \
      --dump E:\\GooseRAG\\data\\rog_qa\\dump_webqsp_test_rwkv_ft200.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict, deque

DATA = r"E:\GooseRAG\data"
ROG_QA = os.path.join(DATA, "rog_qa")
GENERIC_PREFIX = ("common.topic.", "freebase.valuenotation.", "common.webpage.",
                  "common.document.", "type.object.")


def is_generic(rel):
    return rel.startswith(GENERIC_PREFIX)


def build_graph(edges):
    out = defaultdict(lambda: defaultdict(list))
    in_edges = defaultdict(set)
    nodes = set()
    for h, r, t in edges:
        out[h][r].append(t)
        in_edges[t].add((h, r))
        nodes.add(h)
        nodes.add(t)
    return out, in_edges, nodes


def shortest_rel_path(starts, target, out, max_hops, relset, ban_generic=True):
    """返回从任一 start 到 target 的一条最短合法关系序列；不可达返回 None。"""
    seen = {s: () for s in starts}
    depth = {s: 0 for s in starts}
    q = deque(starts)
    while q:
        e = q.popleft()
        if depth[e] >= max_hops:
            continue
        for r, tails in out.get(e, {}).items():
            if r not in relset:
                continue
            if ban_generic and is_generic(r):
                continue
            for t in tails:
                if t in seen:
                    continue
                seen[t] = seen[e] + (r,)
                depth[t] = depth[e] + 1
                if t == target:
                    return seen[t]
                q.append(t)
    return None


def reachable_nodes(starts, out, max_hops, relset, ban_generic=True):
    seen = set(starts)
    cur = set(starts)
    for _ in range(max_hops):
        nxt = set()
        for e in cur:
            for r, tails in out.get(e, {}).items():
                if r not in relset:
                    continue
                if ban_generic and is_generic(r):
                    continue
                nxt.update(tails)
        nxt -= seen
        seen |= nxt
        cur = nxt
        if not nxt:
            break
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max-hops", type=int, default=0,
                    help="0 表示按数据集默认: webqsp=2, cwq=3")
    ap.add_argument("--rerank-top", type=int, default=200)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.max_hops <= 0:
        args.max_hops = 2 if args.dataset == "webqsp" else 3

    rows = [json.loads(l) for l in
            open(os.path.join(ROG_QA, args.dataset, f"{args.split}.jsonl"), encoding="utf-8")]
    id2row = {str(r["id"]): r for r in rows}
    dump = json.load(open(args.dump, encoding="utf-8"))
    relset = set(json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                                  encoding="utf-8")))

    stats = Counter()
    rank_buckets = Counter()
    samples = defaultdict(list)
    multi_gold = 0
    gold_sizes = Counter()

    for d in dump:
        r = id2row.get(str(d.get("id")))
        if r is None:
            stats["missing_row"] += 1
            continue
        out, in_edges, nodes = build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        gold = set(r["a_entity"]) or set(r["answers"])
        if not starts or not gold:
            stats["skipped"] += 1
            continue
        gold_sizes[len(gold)] += 1
        if len(gold) > 1:
            multi_gold += 1

        rank = d.get("gold_rank")
        pred = d.get("pred") or []
        top1 = pred[0][0] if pred else None
        base = {
            "id": r["id"], "question": r["question"],
            "gold": sorted(gold), "top1": top1,
            "gold_rank": rank, "n_cand": d.get("n_cand"),
        }

        if rank == 0:
            stats["correct"] += 1
            continue

        seen = None
        if rank is not None:
            stats["rerank_error"] += 1
            k = int(rank)
            if k == 1:
                bucket = "1"
            elif k <= 5:
                bucket = "2-5"
            elif k <= 20:
                bucket = "6-20"
            elif k <= 100:
                bucket = "21-100"
            elif k <= args.rerank_top:
                bucket = "101-%d" % args.rerank_top
            else:
                bucket = ">%d" % args.rerank_top
            rank_buckets[bucket] += 1
            if k < args.rerank_top:
                stats["rerank_in_pool"] += 1
            else:
                stats["rerank_outside_pool"] += 1

            # 宽松启发式：pred 与 gold 共享同一 (head, relation)，且 head 合法可达
            alt_loose = False
            if top1 is not None:
                seen = reachable_nodes(starts, out, args.max_hops, relset, True)
                gold_in = set()
                for g in gold:
                    gold_in |= in_edges.get(g, set())
                common = in_edges.get(top1, set()) & gold_in
                if any(h in seen for h, _ in common):
                    alt_loose = True
                stats["rerank_possible_alt_loose"] += int(alt_loose)

            # 严格启发式：pred 与某个 gold 的最短合法关系序列完全相同
            alt_strict = False
            if top1 is not None:
                gold_paths = set()
                for g in gold:
                    p = shortest_rel_path(starts, g, out, args.max_hops, relset, True)
                    if p is not None:
                        gold_paths.add(p)
                p_path = shortest_rel_path(starts, top1, out, args.max_hops, relset, True)
                alt_strict = p_path is not None and p_path in gold_paths
                stats["rerank_possible_alt_strict"] += int(alt_strict)

            cat = "rerank_possible_alt_strict" if alt_strict else "rerank_hard"
            stats[cat] += 1
            if len(samples[cat]) < args.sample:
                s = dict(base)
                s["same_parent_alt_loose"] = alt_loose
                s["same_path_alt_strict"] = alt_strict
                samples[cat].append(s)
            continue

        stats["search_miss"] += 1
        if not (gold & nodes):
            stats["search_not_in_graph"] += 1
            cat = "search_not_in_graph"
        else:
            seen = reachable_nodes(starts, out, args.max_hops, relset, True)
            if not (gold & seen):
                stats["search_closed_set_unreachable"] += 1
                cat = "search_closed_set_unreachable"
            else:
                stats["search_budget_miss"] += 1
                cat = "search_budget_miss"
        if len(samples[cat]) < args.sample:
            samples[cat].append(base)

    total = stats["correct"] + stats["rerank_error"] + stats["search_miss"]
    report = {
        "dataset": args.dataset, "split": args.split, "n": total,
        "max_hops": args.max_hops, "rerank_top": args.rerank_top,
        "counts": dict(stats),
        "rates": {k: (v / max(total, 1)) for k, v in stats.items()},
        "rank_buckets": dict(rank_buckets),
        "multi_gold_questions": multi_gold,
        "gold_size_hist": dict(sorted(gold_sizes.items())),
        "samples": samples,
    }
    print(json.dumps({k: report[k] for k in
                      ("dataset", "split", "n", "max_hops", "counts", "rates",
                       "rank_buckets", "multi_gold_questions")},
                     ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
