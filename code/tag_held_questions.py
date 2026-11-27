# -*- coding: utf-8 -*-
"""给 test/val 每道题标出：它的最短 gold 路径是否用到了留出关系。

输出 JSONL: {"id","question","n_paths","held","path":[...],"gold_hop1":rel_or_null}
"""
import argparse
import json
import os
from collections import defaultdict, deque


def build(edges):
    out = defaultdict(lambda: defaultdict(list))
    for h, r, t in edges:
        out[h][r].append(t)
    return out


def bfs(out, starts, gold, max_hops):
    """返回 <=max_hops 的 gold 路径列表（关系序列），最多保留 20 条。"""
    paths = []
    q = deque((s, ()) for s in starts)
    seen = set((s, ()) for s in starts)
    while q:
        e, p = q.popleft()
        if e in gold and p:
            paths.append(p)
            if len(paths) >= 20:
                break
        if len(p) >= max_hops:
            continue
        for r, tails in out.get(e, {}).items():
            for t in tails:
                if (t, p + (r,)) not in seen:
                    seen.add((t, p + (r,)))
                    q.append((t, p + (r,)))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hops", type=int, default=2)
    args = ap.parse_args()

    held = set(json.load(open(args.held, encoding="utf-8")))
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    n_held = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            out = build(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            paths = bfs(out, starts, gold, args.max_hops) if starts and gold else []
            is_held = any(any(rel in held for rel in p) for p in paths)
            hop1 = paths[0][0] if paths and len(paths[0]) == 1 else None
            n_held += int(is_held)
            f.write(json.dumps({"id": r["id"], "question": r["question"],
                                "n_paths": len(paths), "held": is_held,
                                "path": list(paths[0]) if paths else [],
                                "gold_hop1": hop1}, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "n_held": n_held,
                      "held_rate": n_held / max(len(rows), 1),
                      "out": args.out}, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
