# -*- coding: utf-8 -*-
"""P2: build CWQ listwise training groups with different path renderings.

Source: walker_cands_cwq_train_stride4k.jsonl (4,000 questions, <=50 candidates each,
each candidate carries the relation chain that reached it).

Variants
  full  : the whole relation chain        "a.b.c -> d.e.f"
  hop1  : the first relation only         "a.b.c"
  type  : coarse relation domains only    "film | people"   (first segment, deduped, sorted)

All variants keep the same questions, candidates and labels, so any difference is
attributable to the path string alone.
"""
import json
import os

SRC = r"E:\GooseRAG\data\rog_qa\walker_cands_cwq_train_stride4k.jsonl"
OUT = r"E:\GooseRAG\data\rog_qa"


def render(rels, mode):
    rels = [r for r in rels if r]
    if mode == "full":
        return list(rels)
    if mode == "hop1":
        return rels[:1]
    if mode == "type":
        doms = sorted({r.split(".")[0] for r in rels})
        return doms
    raise ValueError(mode)


def main():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print("groups", len(rows), flush=True)

    stats = {}
    for mode in ("full", "hop1", "type"):
        n_cand = n_pos = 0
        lens = []
        path = os.path.join(OUT, f"_cwq_groups_{mode}.jsonl")
        with open(path, "w", encoding="utf-8") as w:
            for r in rows:
                cands = []
                for c in r["cands"]:
                    rels = render(c.get("path") or c.get("rels") or (), mode)
                    cands.append({"entity": c["entity"], "rels": rels,
                                  "label": int(c["label"])})
                    n_cand += 1
                    n_pos += int(c["label"])
                    lens.append(len(rels))
                if not any(x["label"] for x in cands):
                    continue
                w.write(json.dumps({"question": r["question"], "id": r.get("id"),
                                    "cands": cands}, ensure_ascii=False) + "\n")
        stats[mode] = {"candidates": n_cand, "positives": n_pos,
                       "pos_rate": round(n_pos / max(n_cand, 1), 4),
                       "avg_path_len": round(sum(lens) / max(len(lens), 1), 2),
                       "distinct_paths": len({tuple(render(c.get("path") or (), mode))
                                              for r in rows for c in r["cands"]})}
        print(mode, stats[mode], flush=True)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
