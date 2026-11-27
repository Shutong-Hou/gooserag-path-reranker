# -*- coding: utf-8 -*-
"""把 mined 候选转成「带关系路径」的 listwise 训练组。

输出格式与 train_verifier_listwise.py 的 groups 一致，额外带 rels 字段：
  {"question": ..., "cands": [{"entity":..., "label":0/1, "rels":[...]}, ...]}
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cand", type=int, default=200)
    ap.add_argument("--keep-empty", action="store_true",
                    help="保留无正例的组（默认丢弃，省算力）")
    args = ap.parse_args()

    n_q = n_kept = n_cand = n_pos = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for line in open(args.mined, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            n_q += 1
            gold = set(r.get("gold") or [])
            out = []
            for c in (r.get("cand") or [])[:args.max_cand]:
                e, rels = c[0], c[3]
                out.append({"entity": e, "label": int(e in gold), "rels": list(rels)})
            npos = sum(x["label"] for x in out)
            if not out or (npos == 0 and not args.keep_empty):
                continue
            n_kept += 1
            n_cand += len(out)
            n_pos += npos
            w.write(json.dumps({"question": r["question"], "cands": out},
                               ensure_ascii=False) + "\n")
    print(f"questions={n_q} kept={n_kept} cands={n_cand} pos={n_pos} "
          f"pos_rate={n_pos/max(n_cand,1):.4f}", flush=True)


if __name__ == "__main__":
    main()
