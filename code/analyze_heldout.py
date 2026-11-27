# -*- coding: utf-8 -*-
"""创新 C：把预算曲线 dump 按“是否涉及留出关系”拆开报告。

用法:
  python analyze_heldout.py --dump budget_webqsp_test_held.jsonl \
      --test-held heldout/test_held.jsonl --ks 100,200
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--test-held", required=True)
    ap.add_argument("--ks", default="")
    args = ap.parse_args()

    held = {}
    for line in open(args.test_held, encoding="utf-8"):
        r = json.loads(line)
        q = r["question"]
        held[q] = held.get(q, False) or bool(r["held"])

    rows = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    ks = [int(x) for x in args.ks.split(",")] if args.ks else list(rows[0]["ks"])

    def stat(group):
        out = {}
        for k in ks:
            n = len(group)
            h1 = sum(1 for r in group if r["gold_rank"].get(str(k)) == 0) / max(n, 1)
            t5 = sum(1 for r in group
                     if r["gold_rank"].get(str(k)) is not None
                     and r["gold_rank"][str(k)] < 5) / max(n, 1)
            inc = sum(1 for r in group if r["gold_rank"].get(str(k)) is not None) / max(n, 1)
            out[str(k)] = {"hits1": h1, "top5": t5, "in_cand": inc, "n": n}
        return out

    grp_held = [r for r in rows if held.get(r["question"], False)]
    grp_other = [r for r in rows if not held.get(r["question"], False)]
    res = {"dump": args.dump, "n": len(rows),
           "n_held_q": len(grp_held), "n_other_q": len(grp_other),
           "all": stat(rows), "held": stat(grp_held), "other": stat(grp_other)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
