# -*- coding: utf-8 -*-
"""创新 C 的数据划分：留出一批“冷启动关系”。

输出:
  {out}/held_rels.json        被留出的关系列表
  {out}/g1_{split}_held.jsonl 过滤后的 g1（去掉 target 属于 held 的样本）
  {out}/test_held.jsonl       测试题关联: {question, target, held}
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_e2e import prompt  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-train", required=True)
    ap.add_argument("--g1-val", required=True)
    ap.add_argument("--g1-test", required=True)
    ap.add_argument("--rels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rels = json.load(open(args.rels, encoding="utf-8"))
    relset = set(rels)
    tr = load(args.g1_train)
    va = load(args.g1_val)
    te = load(args.g1_test)

    cnt = {}
    for r in tr:
        cnt[r["target"]] = cnt.get(r["target"], 0) + 1
    cand = sorted([r for r, c in cnt.items() if c >= args.min_count and r in relset])
    random.Random(args.seed).shuffle(cand)
    n_hold = max(1, int(len(cand) * args.frac))
    held = set(cand[:n_hold])

    def dump(path, rows, keep_held):
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                if (r["target"] in held) == keep_held:
                    r = dict(r)
                    r["text"] = prompt(r["question"], tuple(r["prefix"]))
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        return n

    n_tr = dump(os.path.join(args.out, "g1_train_held.jsonl"), tr, False)
    n_va = dump(os.path.join(args.out, "g1_val_held.jsonl"), va, False)
    n_va_full = dump(os.path.join(args.out, "g1_val_full.jsonl"), va, True)
    with open(os.path.join(args.out, "test_held.jsonl"), "w", encoding="utf-8") as f:
        for r in te:
            f.write(json.dumps({"question": r["question"], "target": r["target"],
                                "prefix": r["prefix"],
                                "held": r["target"] in held},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "held_rels.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(held), f, ensure_ascii=False, indent=1)

    n_tr_held_rows = sum(1 for r in tr if r["target"] in held)
    print(json.dumps({
        "n_rels": len(rels), "n_cand": len(cand), "n_held": len(held),
        "held_frac_rows_train": n_tr_held_rows / max(len(tr), 1),
        "train_rows_kept": n_tr, "val_rows_kept": n_va, "val_rows_held": n_va_full,
        "test_rows_held": sum(1 for r in te if r["target"] in held),
        "test_rows": len(te),
    }, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
