# -*- coding: utf-8 -*-
"""自动汇总所有实验产物 -> experiments_auto.jsonl

扫描 data/rog_qa 下的结果文件，提取指标与配置，形成机器可读台账。
每行：{file, mtime, kind, metrics, config}
"""
import glob
import json
import os

D = os.environ.get("GOOSERAG_ROGQA", "/root/autodl-tmp/GooseRAG/data/rog_qa")
OUT = "/root/autodl-tmp/GooseRAG/experiments_auto.jsonl"

KEYS_CFG = ["dataset", "backing", "split", "beam", "topk_rel", "max_children",
            "max_hops", "verifier_top", "beta", "gamma", "search", "heads"]


def dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(D, "*.json"))):
        name = os.path.basename(p)
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        m = {}
        # eval_e2e 输出
        if "hits1" in d:
            m["h1"] = d.get("hits1")
            m["h5"] = dig(d, "gold_rank_rate", "top5")
            m["in_cand"] = dig(d, "gold_rank_rate", "in_cand")
            m["n"] = d.get("n")
        # train_setrank 输出
        if "test" in d and isinstance(d["test"], dict):
            m["h1"] = d["test"].get("1")
            m["h5"] = d["test"].get("5")
            m["n"] = d.get("n_test")
        # train_joint 输出
        if "joint_h1" in d:
            m["h1"] = d["joint_h1"]
            m["h5"] = d.get("joint_h5")
            m["stage1_h1"] = d.get("stage1_h1")
            m["n"] = d.get("n")
        if not m:
            continue
        cfg = {k: d.get(k) for k in KEYS_CFG if k in d}
        recs.append({
            "file": name,
            "mtime": int(os.path.getmtime(p)),
            "metrics": m,
            "config": cfg,
        })

    for p in sorted(glob.glob(os.path.join(D, "*.pt"))):
        recs.append({
            "file": os.path.basename(p),
            "mtime": int(os.path.getmtime(p)),
            "kind": "checkpoint",
            "size_mb": round(os.path.getsize(p) / 1024 / 1024, 1),
        })

    with open(OUT, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
