# -*- coding: utf-8 -*-
"""同池隔离 path 判定器的贡献：mined 文件里已存有 plain 判定器分（c[2]）。

同一份候选（VGHN 池 / 默认池）上比较 plain 与 path 两种打分，
唯一变量就是判定器 prompt/权重，候选池完全一致。
"""
import json

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def run(path, tag):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = empty = inc = 0
    h_plain = h_fused = 0
    for r in rows:
        gold = set(r.get("gold") or [])
        cand = r.get("cand") or []
        if not gold:
            continue
        n += 1
        if not cand:
            empty += 1
            continue
        if any(c[0] in gold for c in cand):
            inc += 1
        kp = max(range(len(cand)), key=lambda k: cand[k][2])
        kf = max(range(len(cand)), key=lambda k: 0.5 * cand[k][1] + 2.0 * cand[k][2])
        h_plain += cand[kp][0] in gold
        h_fused += cand[kf][0] in gold
    print(f"{tag}: n={n} empty={empty} in_cand={inc/n:.4f} "
          f"plain_h1={h_plain/n:.4f} fused_h1={h_fused/n:.4f}", flush=True)


run(f"{D}/mined_webqsp_test_s42.jsonl", "default-head pool")
run(f"{D}/mined_webqsp_test_vghn.jsonl", "vghn pool")
