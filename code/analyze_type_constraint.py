# -*- coding: utf-8 -*-
"""原型：答案类型约束对 H@1 的离线影响。

思路：对 who/when/where 类问题，若候选实体带类型且类型与问题不兼容，则降权/过滤。
类型来自子图的 common.topic.notable_types 边。
"""
import argparse
import json
import collections

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"

# 问题类型 -> 兼容类型关键词（不区分大小写子串匹配）
COMPAT = {
    "who": ["person", "profession", "celebrity", "author", "politician", "artist",
            "athlete", "actor", "director", "founder", "president", "singer",
            "musician", "writer", "scientist", "player", "coach", "leader"],
    "where": ["country", "city", "town", "village", "state", "location", "county",
              "province", "region", "place", "continent", "capital"],
    "when": ["date", "year", "time", "event", "season", "championship", "conflict",
             "election", "war"],
    "howmany": ["number", "quantity", "unit", "count", "population"],
}

# 明确不兼容（用于过滤）
INCOMPAT = {
    "where": ["book", "film", "language", "album", "composition", "recording",
              "profession", "person", "author", "character", "genre"],
    "who": ["book", "film", "country", "language", "album", "composition",
            "city", "town", "state", "postal"],
    "when": ["book", "film", "country", "language", "person", "profession"],
}


def qtype(q):
    q = q.lower()
    if q.startswith("who") or "whose" in q:
        return "who"
    if q.startswith("when") or "what year" in q or "what date" in q:
        return "when"
    if q.startswith("where"):
        return "where"
    if "how many" in q or "how much" in q:
        return "howmany"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default=f"{D}/mined_webqsp_test_s42.jsonl")
    ap.add_argument("--raw", default=f"{D}/webqsp/test.jsonl")
    ap.add_argument("--mode", choices=["filter", "penalty"], default="filter")
    ap.add_argument("--penalty", type=float, default=1.0)
    args = ap.parse_args()

    # id -> types
    types_by_id = {}
    for line in open(args.raw, encoding="utf-8"):
        d = json.loads(line)
        t = {}
        for s, r, o in d["edges"]:
            if r == "common.topic.notable_types":
                t.setdefault(s, set()).add(o)
        types_by_id[d["id"]] = t

    n = hit_base = hit_new = 0
    n_affected = n_fixed = n_broken = 0
    for line in open(args.mined, encoding="utf-8"):
        r = json.loads(line)
        gold = set(r.get("gold") or [])
        cand = list(r.get("cand") or [])
        if not cand:
            continue
        types = types_by_id.get(r["id"], {})
        qt = qtype(r["question"])
        n += 1
        # 基线：按 verifier 分数（cand[2]）
        order = sorted(range(len(cand)), key=lambda i: -float(cand[i][2]))
        if cand[order[0]][0] in gold:
            hit_base += 1
        if qt == "other" or qt not in INCOMPAT:
            if cand[order[0]][0] in gold:
                hit_new += 1
            continue

        bad = INCOMPAT[qt]
        scores = [float(c[2]) for c in cand]
        filtered = []
        for i, c in enumerate(cand):
            ts = types.get(c[0], set())
            tstr = " ".join(ts).lower()
            blocked = bool(tstr) and any(k in tstr for k in bad)
            if args.mode == "filter" and blocked:
                n_affected += 1
                continue
            s = scores[i] - (args.penalty if blocked else 0.0)
            filtered.append((i, s))
        if not filtered:
            filtered = [(i, scores[i]) for i in range(len(cand))]
        best = max(filtered, key=lambda x: x[1])[0]
        base_best = order[0]
        new_hit = cand[best][0] in gold
        base_hit = cand[base_best][0] in gold
        if base_hit != new_hit:
            if new_hit:
                n_fixed += 1
            else:
                n_broken += 1
        if new_hit:
            hit_new += 1

    print(json.dumps({
        "n": n,
        "base_h1": hit_base / n,
        "new_h1": hit_new / n,
        "delta_pt": (hit_new - hit_base) / n * 100,
        "affected_candidates": n_affected,
        "fixed": n_fixed,
        "broken": n_broken,
        "mode": args.mode,
        "penalty": args.penalty,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
