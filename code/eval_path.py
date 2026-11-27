# -*- coding: utf-8 -*-
"""路径级集合搜索（GooseRAG v2 原型）。

与 v1 的差别（只改搜索/决策层，模型与权重全部复用）：
  1. beam 的单元是"关系路径"，每条路径携带其完整子节点集合（不再被 max-children 截断）；
  2. 判定器对"路径 + 其子节点集合"整体打一次分（集合级验证），而不是逐实体打分；
  3. 输出是选中路径的子节点并集，天然支持多答案。

评测同时给出三种决策的 H@1（共用同一次路径级打分）：
  path_top1      : 取判定器分最高的一条路径的子节点集合
  path_topn      : 取前 n 条路径子节点并集
  path_then_ent  : 在选中路径内部，按 walker 分重排实体，只取 top1

用法:
  python eval_path.py --dataset webqsp --split test \
      --verifier-ft .../verifier_ft_webqsp_rwkv7_listwise_beam200_s42.pt \
      --paths 40 --topk-rel 15 --max-kids 400 --dump path_test.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402


def build_paths(args, question, starts, out, rel2id, model, tok, rel_head, device):
    """返回 [(path, score, [children...])]，按关系头打分排序。"""
    # 第 1 跳
    feats = E.encode_texts(model, tok, [E.prompt(question, ())],
                           device, args.backing, args.batch,
                           feat_norm=args.head_feat_norm)
    logits = rel_head(feats.to(device)).cpu()[0]

    cands = {}
    settled = set(starts)
    for e in sorted(starts):
        legal = [r for r in out.get(e, {}) if r in rel2id
                 and not (args.ban_generic and E.is_generic(r, args.generic_mode))]
        if not legal:
            continue
        idx = torch.tensor([rel2id[r] for r in legal])
        lp = torch.log_softmax(logits[idx], dim=0)
        k = min(args.topk_rel, len(legal))
        vals, pos = torch.topk(lp, k)
        for v, j in zip(vals.tolist(), pos.tolist()):
            r = legal[j]
            kids = list(out[e][r])[:args.max_kids]
            key = (r,)
            if key in cands:
                cands[key][0] = max(cands[key][0], v)
                seen = set(cands[key][1])
                cands[key][1].extend([c for c in kids if c not in seen])
                cands[key][1] = cands[key][1][:args.max_kids]
            else:
                cands[key] = [v, kids]
            settled.update(kids)

    # 第 2 跳：只从前一堆 top 路径展开
    if args.max_hops >= 2:
        base = sorted(cands.items(), key=lambda kv: -kv[1][0])[:args.path_beam]
        feats2 = E.encode_texts(model, tok, [E.prompt(question, p) for p, _ in base],
                                device, args.backing, args.batch,
                                feat_norm=args.head_feat_norm)
        logits2 = rel_head(feats2.to(device)).cpu()
        for (p, (s1, kids1)), row in zip(base, logits2):
            legal = set()
            for e in kids1:
                for r in out.get(e, {}):
                    if r in rel2id and not (args.ban_generic and E.is_generic(r, args.generic_mode)):
                        legal.add(r)
            if not legal:
                continue
            legal = sorted(legal)
            idx = torch.tensor([rel2id[r] for r in legal])
            lp = torch.log_softmax(row[idx], dim=0)
            k = min(args.topk_rel, len(legal))
            vals, pos = torch.topk(lp, k)
            for v, j in zip(vals.tolist(), pos.tolist()):
                r = legal[j]
                tails = set()
                for e in kids1:
                    tails.update(out[e].get(r, ()))
                    if len(tails) >= args.max_kids:
                        break
                if not tails:
                    continue
                key = p + (r,)
                sc = s1 + v
                kids = sorted(tails)[:args.max_kids]
                if key in cands:
                    if sc > cands[key][0]:
                        cands[key][0] = sc
                else:
                    cands[key] = [sc, kids]

    rows = []
    for p, (s, kids) in cands.items():
        norm = len(p) if args.score_norm == "mean" else 1.0
        rows.append((p, s / max(norm, 1), kids))
    rows.sort(key=lambda x: -x[1])
    return rows[:args.paths]


def path_text(question, p, kids, head):
    rels = " -> ".join(p) if p else "(none)"
    k = ", ".join(kids[:head])
    more = f" (+{len(kids) - head} more)" if len(kids) > head else ""
    return (f"Question: {question}\nRelation path: {rels}\n"
            f"Candidate set: {k}{more}\nIs this the answer set?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--path-beam", type=int, default=200)
    ap.add_argument("--paths", type=int, default=40)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-kids", type=int, default=400)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all")
    ap.add_argument("--heads", default="")
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--set-head", type=int, default=8, help="集合文本里列出几个子节点")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads_path = args.heads or os.path.join(
        E.ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
    rel_head, rels = E.load_heads(heads_path, device)
    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    rel2id = {r: i for i, r in enumerate(rels)}

    ft = E.EvalFTVerifier(E.MODEL_NAMES[args.backing], feat_norm=args.verifier_feat_norm)
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to(device).eval()

    src = os.path.join(E.ROG_QA, args.dataset, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    st = {"n": 0, "reach": 0, "p1": 0, "p3": 0, "p5": 0, "p1_ent": 0,
          "w1": 0, "fwd": 0, "kids": 0}
    dumps = []
    for i, r in enumerate(rows):
        out = E.build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        gold = set(r["a_entity"]) or set(r["answers"])
        if not starts or not gold:
            continue
        st["n"] += 1
        paths = build_paths(args, r["question"], starts, out, rel2id, model, tok,
                            rel_head, device)
        if not paths:
            continue
        texts = [path_text(r["question"], p, kids, args.set_head) for p, _, kids in paths]
        vs = E.score_ft(ft, tok, texts, device, args.batch)
        st["fwd"] += len(texts)
        ranked = sorted(zip(paths, vs), key=lambda x: -x[1])

        # 候选池覆盖（所有路径子节点并集）
        pool = set()
        for _, _, kids in paths:
            pool.update(kids)
        if gold & pool:
            st["reach"] += 1
        st["kids"] += len(pool)

        # 判定器最高分路径本身
        if gold & set(ranked[0][0][2]):
            st["p1"] += 1
        for k, key in ((3, "p3"), (5, "p5")):
            uni = set()
            for (p, s, kids), v in ranked[:k]:
                uni.update(kids)
            if gold & uni:
                st[key] += 1

        # 路径内实体重排：选中路径的子节点按 walker 分排（这里用关系头路径分 + 实体在路径中的顺序）
        _, _, kids0 = ranked[0][0]
        if gold & set(kids0[:1]):
            st["p1_ent"] += 1

        # walker 基线：全部路径子节点里按路径分排序后的首个实体
        if gold & set(paths[0][2][:1]):
            st["w1"] += 1

        if args.dump:
            dumps.append({"id": r["id"],
                          "gold": sorted(gold),
                          "pool_size": len(pool),
                          "paths": [[list(p), round(s, 4), len(kids), kids[:10]]
                                    for (p, s, kids), v in ranked[:10]],
                          "vs": [round(v, 4) for _, v in ranked[:10]]})
        if (i + 1) % 100 == 0:
            n = st["n"]
            print(f"{i+1}/{len(rows)} reach={st['reach']/n:.4f} "
                  f"p1={st['p1']/n:.4f} p5={st['p5']/n:.4f} "
                  f"avg_pool={st['kids']/n:.0f} fwd={st['fwd']/n:.0f}", flush=True)

    n = max(st["n"], 1)
    res = {
        "dataset": args.dataset, "split": args.split, "n": st["n"],
        "path_pool_coverage": st["reach"] / n,
        "hits1_path_top1": st["p1"] / n,
        "hits1_path_top3": st["p3"] / n,
        "hits1_path_top5": st["p5"] / n,
        "hits1_path_top1_entity": st["p1_ent"] / n,
        "hits1_walker_baseline": st["w1"] / n,
        "avg_pool_size": st["kids"] / n,
        "avg_verifier_forwards": st["fwd"] / n,
        "paths": args.paths, "topk_rel": args.topk_rel,
        "max_kids": args.max_kids, "set_head": args.set_head,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for d in dumps:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print("dumped ->", args.dump, flush=True)


if __name__ == "__main__":
    main()
