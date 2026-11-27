# -*- coding: utf-8 -*-
"""创新 B 的数据采集：train 全量 walker 候选 + 判定器分数 + 关系路径。

输出 JSONL，一行一题:
  {"id","question","gold":[...],"n_cand":N,
   "cand":[[entity, walker_score, verifier_score, [rel,...]], ...]}   # top-Kmax
"""
import argparse
import json
import os

import torch
import torch.nn as nn

import eval_e2e as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kmax", type=int, default=200)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--length-mode", choices=["final", "all"], default="all")
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--heads", default="")
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--search", choices=["entity", "set"], default="entity")
    ap.add_argument("--max-set", type=int, default=500)
    ap.add_argument("--policy", choices=["model", "random", "prior"], default="model")
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads_path = args.heads or os.path.join(
        E.ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
    rel_head, rels = E.load_heads(heads_path, device)
    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    rel2id = {r: i for i, r in enumerate(rels)}

    ft_model = E.EvalFTVerifier(E.MODEL_NAMES[args.backing],
                                feat_norm=args.verifier_feat_norm)
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft_model.lm.load_state_dict(fb["lm_state"])
    ft_model.head.load_state_dict(fb["head_state"])
    ft_model.to(device).eval()
    print(f"verifier={os.path.basename(args.verifier_ft)} val_acc={fb.get('val_acc')}", flush=True)

    src = os.path.join(E.ROG_QA, args.dataset, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    n_gold_in = 0
    tot = 0
    with open(args.dump, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            out = E.build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            if not starts or not gold:
                continue
            tot += 1
            ranked, _, path_of = E.run_beam(
                args, r["question"], starts, out, rel2id, model, tok,
                rel_head, None, device)
            cand = ranked[:args.kmax]
            texts = [E.prompt_verifier(r["question"], e) for e, _ in cand]
            vs = E.score_ft(ft_model, tok, texts, device, args.batch) if cand else []
            items = [[e, float(s), float(v), list(path_of.get(e, ()))]
                     for (e, s), v in zip(cand, vs)]
            if any(e in gold for e, _, _, _ in items):
                n_gold_in += 1
            fout.write(json.dumps({
                "id": r["id"], "question": r["question"], "gold": sorted(gold),
                "n_cand": len(ranked), "cand": items,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)} gold_in_cand={n_gold_in/max(tot,1):.4f}",
                      flush=True)

    print(json.dumps({"n": tot, "gold_in_cand_rate": n_gold_in / max(tot, 1),
                      "dump": args.dump}, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
