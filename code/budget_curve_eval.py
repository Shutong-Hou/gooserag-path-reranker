# -*- coding: utf-8 -*-
"""创新 A 的数据生成：单次前向得到多预算逐题曲线。

对每题：walker beam 一次 + 判定器对 top-Kmax 候选打分一次，
然后模拟 verifier-top k ∈ --ks 的融合排序，输出逐题 gold_rank 与特征。

输出 JSONL，一行一题：
  {"id","question","n_cand","walker_gold_rank","ks":[...],
   "gold_rank":{k:v},"feat":{k:{...}}}
用法（云端）:
  python budget_curve_eval.py --dataset webqsp --split train \
      --verifier-ft .../verifier_ft_webqsp_rwkv7_listwise_s42.pt \
      --ks 25,50,100,150,200 --dump budget_webqsp_train.jsonl
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

import eval_e2e as E


def entropy(xs):
    if not xs:
        return 0.0
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    ps = [x / s for x in ex]
    return -sum(p * math.log(p + 1e-12) for p in ps if p > 0)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ks", default="25,50,100,150,200")
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--length-mode", choices=["final", "all"], default="all")
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-top", type=int, default=100)
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--verifier", action="store_true")
    ap.add_argument("--verifier-file", default="")
    ap.add_argument("--verifier-ft", default="")
    ap.add_argument("--verifier-path", action="store_true")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--search", choices=["entity", "set"], default="entity")
    ap.add_argument("--max-set", type=int, default=500)
    ap.add_argument("--policy", choices=["model", "random", "prior"], default="model")
    ap.add_argument("--heads", default="", help="关系头 ckpt；默认 {dataset}_{backing}_heads.pt")
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    kmax = max(ks)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    heads_path = args.heads or os.path.join(
        E.ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
    rel_head, rels = E.load_heads(heads_path, device)
    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    rel2id = {r: i for i, r in enumerate(rels)}

    ver_head = ft_model = None
    if args.verifier_ft:
        ft_model = E.EvalFTVerifier(E.MODEL_NAMES[args.backing],
                                    feat_norm=args.verifier_feat_norm)
        fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
        ft_model.lm.load_state_dict(fb["lm_state"])
        ft_model.head.load_state_dict(fb["head_state"])
        ft_model.to(device).eval()
        print(f"verifier_ft={os.path.basename(args.verifier_ft)} "
              f"val_acc={fb.get('val_acc')} feat_norm={args.verifier_feat_norm}", flush=True)
    elif args.verifier:
        vp = args.verifier_file or os.path.join(
            E.ROG_QA, f"verifier_{args.dataset}_{args.backing}.pt")
        vb = torch.load(vp, map_location="cpu", weights_only=False)
        ver_head = nn.Linear(vb["hidden"], 2)
        ver_head.load_state_dict(vb["state"])
        ver_head.to(device).eval()
        print(f"verifier={os.path.basename(vp)} val_acc={vb.get('val_acc')}", flush=True)
    assert ft_model is not None or ver_head is not None, "必须给 --verifier-ft 或 --verifier"

    src = os.path.join(E.ROG_QA, args.dataset, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    hit_at = {k: 0 for k in ks}
    top5_at = {k: 0 for k in ks}
    walker_hit1 = 0
    tot = 0
    oracle_cost = 0
    oracle_hit = 0
    n_oracle_miss = 0

    with open(args.dump, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            out = E.build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            gold = set(r["a_entity"]) or set(r["answers"])
            if not starts or not gold:
                continue
            tot += 1
            if args.search == "set":
                ranked, beams, path_of = E.run_set_beam(
                    args, r["question"], starts, out, rel2id, model, tok,
                    rel_head, None, device)
            else:
                ranked, beams, path_of = E.run_beam(
                    args, r["question"], starts, out, rel2id, model, tok,
                    rel_head, None, device)

            walker_rank = None
            for j, (e, _) in enumerate(ranked):
                if e in gold:
                    walker_rank = j
                    break
            if walker_rank == 0:
                walker_hit1 += 1

            cand = ranked[:kmax]
            if args.verifier_path:
                texts = [E.prompt_verifier_path(r["question"], path_of.get(e, ()), e)
                         for e, _ in cand]
            else:
                texts = [E.prompt_verifier(r["question"], e) for e, _ in cand]
            if ft_model is not None:
                vs = E.score_ft(ft_model, tok, texts, device, args.batch)
            else:
                feats = E.encode_texts(model, tok, texts, device, args.backing,
                                       args.batch, feat_norm=args.verifier_feat_norm)
                vl = ver_head(feats.to(device)).cpu()
                vs = (vl[:, 1] - vl[:, 0]).tolist()

            w_scores = [s for _, s in cand]
            gold_rank = {}
            feat = {}
            for k in ks:
                kk = min(k, len(cand))
                fused = [(e, args.gamma * s + args.beta * float(v))
                         for (e, s), v in zip(cand[:kk], vs[:kk])]
                fused.sort(key=lambda x: -x[1])
                full = fused + list(ranked[kk:])
                gr = None
                for j, (e, _) in enumerate(full):
                    if e in gold:
                        gr = j
                        break
                gold_rank[k] = gr
                if gr == 0:
                    hit_at[k] += 1
                if gr is not None and gr < 5:
                    top5_at[k] += 1

                vk = vs[:kk]
                fused_top = [s for _, s in fused]
                hops = [len(path_of.get(e, ())) for e, _ in cand[:kk]]
                first_rels = {p[0] for e, _ in cand[:kk]
                              if (p := path_of.get(e, ()))}
                feat[k] = {
                    "n_cand": len(ranked),
                    "n_scored": kk,
                    "w_top1": w_scores[0] if w_scores else 0.0,
                    "w_margin": (w_scores[0] - w_scores[1]) if len(w_scores) > 1 else 0.0,
                    "v_top1": max(vk) if vk else 0.0,
                    "v_margin": (max(vk) - sorted(vk)[-2]) if len(vk) > 1 else 0.0,
                    "v_entropy": entropy(vk),
                    "v_std": std(vk),
                    "f_top1": fused_top[0] if fused_top else 0.0,
                    "f_margin": (fused_top[0] - fused_top[1]) if len(fused_top) > 1 else 0.0,
                    "hop_top1": hops[0] if hops else 0,
                    "frac_maxhop": (sum(1 for h in hops if h >= args.max_hops) / len(hops)
                                    if hops else 0.0),
                    "n_first_rel": len(first_rels),
                    "q_len": len(r["question"].split()),
                    "n_topic": len(r["q_entity"]),
                }

            best_k = None
            for k in ks:
                if gold_rank[k] == 0:
                    best_k = k
                    break
            if best_k is None:
                n_oracle_miss += 1
            else:
                oracle_hit += 1
                oracle_cost += best_k

            fout.write(json.dumps({
                "id": r["id"], "question": r["question"], "gold": sorted(gold),
                "n_cand": len(ranked), "walker_gold_rank": walker_rank,
                "ks": ks, "gold_rank": {str(k): v for k, v in gold_rank.items()},
                "feat": {str(k): v for k, v in feat.items()},
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)} " +
                      " ".join(f"h1@{k}={hit_at[k]/max(tot,1):.4f}" for k in ks),
                      flush=True)

    res = {
        "dataset": args.dataset, "split": args.split, "n": tot, "ks": ks,
        "hits1_at": {k: hit_at[k] / max(tot, 1) for k in ks},
        "top5_at": {k: top5_at[k] / max(tot, 1) for k in ks},
        "walker_hits1": walker_hit1 / max(tot, 1),
        "oracle_hits1": oracle_hit / max(tot, 1),
        "oracle_avg_cost": oracle_cost / max(oracle_hit, 1),
        "oracle_miss_rate": n_oracle_miss / max(tot, 1),
        "verifier_ft": args.verifier_ft or None,
        "beta": args.beta, "gamma": args.gamma,
        "beam": args.beam, "max_children": args.max_children,
        "max_hops": args.max_hops, "verifier_feat_norm": args.verifier_feat_norm,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    with open(args.dump + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
