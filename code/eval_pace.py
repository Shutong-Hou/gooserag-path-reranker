# -*- coding: utf-8 -*-
"""PACE 端到端评测：walker -> 判定器隐状态 -> SetRank 重排。

用于把离线 0.4898 复现到真实管线上（逐题可对）。
"""
import argparse
import json
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402
from train_setrank import SetRank  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--length-mode", choices=["final", "all"], default="all")
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--policy", choices=["model", "random", "prior"], default="model")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all")
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--heads", default="")
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--setrank", required=True)
    ap.add_argument("--setrank-d", type=int, default=256)
    ap.add_argument("--setrank-layers", type=int, default=2)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--path", action="store_true",
                    help="与 path-aware 判定器配套：编码/打分都用带关系路径的 prompt")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads_path = args.heads or os.path.join(
        E.ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
    rel_head, rels = E.load_heads(heads_path, device)
    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    rel2id = {r: i for i, r in enumerate(rels)}

    ft = E.EvalFTVerifier(E.MODEL_NAMES[args.backing],
                          feat_norm=args.verifier_feat_norm)
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to(device).eval()

    sr = SetRank(d=args.setrank_d, layers=args.setrank_layers)
    sr.load_state_dict(torch.load(args.setrank, map_location="cpu", weights_only=True))
    sr.to(device).eval()

    src = os.path.join(E.ROG_QA, args.dataset, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    stats = {"n": 0, "h1": 0, "h5": 0, "h10": 0, "base_h1": 0, "base_h5": 0}
    dumps = []
    for i, r in enumerate(rows):
        out = E.build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        gold = set(r["a_entity"]) or set(r["answers"])
        if not starts or not gold:
            continue
        ranked, _, path_of = E.run_beam(args, r["question"], starts, out, rel2id,
                                        model, tok, rel_head, None, device)
        stats["n"] += 1
        cand = ranked[:args.top]
        if not cand:
            continue
        paths = [path_of.get(e, ()) for e, _ in cand]
        if args.path:
            texts = [E.prompt_verifier_path(r["question"], p, e)
                     for (e, _), p in zip(cand, paths)]
        else:
            texts = [E.prompt_verifier(r["question"], e) for e, _ in cand]
        feats_list = []
        hs = []
        vs = []
        for j in range(0, len(texts), args.batch):
            chunk = texts[j:j + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=160).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = ft.lm(**enc, output_hidden_states=True, use_cache=False)
            h = o.hidden_states[-1]
            if args.verifier_feat_norm == "double":
                nrm = E._final_norm_module(ft.lm)
                w = next(nrm.parameters(), None)
                h = nrm(h.to(w.dtype) if w is not None else h)
            last = enc["attention_mask"].sum(dim=1) - 1
            hl = h[torch.arange(h.size(0), device=h.device), last].float()
            logit = ft.head(hl)
            vs.extend((logit[:, 1] - logit[:, 0]).float().cpu().tolist())
            hs.append(hl)

        q = toks(r["question"])
        cnt = {}
        for p in paths:
            cnt[p] = cnt.get(p, 0) + 1
        for (e, s), v, p in zip(cand, vs, paths):
            et = toks(e)
            feats_list.append([float(s), float(v), float(len(p)),
                               len(q & et) / max(len(et), 1), float(cnt[p])])

        H = torch.cat(hs, 0)
        F = torch.tensor(feats_list, dtype=torch.float32, device=device)
        with torch.no_grad():
            sc = sr(H, F).cpu().tolist()
        order = sorted(range(len(cand)), key=lambda k: -sc[k])
        pred = [cand[k][0] for k in order]

        # 基线：walker+verifier 融合
        fused = [(e, 0.5 * s + 2.0 * v) for (e, s), v in zip(cand, vs)]
        fused.sort(key=lambda x: -x[1])
        bpred = [e for e, _ in fused]

        if pred and pred[0] in gold:
            stats["h1"] += 1
        if set(pred[:5]) & gold:
            stats["h5"] += 1
        if set(pred[:10]) & gold:
            stats["h10"] += 1
        if bpred and bpred[0] in gold:
            stats["base_h1"] += 1
        if set(bpred[:5]) & gold:
            stats["base_h5"] += 1
        if args.dump:
            gr = next((j for j, e in enumerate(pred) if e in gold), None)
            dumps.append({"id": r["id"], "gold": sorted(gold), "pred": pred[:20],
                          "gold_rank": gr,
                          "hit": int(bool(pred) and pred[0] in gold),
                          "base_hit": int(bool(bpred) and bpred[0] in gold)})
        if (i + 1) % 200 == 0:
            n = stats["n"]
            print(f"  {i+1}/{len(rows)} h1={stats['h1']/n:.4f} h5={stats['h5']/n:.4f} "
                  f"base_h1={stats['base_h1']/n:.4f}", flush=True)
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    n = max(stats["n"], 1)
    res = {k: (v / n if k != "n" else v) for k, v in stats.items()}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(dumps, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
