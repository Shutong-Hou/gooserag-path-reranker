# -*- coding: utf-8 -*-
"""效率实测：参数量 / 分阶段延迟 / 前向次数 / 峰值显存。

阶段划分（与论文口径一致）：
  1) walker  : 关系头编码 + beam 扩展（候选生成）
  2) verifier: 对 top-K 候选逐条打分（K = --verifier-top）
  3) setrank : 兼容层重排（若给 --setrank）

前向次数用 monkey patch 统计 encode_texts / score_ft 实际编码的文本条数。
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402
from train_setrank import SetRank  # noqa: E402

COUNT = {"walker_texts": 0, "verifier_texts": 0, "walker_calls": 0, "verifier_calls": 0}
_enc, _sft = E.encode_texts, E.score_ft


def enc_counted(*a, **k):
    texts = a[2]
    COUNT["walker_texts"] += len(texts)
    COUNT["walker_calls"] += 1
    return _enc(*a, **k)


def sft_counted(ft, tok, texts, *a, **k):
    COUNT["verifier_texts"] += len(texts)
    COUNT["verifier_calls"] += 1
    return _sft(ft, tok, texts, *a, **k)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--verifier-top", type=int, default=200)
    ap.add_argument("--verifier-ft", default="")
    ap.add_argument("--setrank", default="")
    ap.add_argument("--setrank-d", type=int, default=256)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cpu-n", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--heads", default="")
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--length-mode", choices=["all", "final"], default="all")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--policy", choices=["model", "random", "prior"], default="model")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all")
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double")
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double")
    args = ap.parse_args()

    E.encode_texts = enc_counted
    E.score_ft = sft_counted

    rows = [json.loads(l) for l in
            open(os.path.join(E.ROG_QA, args.dataset, f"{args.split}.jsonl"), encoding="utf-8")]
    rows = rows[:max(args.n, args.cpu_n) + 8]

    res = {"params": {}, "gpu": {}, "cpu": {}, "io": {}}

    def build(device):
        torch.cuda.empty_cache() if device == "cuda" else None
        heads = args.heads if hasattr(args, "heads") and args.heads else os.path.join(
            E.ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
        rh, rels = E.load_heads(heads, device)
        model, tok, _ = E.load_model(args.backing)
        model.to(device).eval()
        ft = None
        if args.verifier_ft:
            ft = E.EvalFTVerifier(E.MODEL_NAMES[args.backing], feat_norm="double")
            fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
            ft.lm.load_state_dict(fb["lm_state"])
            ft.head.load_state_dict(fb["head_state"])
            ft.to(device).eval()
        sr = None
        if args.setrank:
            sr = SetRank(d=args.setrank_d).to(device)
            sr.load_state_dict(torch.load(args.setrank, map_location="cpu", weights_only=False))
            sr.eval()
        return rels, rh, model, tok, ft, sr

    def run_split(device, n, tag):
        for k in COUNT:
            COUNT[k] = 0
        rels, rh, model, tok, ft, sr = build(device)
        rel2id = {r: i for i, r in enumerate(rels)}
        if device == "cuda":
            res["params"] = {
                "lm": nparams(model), "rel_head": nparams(rh),
                "verifier_total": (nparams(ft) if ft else 0),
                "verifier_backbone": (nparams(ft.lm) if ft else 0),
                "verifier_head": (nparams(ft.head) if ft else 0),
                "setrank": (nparams(sr) if sr else 0),
            }
        t_w = t_v = t_s = 0.0
        done = 0
        for r in rows:
            if done >= n + 2:
                break
            out = E.build_graph(r["edges"])
            starts = [e for e in r["q_entity"] if e in out]
            if not starts:
                continue
            done += 1
            sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
            sync(); a = time.time()
            cand, _, _ = E.run_beam(args, r["question"], starts, out, rel2id,
                                    model, tok, rh, None, device)
            sync(); b = time.time()
            if ft is not None and cand:
                texts = [E.prompt_verifier(r["question"], e) for e, _ in cand[:args.verifier_top]]
                vs = E.score_ft(ft, tok, texts, device, args.batch)
                fused = [(e, 0.5 * s + 2.0 * float(v)) for (e, s), v in zip(cand[:args.verifier_top], vs)]
                fused.sort(key=lambda x: -x[1])
                top = fused
            else:
                top = cand[:args.verifier_top]
            sync(); c = time.time()
            if sr is not None and top:
                # 用判定器的隐状态做 SetRank（这里复用 verifier 编码，真实管线也如此）
                toks = [E.prompt_verifier(r["question"], e) for e, _ in top]
                hs = []
                for j in range(0, len(toks), args.batch):
                    feats = E.encode_texts(model, tok, toks[j:j + args.batch], device,
                                           args.backing, args.batch, feat_norm="double")
                    hs.append(feats)
                h = torch.cat(hs, 0)
                f = torch.tensor([[s, 0.5 * s + 2.0 * 0, 1.0, 0.0, 0.0] for _, s in top],
                                 dtype=torch.float32, device=device)
                with torch.no_grad():
                    _ = sr(h, f)
            sync(); d = time.time()
            if done > 2:  # 前 2 题当 warmup
                t_w += b - a
                t_v += c - b
                t_s += d - c
        m = max(min(n, max(done - 2, 1)), 1)
        res[tag] = {
            "n": m,
            "walker_ms": t_w / m * 1000,
            "verifier_ms": t_v / m * 1000,
            "setrank_ms": t_s / m * 1000,
            "total_ms": (t_w + t_v + t_s) / m * 1000,
        }
        if device == "cuda":
            res["gpu"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        cnt = dict(COUNT)
        res["io"] = {
            "walker_texts_per_q": cnt["walker_texts"] / max(m, 1),
            "verifier_texts_per_q": cnt["verifier_texts"] / max(m, 1),
            "total_texts_per_q": (cnt["walker_texts"] + cnt["verifier_texts"]) / max(m, 1),
        }

    if torch.cuda.is_available():
        run_split("cuda", args.n, "gpu")
    if args.cpu_n > 0:
        run_split("cpu", args.cpu_n, "cpu")

    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
