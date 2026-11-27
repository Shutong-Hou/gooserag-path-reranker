# -*- coding: utf-8 -*-
"""创新 C：用“关系名 -> 关系头权重”的线性映射给冷启动关系补权重。

同时在已见关系上做交叉验证探针：测关系名嵌入能在多大程度上预测训练出的头权重
（mean cosine）。若探针成立，说明关系方向具有可迁移的语义结构。
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402


def name_text(rel):
    return rel.replace(".", " ").replace("_", " ")


def ridge(X, Y, lam):
    # X: n x d, Y: n x d  ->  d x d
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--probe-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.heads, map_location="cpu", weights_only=False)
    rels = blob["rels"]
    held = set(json.load(open(args.held, encoding="utf-8")))
    W = blob["rel_state"]["weight"].float()
    b = blob["rel_state"]["bias"].float()
    seen = [i for i, r in enumerate(rels) if r not in held]
    held_i = [i for i, r in enumerate(rels) if r in held]
    print(f"rels={len(rels)} seen={len(seen)} held={len(held_i)}", flush=True)

    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    texts = [name_text(r) for r in rels]
    X = E.encode_texts(model, tok, texts, device, args.backing,
                       batch_size=64, max_len=64, feat_norm="double").float()

    # 探针：在已见关系内部再留一部分，拟合后测权重余弦相似度
    rng = random.Random(args.seed)
    idx = list(seen)
    rng.shuffle(idx)
    n_pr = max(5, int(len(idx) * args.probe_frac))
    pr, fit = idx[:n_pr], idx[n_pr:]
    Wm = ridge(X[fit], W[fit], args.lam)
    cs = [cos(X[i] @ Wm, W[i]) for i in pr]
    probe = {"n_fit": len(fit), "n_probe": n_pr,
             "mean_cos": sum(cs) / len(cs), "max_cos": max(cs), "min_cos": min(cs)}
    print("probe " + json.dumps(probe, ensure_ascii=False), flush=True)

    Wm = ridge(X[seen], W[seen], args.lam)
    W2 = W.clone()
    for i in held_i:
        W2[i] = X[i] @ Wm
    b2 = b.clone()
    b2[held_i] = b[seen].mean()

    state = dict(blob["rel_state"])
    state["weight"] = W2
    state["bias"] = b2
    torch.save({
        "rel_state": state, "rels": rels, "hidden": blob["hidden"],
        "backing": blob.get("backing", args.backing),
        "imputed": {"held": len(held_i), "lam": args.lam, "probe": probe},
    }, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
