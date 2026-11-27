# -*- coding: utf-8 -*-
"""零样本 SetRank 评测：用 A 数据集训练的重排器直接评 B 数据集缓存。

同时输出基线融合（gamma*walker + beta*verifier）作为对照，以及覆盖率。
"""
import argparse
import json

import torch

from train_setrank import SetRank, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n = len(cache["ids"])
    print(f"n={n} avg_cand={sum(len(e) for e in cache['ents'])/max(n,1):.1f}", flush=True)

    # 覆盖率：gold 是否在候选池
    inc = sum(1 for i in range(n) if set(cache["ents"][i]) & set(cache["gold"][i]))
    print(f"in_cand={inc/max(n,1):.4f}", flush=True)

    # 基线融合
    hits = {1: 0, 5: 0}
    for i in range(n):
        feat = cache["feat"][i]
        sc = [args.gamma * f[0] + args.beta * f[1] for f in feat]
        order = sorted(range(len(sc)), key=lambda j: -sc[j])
        gold = set(cache["gold"][i])
        for k in hits:
            if set(cache["ents"][i][j] for j in order[:k]) & gold:
                hits[k] += 1
    print("baseline_fusion " + json.dumps({f"h{k}": v / max(n, 1) for k, v in hits.items()}),
          flush=True)

    model = SetRank(d=args.d, layers=args.layers).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    m, nn_ = evaluate(model, cache, dev, ks=(1, 5, 10))
    print("setrank_zeroshot " + json.dumps({f"h{k}": v for k, v in m.items()}) +
          f" n={nn_}", flush=True)


if __name__ == "__main__":
    main()
