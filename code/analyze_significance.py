# -*- coding: utf-8 -*-
"""5-seed 配对显著性分析（纯 CPU，直接用 setrank cache）。

对照关系（同一候选集、同一测试题，唯一变量=排序函数）：
  baseline : path-aware 判定器单次打分 argmax(feat[1])
  full     : 判定器隐状态 + 特征 -> SetRank 重排
输出：H@1 均值/标准差、McNemar 精确检验、bootstrap 95% CI、候选内 oracle。
"""
import json
import math
import sys

import numpy as np
import torch

# 小矩阵在 128 核上会被线程同步拖死，限到 8 线程
torch.set_num_threads(8)

sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
from train_setrank import SetRank, load_cache

CACHE = "/root/autodl-tmp/GooseRAG/data/rog_qa/setrank_cache_vp_test.pt"
SEEDS = [42, 43, 44, 45, 46]
DEV = torch.device("cpu")


def mcnemar(b, c):
    """b = baseline对/ours错, c = baseline错/ours对；精确二项双侧检验。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    cache = load_cache(CACHE)
    n = len(cache["ids"])
    print(f"n_test={n}", flush=True)

    # 预转换一次，避免每个 seed 重复 float() 拷贝
    H32 = [t.float() for t in cache["h"]]
    F32 = [torch.tensor(f, dtype=torch.float32) for f in cache["feat"]]
    GOLD = [set(g) for g in cache["gold"]]

    oracle = 0
    base_hits = []
    for i in range(n):
        gold = GOLD[i]
        ents = cache["ents"][i]
        if set(ents) & gold:
            oracle += 1
        f = cache["feat"][i]
        k = int(np.argmax([row[1] for row in f]))
        base_hits.append(1 if ents[k] in gold else 0)
    base_hits = np.array(base_hits, dtype=np.int8)
    print(f"in_cand_oracle={oracle/n:.4f}  base_h1={base_hits.mean():.4f}", flush=True)

    results = {}
    all_hits = []
    for s in SEEDS:
        model = SetRank()
        sd = torch.load(f"/root/autodl-tmp/GooseRAG/data/rog_qa/setrank_vp_d256_s{s}.pt",
                        map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        model.eval()
        hits = []
        with torch.no_grad():
            for i in range(n):
                sc = model(H32[i], F32[i])
                k = int(sc.argmax())
                hits.append(1 if cache["ents"][i][k] in GOLD[i] else 0)
        hits = np.array(hits, dtype=np.int8)
        all_hits.append(hits)
        b = int(((base_hits == 1) & (hits == 0)).sum())
        c = int(((base_hits == 0) & (hits == 1)).sum())
        lo, hi = boot_ci((hits - base_hits).astype(np.float64))
        results[s] = {"h1": float(hits.mean()), "b": b, "c": c,
                      "p": mcnemar(b, c), "ci": [lo, hi]}
        print(f"seed {s}: h1={hits.mean():.4f} delta={hits.mean()-base_hits.mean():+.4f} "
              f"b={b} c={c} p={results[s]['p']:.3e} CI=[{lo:+.4f},{hi:+.4f}]", flush=True)

    H = np.stack(all_hits)
    mean = float(H.mean(axis=1).mean())
    std = float(H.mean(axis=1).std(ddof=1))
    # 5-seed 多数投票（更强的报告口径，仅作参考）
    vote = (H.sum(axis=0) >= 3).astype(np.int8)
    b = int(((base_hits == 1) & (vote == 0)).sum())
    c = int(((base_hits == 0) & (vote == 1)).sum())
    lo, hi = boot_ci((vote - base_hits).astype(np.float64))
    print(f"5seed mean={mean:.4f} +- {std:.4f}", flush=True)
    print(f"majority-vote h1={vote.mean():.4f} delta={vote.mean()-base_hits.mean():+.4f} "
          f"b={b} c={c} p={mcnemar(b, c):.3e} CI=[{lo:+.4f},{hi:+.4f}]", flush=True)

    out = {"dataset": "webqsp", "split": "test", "n": n, "n_full": 1627,
           "in_cand_oracle": oracle / n,
           "baseline_path_verifier_h1": float(base_hits.mean()),
           "baseline_full_h1": float(base_hits.sum() / 1627),
           "seeds": {str(k): v for k, v in results.items()},
           "mean_h1": mean, "std_h1": std,
           "mean_full_h1": float(H.sum() / 1627 / len(SEEDS)),
           "majority_vote_h1": float(vote.mean()),
           "majority_vote_full_h1": float(vote.sum() / 1627)}
    with open("/root/autodl-tmp/GooseRAG/data/rog_qa/significance_5seed.json", "w") as w:
        json.dump(out, w, indent=1)
    print("saved significance_5seed.json", flush=True)


if __name__ == "__main__":
    main()
