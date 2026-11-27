# -*- coding: utf-8 -*-
"""创新 A 的控制器训练与评测（CPU 可跑）。

输入: budget_curve_eval.py 生成的 train/val/test JSONL。
协议:
  - 控制器只在 train 上训练
  - 温度 lambda 只在 val 上选（选择规则固定，见 select_lambda）
  - test 只报最终点 + 固定预算参照

成本口径: 判定器前向次数 = verifier-top k（walker/beam 固定，单独报告）
"""
import argparse
import json
import math

import torch
import torch.nn as nn

FEATS = ["n_cand", "n_scored", "w_top1", "w_margin", "v_top1", "v_margin",
         "v_entropy", "v_std", "f_top1", "f_margin", "hop_top1", "frac_maxhop",
         "n_first_rel", "q_len", "n_topic"]


def load(path, ks=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ks:
                r["ks"] = ks
            rows.append(r)
    return rows


def vec(feat, k):
    x = [float(feat.get(name, 0.0)) for name in FEATS]
    x += [math.log1p(k), k / 200.0, 1.0 if feat.get("frac_maxhop", 0) >= 0.99 else 0.0]
    return x


def build(rows):
    xs, ys, meta = [], [], []
    for r in rows:
        for k in r["ks"]:
            gr = r["gold_rank"].get(str(k))
            xs.append(vec(r["feat"][str(k)], k))
            ys.append(1.0 if gr == 0 else 0.0)
            meta.append((r["id"], k))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32), meta


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, rows):
    model.eval()
    out = {}
    with torch.no_grad():
        for r in rows:
            for k in r["ks"]:
                x = torch.tensor([vec(r["feat"][str(k)], k)], dtype=torch.float32)
                out[(r["id"], k)] = torch.sigmoid(model(x)).item()
    return out


def eval_policy(rows, probs, lam):
    """每题选 argmax_k p(correct|k) - lam * k/200；返回 hits1 与平均成本。"""
    hits, cost, n = 0, 0.0, 0
    for r in rows:
        best_k, best_u = None, -1e9
        for k in r["ks"]:
            u = probs[(r["id"], k)] - lam * (k / 200.0)
            if u > best_u:
                best_u, best_k = u, k
        cost += best_k
        gr = r["gold_rank"].get(str(best_k))
        if gr == 0:
            hits += 1
        n += 1
    return hits / max(n, 1), cost / max(n, 1)


def fixed(rows, k):
    hits, n = 0, 0
    for r in rows:
        if k not in r["ks"]:
            continue
        n += 1
        if r["gold_rank"].get(str(k)) == 0:
            hits += 1
    return hits / max(n, 1), k


def heuristic(rows, t, k_lo, k_hi):
    """简单规则基线: f_margin>=t 用 k_lo，否则 k_hi。"""
    hits, cost, n = 0, 0.0, 0
    for r in rows:
        f = r["feat"][str(k_hi)]
        k = k_lo if f.get("f_margin", 0.0) >= t else k_hi
        cost += k
        if r["gold_rank"].get(str(k)) == 0:
            hits += 1
        n += 1
    return hits / max(n, 1), cost / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help=">0 时从 train 内部切一份选 lambda；0 表示用官方 val（测试分布更近）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tr, va, te = load(args.train), load(args.val), load(args.test)
    va_ext = va
    if args.val_frac > 0:
        import random as _random
        order = list(range(len(tr)))
        _random.Random(args.seed).shuffle(order)
        n_sel = max(50, int(len(tr) * args.val_frac))
        va = [tr[i] for i in order[:n_sel]]
        tr = [tr[i] for i in order[n_sel:]]
        print(f"controller split: train={len(tr)} select={len(va)} official_val={len(va_ext)}",
              flush=True)
    ks = sorted({k for r in tr for k in r["ks"]})
    x, y, _ = build(tr)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x = (x - mu) / sd

    model = MLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = y.mean().item()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-6)]))
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, x.shape[0], 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = lossf(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
        if (ep + 1) % 50 == 0:
            print(f"epoch {ep+1} loss={loss.item():.4f}", flush=True)

    def probs_of(rows):
        out = {}
        model.eval()
        with torch.no_grad():
            for r in rows:
                for k in r["ks"]:
                    xv = torch.tensor([vec(r["feat"][str(k)], k)], dtype=torch.float32)
                    xv = (xv - mu) / sd
                    out[(r["id"], k)] = torch.sigmoid(model(xv)).item()
        return out

    pv, pt = probs_of(va), probs_of(te)
    pv_ext = probs_of(va_ext)

    # 人工规则基线（在 val 上调阈值，两档预算 50/200）
    k_lo, k_hi = (50, 200) if 50 in ks and 200 in ks else (ks[0], ks[-1])
    best_h, best_t = -1, None
    for t in [i * 0.5 for i in range(-40, 41)]:
        h, c = heuristic(va, t, k_lo, k_hi)
        if h > best_h:
            best_h, best_t = h, t

    # lambda 选择规则（只用 val）: 在 val 精度不低于固定 200 的 -0.2pt 前提下，取最小成本
    fixed200_val, _ = fixed(va, ks[-1])
    target = fixed200_val - 0.002
    grid = [i * 0.01 for i in range(0, 200)]
    rows_curve = []
    lam_sel, cost_sel = None, None
    for lam in grid:
        h, c = eval_policy(va, pv, lam)
        rows_curve.append({"lam": lam, "val_hits1": h, "val_cost": c})
        if h >= target and (lam_sel is None or c < cost_sel):
            lam_sel, cost_sel = lam, c
    if lam_sel is None:
        lam_sel = min(rows_curve, key=lambda z: -z["val_hits1"])["lam"]

    val_h, val_c = eval_policy(va, pv, lam_sel)
    test_h, test_c = eval_policy(te, pt, lam_sel)

    # 成本匹配表：在 val 上按目标平均成本选 lambda，再看 test（只按成本选，不看 test 精度）
    matched = {}
    for target in ks:
        c = min(rows_curve, key=lambda z: (abs(z["val_cost"] - target), -z["val_hits1"]))
        th, tc = eval_policy(te, pt, c["lam"])
        matched[str(target)] = {"lam": c["lam"], "val_cost": c["val_cost"],
                                "test_hits1": th, "test_cost": tc}

    res = {
        "ks": ks, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "lambda_selected": lam_sel,
        "val": {"adaptive_hits1": val_h, "adaptive_cost": val_c,
                "rule_hits1": best_h, "rule_threshold": best_t, "rule_cost":
                    heuristic(va, best_t, k_lo, k_hi)[1]},
        "test": {"adaptive_hits1": test_h, "adaptive_cost": test_c,
                 "rule_hits1": heuristic(te, best_t, k_lo, k_hi)[0],
                 "rule_cost": heuristic(te, best_t, k_lo, k_hi)[1]},
        "test_fixed": {str(k): fixed(te, k)[0] for k in ks},
        "test_matched": matched,
        "val_fixed": {str(k): fixed(va, k)[0] for k in ks},
        "val_official": {
            "adaptive_hits1": eval_policy(va_ext, pv_ext, lam_sel)[0],
            "adaptive_cost": eval_policy(va_ext, pv_ext, lam_sel)[1],
            "fixed": {str(k): fixed(va_ext, k)[0] for k in ks},
            "n": len(va_ext),
        },
        "oracle": {
            "test_hits1": sum(1 for r in te if any(
                r["gold_rank"].get(str(k)) == 0 for k in r["ks"])) / max(len(te), 1),
            "test_cost": (sum(min([k for k in r["ks"]
                                   if r["gold_rank"].get(str(k)) == 0], default=max(r["ks"]))
                              for r in te) / max(len(te), 1)),
        },
        "val_curve": rows_curve[::10],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
