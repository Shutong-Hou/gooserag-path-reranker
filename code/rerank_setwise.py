# -*- coding: utf-8 -*-
"""PACE 的离线重排试点。

输入: mine_train_candidates.py 的缓存（每题候选含 walker 分、判定器分、关系路径）。
模型: 逐候选特征 -> MLP 打分 -> 题内 listwise softmax（gold 为监督）。
评测: 只重排、不重跑搜索，因此可分钟级迭代。

特征（题内 z-score 后使用）:
  v, v-max, v-second, v-rank, s, s-max, s-rank, path_len,
  n_same_path, n_same_first_rel, name_overlap, name_len, n_cand
"""
import argparse
import json
import math
import re

import torch
import torch.nn as nn

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(s.lower()))


def build_rows(rows):
    out = []
    for r in rows:
        gold = set(r["gold"])
        cand = r["cand"]
        if not cand:
            continue
        vs = [c[2] for c in cand]
        ss = [c[1] for c in cand]
        vmax, smax = max(vs), max(ss)
        vsorted = sorted(vs, reverse=True)
        v2 = vsorted[1] if len(vsorted) > 1 else vsorted[0]
        sv = torch.tensor([[c[2], c[1]] for c in cand])
        vrank = (-sv[:, 0]).argsort().argsort().float() / max(len(cand) - 1, 1)
        srank = (-sv[:, 1]).argsort().argsort().float() / max(len(cand) - 1, 1)
        q = toks(r["question"])
        paths = [tuple(c[3]) for c in cand]
        cnt = {}
        cnt1 = {}
        for p in paths:
            cnt[p] = cnt.get(p, 0) + 1
            if p:
                cnt1[p[0]] = cnt1.get(p[0], 0) + 1
        feats = []
        for i, (e, s, v, p) in enumerate(cand):
            et = toks(e)
            ov = len(q & et) / max(len(et), 1)
            feats.append([
                v, v - vmax, v - v2, float(vrank[i]),
                s, s - smax, float(srank[i]),
                float(len(p)), float(cnt[tuple(p)]), float(cnt1.get(tuple(p)[0], 0) if p else 0),
                ov, float(len(et)), float(len(cand)),
            ])
        x = torch.tensor(feats, dtype=torch.float32)
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        y = torch.tensor([1.0 if e in gold else 0.0 for e, _, _, _ in cand])
        out.append({"id": r["id"], "x": x, "y": y, "v": torch.tensor(vs),
                    "s": torch.tensor(ss),
                    "fused": 0.5 * torch.tensor(ss) + 2.0 * torch.tensor(vs),
                    "gold": gold, "ents": [c[0] for c in cand],
                    "q_len": float(len(r["question"].split())),
                    "hop": [float(len(c[3])) for c in cand]})
    return out


class Scorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate(model, rows, lam, ks=(1, 5, 10, 20, 50, 100, 200)):
    """score = z(mlp) + lam * z(v)；返回各 k 的 hits 率。"""
    hits = {k: 0 for k in ks}
    n = 0
    with torch.no_grad():
        for r in rows:
            sc = model(r["x"])
            sc = (sc - sc.mean()) / (sc.std() + 1e-6)
            v = r["v"]
            v = (v - v.mean()) / (v.std() + 1e-6)
            fused = sc + lam * v
            order = fused.argsort(descending=True).tolist()
            n += 1
            for k in ks:
                top = set(r["ents"][i] for i in order[:k])
                if top & r["gold"]:
                    hits[k] += 1
    return {k: hits[k] / max(n, 1) for k in ks}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--budget-dump", default="")
    ap.add_argument("--budget-ms", default="25,50,100,150,200")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = build_rows([json.loads(l) for l in open(args.train, encoding="utf-8")])
    va = build_rows([json.loads(l) for l in open(args.val, encoding="utf-8")])
    te = build_rows([json.loads(l) for l in open(args.test, encoding="utf-8")])
    for rows in (tr, va, te):
        for r in rows:
            for key in ("x", "y", "v", "s", "fused"):
                r[key] = r[key].to(dev)
    print(f"rows train={len(tr)} val={len(va)} test={len(te)} device={dev}", flush=True)

    model = Scorer(tr[0]["x"].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        order = torch.randperm(len(tr)).tolist()
        for i in range(0, len(order), 64):
            batch = [tr[j] for j in order[i:i + 64]]
            batch = [r for r in batch if r["y"].sum() > 0]
            if not batch:
                continue
            loss = 0.0
            for r in batch:
                sc = model(r["x"])
                tgt = r["y"] / (r["y"].sum() + 1e-9)
                loss = loss - (tgt * torch.log_softmax(sc, dim=0)).sum()
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1} loss={tot/max(nb,1):.4f}", flush=True)

    # 以 val 选 lam（只看 hits@1）
    best = (-1, 0.0)
    for lam in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        m, _ = evaluate(model, va, lam)
        if m[1] > best[0]:
            best = (m[1], lam)
    lam = best[1]

    val_m, nv = evaluate(model, va, lam)
    test_m, nt = evaluate(model, te, lam)
    base_val, _ = evaluate_zero(va)
    base_test, _ = evaluate_zero(te)
    sys_val, _ = evaluate_key(va, "fused")
    sys_test, _ = evaluate_key(te, "fused")
    res = {
        "lam": lam, "n_train": len(tr), "n_val": nv, "n_test": nt,
        "val": val_m, "test": test_m,
        "base_val_verifier_only": base_val, "base_test_verifier_only": base_test,
        "sys_val_fused": sys_val, "sys_test_fused": sys_test,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if args.save:
        torch.save(model.state_dict(), args.save)
        print("saved ->", args.save, flush=True)
    if args.budget_dump:
        ms = [int(x) for x in args.budget_ms.split(",")]
        budget_dump(te, model, lam, ms, args.budget_dump)
        print("budget dump ->", args.budget_dump, flush=True)


def evaluate_zero(rows, ks=(1, 5, 10, 20, 50, 100, 200)):
    hits = {k: 0 for k in ks}
    n = 0
    for r in rows:
        order = r["v"].argsort(descending=True).tolist()
        n += 1
        for k in ks:
            if set(r["ents"][i] for i in order[:k]) & r["gold"]:
                hits[k] += 1
    return {k: hits[k] / max(n, 1) for k in ks}, n


def budget_dump(rows, model, lam, ms, path):
    """按 verifier-top=m 截断候选池，输出可供控制器训练的逐题曲线。"""
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for r in rows:
                sc = model(r["x"])
                sc = (sc - sc.mean()) / (sc.std() + 1e-6)
                v = r["v"]
                vz = (v - v.mean()) / (v.std() + 1e-6)
                fused = sc + lam * vz
                gold_rank, feat = {}, {}
                for m in ms:
                    kk = min(m, len(r["ents"]))
                    order = fused[:kk].argsort(descending=True).tolist()
                    gr = None
                    for j, i in enumerate(order):
                        if r["ents"][i] in r["gold"]:
                            gr = j
                            break
                    gold_rank[str(m)] = gr
                    fk = fused[:kk].sort(descending=True).values
                    vk = v[:kk].sort(descending=True).values
                    hops = r["hop"][:kk]
                    feat[str(m)] = {
                        "n_cand": kk, "n_scored": kk,
                        "w_top1": float(vk[0]) if len(vk) else 0.0,
                        "w_margin": float(vk[0] - vk[1]) if len(vk) > 1 else 0.0,
                        "v_top1": float(vk[0]) if len(vk) else 0.0,
                        "v_margin": float(vk[0] - vk[1]) if len(vk) > 1 else 0.0,
                        "v_entropy": float(-(torch.softmax(vk, 0) *
                                             torch.log_softmax(vk, 0)).sum()),
                        "v_std": float(vk.std()) if len(vk) > 1 else 0.0,
                        "f_top1": float(fk[0]) if len(fk) else 0.0,
                        "f_margin": float(fk[0] - fk[1]) if len(fk) > 1 else 0.0,
                        "hop_top1": int(hops[order[0]]) if order else 0,
                        "frac_maxhop": float(sum(1 for h in hops if h >= 2) / max(len(hops), 1)),
                        "n_first_rel": int(len(set(hops))),
                        "q_len": r["q_len"], "n_topic": 1,
                    }
                f.write(_json.dumps({"id": r["id"], "question": "",
                                     "gold": sorted(r["gold"]),
                                     "n_cand": len(r["ents"]),
                                     "walker_gold_rank": None,
                                     "ks": ms,
                                     "gold_rank": gold_rank,
                                     "feat": feat}, ensure_ascii=False) + "\n")


def evaluate_key(rows, key, ks=(1, 5, 10, 20, 50, 100, 200)):
    hits = {k: 0 for k in ks}
    n = 0
    for r in rows:
        order = r[key].argsort(descending=True).tolist()
        n += 1
        for k in ks:
            if set(r["ents"][i] for i in order[:k]) & r["gold"]:
                hits[k] += 1
    return {k: hits[k] / max(n, 1) for k in ks}, n


if __name__ == "__main__":
    main()
