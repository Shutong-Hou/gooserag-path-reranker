# -*- coding: utf-8 -*-
"""SetRank：候选集合上的交叉编码器重排。

输入 cache_cand_states.py 的缓存；每题候选为一个 token 序列（隐状态 + 标量特征），
用 Transformer 自注意力在候选间比较，题内 listwise softmax 训练。
"""
import argparse
import json

import torch
import torch.nn as nn


class SetRank(nn.Module):
    def __init__(self, d_in=768, n_feat=5, d=256, layers=2, heads=4,
                 use_h=True, use_f=True, use_enc=True):
        super().__init__()
        self.use_h, self.use_f = use_h, use_f
        in_dim = (d_in if use_h else 0) + (n_feat if use_f else 0)
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
        if use_enc:
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=512,
                                             dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        else:
            self.enc = None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, f):
        xs = []
        if self.use_h:
            xs.append(h)
        if self.use_f:
            xs.append(f)
        x = self.proj(torch.cat(xs, dim=-1))
        if self.enc is not None:
            x = self.enc(x)
        return self.head(x).squeeze(-1) * self.logit_scale


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(model, cache, dev, ks=(1, 5, 10)):
    model.eval()
    hits = {k: 0 for k in ks}
    n = 0
    with torch.no_grad():
        for i in range(len(cache["ids"])):
            h = cache["h"][i].float().to(dev)
            f = torch.tensor(cache["feat"][i], dtype=torch.float32, device=dev)
            sc = model(h, f)
            order = sc.argsort(descending=True).tolist()
            gold = set(cache["gold"][i])
            n += 1
            for k in ks:
                if set(cache["ents"][i][j] for j in order[:k]) & gold:
                    hits[k] += 1
    return {k: v / max(n, 1) for k, v in hits.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    # 预注册模式：不做 val 选优/早停，直接用固定轮数的最终模型评测（避免 val 泄漏）
    ap.add_argument("--no-select", action="store_true")
    ap.add_argument("--no-h", action="store_true", help="消融：不使用判定器隐状态")
    ap.add_argument("--no-f", action="store_true", help="消融：不使用 5 维标量特征")
    ap.add_argument("--no-enc", action="store_true", help="消融：去掉集合自注意力（退化为逐项 MLP）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = load_cache(args.train), load_cache(args.val), load_cache(args.test)
    print(f"train={len(tr['ids'])} val={len(va['ids'])} test={len(te['ids'])}", flush=True)

    model = SetRank(d=args.d, layers=args.layers, use_h=not args.no_h,
                    use_f=not args.no_f, use_enc=not args.no_enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = (-1.0, None, 0)
    n = len(tr["ids"])
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n).tolist()
        tot, nb = 0.0, 0
        for i in perm:
            h = tr["h"][i].float().to(dev)
            f = torch.tensor(tr["feat"][i], dtype=torch.float32, device=dev)
            gold = set(tr["gold"][i])
            y = torch.tensor([1.0 if e in gold else 0.0 for e in tr["ents"][i]],
                             device=dev)
            if y.sum() == 0:
                continue
            sc = model(h, f)
            loss = -(y / y.sum() * torch.log_softmax(sc, dim=0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vm, _ = evaluate(model, va, dev)
        print(f"epoch {ep+1} loss={tot/max(nb,1):.4f} val_h1={vm[1]:.4f} val_h5={vm[5]:.4f}",
              flush=True)
        if args.no_select:
            continue
        if vm[1] > best[0]:
            best = (vm[1], {k: v.detach().clone() for k, v in model.state_dict().items()}, 0)
        else:
            best = (best[0], best[1], best[2] + 1)
            if best[2] >= args.patience:
                print(f"early stop at epoch {ep+1}", flush=True)
                break

    if best[1] is not None:
        model.load_state_dict(best[1])

    vm, nv = evaluate(model, va, dev)
    tm, nt = evaluate(model, te, dev)
    res = {"val": vm, "test": tm, "n_val": nv, "n_test": nt}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.out:
        torch.save(model.state_dict(), args.out)
        print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
