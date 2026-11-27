# -*- coding: utf-8 -*-
"""两阶段判定器的联合训练（架构级创新）。

与现有流程的区别：
  现有：stage-1（逐步打分）先训完并冻结 -> 再单独训 stage-2（用冻结隐状态）
  本脚本：stage-1 与 stage-2 共享主干，联合优化
         L = L_stage1(pointwise listwise) + lam * L_stage2(二阶 listwise)
         stage-2 的输入包含 stage-1 的输出分，梯度因此回流到 stage-1 与主干。

用法:
  python train_joint.py --train mined_webqsp_train_s42.jsonl \
      --test mined_webqsp_test_s42.jsonl --epochs 3 --lam 1.0
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn

DATA = os.environ.get("GOOSERAG_ROGQA", r"E:\GooseRAG\data\rog_qa")
MODELS = {
    "rwkv7": os.environ.get("GOOSERAG_RWKV7",
                            r"E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF"),
}


def prompt_path(q, rels, e):
    p = " -> ".join(rels) if rels else "(none)"
    return f"Question: {q}\nRelation path: {p}\nCandidate: {e}\nIs this the final answer?"


class Joint(nn.Module):
    def __init__(self, lm, s1_head, feat_norm="double", d=256, n_feat=5, hidden=768):
        super().__init__()
        self.lm = lm
        self.s1 = s1_head
        self.feat_norm = feat_norm
        self.s2 = nn.Sequential(
            nn.LayerNorm(hidden + n_feat),
            nn.Linear(hidden + n_feat, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, 1),
        )

    def encode(self, ids, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.lm(input_ids=ids, attention_mask=mask,
                          output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1]
            if self.feat_norm == "double":
                from train_verifier_ft import _final_norm_module
                nrm = _final_norm_module(self.lm)
                w = next(nrm.parameters(), None)
                h = nrm(h.to(w.dtype) if w is not None else h)
        last = mask.sum(dim=1) - 1
        idx = torch.arange(h.size(0), device=h.device)
        return h[idx, last].float()

    def forward(self, ids, mask, feats):
        hl = self.encode(ids, mask)
        logits1 = self.s1(hl)
        v = logits1[:, 1] - logits1[:, 0]
        feat = torch.cat([feats, v.unsqueeze(-1)], dim=-1)
        s2 = self.s2(torch.cat([hl, feat], dim=-1)).squeeze(-1)
        return v, s2


def group_loss(scores, y):
    pos = y.bool()
    if not pos.any():
        return scores.new_zeros(())
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores[pos], 0)


def make_inputs(tok, q, cand, pad_id, max_len):
    texts = [prompt_path(q, c[3], c[0]) for c in cand]
    ids = [tok(t)["input_ids"][:max_len] for t in texts]
    L = max(len(x) for x in ids)
    out = torch.full((len(ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return out, mask


def make_feats(tok, q, cand):
    """4 维静态特征：walker 分 / 路径长度 / 词面重合 / 同路径候选数。第 5 维在模型内由 stage-1 输出填充。"""
    qset = set(tok(q, add_special_tokens=False)["input_ids"])
    cnt = {}
    for c in cand:
        p = tuple(c[3])
        cnt[p] = cnt.get(p, 0) + 1
    outs = []
    for c in cand:
        et = set(tok(c[0], add_special_tokens=False)["input_ids"])
        outs.append([float(c[1]), float(len(c[3])),
                     len(qset & et) / max(len(et), 1), float(cnt[tuple(c[3])])])
    return outs


def build_group(tok, r, max_cands, max_len, pad_id, rng):
    """把一道题整理成一个候选组；无正例或无候选返回 None。"""
    gold = set(r.get("gold") or [])
    cand = list(r.get("cand") or [])
    if not cand:
        return None
    pos = [c for c in cand if c[0] in gold]
    if not pos:
        return None
    neg = [c for c in cand if c[0] not in gold]
    rng.shuffle(neg)
    keep = pos + neg[:max(max_cands - len(pos), 0)]
    rng.shuffle(keep)
    ids, mask = make_inputs(tok, r["question"], keep, pad_id, max_len)
    feats = torch.tensor(make_feats(tok, r["question"], keep), dtype=torch.float32)
    y = torch.tensor([1.0 if c[0] in gold else 0.0 for c in keep])
    return ids, mask, feats, y


def collate(groups, pad_id):
    L = max(g[0].size(1) for g in groups)
    pad = torch.nn.functional.pad
    ids = torch.cat([pad(g[0], (0, L - g[0].size(1)), value=pad_id) for g in groups], 0)
    mask = torch.cat([pad(g[1], (0, L - g[1].size(1)), value=0) for g in groups], 0)
    feats = torch.cat([g[2] for g in groups], 0)
    y = torch.cat([g[3] for g in groups], 0)
    return ids, mask, feats, y, [g[0].size(0) for g in groups]


def evaluate(model, test_rows, dev, tok, pad_id, max_len, tag=""):
    model.eval()
    h1 = h5 = hs1 = h5s1 = 0
    n = 0
    with torch.no_grad():
        for r in test_rows:
            gold = set(r.get("gold") or [])
            cand = list(r.get("cand") or [])
            if not cand:
                continue
            ids, mask = make_inputs(tok, r["question"], cand, pad_id, max_len)
            feats = torch.tensor(make_feats(tok, r["question"], cand),
                                 dtype=torch.float32, device=dev)
            v, s2 = model(ids.to(dev), mask.to(dev), feats)
            n += 1
            order = s2.argsort(descending=True).tolist()
            if cand[order[0]][0] in gold:
                h1 += 1
            if any(cand[k][0] in gold for k in order[:5]):
                h5 += 1
            o1 = v.argsort(descending=True).tolist()
            if cand[o1[0]][0] in gold:
                hs1 += 1
            if any(cand[k][0] in gold for k in o1[:5]):
                h5s1 += 1
            if n % 500 == 0:
                print(f"  eval[{tag}] {n}/{len(test_rows)} h1={h1/n:.4f} "
                      f"stage1={hs1/n:.4f}", flush=True)
    return {"n": n, "h1": h1 / max(n, 1), "h5": h5 / max(n, 1),
            "stage1_h1": hs1 / max(n, 1), "stage1_h5": h5s1 / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr-lm", type=float, default=2e-6)
    ap.add_argument("--lr-s1", type=float, default=1e-4)
    ap.add_argument("--lr-s2", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-cands", type=int, default=48)
    ap.add_argument("--batch-groups", type=int, default=4,
                    help="每组独立做 listwise，可拼接成一个 batch 提升吞吐")
    ap.add_argument("--groups-per-epoch", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--skip-init-eval", action="store_true",
                    help="跳过训练前的 stage-1 评测（多 seed 时只需做一次）")
    ap.add_argument("--init-ft", default="",
                    help="从已训练的 path-aware 判定器初始化 LM + stage-1 头（必须先做，否则 stage-1 从随机头重训）")
    ap.add_argument("--load-joint", default="",
                    help="载入已联合训练的 checkpoint（lm/s1/s2），配合 --epochs 0 做评测")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    from train_verifier_ft import FTVerifier
    tok = AutoTokenizer.from_pretrained(MODELS["rwkv7"], trust_remote_code=True,
                                       local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    ft = FTVerifier(MODELS["rwkv7"], "double")
    if args.init_ft:
        fb = torch.load(args.init_ft, map_location="cpu", weights_only=False)
        ft.lm.load_state_dict(fb["lm_state"])
        ft.head.load_state_dict(fb["head_state"])
        print(f"init from {args.init_ft}", flush=True)
    model = Joint(ft.lm, ft.head).to(dev)
    if args.load_joint:
        ck = torch.load(args.load_joint, map_location="cpu", weights_only=False)
        model.lm.load_state_dict(ck["lm"])
        model.s1.load_state_dict(ck["s1"])
        model.s2.load_state_dict(ck["s2"])
        print(f"loaded joint {args.load_joint}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.lm.parameters(), "lr": args.lr_lm},
        {"params": model.s1.parameters(), "lr": args.lr_s1},
        {"params": model.s2.parameters(), "lr": args.lr_s2},
    ], weight_decay=1e-4)

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    test_rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit_test:
        test_rows = test_rows[:args.limit_test]

    rng = random.Random(args.seed)
    if args.skip_init_eval:
        init_stage1 = float("nan")
    else:
        pre = evaluate(model, test_rows, dev, tok, pad_id, args.max_len, tag="init")
        init_stage1 = pre["stage1_h1"]
        print(f"init stage1_h1={init_stage1:.4f} joint_h1(random s2)={pre['h1']:.4f}",
              flush=True)
    for ep in range(args.epochs):
        sample = rng.sample(rows, min(args.groups_per_epoch, len(rows)))
        model.train()
        tot1 = tot2 = 0.0
        nb = 0
        buf = []
        t_ep = time.time()
        for r in sample:
            g = build_group(tok, r, args.max_cands, args.max_len, pad_id, rng)
            if g is None:
                continue
            buf.append(g)
            if len(buf) < args.batch_groups:
                continue

            def run_batch(groups):
                ids, mask, feats, y, sizes = collate(groups, pad_id)
                v, s2 = model(ids.to(dev), mask.to(dev), feats.to(dev))
                y = y.to(dev)
                a, b, off = [], [], 0
                for k in sizes:
                    a.append(group_loss(v[off:off + k], y[off:off + k]))
                    b.append(group_loss(s2[off:off + k], y[off:off + k]))
                    off += k
                return torch.stack(a).mean(), torch.stack(b).mean()

            try:
                l1, l2 = run_batch(buf)
                loss = l1 + args.lam * l2
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot1 += l1.item()
                tot2 += l2.item()
                nb += 1
            except torch.cuda.OutOfMemoryError:
                # 长批次会爆显存：退化为逐组（语义等价，只是少了组间平均的并行）
                torch.cuda.empty_cache()
                opt.zero_grad()
                for g in buf:
                    l1, l2 = run_batch([g])
                    loss = l1 + args.lam * l2
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot1 += l1.item()
                    tot2 += l2.item()
                    nb += 1
                torch.cuda.empty_cache()
            buf = []
            if nb % 50 == 0:
                el = time.time() - t_ep
                print(f"  ep{ep} step{nb} L1={tot1/nb:.4f} L2={tot2/nb:.4f} "
                      f"{el/nb:.2f}s/step", flush=True)
        print(f"epoch {ep+1} L1={tot1/max(nb,1):.4f} L2={tot2/max(nb,1):.4f} steps={nb} "
              f"{(time.time()-t_ep)/60:.1f}min", flush=True)

    # 评估：test 全量，stage-2 排序
    ev = evaluate(model, test_rows, dev, tok, pad_id, args.max_len, tag="after")
    res = {"n": ev["n"], "joint_h1": ev["h1"], "joint_h5": ev["h5"],
           "stage1_h1": ev["stage1_h1"], "stage1_h5": ev["stage1_h5"],
           "init_stage1_h1": init_stage1,
           "lam": args.lam, "epochs": args.epochs, "max_cands": args.max_cands,
           "batch_groups": args.batch_groups,
           "groups_per_epoch": args.groups_per_epoch,
           "lr_lm": args.lr_lm, "lr_s1": args.lr_s1, "lr_s2": args.lr_s2,
           "init_ft": args.init_ft, "seed": args.seed}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if args.out:
        torch.save({"lm": model.lm.state_dict(), "s1": model.s1.state_dict(),
                    "s2": model.s2.state_dict(), "meta": res}, args.out)


if __name__ == "__main__":
    main()
