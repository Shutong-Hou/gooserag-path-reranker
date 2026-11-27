# -*- coding: utf-8 -*-
"""Listwise 判定器：在同一问题的 walker 候选组内做 softmax 重排。

训练数据: walker_cands_{dataset}_{split}.jsonl（由 gen_walker_cands.py 生成）
输出: verifier_ft_{dataset}_{backing}_listwise.pt

用法:
  python train_verifier_listwise.py --dataset webqsp --backing rwkv7 --epochs 1
"""
import argparse
import contextlib
import hashlib
import json
import os
import random

import torch

from train_verifier_ft import FTVerifier, collate, norm_q, MODELS

DATA = os.environ.get("GOOSERAG_ROGQA", r"E:\GooseRAG\data\rog_qa")


USE_PATH = False
PATH_STYLE = "rel"


def fmt_path(rels, ents=None):
    if not rels:
        return "(none)"
    if PATH_STYLE == "entity" and ents:
        parts = [str(ents[0])]
        for r, nxt in zip(rels, ents[1:]):
            parts.append(f" --[{r}]--> {nxt}")
        return "".join(parts)
    if PATH_STYLE == "short":
        return " -> ".join(r.split(".")[-1] for r in rels)
    if PATH_STYLE == "natural":
        return "The answer is reached via " + ", then ".join(rels) + "."
    return " -> ".join(rels)


def prompt(q, e, rels=(), ents=None):
    if USE_PATH:
        p = fmt_path(rels, ents)
        return (f"Question: {q}\nRelation path: {p}\n"
                f"Candidate: {e}\nIs this the final answer?")
    return f"Question: {q}\nCandidate: {e}\nIs this the final answer?"


def load_groups(dataset, split, path=""):
    p = path or os.path.join(DATA, f"walker_cands_{dataset}_{split}.jsonl")
    groups = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("cands"):
            continue
        groups.append(r)
    return groups


def make_batch(groups, tok, pad_id, max_len):
    flat = []
    sizes = []
    for g in groups:
        for c in g["cands"]:
            flat.append((tok(prompt(g["question"], c["entity"], c.get("rels", ()),
                                    c.get("ents")))["input_ids"],
                         int(c["label"])))
        sizes.append(len(g["cands"]))
    ids, mask, ys = collate(flat, pad_id, max_len)
    return ids, mask, ys, sizes


def group_loss(scores, ys, sizes):
    loss = scores.new_zeros(())
    n = 0
    start = 0
    for gs in sizes:
        s = scores[start:start + gs]
        y = ys[start:start + gs].to(s.device)
        pos = y.bool()
        if pos.any():
            loss = loss + (torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0))
            n += 1
        start += gs
    return loss / max(n, 1)


@torch.no_grad()
def evaluate(model, groups, tok, pad_id, max_len, device, batch_groups, autocast):
    model.eval()
    hit = tot = 0
    mrr = 0.0
    for j in range(0, len(groups), batch_groups):
        b = groups[j:j + batch_groups]
        ids, mask, ys, sizes = make_batch(b, tok, pad_id, max_len)
        with autocast():
            logits = model(ids.to(device), mask.to(device))
        scores = (logits[:, 1] - logits[:, 0]).float().cpu()
        start = 0
        for gs in sizes:
            s = scores[start:start + gs]
            y = ys[start:start + gs]
            start += gs
            pos = (y == 1).nonzero(as_tuple=False).flatten().tolist()
            if not pos:
                continue
            pred = int(s.argmax().item())
            hit += int(pred in pos)
            order = s.argsort(descending=True).tolist()
            first = next(i for i, idx in enumerate(order) if idx in pos) + 1
            mrr += 1.0 / first
            tot += 1
    model.train()
    return hit / max(tot, 1), mrr / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq", "metaqa"])
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2", "pythia"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-groups", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-norm", choices=["single", "double"], default="double",
                    help="double=复现历史 ckpt 口径（主表）；single=修复后口径（敏感性）")
    ap.add_argument("--cands", default="", help="候选 JSONL; 默认 walker_cands_{dataset}_{split}.jsonl")
    ap.add_argument("--path", action="store_true", help="在 prompt 中加入关系路径")
    ap.add_argument("--max-cands", type=int, default=0,
                    help="每组最多保留的候选数（保留全部正例 + 采样负例；0=不限制）")
    ap.add_argument("--out", default="")
    ap.add_argument("--path-style", choices=["rel", "short", "natural", "entity"], default="rel",
                    help="关系路径的呈现方式（P1-1 消融）：rel=全名链, short=去命名空间, "
                         "natural=自然语言, entity=含中间实体")
    args = ap.parse_args()
    global USE_PATH, PATH_STYLE
    USE_PATH = args.path
    PATH_STYLE = args.path_style
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = MODELS[args.backing]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    groups = load_groups(args.dataset, args.split, args.cands)
    if args.max_cands > 0:
        rng = random.Random(args.seed)
        kept = []
        for g in groups:
            pos = [c for c in g["cands"] if int(c["label"]) == 1]
            neg = [c for c in g["cands"] if int(c["label"]) == 0]
            rng.shuffle(neg)
            cands = pos + neg[:max(args.max_cands - len(pos), 0)]
            rng.shuffle(cands)
            kept.append({"question": g["question"], "cands": cands})
        groups = kept
        print(f"truncated to <= {args.max_cands} cands/group", flush=True)
    # 去泄漏：训练组问题不得与 val/test 归一化后重合
    eval_q = set()
    for sp in ("validation", "test"):
        p = os.path.join(DATA, args.dataset, f"{sp}.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    q = json.loads(line).get("question")
                    if q:
                        eval_q.add(norm_q(q))
    n0 = len(groups)
    groups = [g for g in groups if norm_q(g["question"]) not in eval_q]
    print(f"groups={len(groups)} (removed {n0-len(groups)} leaked)", flush=True)

    tr, va = [], []
    for g in groups:
        h = int(hashlib.md5(g["question"].encode()).hexdigest(), 16) % 10
        (va if h == 0 else tr).append(g)
    print(f"train_groups={len(tr)} val_groups={len(va)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" \
        else (lambda: contextlib.nullcontext())
    model = FTVerifier(model_path, args.feat_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_hit, best_lm, best_head = -1.0, None, None

    for ep in range(args.epochs):
        random.shuffle(tr)
        model.train()
        opt.zero_grad()
        for i in range(0, len(tr), args.batch_groups):
            b = tr[i:i + args.batch_groups]
            ids, mask, ys, sizes = make_batch(b, tok, pad_id, args.max_len)
            with autocast():
                logits = model(ids.to(device), mask.to(device))
            scores = logits[:, 1] - logits[:, 0]
            loss = group_loss(scores.float(), ys, sizes) / args.accum
            loss.backward()
            if (i // args.batch_groups + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            if (i // args.batch_groups) % 50 == 0:
                print(f"ep{ep} step{i//args.batch_groups} loss={loss.item()*args.accum:.4f}",
                      flush=True)
        opt.step()
        opt.zero_grad()
        hit, mrr, n = evaluate(model, va, tok, pad_id, args.max_len,
                               device, args.batch_groups, autocast)
        print(f"ep{ep} val_hit1={hit:.4f} mrr={mrr:.4f} n={n}", flush=True)
        if hit > best_hit:
            best_hit = hit
            best_lm = {k: v.detach().cpu().clone() for k, v in model.lm.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}

    out = args.out or os.path.join(DATA, f"verifier_ft_{args.dataset}_{args.backing}_listwise.pt")
    torch.save({"lm_state": best_lm, "head_state": best_head,
                "dataset": args.dataset, "backing": args.backing,
                "val_acc": best_hit, "seed": args.seed, "epochs": args.epochs,
                "lr": args.lr, "mode": "listwise",
                "feat_norm": args.feat_norm}, out)
    print(f"best val_hit1={best_hit:.4f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
