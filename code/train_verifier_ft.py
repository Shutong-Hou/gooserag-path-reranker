# -*- coding: utf-8 -*-
"""端到端微调答案判定器: (问题, 候选) -> 是/否。

与冻结特征线性头不同, 这里让 RWKV-7 主干也参与训练, 学习问题-候选交互。
训练数据: verifier_{dataset}_train.jsonl + verifier_{dataset}_hard.jsonl
输出: E:\\GooseRAG\\data\\rog_qa\\verifier_ft_{dataset}_rwkv7.pt
"""
import argparse
import hashlib
import json
import os
import random
import re

import torch
import torch.nn as nn

DATA = os.environ.get("GOOSERAG_ROGQA", r"E:\GooseRAG\data\rog_qa")
MODELS = {
    "rwkv7": os.environ.get(
        "GOOSERAG_RWKV7", r"E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF"),
    "gpt2": os.environ.get("GOOSERAG_GPT2", r"E:\GooseRAG\models\gpt2"),
    "pythia": os.environ.get("GOOSERAG_PYTHIA", r"E:\GooseRAG\models\pythia-160m"),
}


def _final_norm_module(model):
    """主干最后的归一化层；仅 --feat-norm double 时再套一次（复现历史 ckpt 口径）。"""
    for path in ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    last = None
    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            last = m
    return last


class FTVerifier(nn.Module):
    def __init__(self, model_path, feat_norm="double"):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32,
            local_files_only=True)
        self.head = nn.Linear(self.lm.config.hidden_size, 2)
        self.feat_norm = feat_norm

    def forward(self, ids, mask):
        out = self.lm(input_ids=ids, attention_mask=mask,
                      output_hidden_states=True, use_cache=False)
        # GPT-2 / GPT-NeoX / RWKV7 的 hidden_states[-1] 均已是 final norm 之后的表示
        # （与 last_hidden_state 逐元素相同），不能再手动做一次 final norm。
        # 主表 ckpt 历史上按 double 训练，默认 double 以保持一致；single 仅作敏感性。
        h = out.hidden_states[-1]
        if self.feat_norm == "double":
            nrm = _final_norm_module(self.lm)
            w = next(nrm.parameters(), None)
            h = nrm(h.to(w.dtype) if w is not None else h)
        last = mask.sum(dim=1) - 1
        h = h[torch.arange(h.size(0), device=h.device), last]
        return self.head(h.float())


def collate(batch, pad_id, max_len):
    ids = [b[0][:max_len] for b in batch]
    ys = torch.tensor([b[1] for b in batch])
    L = max(len(x) for x in ids)
    out = torch.full((len(ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return out, mask, ys


def load_rows(dataset, path_only=False):
    rows = []
    names = [f"verifier_path_{dataset}_train.jsonl"] if path_only else [
        f"verifier_{dataset}_train.jsonl", f"verifier_{dataset}_hard.jsonl"]
    for name in names:
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                q = r.get("question") or r["text"].split("\n")[0].replace("Question: ", "")
                rows.append((r["text"], int(r["label"]), q))
    return rows


def norm_q(q):
    """大小写/标点归一化，避免近似重复题跨 split 泄漏。"""
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def eval_questions(dataset):
    """val/test 问题集合（归一化），用于剔除训练对里的泄漏题。"""
    qs = set()
    for sp in ("validation", "test"):
        p = os.path.join(DATA, dataset, f"{sp}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line).get("question")
            if q:
                qs.add(norm_q(q))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq", "metaqa"])
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2", "pythia"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--path", action="store_true",
                    help="使用路径感知数据 verifier_path_*")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-norm", choices=["single", "double"], default="double",
                    help="double=复现历史 ckpt 口径（主表）；single=修复后口径（敏感性）")
    ap.add_argument("--out", default="", help="输出路径; 默认 verifier_ft_{dataset}_{backing}.pt")
    args = ap.parse_args()
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

    rows = load_rows(args.dataset, args.path)
    leak_q = eval_questions(args.dataset)
    if leak_q:
        n0 = len(rows)
        rows = [r for r in rows if norm_q(r[2]) not in leak_q]
        print(f"drop leaked eval questions: {n0 - len(rows)} pairs "
              f"({n0} -> {len(rows)})", flush=True)
    tr, va = [], []
    for text, y, q in rows:
        h = int(hashlib.md5(q.encode()).hexdigest(), 16) % 10
        (va if h == 0 else tr).append((tok(text)["input_ids"], y))
    print(f"train={len(tr)} val={len(va)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FTVerifier(model_path, args.feat_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        hit = tot = 0
        with torch.no_grad():
            for j in range(0, len(va), args.batch):
                b = va[j:j + args.batch]
                ids, mask, ys = collate(b, tok.pad_token_id, args.max_len)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids.to(device), mask.to(device))
                hit += (logits.argmax(1).cpu() == ys).sum().item()
                tot += len(ys)
        model.train()
        return hit / max(tot, 1)

    best_acc = -1.0
    best_lm = best_head = None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for j in range(0, len(perm), args.batch):
            b = [tr[i] for i in perm[j:j + args.batch].tolist()]
            ids, mask, ys = collate(b, tok.pad_token_id, args.max_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids.to(device), mask.to(device))
            loss = lossf(logits, ys.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (j // args.batch) % 200 == 0:
                print(f"ep{ep} step{j} loss={loss.item():.4f}", flush=True)
        acc = evaluate()
        print(f"ep{ep} val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_lm = {k: v.detach().cpu().clone() for k, v in model.lm.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}

    tag = "path_" if args.path else ""
    out = args.out or os.path.join(DATA, f"verifier_ft_{tag}{args.dataset}_{args.backing}.pt")
    torch.save({"lm_state": best_lm,
                "head_state": best_head,
                "dataset": args.dataset, "backing": args.backing,
                "val_acc": best_acc, "seed": args.seed,
                "epochs": args.epochs, "lr": args.lr,
                "feat_norm": args.feat_norm}, out)
    print(f"best val_acc={best_acc:.4f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
