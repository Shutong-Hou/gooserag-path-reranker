# -*- coding: utf-8 -*-
"""M1: external reranker baselines on the *same* candidate pool.

Three families:
  (a) monoBERT-style cross-encoder (bert-base-uncased + linear head), trained listwise
      on the same groups as our verifier, optionally with the relation path in the text.
  (b) T5 encoder cross-encoder (t5-base encoder + linear head), same recipe.
  (c) small LLM listwise reranker (zero-shot, chat template), optionally with paths.

All arms score exactly the candidates stored in a `mined_*_vghn.jsonl` file, so no
re-mining and no pool difference is involved.

Examples
--------
# train (path-conditioned) and evaluate
python ext_rerank.py train --model bert-base-uncased --groups path_groups_webqsp_train.jsonl \
    --path --epochs 2 --max-cands 64 --max-len 192 --out bert_path.pt
python ext_rerank.py eval  --model bert-base-uncased --ckpt bert_path.pt \
    --mined mined_webqsp_test_vghn.jsonl --path --max-len 192 --out bert_path_test.json

# small LLM, zero-shot listwise
python ext_rerank.py llm --model Qwen/Qwen2.5-0.5B-Instruct \
    --mined mined_webqsp_test_vghn.jsonl --path --out qwen_path_test.json
"""
import argparse
import json
import os
import re
import time

import torch
import torch.nn as nn


def fmt_path(rels):
    return " -> ".join(rels) if rels else "(none)"


def build_text(q, rels, ent, use_path):
    if use_path:
        return (f"Question: {q}\nRelation path: {fmt_path(rels)}\n"
                f"Candidate: {ent}\nIs this the final answer?")
    return f"Question: {q}\nCandidate: {ent}\nIs this the final answer?"


def enc_kwargs(model_name):
    if "t5" in model_name.lower():
        return {"encoder_only": True}
    return {}


class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if "t5" in model_name.lower():
            from transformers import T5EncoderModel
            self.enc = T5EncoderModel.from_pretrained(model_name)
        else:
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.head = nn.Linear(h, 2)
        # decoder-only backbones must pool the last non-pad token: the first position
        # of a causal LM attends to itself only.
        mt = getattr(self.enc.config, "model_type", "")
        self.decoder = mt in {"qwen2", "llama", "gpt2", "mistral", "gemma", "gemma2", "phi3"}

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        if self.decoder:
            idx = mask.sum(dim=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            pooled = h[:, 0]  # [CLS] / first token
        return self.head(pooled.float())


def load_groups(path, max_cands):
    groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cands = r.get("cands") or []
            if max_cands:
                cands = cands[:max_cands]
            if not cands or not any(c.get("label") for c in cands):
                continue
            groups.append((r["question"], cands))
    return groups


def tokenize(tok, texts, max_len):
    return tok(texts, padding=True, truncation=True, max_length=max_len,
               return_tensors="pt")


def train(args):
    from transformers import AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.pad_token
    model = CrossEncoder(args.model).to(dev)
    if getattr(args, "grad_ckpt", False) and model.decoder:
        model.enc.config.use_cache = False
        model.enc.gradient_checkpointing_enable()
        model.enc.enable_input_require_grads()
        print("gradient checkpointing on", flush=True)
    groups = load_groups(args.groups, args.max_cands)
    print(f"groups={len(groups)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accum = max(1, args.accum)
    bf16 = torch.bfloat16
    for ep in range(args.epochs):
        t0 = time.time()
        tot = 0.0
        opt.zero_grad()
        for gi, (q, cands) in enumerate(groups):
            texts = [build_text(q, c["rels"], c["entity"], args.path) for c in cands]
            labels = torch.tensor([1 if c["label"] else 0 for c in cands], device=dev)
            pos = labels.nonzero().flatten()
            if len(pos) == 0:
                continue
            chunk = getattr(args, "chunk", 0) or len(texts)
            parts = []
            for i in range(0, len(texts), chunk):
                batch = tokenize(tok, texts[i:i + chunk], args.max_len)
                batch = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=bf16):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                parts.append(logits[:, 1] - logits[:, 0])
            scores = torch.cat(parts)
            # listwise softmax over candidates, CE on the gold candidates
            logp = torch.log_softmax(scores, dim=0)
            loss = -logp[pos].mean() / accum
            loss.backward()
            if (gi + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            tot += float(loss) * accum
            if (gi + 1) % 200 == 0:
                print(f"ep{ep} {gi+1}/{len(groups)} loss={tot/(gi+1):.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        print(f"epoch {ep+1} loss={tot/max(len(groups),1):.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": args.model, "state": model.state_dict()}, args.out)
    print("saved", args.out, flush=True)


@torch.no_grad()
def eval_ce(args):
    from transformers import AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.pad_token
    model = CrossEncoder(args.model).to(dev)
    blob = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(blob["state"])
    model.eval()
    n = hit1 = 0
    preds = []
    t0 = time.time()
    with open(args.mined, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cand = r["cand"][: args.max_pool]
            gold = set(r["gold"])
            texts = [build_text(r["question"], c[3], c[0], args.path) for c in cand]
            if not texts:
                n += 1
                preds.append({"id": r.get("id"), "gold": sorted(gold), "pred": None, "hit": 0})
                continue
            batch = tokenize(tok, texts, args.max_len)
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            scores = (logits[:, 1] - logits[:, 0]).float().cpu()
            order = torch.argsort(-scores)
            top = cand[int(order[0])][0]
            ok = int(top in gold)
            hit1 += ok
            n += 1
            preds.append({"id": r.get("id"), "gold": sorted(gold), "pred": top, "hit": ok})
            if n % 200 == 0:
                print(f"  {n} hit1={hit1/n:.4f} {time.time()-t0:.0f}s", flush=True)
    out = {"model": args.model, "ckpt": args.ckpt, "path": bool(args.path),
           "n": n, "hits1": hit1 / max(n, 1)}
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=1)
    with open(args.out + ".preds.jsonl", "w", encoding="utf-8") as w:
        for p in preds:
            w.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False), flush=True)


@torch.no_grad()
def eval_llm(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    n = hit1 = 0
    preds = []
    t0 = time.time()
    with open(args.mined, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cand = r["cand"][: args.max_pool]
            gold = set(r["gold"])
            lines = []
            for i, c in enumerate(cand, 1):
                if args.path:
                    lines.append(f"{i}. {c[0]} (path: {fmt_path(c[3])})")
                else:
                    lines.append(f"{i}. {c[0]}")
            if not lines:
                n += 1
                preds.append({"id": r.get("id"), "gold": sorted(gold), "pred": None, "hit": 0})
                continue
            prompt = (
                "You are a reranker for a knowledge-graph question answering system. "
                "Given the question and the numbered candidate answers"
                + (" with the relation path that reached each candidate" if args.path else "")
                + ", reply with the number of the single most likely answer and nothing else.\n\n"
                f"Question: {r['question']}\n\nCandidates:\n" + "\n".join(lines) +
                "\n\nAnswer (number):"
            )
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_len).to(dev)
            out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+", gen)
            idx = int(m.group(0)) - 1 if m else -1
            pred = cand[idx][0] if 0 <= idx < len(cand) else None
            ok = int(pred in gold) if pred is not None else 0
            hit1 += ok
            n += 1
            preds.append({"id": r.get("id"), "gold": sorted(gold), "pred": pred,
                          "raw": gen.strip()[:20], "hit": ok})
            if n % 100 == 0:
                print(f"  {n} hit1={hit1/n:.4f} {time.time()-t0:.0f}s", flush=True)
    out = {"model": args.model, "path": bool(args.path), "n": n,
           "hits1": hit1 / max(n, 1), "max_pool": args.max_pool}
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=1)
    with open(args.out + ".preds.jsonl", "w", encoding="utf-8") as w:
        for p in preds:
            w.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default="bert-base-uncased")
    t.add_argument("--groups", required=True)
    t.add_argument("--path", action="store_true")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--max-cands", type=int, default=64)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--max-len", type=int, default=192)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--chunk", type=int, default=0)
    t.add_argument("--out", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--model", default="bert-base-uncased")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--mined", required=True)
    e.add_argument("--path", action="store_true")
    e.add_argument("--max-len", type=int, default=160)
    e.add_argument("--max-pool", type=int, default=200)
    e.add_argument("--out", required=True)

    l = sub.add_parser("llm")
    l.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    l.add_argument("--mined", required=True)
    l.add_argument("--path", action="store_true")
    l.add_argument("--max-len", type=int, default=8192)
    l.add_argument("--max-pool", type=int, default=200)
    l.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_ce(args)
    else:
        eval_llm(args)


if __name__ == "__main__":
    main()
