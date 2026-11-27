# -*- coding: utf-8 -*-
"""生成式证据打分：用冻结 LM 计算 P(entity | question) 的对数似然。

输入: mined_*.jsonl（每题候选 + 判定器分）
输出: {"id","gen":[[entity, sum_lp, mean_lp], ...]}  仅对判定器分最高的 top-m 个候选
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_e2e as E  # noqa: E402


@torch.no_grad()
def score_batch(model, tok, prompts, entities, device, backing):
    texts = [p + " " + e for p, e in zip(prompts, entities)]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=320).to(device)
    n_prompt = [len(tok(p).input_ids) for p in prompts]
    if backing == "rwkv7":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**enc, use_cache=False)
    else:
        out = model(**enc, use_cache=False)
    logits = out.logits.float()
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = enc["input_ids"][:, 1:]
    tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).cpu()
    mask = enc["attention_mask"][:, 1:].cpu()
    res = []
    for i in range(len(texts)):
        start = max(n_prompt[i] - 1, 0)
        seg = tok_lp[i, start:]
        m = mask[i, start:]
        vals = seg[m.bool()]
        if vals.numel() == 0:
            res.append((0.0, 0.0))
        else:
            res.append((float(vals.sum()), float(vals.mean())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.mined, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    model, tok, _ = E.load_model(args.backing)
    model.to(device)
    print(f"rows={len(rows)} top={args.top} device={device}", flush=True)

    out = []
    pairs = []
    for qi, r in enumerate(rows):
        cand = sorted(r["cand"], key=lambda x: -x[2])[:args.top]
        out.append({"id": r["id"], "gen": []})
        for e, _s, _v, _p in cand:
            pairs.append((qi, r["question"], e))

    for start in range(0, len(pairs), args.batch):
        chunk = pairs[start:start + args.batch]
        sc = score_batch(model, tok,
                         [f"Question: {q}\nAnswer:" for _, q, _ in chunk],
                         [e for _, _, e in chunk], device, args.backing)
        for (qi, _q, e), (s, m) in zip(chunk, sc):
            out[qi]["gen"].append([e, s, m])
        if (start // args.batch + 1) % 100 == 0:
            print(f"  {start + len(chunk)}/{len(pairs)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
