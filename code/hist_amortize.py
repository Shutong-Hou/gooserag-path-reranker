# -*- coding: utf-8 -*-
"""长会话摊销 pilot（逐候选调用，部署真实形态）。

每条候选一次前向（batch=1）：
  A none : prompt = 当前问题 + 路径 + 候选
  B cat  : prompt = 历史 + 当前问题 + 路径 + 候选（Transformer 做法：每条候选都重算历史）
  C state: 历史编码一次 -> RWKV 状态；每条候选从该状态续算（历史只付一次）

指标：H@1、每题延迟、峰值显存，随历史轮数 H 的变化。
"""
import argparse
import json
import random
import time

import torch

import eval_e2e as E
from transformers import AutoTokenizer

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"


def clone_state(state):
    """FLA Cache 原地更新，每个候选必须用独立副本；状态很小（约 9MB）。"""
    from fla.models.utils import Cache
    new = Cache(seen_tokens=getattr(state, "_seen_tokens", 0))
    out = []
    for d in state.states:
        nd = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                nd[k] = v.detach().clone()
            elif isinstance(v, tuple):
                nd[k] = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in v)
            else:
                nd[k] = v
        out.append(nd)
    new.states = out
    return new


def last_hidden(ft, out, mask):
    h = out.hidden_states[-1]
    if ft.feat_norm == "double":
        nrm = E._final_norm_module(ft.lm)
        w = next(nrm.parameters(), None)
        h = nrm(h.to(w.dtype) if w is not None else h)
    idx = (mask.sum(dim=1) - 1).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, idx].float()


def score1(ft, tok, text, max_len, state=None):
    enc = tok([text], return_tensors="pt", truncation=True,
              max_length=max_len).to("cuda")
    kw = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
              output_hidden_states=True, use_cache=state is not None)
    if state is not None:
        kw["past_key_values"] = state
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = ft.lm(**kw)
    hl = last_hidden(ft, out, enc["attention_mask"])
    lg = ft.head(hl)
    return float(lg[0, 1] - lg[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--hs", default="0,8,16,32")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--ckpt", default=f"{D}/verifier_ft_webqsp_joint_s42.pt")
    ap.add_argument("--out", default=f"{D}/hist_amortize.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{D}/mined_webqsp_test_vghn.jsonl",
                                        encoding="utf-8")]
    rows = [r for r in rows if r.get("cand")][:args.limit]
    raw = {}
    for l in open(f"{D}/webqsp/test.jsonl", encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            raw[x["id"]] = x
    for r in rows:
        r["q_entity"] = (raw.get(r["id"], {}) or {}).get("q_entity") or []

    by_ent, all_turns = {}, []
    for l in open(f"{D}/webqsp/train.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        a = (r.get("a_entity") or r.get("answers") or [""])[0]
        all_turns.append((r["question"], a))
        e = (r.get("q_entity") or [None])[0]
        if e:
            by_ent.setdefault(e, []).append((r["question"], a))
    rng = random.Random(0)
    print(f"questions={len(rows)} history_pool={len(by_ent)}", flush=True)

    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    fb = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to("cuda").eval()

    res = []
    for H in [int(x) for x in args.hs.split(",")]:
        hists = []
        for r in rows:
            e = (r.get("q_entity") or [None])[0]
            turns = list(by_ent.get(e, []))[:H]
            while len(turns) < H:          # 补齐到恰好 H 轮（会话长度可控）
                turns.append(rng.choice(all_turns))
            hists.append("".join(f"User: {q}\nAssistant: {a}\n" for q, a in turns))

        per_mode = {}
        for mode in ("none", "cat", "state"):
            with torch.no_grad():
                for _ in range(3):
                    score1(ft, tok, "Question: warmup\nCandidate: X\nIs this the final answer?",
                           args.max_len)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            hit = n = 0
            with torch.no_grad():
                for i, r in enumerate(rows):
                    ents = [c[0] for c in r["cand"][:args.topk]]
                    rels = [c[3] for c in r["cand"][:args.topk]]
                    gold = set(r["gold"])
                    q = r["question"]
                    texts = [E.prompt_verifier_path(q, rels[k], ents[k])
                             for k in range(len(ents))]
                    state = None
                    if mode == "cat" and H:
                        texts = [hists[i] + t for t in texts]
                    elif mode == "state" and H:
                        enc = tok([hists[i]], return_tensors="pt", truncation=True,
                                  max_length=args.max_len).to("cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            o = ft.lm(input_ids=enc["input_ids"],
                                      attention_mask=enc["attention_mask"],
                                      use_cache=True)
                        state = o.past_key_values
                    if state is not None:
                        scs = [score1(ft, tok, t, args.max_len, clone_state(state))
                               for t in texts]
                    else:
                        scs = [score1(ft, tok, t, args.max_len) for t in texts]
                    n += 1
                    hit += int(ents[max(range(len(scs)), key=lambda k: scs[k])] in gold)
            torch.cuda.synchronize()
            dt = time.time() - t0
            per_mode[mode] = {"h1": hit / max(n, 1), "sec": dt,
                              "ms_q": dt / max(n, 1) * 1000,
                              "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"  H={H:2d} {mode:5s} h1={hit/max(n,1):.4f} "
                  f"{dt/max(n,1)*1000:7.1f} ms/q  peak={per_mode[mode]['peak_gb']:.2f}GB",
                  flush=True)
        res.append({"H": H, "modes": per_mode})

    json.dump(res, open(args.out, "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
