# -*- coding: utf-8 -*-
"""M2: RoG reader end-to-end after retrieval-stage reranking.

Pipeline per question:
  1. RoG-7B generates relation paths (beam = n_beam) -> candidate entities,
     each with one representative entity chain.
  2. Two orderings of the *same* candidate set: RoG's own (coverage / beam score)
     and ours (path-conditioned verifier).
  3. Take top-k chains under each ordering, render them exactly as RoG's reader
     expects ("Reasoning Paths"), and let the same RoG reader answer.
  4. End-to-end H@1 = reader answer contains a gold answer (and list-level match).

Usage (smoke):
python rog_reader_eval.py --limit 30 --n-beam 3 --topk 5 \
  --verifier-ft verifier_ft_webqsp_rwkv7_listwise_path_s42.pt --out rog_reader_smoke.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import deque

import torch

ROOT = os.environ.get("GOOSERAG_ROOT", r"E:\GooseRAG")
D = os.path.join(ROOT, "data", "rog_qa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PATH_RE = r"<PATH>(.*)</PATH>"
GEN_INSTRUCTION = ("Please generate a valid relation path that can be helpful "
                   "for answering the following question: ")
READ_INSTRUCTION = ("Based on the reasoning paths, please answer the given question. "
                    "Please keep the answer as simple as possible and return all the "
                    "possible answers as a list.")
PROMPT_TMPL = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"
MAX_PATHS_PER_RULE = 200


def build_adj(graph):
    adj = {}
    for h, r, t in graph:
        adj.setdefault(str(h).strip(), []).append((str(r).strip(), str(t).strip()))
    return adj


def bfs_with_rule(adj, start, rule, max_num=MAX_PATHS_PER_RULE):
    out = []
    queue = deque([(start, [])])
    while queue and len(out) < max_num:
        cur, path = queue.popleft()
        if len(path) == len(rule):
            out.append(path)
            continue
        for r, t in adj.get(cur, []):
            if r != rule[len(path)]:
                continue
            queue.append((t, path + [(cur, r, t)]))
    return out


def chain_to_string(chain):
    s = ""
    for i, (h, r, t) in enumerate(chain):
        s += f"{h} -> {r} -> {t}" if i == 0 else f" -> {r} -> {t}"
    return s.strip()


def load_rog(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    model.eval()
    return tok, model


@torch.no_grad()
def gen_paths(tok, model, question, n_beam, max_new_tokens=100):
    text = PROMPT_TMPL.format(instruction=GEN_INSTRUCTION, input=question)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, num_return_sequences=n_beam, num_beams=n_beam,
                         early_stopping=False, do_sample=False,
                         max_new_tokens=max_new_tokens, output_scores=True,
                         return_dict_in_generate=True)
    decoded = tok.batch_decode(out.sequences[:, ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
    norm = torch.softmax(out.sequences_scores.float(), dim=0).tolist()
    rules, scores = [], []
    for p, s in zip(decoded, norm):
        m = re.search(PATH_RE, p.strip())
        if not m:
            continue
        rels = [r.strip() for r in m.group(1).split("<SEP>") if r.strip()]
        if rels:
            rules.append(rels)
            scores.append(float(s))
    return rules, scores


@torch.no_grad()
def reader_answer(tok, model, question, chains, max_new_tokens=64):
    q = question if question.endswith("?") else question + "?"
    ctx = "\n".join(chain_to_string(c) for c in chains)
    inp = f"Reasoning Paths:\n{ctx}\n\nQuestion:\n{q}"
    text = PROMPT_TMPL.format(instruction=READ_INSTRUCTION, input=inp)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def match_answer(ans, gold):
    na = norm(ans)
    toks = set(x for x in re.split(r"[,;\n]", na) if x)
    # list-level exact match, then containment of a gold string
    for g in gold:
        ng = norm(g)
        if ng and (ng in na):
            return 1
        if ng and ng in toks:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-beam", type=int, default=3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--data", default="webqsp")
    ap.add_argument("--rog-model", default=os.path.join(ROOT, "models", "RoG"))
    ap.add_argument("--rog-data", default=os.path.join(ROOT, "data", "rog_hf", "webqsp"))
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    files = sorted(glob.glob(os.path.join(args.rog_data, "test-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"rows={len(df)}", flush=True)

    tok, model = load_rog(args.rog_model)
    print("rog loaded", flush=True)

    import eval_e2e as E
    from transformers import AutoTokenizer
    rwkv_tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"],
                                             trust_remote_code=True, local_files_only=True)
    if rwkv_tok.pad_token is None:
        rwkv_tok.pad_token = rwkv_tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to("cuda").eval()
    print("verifier loaded", flush=True)

    recs, t0 = [], time.time()
    for i, row in df.iterrows():
        q = row["question"]
        gold = set(str(a) for a in row["a_entity"])
        adj = build_adj(row["graph"])
        rules, scores = gen_paths(tok, model, q, args.n_beam)
        cand = {}  # entity -> [count, score_sum, chain]
        for rule, sc in zip(rules, scores):
            for start in row["q_entity"]:
                for path in bfs_with_rule(adj, str(start).strip(), rule):
                    ent = path[-1][2]
                    c = cand.setdefault(ent, [0, 0.0, path])
                    c[0] += 1
                    c[1] += sc
                    if len(c[2]) == 0:
                        c[2] = path
        if not cand:
            recs.append({"id": row["id"], "n_cand": 0, "hit_rog": 0, "hit_ours": 0,
                         "gold": sorted(gold), "ans_rog": "", "ans_ours": ""})
            continue
        ents = list(cand.keys())
        rels = [cand[e][2] and [p[1] for p in cand[e][2]] or [] for e in ents]
        texts = [E.prompt_verifier_path(q, rels[k], ents[k]) for k in range(len(ents))]
        vs = E.score_ft(ft, rwkv_tok, texts, "cuda", 32, 160)
        order_rog = sorted(range(len(ents)), key=lambda k: (cand[ents[k]][0], cand[ents[k]][1]), reverse=True)
        order_ours = sorted(range(len(ents)), key=lambda k: vs[k], reverse=True)
        top_rog = [cand[ents[k]][2] for k in order_rog[:args.topk]]
        top_ours = [cand[ents[k]][2] for k in order_ours[:args.topk]]
        ans_rog = reader_answer(tok, model, q, top_rog)
        ans_ours = reader_answer(tok, model, q, top_ours)
        recs.append({
            "id": row["id"], "n_cand": len(ents),
            "hit_rog": match_answer(ans_rog, gold),
            "hit_ours": match_answer(ans_ours, gold),
            "gold": sorted(gold), "ans_rog": ans_rog, "ans_ours": ans_ours,
            "top1_rog": ents[order_rog[0]] if order_rog else None,
            "top1_ours": ents[order_ours[0]] if order_ours else None,
        })
        if (i + 1) % 20 == 0:
            n = len(recs)
            hr = sum(r["hit_rog"] for r in recs) / n
            ho = sum(r["hit_ours"] for r in recs) / n
            print(f"  {i+1}/{len(df)} reader_rog={hr:.4f} reader_ours={ho:.4f} "
                  f"{(time.time()-t0)/(i+1):.1f}s/q", flush=True)

    with open(args.out, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(recs)
    res = {"n": n, "n_beam": args.n_beam, "topk": args.topk,
           "reader_h1_rog": sum(r["hit_rog"] for r in recs) / n,
           "reader_h1_ours": sum(r["hit_ours"] for r in recs) / n,
           "avg_cand": sum(r["n_cand"] for r in recs) / n,
           "empty_rate": sum(1 for r in recs if r["n_cand"] == 0) / n}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    with open(args.out + ".summary.json", "w", encoding="utf-8") as w:
        json.dump(res, w, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
