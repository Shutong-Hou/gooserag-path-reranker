# -*- coding: utf-8 -*-
"""底座 #2：纯词面关系检索器（不训练、不用 LLM）。

思路：把问题与关系名做词面匹配，取 top-k 关系构造 1/2 跳路径，在图上前缀匹配执行，
得到候选实体；再对比「词面分排序」vs「我们的 path 判定器重排」。
这是最弱的底座，用来检验重排层是否只对强检索器有效。
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = "/root/autodl-tmp/GooseRAG"
sys.path.insert(0, f"{ROOT}/code/rog_qa")
import eval_e2e as E  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(s.lower()))


def rel_tokens(rel):
    return toks(rel.replace(".", " ").replace("_", " "))


def walk(g, node, rule, depth=0, cap=50):
    """按关系序列在图上展开，返回路径（有上限，防爆炸）。"""
    if depth == len(rule):
        yield []
        return
    n = 0
    for nxt in g.get(node, {}).get(rule[depth], []):
        for rest in walk(g, nxt, rule, depth + 1, cap):
            yield [(node, rule[depth], nxt)] + rest
            n += 1
            if n >= cap:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-paths", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verifier-ft",
                    default=f"{ROOT}/data/rog_qa/verifier_ft_webqsp_joint_s42.pt")
    ap.add_argument("--out", default=f"{ROOT}/data/rog_qa/lexbase_webqsp.json")
    args = ap.parse_args()

    rels = json.load(open(f"{ROOT}/data/g1_webqsp_relations.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(f"{ROOT}/data/rog_qa/{args.dataset}/{args.split}.jsonl",
                                        encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    tok = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to("cuda").eval()

    rel_texts = [f"Relation: {r}" for r in rels]
    rv = E.encode_texts(ft.lm, tok, rel_texts, "cuda", "rwkv7", batch_size=64,
                        max_len=64, feat_norm="double").float()
    rel_mat = (rv / rv.norm(dim=1, keepdim=True)).cuda()

    recs = []
    for i, r in enumerate(rows):
        gold = set(r["a_entity"]) or set(r["answers"])
        starts = r["q_entity"]
        if not gold or not starts:
            continue
        g = E.build_graph(r["edges"])
        qv = E.encode_texts(ft.lm, tok, [f"Question: {r['question']}"], "cuda", "rwkv7",
                            batch_size=1, max_len=64, feat_norm="double")[0].float().cuda()
        qv = qv / qv.norm()
        sims = (rel_mat @ qv).tolist()
        order = sorted(range(len(rels)), key=lambda j: -sims[j])
        top = [(sims[j], rels[j]) for j in order[:args.topk]]
        rules = [([rel], sc) for sc, rel in top]
        half = top[:max(args.topk // 2, 1)]
        for sa, ra in half:
            for sb, rb in half:
                if ra != rb:
                    rules.append(([ra, rb], (sa + sb) / 2))

        cand = {}
        for rule, w in rules:
            for s in starts:
                for path in walk(g, s, rule):
                    e = path[-1][2]
                    cur = cand.get(e)
                    if cur is None or w > cur[0]:
                        cand[e] = [w, rule]
        if not cand:
            recs.append({"id": r["id"], "n": 0, "lex_hit": 0, "ours_hit": 0})
            continue
        ents = sorted(cand.keys(), key=lambda e: -cand[e][0])[:args.max_paths]
        lex = [cand[e][0] for e in ents]
        rels_ = [cand[e][1] for e in ents]
        texts = [E.prompt_verifier_path(r["question"], rels_[k], ents[k])
                 for k in range(len(ents))]
        vs = E.score_ft(ft, tok, texts, "cuda", 32, 160)
        kl = max(range(len(ents)), key=lambda k: lex[k])
        ko = max(range(len(ents)), key=lambda k: vs[k])
        recs.append({"id": r["id"], "n": len(ents),
                     "lex_hit": int(ents[kl] in gold),
                     "ours_hit": int(ents[ko] in gold)})
        if (i + 1) % 200 == 0:
            n = len(recs)
            print(f"  {i+1}/{len(rows)} lex={sum(x['lex_hit'] for x in recs)/n:.4f} "
                  f"ours={sum(x['ours_hit'] for x in recs)/n:.4f}", flush=True)

    n = len(recs)
    res = {"n": n, "lex_h1": sum(x["lex_hit"] for x in recs) / n,
           "ours_h1": sum(x["ours_hit"] for x in recs) / n,
           "avg_cand": sum(x["n"] for x in recs) / n, "recs": recs}
    print(json.dumps({k: v for k, v in res.items() if k != "recs"}, indent=1), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
