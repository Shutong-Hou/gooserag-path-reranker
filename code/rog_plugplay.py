# -*- coding: utf-8 -*-
"""T4 可插拔验证：RoG 的 relation paths -> 候选实体 -> 用 GooseRAG 的判定器重排。

对比三种排序（同一候选集，公平比较）：
  A. coverage    : 按路径被覆盖的频次（RoG "retrieval only" 近似）
  B. score       : 按 RoG beam 归一化分数累加（RoG 自己的路径置信度）
  C. ours        : 用 path-aware 判定器打分（本题核心：零额外成本的重排层）

所有候选集来自 RoG 生成的路径，因此差异只来自排序函数。
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import deque

import networkx as nx
import torch

ROOT = "/root/autodl-tmp/GooseRAG"
sys.path.insert(0, f"{ROOT}/code/rog_qa")

PATH_RE = r"<PATH>(.*)</PATH>"
INSTRUCTION = "Please generate a valid relation path that can be helpful for answering the following question: "
PROMPT_TMPL = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"
MAX_PATHS_PER_RULE = 200


def build_graph(graph):
    G = nx.Graph()
    for h, r, t in graph:
        G.add_edge(str(h).strip(), str(t).strip(), relation=str(r).strip())
    return G


def bfs_with_rule(graph, start_node, target_rule, max_num=MAX_PATHS_PER_RULE):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue and len(result_paths) < max_num:
        cur, path = queue.popleft()
        if len(path) == len(target_rule):
            result_paths.append(path)
            continue
        if cur not in graph:
            continue
        for nb in graph.neighbors(cur):
            if graph[cur][nb]["relation"] != target_rule[len(path)]:
                continue
            queue.append((nb, path + [(cur, target_rule[len(path)], nb)]))
    return result_paths


def load_rog_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    model.eval()
    return tok, model


@torch.no_grad()
def gen_paths(tok, model, question, n_beam=3, max_new_tokens=100):
    """返回 (rules, scores)：rules 为关系列表，scores 为 beam 归一化置信度。"""
    text = PROMPT_TMPL.format(instruction=INSTRUCTION, input=question)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, num_return_sequences=n_beam, num_beams=n_beam,
        early_stopping=False, do_sample=False,
        max_new_tokens=max_new_tokens, output_scores=True,
        return_dict_in_generate=True,
    )
    decoded = tok.batch_decode(out.sequences[:, ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
    norm = torch.softmax(out.sequences_scores.float(), dim=0).tolist()
    rules, scores = [], []
    for p, s in zip(decoded, norm):
        m = re.search(PATH_RE, p.strip())
        if not m:
            continue
        rels = [r.strip() for r in m.group(1).split("<SEP>")]
        rels = [r for r in rels if r]
        if rels:
            rules.append(rels)
            scores.append(float(s))
    return rules, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n-beam", type=int, default=3)
    ap.add_argument("--data", default="webqsp", choices=("webqsp", "cwq"))
    ap.add_argument("--verifier-ft", default=f"{ROOT}/data/rog_qa/verifier_ft_webqsp_rwkv7_listwise_path_s42.pt")
    ap.add_argument("--out", default=f"{ROOT}/data/rog_qa/rog_plugplay.jsonl")
    args = ap.parse_args()

    import pandas as pd
    files = sorted(glob.glob(f"{ROOT}/RoG_ext/{args.data}/data/test-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"rows={len(df)}", flush=True)

    tok, model = load_rog_model(f"{ROOT}/RoG_ext/model")
    print("model loaded", flush=True)

    import eval_e2e as E
    from transformers import AutoTokenizer
    rwkv_tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"],
                                             trust_remote_code=True,
                                             local_files_only=True)
    if rwkv_tok.pad_token is None:
        rwkv_tok.pad_token = rwkv_tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to("cuda").eval()
    print("verifier loaded", flush=True)

    recs = []
    t0 = time.time()
    for i, row in df.iloc[:args.limit].iterrows():
        q = row["question"]
        gold = set(str(a) for a in row["a_entity"])
        graph = build_graph(row["graph"])
        rules, scores = gen_paths(tok, model, q, args.n_beam)
        # 执行路径 -> 实体；同时累计频次与 RoG 置信度
        cand = {}   # entity -> [count, score_sum, best_rule, best_rule_score]
        for rule, sc in zip(rules, scores):
            for start in row["q_entity"]:
                for path in bfs_with_rule(graph, str(start).strip(), rule):
                    ent = path[-1][2]
                    c = cand.setdefault(ent, [0, 0.0, rule, -1.0])
                    c[0] += 1
                    c[1] += sc
                    if sc > c[3]:
                        c[3] = sc
                        c[2] = rule
        if not cand:
            recs.append({"id": row["id"], "n_cand": 0, "hit_cov": 0, "hit_score": 0,
                         "hit_ours": 0, "gold": sorted(gold)})
            continue
        ents = list(cand.keys())
        cov = [cand[e][0] for e in ents]
        ssum = [cand[e][1] for e in ents]
        rels = [cand[e][2] for e in ents]
        texts = [E.prompt_verifier_path(q, rels[k], ents[k]) for k in range(len(ents))]
        vs = E.score_ft(ft, rwkv_tok, texts, "cuda", 32, 160)
        cov_best = max(range(len(ents)), key=lambda k: (cov[k], ssum[k]))
        sc_best = max(range(len(ents)), key=lambda k: ssum[k])
        ours_best = max(range(len(ents)), key=lambda k: vs[k])
        recs.append({
            "id": row["id"], "n_cand": len(ents),
            "hit_cov": int(ents[cov_best] in gold),
            "hit_score": int(ents[sc_best] in gold),
            "hit_ours": int(ents[ours_best] in gold),
            "gold": sorted(gold), "rules": rules,
            "pred_cov": ents[cov_best], "pred_score": ents[sc_best],
            "pred_ours": ents[ours_best],
        })
        if (i + 1) % 25 == 0:
            n = len(recs)
            hc = sum(r["hit_cov"] for r in recs) / n
            hs = sum(r["hit_score"] for r in recs) / n
            ho = sum(r["hit_ours"] for r in recs) / n
            el = time.time() - t0
            print(f"  {i+1}/{args.limit} cov={hc:.4f} score={hs:.4f} "
                  f"ours={ho:.4f} {el/(i+1):.2f}s/q", flush=True)

    with open(args.out, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(recs)
    res = {"dataset": args.data, "n": n, "n_beam": args.n_beam,
           "coverage_h1": sum(r["hit_cov"] for r in recs) / n,
           "rog_score_h1": sum(r["hit_score"] for r in recs) / n,
           "ours_h1": sum(r["hit_ours"] for r in recs) / n,
           "avg_cand": sum(r["n_cand"] for r in recs) / n,
           "empty_rate": sum(1 for r in recs if r["n_cand"] == 0) / n}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
