# -*- coding: utf-8 -*-
"""M0: second-host evaluation. Same candidate set, only the ranking function changes.

Host      : GNN-RAG's released ReaRev-webqsp candidates (MID + GNN score)
Ours      : our path-conditioned verifier, applied without retraining
Matching  : exact entity id (Freebase MID), the native space of GNN-RAG's release

Outputs per-question predictions and an exact McNemar test between the two rankings.
"""
import argparse
import json
import os
import time

import torch

ROOT = r"E:\GooseRAG"
import sys
sys.path.insert(0, os.path.join(ROOT, "code", "rog_qa_1127"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "rog_qa", "gnnrag_webqsp_test.jsonl"))
    ap.add_argument("--verifier-ft", default=os.path.join(ROOT, "data", "rog_qa", "verifier_ft_webqsp_rwkv7_listwise_path_s42.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rog_qa", "m0_gnnrag_host_eval.jsonl"))
    args = ap.parse_args()

    import eval_e2e as E
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(fb["lm_state"])
    ft.head.load_state_dict(fb["head_state"])
    ft.to("cuda").eval()
    print("verifier loaded", flush=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    recs, t0 = [], time.time()
    for i, r in enumerate(rows):
        gold = set(r["gold"])
        cand = r["cand"]
        texts = [E.prompt_verifier_path(r["question"], c[3], c[1]) for c in cand]
        vs = E.score_ft(ft, tok, texts, "cuda", 32, 160)
        k_ours = max(range(len(cand)), key=lambda k: vs[k])
        k_gnn = max(range(len(cand)), key=lambda k: cand[k][2])
        k_len = max(range(len(cand)), key=lambda k: (-len(cand[k][3]), cand[k][2]))
        rec = {
            "id": r["id"],
            "n_cand": len(cand),
            "gold_in_pool": int(bool(gold & {c[0] for c in cand})),
            "hit_gnn": int(cand[k_gnn][0] in gold),
            "hit_ours": int(cand[k_ours][0] in gold),
            "hit_len": int(cand[k_len][0] in gold),
            "pred_gnn": cand[k_gnn][0], "pred_ours": cand[k_ours][0],
            "gnn_scores": [c[2] for c in cand],
            "our_scores": [float(v) for v in vs],
            "mids": [c[0] for c in cand],
        }
        recs.append(rec)
        if (i + 1) % 200 == 0:
            n = len(recs)
            print(f"  {i+1}/{len(rows)} gnn={sum(x['hit_gnn'] for x in recs)/n:.4f} "
                  f"ours={sum(x['hit_ours'] for x in recs)/n:.4f} {(time.time()-t0)/(i+1):.2f}s/q", flush=True)

    with open(args.out, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    hg = sum(r["hit_gnn"] for r in recs) / n
    ho = sum(r["hit_ours"] for r in recs) / n
    hl = sum(r["hit_len"] for r in recs) / n
    b = sum(1 for r in recs if not r["hit_gnn"] and r["hit_ours"])
    c = sum(1 for r in recs if r["hit_gnn"] and not r["hit_ours"])
    from math import comb
    m = b + c
    p = 2 * sum(comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m) if m else 1.0
    res = {"n": n, "avg_cand": sum(r["n_cand"] for r in recs) / n,
           "gold_in_pool_rate": sum(r["gold_in_pool"] for r in recs) / n,
           "h1_gnn_score": hg, "h1_ours": ho, "h1_length_first": hl,
           "delta": ho - hg, "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p}
    print(json.dumps(res, indent=1), flush=True)
    with open(args.out + ".summary.json", "w", encoding="utf-8") as w:
        json.dump(res, w, indent=1)


if __name__ == "__main__":
    main()
