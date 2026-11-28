# -*- coding: utf-8 -*-
"""P2: re-score one stored CWQ candidate pool with several path renderings.

The pool file is the `--dump` of eval_e2e.py, which now carries, for every question,
the top-`verifier_top` candidates with the relation path that reached them and the
beam score.  Ranking replicates eval_e2e exactly: fused = gamma * beam + beta * verifier,
sorted descending, H@1 = top entity in the gold set.

Every arm uses the *same* pool, so differences are attributable to the verifier only.
"""
import argparse
import json
import os
import sys
from math import comb

import torch

ROOT = r"E:\GooseRAG"
sys.path.insert(0, os.path.join(ROOT, "code", "rog_qa_1127"))


def render(rels, mode):
    rels = [r for r in rels if r]
    if mode == "full":
        return rels
    if mode == "hop1":
        return rels[:1]
    if mode == "type":
        return sorted({r.split(".")[0] for r in rels})
    return []


def mcnemar(b, c):
    m = b + c
    if not m:
        return 1.0
    return 2 * sum(comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "rog_qa", "_cwq_test_pool.json"))
    ap.add_argument("--arms", default="plain,full,hop1,type")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rog_qa", "p2_cwq_truncation.json"))
    args = ap.parse_args()

    import eval_e2e as E
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"],
                                        trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = json.load(open(args.pool, encoding="utf-8"))
    print("questions", len(rows), flush=True)

    results = {}
    hits = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        ckpt = os.path.join(ROOT, "data", "rog_qa", f"verifier_ft_cwq_p2_{arm}.pt")
        if not os.path.exists(ckpt):
            print("skip (missing)", arm, flush=True)
            continue
        ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        ft.lm.load_state_dict(blob["lm_state"])
        ft.head.load_state_dict(blob["head_state"])
        ft.to("cuda").eval()

        hit1 = 0
        arm_hits = []
        for i, r in enumerate(rows):
            gold = set(r["gold"])
            cand = r["cand"]
            if not cand:
                arm_hits.append(0)
                continue
            if arm == "plain":
                texts = [E.prompt_verifier(r["question"], c["e"]) for c in cand]
            else:
                texts = [E.prompt_verifier_path(r["question"], render(c["rels"], arm), c["e"])
                         for c in cand]
            vs = E.score_ft(ft, tok, texts, "cuda", 64, args.max_len)
            fused = [(c["e"], args.gamma * c["s"] + args.beta * float(v))
                     for c, v in zip(cand, vs)]
            fused.sort(key=lambda x: -x[1])
            ok = int(fused[0][0] in gold)
            hit1 += ok
            arm_hits.append(ok)
            if (i + 1) % 500 == 0:
                print(f"  {arm} {i+1}/{len(rows)} h1={hit1/(i+1):.4f}", flush=True)
        n = len(rows)
        results[arm] = {"n": n, "hits1": hit1 / n,
                        "val_acc": blob.get("val_acc"), "epochs": blob.get("epochs")}
        hits[arm] = arm_hits
        print(arm, results[arm], flush=True)
        del ft
        torch.cuda.empty_cache()

    # paired comparisons against plain and against full
    for a in ("plain", "full"):
        if a not in hits:
            continue
        for b in hits:
            if b == a:
                continue
            bc = sum(1 for x, y in zip(hits[a], hits[b]) if x and not y)
            cb = sum(1 for x, y in zip(hits[a], hits[b]) if not x and y)
            results.setdefault("paired", {})[f"{b}_vs_{a}"] = {
                "delta_pt": round((sum(hits[b]) - sum(hits[a])) / len(hits[a]) * 100, 2),
                "b_onlyA": bc, "c_onlyB": cb, "mcnemar_p": mcnemar(bc, cb)}

    print(json.dumps(results, indent=1, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(results, w, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
