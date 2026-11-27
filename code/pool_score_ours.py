# -*- coding: utf-8 -*-
"""Score our path-conditioned verifier on a stored mined pool (no re-mining),
so that external rerankers and ours can be paired question by question.

Usage:
  python pool_score_ours.py --mined mined_webqsp_test_vghn.jsonl \
      --verifier-ft verifier_ft_webqsp_rwkv7_listwise_path_s42.pt --out ours_pool.jsonl
"""
import argparse
import json

import torch

import eval_e2e as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--max-pool", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.MODEL_NAMES["rwkv7"], trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ft = E.EvalFTVerifier(E.MODEL_NAMES["rwkv7"], feat_norm="double")
    blob = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
    ft.lm.load_state_dict(blob["lm_state"])
    ft.head.load_state_dict(blob["head_state"])
    ft.to("cuda").eval()

    n = hit = 0
    with open(args.out, "w", encoding="utf-8") as w, open(args.mined, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cand = (r.get("cand") or [])[: args.max_pool]
            gold = set(r.get("gold") or [])
            texts = [E.prompt_verifier_path(r["question"], c[3], c[0]) for c in cand]
            if texts:
                vs = E.score_ft(ft, tok, texts, "cuda", 32, 160)
                best = cand[int(max(range(len(cand)), key=lambda k: vs[k]))][0]
            else:
                best = None
            ok = int(best in gold) if best is not None else 0
            n += 1
            hit += ok
            w.write(json.dumps({"id": r.get("id"), "gold": sorted(gold),
                                "pred": [best] if best else [], "hit": ok},
                               ensure_ascii=False) + "\n")
            if n % 200 == 0:
                print(f"  {n} hit1={hit/n:.4f}", flush=True)
    print(json.dumps({"n": n, "hits1": hit / max(n, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
