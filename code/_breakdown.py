# -*- coding: utf-8 -*-
"""P0-4 端到端延迟 / 显存分解：beam 搜索 vs 判定器打分。

复用 eval_e2e 的加载与函数，逐题计时，输出均值（ms/题）与峰值显存。
"""
import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
import eval_e2e as E

ROG_QA = os.environ.get("GOOSERAG_ROGQA", "/root/autodl-tmp/GooseRAG/data/rog_qa")
DEVICE = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--topk-rel", type=int, default=15)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--verifier-top", type=int, default=200)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--verifier-ft", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rel_head, rels = E.load_heads(a.heads, DEVICE)
    model, tok, _ = E.load_model(a.backing)
    model.to(DEVICE)
    ft_model = E.EvalFTVerifier(E.MODEL_NAMES[a.backing], feat_norm="double")
    fb = torch.load(a.verifier_ft, map_location="cpu", weights_only=False)
    ft_model.lm.load_state_dict(fb["lm_state"])
    ft_model.head.load_state_dict(fb["head_state"])
    ft_model.to(DEVICE).eval()
    rel2id = {r: i for i, r in enumerate(rels)}

    args = SimpleNamespace(
        max_hops=a.max_hops, policy="model", cond=False, backing=a.backing,
        batch=a.batch, head_feat_norm="double", ban_generic=True,
        generic_mode="all", topk_rel=a.topk_rel, max_children=a.max_children,
        beam=a.beam, score_norm="mean", length_mode="all", search="entity",
    )

    src = os.path.join(ROG_QA, a.dataset, f"{a.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")][:a.limit]

    t_beam, t_ver, n_cand = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for i, r in enumerate(rows):
        out = E.build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        if not starts:
            continue
        torch.cuda.synchronize()
        t0 = time.time()
        ranked, beams, path_of = E.run_beam(args, r["question"], starts, out, rel2id,
                                            model, tok, rel_head, None, DEVICE)
        torch.cuda.synchronize()
        t1 = time.time()
        cand = ranked[:a.verifier_top]
        texts = [E.prompt_verifier_path(r["question"], path_of.get(e, ()), e)
                 for e, _ in cand]
        if texts:
            E.score_ft(ft_model, tok, texts, DEVICE, a.batch)
        torch.cuda.synchronize()
        t2 = time.time()
        t_beam.append((t1 - t0) * 1000)
        t_ver.append((t2 - t1) * 1000)
        n_cand.append(len(cand))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} beam={statistics.mean(t_beam):.0f}ms "
                  f"ver={statistics.mean(t_ver):.0f}ms", flush=True)

    res = {
        "dataset": a.dataset, "split": a.split, "n": len(t_beam),
        "beam_ms_mean": statistics.mean(t_beam),
        "verify_ms_mean": statistics.mean(t_ver),
        "total_ms_mean": statistics.mean(t_beam) + statistics.mean(t_ver),
        "cand_mean": statistics.mean(n_cand),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "note": "walker 与 verifier 分别实例化；共享权重部署下只保留一份 LM",
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
