# -*- coding: utf-8 -*-
"""路径感知判定头数据: (问题, 关系路径, 候选) -> 是否最终答案。

正例: RoG 路径终点 == 答案。
负例: 路径终点不是答案 / 路径中间实体 / 同问题其他路径。
输出: E:\\GooseRAG\\data\\rog_qa\\verifier_path_{dataset}_train.jsonl
"""
import argparse
import json
import os
import random
import re

ROG = r"E:\GooseRAG\RoG\datasets\datasets\joint_training\qa"
OUT = r"E:\GooseRAG\data\rog_qa"
BLOCK_RE = re.compile(
    r"Reasoning Paths:\s*\n(.*?)\n\s*Question:\s*\n(.*?)\s*\[/INST\]", re.DOTALL)


def parse_file(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
        except json.JSONDecodeError:
            continue
        m = BLOCK_RE.search(text)
        if not m:
            continue
        path_block, question = m.group(1), m.group(2).strip()
        ans_block = text.split("[/INST]", 1)[-1].replace("</s>", "").strip()
        answers = [a.strip() for a in ans_block.split("\n") if a.strip()]
        paths = []
        for raw in path_block.split("\n"):
            raw = raw.strip()
            if not raw or "->" not in raw:
                continue
            parts = [p.strip() for p in raw.split("->")]
            if len(parts) < 3 or len(parts) % 2 == 0:
                continue
            ents, rels = parts[0::2], parts[1::2]
            if all(ents) and all(rels):
                paths.append((ents, rels))
        if paths and answers:
            out.append((question, paths, answers))
    return out


def make_text(q, rels, e):
    return (f"Question: {q}\nRelation path: {' -> '.join(rels)}\n"
            f"Candidate: {e}\nIs this the final answer?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=6)
    ap.add_argument("--max-neg", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = parse_file(os.path.join(ROG, args.dataset, f"{args.dataset}_train.jsonl"))
    n_pos = n_neg = 0
    with open(os.path.join(OUT, f"verifier_path_{args.dataset}_train.jsonl"), "w",
              encoding="utf-8") as w:
        for q, paths, answers in rows:
            pos, neg = [], []
            for ents, rels in paths:
                if ents[-1] in answers:
                    pos.append((rels, ents[-1]))
                else:
                    neg.append((rels, ents[-1]))
                for e in ents[1:-1]:
                    neg.append((rels, e))
            pos = list({(tuple(r), e) for r, e in pos})
            neg = list({(tuple(r), e) for r, e in neg} - set(pos))
            rng.shuffle(pos)
            rng.shuffle(neg)
            for rels, e in pos[:args.max_pos]:
                w.write(json.dumps({"text": make_text(q, rels, e), "label": 1},
                                   ensure_ascii=False) + "\n")
                n_pos += 1
            for rels, e in neg[:args.max_neg]:
                w.write(json.dumps({"text": make_text(q, rels, e), "label": 0},
                                   ensure_ascii=False) + "\n")
                n_neg += 1
    print(f"questions={len(rows)} pos={n_pos} neg={n_neg} total={n_pos+n_neg}", flush=True)


if __name__ == "__main__":
    main()
