# -*- coding: utf-8 -*-
"""构建答案判定头训练对: (问题, 候选实体) -> 是否最终答案。

正例: RoG 路径的最后一个实体 (答案)。
负例: 同一问题路径中的中间实体 + 其他问题的随机实体。
输入:
  E:\\GooseRAG\\RoG\\datasets\\datasets\\joint_training\\qa\\{dataset}\\{dataset}_train.jsonl
输出:
  E:\\GooseRAG\\data\\rog_qa\\verifier_{dataset}_train.jsonl
    {text, label, question, entity}
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
    with open(path, encoding="utf-8") as f:
        for line in f:
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
            # [/INST] 之后是答案块: 换行分隔的多答案, 不能用逗号切(标题内含逗号)
            ans_block = text.split("[/INST]", 1)[-1].replace("</s>", "").strip()
            answers = [a.strip() for a in ans_block.split("\n") if a.strip()]
            if not answers:
                continue
            paths = []
            for raw in path_block.split("\n"):
                raw = raw.strip()
                if not raw or "->" not in raw:
                    continue
                parts = [p.strip() for p in raw.split("->")]
                if len(parts) < 3 or len(parts) % 2 == 0:
                    continue
                ents = parts[0::2]
                if all(ents):
                    paths.append(ents)
            if paths:
                out.append((question, paths, answers))
    return out


def make_text(q, e):
    return f"Question: {q}\nCandidate: {e}\nIs this the final answer?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs-per-q", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = parse_file(os.path.join(ROG, args.dataset, f"{args.dataset}_train.jsonl"))
    print(f"questions with paths: {len(rows)}", flush=True)
    pool = []
    parsed = []
    for q, paths, answers in rows:
        pos = {a for a in answers if a}
        neg = {e for p in paths for e in p} - pos
        if not pos:
            continue
        parsed.append((q, sorted(pos), sorted(neg)))
        pool.extend(pos)
        pool.extend(neg)
    pool = list(set(pool))
    print(f"entity pool: {len(pool)}", flush=True)

    n_pos = n_neg = 0
    with open(os.path.join(OUT, f"verifier_{args.dataset}_train.jsonl"), "w",
              encoding="utf-8") as w:
        for q, pos, neg in parsed:
            pos_pairs = [(e, 1) for e in pos]
            neg_pairs = [(e, 0) for e in neg]
            rng.shuffle(neg_pairs)
            neg_pairs = neg_pairs[:max(0, args.max_pairs_per_q - len(pos_pairs))]
            pairs = pos_pairs + neg_pairs
            # 补 2 个跨问题随机负例, 让判定头见到更宽的实体分布
            for _ in range(2):
                e = rng.choice(pool)
                if e not in pos:
                    pairs.append((e, 0))
            for e, y in pairs:
                w.write(json.dumps({"text": make_text(q, e), "label": y,
                                    "question": q, "entity": e},
                                   ensure_ascii=False) + "\n")
                n_pos += y
                n_neg += (1 - y)
    print(f"pairs: pos={n_pos} neg={n_neg} total={n_pos + n_neg}", flush=True)


if __name__ == "__main__":
    main()
