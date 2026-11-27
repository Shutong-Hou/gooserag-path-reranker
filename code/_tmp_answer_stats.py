# -*- coding: utf-8 -*-
"""临时: 统计 RoG 答案块的分隔符与长答案样例。"""
import json
from collections import Counter

P = r"E:\GooseRAG\RoG\datasets\datasets\joint_training\qa\webqsp\webqsp_train.jsonl"
seps = Counter()
longs = []
for line in open(P, encoding="utf-8"):
    text = json.loads(line)["text"]
    tail = text.split("[/INST]", 1)[-1].replace("</s>", "").strip()
    for s in (", ", "; ", " | ", "\n", " and "):
        if s in tail:
            seps[s] += 1
    if len(tail) > 80:
        longs.append(tail)
print("separators:", dict(seps))
print("long answers:", len(longs))
for t in longs[:5]:
    print("---", t[:300])
