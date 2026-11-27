# -*- coding: utf-8 -*-
"""临时: 打印 RoG qa 行 [/INST] 之后的答案块。"""
import json

P = r"E:\GooseRAG\RoG\datasets\datasets\joint_training\qa\webqsp\webqsp_train.jsonl"
for i, line in enumerate(open(P, encoding="utf-8")):
    text = json.loads(line)["text"]
    tail = text.split("[/INST]", 1)[-1]
    print("=== line", i, "===")
    print(tail[:500])
    if i >= 2:
        break
