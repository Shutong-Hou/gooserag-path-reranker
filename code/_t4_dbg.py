# -*- coding: utf-8 -*-
"""T4 调试：检查 RoG 生成的原始文本与路径解析。"""
import json
import re
import sys

import pandas as pd
import torch

ROOT = "/root/autodl-tmp/GooseRAG"
sys.path.insert(0, f"{ROOT}/code/rog_qa")

import rog_plugplay as P

files = sorted(__import__("glob").glob(f"{ROOT}/RoG_ext/webqsp/data/test-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print("rows", len(df), flush=True)
print("cols", list(df.columns), flush=True)
row = df.iloc[0]
print("q_entity", row["q_entity"], "a_entity", row["a_entity"], flush=True)
print("graph head", row["graph"][:3], type(row["graph"]), flush=True)

tok, model = P.load_rog_model(f"{ROOT}/RoG_ext/model")
for i in range(3):
    q = df.iloc[i]["question"]
    text = P.PROMPT_TMPL.format(instruction=P.INSTRUCTION, input=q)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, num_beams=3, num_return_sequences=3,
                         early_stopping=False, do_sample=False,
                         max_new_tokens=100, return_dict_in_generate=True,
                         output_scores=True)
    dec = tok.batch_decode(out.sequences[:, ids["input_ids"].shape[1]:],
                           skip_special_tokens=True)
    print("=== Q", i, q, flush=True)
    for d in dec:
        print("  RAW:", repr(d), flush=True)
    print("  RULES:", P.gen_paths(tok, model, q, 3), flush=True)
