# -*- coding: utf-8 -*-
"""把联合训练的 checkpoint 导出成 EvalFTVerifier 可加载的格式（lm_state/head_state）。"""
import sys

import torch

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
src = sys.argv[1] if len(sys.argv) > 1 else f"{D}/joint_full_s42.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else f"{D}/verifier_ft_webqsp_joint_s42.pt"
ck = torch.load(src, map_location="cpu", weights_only=False)
torch.save({"lm_state": ck["lm"], "head_state": ck["s1"]}, dst)
print("saved", dst, "meta", ck.get("meta"))
