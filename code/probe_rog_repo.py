# -*- coding: utf-8 -*-
"""探查 rmanluo/RoG-webqsp 数据集内容，判断是否已有现成的 relation paths。"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
files = api.list_repo_files("rmanluo/RoG-webqsp", repo_type="dataset")
print(f"total files: {len(files)}")
for f in files:
    print(" ", f)

info = api.dataset_info("rmanluo/RoG-webqsp")
print("\nsiblings with size:")
for s in (info.siblings or [])[:80]:
    print(f"  {s.rfilename}  {getattr(s, 'size', '?')}")
