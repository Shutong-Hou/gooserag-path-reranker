# -*- coding: utf-8 -*-
"""下载 RoG 模型与 test 数据。"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

what = sys.argv[1] if len(sys.argv) > 1 else "data"
ROOT = "/root/autodl-tmp/GooseRAG/RoG_ext"

if what == "data":
    p = snapshot_download("rmanluo/RoG-webqsp", repo_type="dataset",
                          local_dir=f"{ROOT}/webqsp",
                          allow_patterns=["data/test-*", "README.md"])
    print("data ->", p)
else:
    p = snapshot_download("rmanluo/RoG", local_dir=f"{ROOT}/model",
                          allow_patterns=["*.json", "*.safetensors", "*.model",
                                          "*.txt", "tokenizer*"])
    print("model ->", p)
