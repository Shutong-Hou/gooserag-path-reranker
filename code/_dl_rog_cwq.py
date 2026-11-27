# -*- coding: utf-8 -*-
"""下载 RoG-CWQ 7B 到 $ROOT/RoG_ext/model（走 hf-mirror）；可断点续传。"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = "/root/autodl-tmp/GooseRAG"
LOCAL = f"{ROOT}/RoG_ext/model"
os.makedirs(LOCAL, exist_ok=True)

from huggingface_hub import snapshot_download

print("start", flush=True)
p = snapshot_download(
    repo_id="rmanluo/RoG-cwq",
    local_dir=LOCAL,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    ignore_patterns=["*.gguf", "*.bin.index.json"],
)
print("done ->", p, flush=True)
