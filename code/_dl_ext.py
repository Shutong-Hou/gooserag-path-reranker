# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
from huggingface_hub import snapshot_download

root = r"E:\GooseRAG\models"
pats = ["*.json", "*.txt", "*.safetensors", "*.model", "*.bin", "*.py"]
for mid in ["bert-base-uncased", "Qwen/Qwen2.5-0.5B-Instruct", "t5-base"]:
    dst = os.path.join(root, mid.split("/")[-1])
    try:
        p = snapshot_download(mid, local_dir=dst, local_dir_use_symlinks=False,
                              allow_patterns=pats, max_workers=2)
        print("OK", mid, p, flush=True)
    except Exception as e:
        print("FAIL", mid, repr(e)[:300], flush=True)
