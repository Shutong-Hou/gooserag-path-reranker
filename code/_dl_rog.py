# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
from huggingface_hub import snapshot_download

p = snapshot_download("rmanluo/RoG", local_dir=r"E:\GooseRAG\models\RoG",
                      local_dir_use_symlinks=False, max_workers=2)
print("DONE", p, flush=True)
