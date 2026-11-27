import torch

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
for name in ("verifier_ft_webqsp_rwkv7_listwise_path_s42.pt",
             "verifier_ft_webqsp_p11_rel_s42.pt",
             "verifier_ft_metaqa_path_s42.pt"):
    a = torch.load(f"{D}/{name}", map_location="cpu", weights_only=False)
    meta = {k: v for k, v in a.items() if k not in ("lm_state", "head_state")}
    print(name, meta)
