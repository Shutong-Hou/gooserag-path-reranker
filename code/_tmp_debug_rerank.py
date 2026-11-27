import json
import sys

sys.path.insert(0, "/root/autodl-tmp/GooseRAG/code/rog_qa")
import rerank_setwise as R

D = "/root/autodl-tmp/GooseRAG/data/rog_qa"
te = R.build_rows([json.loads(l) for l in open(f"{D}/mined_webqsp_test_s42.jsonl", encoding="utf-8")])
print("te", len(te))
print("fused", R.evaluate_key(te, "fused")[0])
print("v", R.evaluate_key(te, "v")[0])

# 手工核对第一题
r = te[0]
print("id", r["id"], "gold", list(r["gold"])[:3])
print("fused head", r["fused"][:5].tolist())
print("ents head", r["ents"][:5])
order = r["fused"].argsort(descending=True).tolist()
print("order head", order[:5], "top1 in gold", r["ents"][order[0]] in r["gold"])
