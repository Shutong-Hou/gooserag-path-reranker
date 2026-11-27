import json

base = "/root/autodl-tmp/GooseRAG/data/rog_qa"
for s in ["test", "validation", "train"]:
    p = f"{base}/budget_webqsp_{s}_s42.jsonl.summary.json"
    try:
        d = json.load(open(p))
    except Exception as e:
        print(s, "ERR", e)
        continue
    print(s, "n=", d["n"],
          "h1=", {k: round(v, 4) for k, v in d["hits1_at"].items()},
          "top5@200=", round(d["top5_at"]["200"], 4),
          "oracle=", round(d["oracle_hits1"], 4),
          "oracle_cost=", round(d["oracle_avg_cost"], 1))

for s in ["validation", "test"]:
    p = f"{base}/budget_webqsp_{s}_vghn.jsonl.summary.json"
    try:
        d = json.load(open(p))
        print("vghn", s, "n=", d["n"],
              "h1=", {k: round(v, 4) for k, v in d["hits1_at"].items()})
    except Exception as e:
        print("vghn", s, "not ready")
