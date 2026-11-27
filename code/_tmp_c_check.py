import json

base = "/root/autodl-tmp/GooseRAG/data/rog_qa"
held_rows = [json.loads(l) for l in open(f"{base}/heldout/test_held.jsonl", encoding="utf-8")]
print("test_held rows", len(held_rows),
      "held rows", sum(1 for r in held_rows if r["held"]),
      "unique q", len({r["question"] for r in held_rows}),
      "unique held q", len({r["question"] for r in held_rows if r["held"]}))
dump = [json.loads(l) for l in open(f"{base}/budget_webqsp_test_s42.jsonl", encoding="utf-8")]
dq = {r["question"] for r in dump}
hq = {r["question"] for r in held_rows if r["held"]}
print("dump q", len(dq), "held q", len(hq), "intersect", len(dq & hq))
print("sample held q:", list(hq)[:3])
print("sample dump q:", list(dq)[:3])
rels = json.load(open(f"{base}/heldout/held_rels.json", encoding="utf-8"))
print("n held rels", len(rels), "sample", rels[:3])
