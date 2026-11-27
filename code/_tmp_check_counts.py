import json

base = "/root/autodl-tmp/GooseRAG/data/rog_qa"
for name in ["budget_webqsp_test_s42.jsonl", "budget_webqsp_test_vghn.jsonl"]:
    rows = [json.loads(l) for l in open(f"{base}/{name}", encoding="utf-8")]
    n = len(rows)
    for k in ["100", "200"]:
        h1 = sum(1 for r in rows if r["gold_rank"].get(k) == 0)
        inc = sum(1 for r in rows if r["gold_rank"].get(k) is not None)
        print(name, "k=", k, "n=", n, "h1_cnt=", h1, "h1=", round(h1 / n, 6),
              "in_cand=", round(inc / n, 4))
    orc = sum(1 for r in rows if any(r["gold_rank"].get(k) == 0 for k in r["ks"]))
    print(name, "oracle_cnt=", orc, "oracle=", round(orc / n, 6),
          "walker_top1=", round(sum(1 for r in rows if r["walker_gold_rank"] == 0) / n, 4))
