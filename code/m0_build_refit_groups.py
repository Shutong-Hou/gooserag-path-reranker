# -*- coding: utf-8 -*-
"""Build listwise training groups from GNN-RAG's ReaRev output on the WebQSP *train* split,
so our verifier can be re-fitted to the second host in ~7 minutes.
"""
import json, os, re
from collections import deque

ROOT = r"E:\GooseRAG"
GNN = os.path.join(ROOT, "models", "GNN-RAG")
G = os.path.join(ROOT, "third_party", "cmavro-GNN-RAG-28d31bc", "gnn")
INFO = os.path.join(G, "checkpoint", "pretrain", "webqsp_train_eval_test.info")
DATA = os.path.join(G, "data", "webqsp_train")
NAMES = os.path.join(ROOT, "third_party", "cmavro-GNN-RAG-28d31bc", "llm", "entities_names.json")
OUT = os.path.join(ROOT, "data", "rog_qa", "gnnrag_train_groups.jsonl")


def norm_q(s):
    s = s.replace(" ##", "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def load_names(path):
    s = open(path, encoding="utf-8").read()
    pat = re.compile(r'"(m\.[A-Za-z0-9_]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')
    d = {}
    for m in pat.finditer(s):
        try:
            d[m.group(1)] = json.loads('"' + m.group(2) + '"')
        except Exception:
            d[m.group(1)] = m.group(2)
    return d


def shortest_path(adj, src, dst, max_hops=3):
    if src == dst:
        return []
    seen = {src}
    q = deque([(src, [])])
    while q:
        node, rels = q.popleft()
        if len(rels) >= max_hops:
            continue
        for rel, nxt in adj.get(node, []):
            if nxt in seen:
                continue
            nr = rels + [rel]
            if nxt == dst:
                return nr
            seen.add(nxt)
            q.append((nxt, nr))
    return None


def main():
    ents = [l.strip() for l in open(os.path.join(DATA, "entities.txt"), encoding="utf-8")]
    ent2id = {e: i for i, e in enumerate(ents)}
    rels = [l.strip() for l in open(os.path.join(DATA, "relations.txt"), encoding="utf-8")]
    mid2name = load_names(NAMES)
    info = {}
    for line in open(INFO, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            info[norm_q(r["question"])] = r
    n_groups = n_cand = n_pos = 0
    with open(OUT, "w", encoding="utf-8") as w:
        for line in open(os.path.join(DATA, "test.json"), encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            r = info.get(norm_q(row["question"]))
            if r is None:
                continue
            gold = set(r.get("answers") or [])
            if not gold:
                continue
            adj = {}
            for h, rl, t in row["subgraph"]["tuples"]:
                rel = rels[rl] if rl < len(rels) else str(rl)
                adj.setdefault(h, []).append((rel, t))
                adj.setdefault(t, []).append((rel, h))
            cands = []
            for mid, score in (r.get("cand") or []):
                cid = ent2id.get(mid)
                if cid is None:
                    continue
                path = None
                for t in row["entities"]:
                    path = shortest_path(adj, t, cid)
                    if path is not None:
                        break
                if path is None:
                    path = []
                cands.append({"entity": mid2name.get(mid, mid), "mid": mid,
                              "rels": path, "label": int(mid in gold)})
            if not cands or not any(c["label"] for c in cands):
                continue
            w.write(json.dumps({"question": row["question"].strip(),
                                "id": row["id"], "cands": cands}, ensure_ascii=False) + "\n")
            n_groups += 1
            n_cand += len(cands)
            n_pos += sum(c["label"] for c in cands)
    print(json.dumps({"groups": n_groups, "candidates": n_cand,
                      "avg_cand": round(n_cand / max(n_groups, 1), 2),
                      "positives": n_pos,
                      "avg_pos": round(n_pos / max(n_groups, 1), 2)}, indent=1))


if __name__ == "__main__":
    main()
