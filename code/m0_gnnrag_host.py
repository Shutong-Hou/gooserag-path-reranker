# -*- coding: utf-8 -*-
"""M0: build a *second-host* candidate set from GNN-RAG's released ReaRev outputs.

Inputs (all public):
  - llm/results/gnn/RoG-webqsp/rearev-sbert/test.info   (per-question GNN candidates + scores)
  - data/webqsp/test.json                                (per-question KG subgraph)
  - data/webqsp/entities.txt                             (internal id -> Freebase MID)
  - data/webqsp/relations.txt                            (internal id -> relation name)
  - llm/entities_names.json                              (MID -> surface name)

Output: a JSONL in the same shape as our `mined_*` files, so the same verifier and the
same H@1 code can be applied without retraining:
  {id, question, gold: [names], cand: [[name, gnn_score, hops, [rel names]]]}
"""
import json
import os
import re
import sys
from collections import deque

ROOT = r"E:\GooseRAG"
GNN = os.path.join(ROOT, "models", "GNN-RAG")
INFO = os.path.join(GNN, "results", "gnn", "rearev-sbert_test.info")
DATA = os.path.join(GNN, "data", "webqsp")
NAMES = os.path.join(ROOT, "third_party", "cmavro-GNN-RAG-28d31bc", "llm", "entities_names.json")
OUT = os.path.join(ROOT, "data", "rog_qa", "gnnrag_webqsp_test.jsonl")


def load_lines(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_q(s):
    """test.info questions are WordPiece-decoded ('jam ##ai ##can'), so undo the markers."""
    s = s.replace(" ##", "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def load_names(path):
    """The released file is a single JSON object but contains raw quotes that break json.load;
    recover pairs with a regex instead (covers every well-formed pair)."""
    s = open(path, encoding="utf-8").read()
    pat = re.compile(r'"(m\.[A-Za-z0-9_]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')
    d = {}
    for m in pat.finditer(s):
        try:
            v = json.loads('"' + m.group(2) + '"')
        except Exception:
            v = m.group(2)
        d[m.group(1)] = v
    return d


def shortest_path(adj, src, dst, max_hops=3):
    """BFS over (head, rel, tail) edges; returns the relation-name list or None."""
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
    print("entity ids", len(ents), "relations", len(rels), "names", len(mid2name), flush=True)

    info = {norm_q(r["question"]): r for r in load_lines(INFO)}
    data = load_lines(os.path.join(DATA, "test.json"))
    print("info", len(info), "data", len(data), flush=True)

    out = []
    nopath = 0
    noname = 0
    nocand = 0
    unmatched = 0
    for row in data:
        q = row["question"].strip()
        r = info.get(norm_q(q))
        if r is None:
            unmatched += 1
            continue
        gold = list(r.get("answers") or [])
        if not gold:
            continue
        # adjacency over the local subgraph, keyed by internal entity ids
        adj = {}
        for h, rl, t in row["subgraph"]["tuples"]:
            adj.setdefault(h, []).append((rels[rl] if rl < len(rels) else str(rl), t))
            adj.setdefault(t, []).append((rels[rl] if rl < len(rels) else str(rl), h))
        topics = [e for e in row["entities"]]
        cand = []
        for mid, score in (r.get("cand") or []):
            try:
                cid = ent2id[mid]
            except KeyError:
                noname += 1
                continue
            name = mid2name.get(mid)
            if name is None:
                noname += 1
                name = mid  # OOD fallback: keep the candidate, mark it by its id
            path = None
            for t in topics:
                path = shortest_path(adj, t, cid)
                if path is not None:
                    break
            if path is None:
                nopath += 1
                path = []
            cand.append([mid, name, float(score), path])
        if not cand:
            nocand += 1
            continue
        out.append({"id": row["id"], "question": q, "gold": gold, "cand": cand})

    with open(OUT, "w", encoding="utf-8") as w:
        for r in out:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out)
    cands = [len(r["cand"]) for r in out]
    withp = sum(1 for r in out if any(c[3] for c in r["cand"]))
    gold_in = sum(1 for r in out if set(r["gold"]) & {c[0] for c in r["cand"]})
    print(json.dumps({
        "questions": n,
        "avg_cand": round(sum(cands) / max(n, 1), 2),
        "questions_with_any_path": withp,
        "gold_in_pool": gold_in,
        "gold_in_pool_rate": round(gold_in / max(n, 1), 4),
        "cand_without_path": nopath,
        "cand_without_name": noname,
        "questions_dropped_no_cand": nocand,
        "questions_unmatched": unmatched,
    }, indent=1), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
