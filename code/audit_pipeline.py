# -*- coding: utf-8 -*-
"""RoG KGQA 管线审计：词表泄漏、训练/评测重叠、gold 表示一致性。

只读检查，不修改任何数据。输出 JSON 便于记录。
用法:
  python audit_pipeline.py --dataset webqsp
  python audit_pipeline.py --dataset cwq
"""
import argparse
import json
import os
import re
from collections import Counter

DATA = r"E:\GooseRAG\data"
ROG_QA = os.path.join(DATA, "rog_qa")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def split_relations(rows):
    rels = Counter()
    nodes = set()
    for r in rows:
        for h, rel, t in r["edges"]:
            rels[rel] += 1
            nodes.add(h)
            nodes.add(t)
    return set(rels), nodes


def gold_report(rows):
    """逐题检查 gold 是否在**该题自己的子图**里（不能用 split 节点并集）。"""
    n = len(rows)
    a_nonempty = a_empty_ans_nonempty = a_empty_ans_empty = 0
    a_subset_nodes = a_partial_nodes = a_none_nodes = 0
    fallback_ans_in_nodes = fallback_ans_not_nodes = 0
    samples = []
    for r in rows:
        nodes = set()
        for h, _, t in r.get("edges", []):
            nodes.add(h)
            nodes.add(t)
        a = [str(x) for x in r.get("a_entity", [])]
        ans = [str(x) for x in r.get("answers", [])]
        if a:
            a_nonempty += 1
            hit = sum(1 for x in a if x in nodes)
            if hit == len(a):
                a_subset_nodes += 1
            elif hit > 0:
                a_partial_nodes += 1
            else:
                a_none_nodes += 1
        else:
            if ans:
                a_empty_ans_nonempty += 1
                hit = sum(1 for x in ans if x in nodes)
                if hit:
                    fallback_ans_in_nodes += 1
                else:
                    fallback_ans_not_nodes += 1
                if len(samples) < 5:
                    samples.append({
                        "id": r.get("id"), "answers": ans[:5],
                        "a_entity": a, "n_nodes": len(nodes),
                        "ans_in_nodes": hit,
                    })
            else:
                a_empty_ans_empty += 1
    return {
        "n": n,
        "a_entity_nonempty": a_nonempty,
        "a_entity_empty_answers_nonempty": a_empty_ans_nonempty,
        "a_entity_empty_answers_empty": a_empty_ans_empty,
        "a_entity_all_in_graph_nodes": a_subset_nodes,
        "a_entity_partial_in_graph_nodes": a_partial_nodes,
        "a_entity_none_in_graph_nodes": a_none_nodes,
        "fallback_samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                           encoding="utf-8"))
    relset = set(rels)
    splits = ["train", "validation", "test"]
    rows = {}
    rels_by_split = {}
    nodes_by_split = {}
    missing = []
    for sp in splits:
        p = os.path.join(ROG_QA, args.dataset, f"{sp}.jsonl")
        if not os.path.exists(p):
            missing.append(sp)
            rows[sp] = []
            rels_by_split[sp] = set()
            nodes_by_split[sp] = set()
            continue
        rows[sp] = load_jsonl(p)
        rr, nn = split_relations(rows[sp])
        rels_by_split[sp] = rr
        nodes_by_split[sp] = nn

    # train 子图缺失时，用 G1 训练数据的 target 关系作为训练关系代理
    train_rel_source = "subgraph"
    if "train" in missing:
        g1p = os.path.join(DATA, f"g1_{args.dataset}_train.jsonl")
        train_rels = set()
        if os.path.exists(g1p):
            for r in load_jsonl(g1p):
                if r.get("target"):
                    train_rels.add(r["target"])
            train_rel_source = f"g1_targets({g1p})"
        rels_by_split["train"] = train_rels

    rel_unseen = {sp: sorted(rels_by_split[sp] - relset) for sp in splits}
    rel_only_test = sorted((rels_by_split["test"] - rels_by_split["train"]) & relset)
    rel_only_val = sorted((rels_by_split["validation"] - rels_by_split["train"]) & relset)

    unseen_rel_edges = {}
    unseen_rel_questions = {}
    for sp in splits:
        bad = set(rel_only_test if sp == "test" else rel_only_val)
        ne = nq = 0
        for r in rows[sp]:
            hit = False
            for _, rel, _ in r["edges"]:
                if rel in bad:
                    ne += 1
                    hit = True
            if hit:
                nq += 1
        unseen_rel_edges[sp] = ne
        unseen_rel_questions[sp] = nq

    qsets = {sp: {r["question"].strip() for r in rows[sp]} for sp in splits}
    overlaps = {}
    for a in splits:
        for b in splits:
            if a < b:
                overlaps[f"{a}&{b}"] = len(qsets[a] & qsets[b])

    verifier_q = {}
    for name in (f"verifier_{args.dataset}_train.jsonl",
                 f"verifier_{args.dataset}_hard.jsonl"):
        p = os.path.join(ROG_QA, name)
        if not os.path.exists(p):
            verifier_q[name] = None
            continue
        qs = set()
        for r in load_jsonl(p):
            q = r.get("question")
            if not q:
                q = r.get("text", "").split("\n")[0].replace("Question: ", "")
            qs.add(q.strip())
        nqs = {norm_q(q) for q in qs}
        nval = {norm_q(q) for q in qsets["validation"]}
        ntest = {norm_q(q) for q in qsets["test"]}
        verifier_q[name] = {
            "n_questions": len(qs),
            "overlap_with_validation": len(qs & qsets["validation"]),
            "overlap_with_test": len(qs & qsets["test"]),
            "norm_overlap_with_validation": len(nqs & nval),
            "norm_overlap_with_test": len(nqs & ntest),
            "overlap_validation_examples": sorted(qs & qsets["validation"])[:5],
            "overlap_test_examples": sorted(qs & qsets["test"])[:5],
        }

    g1_checks = {}
    for sp in splits:
        p = os.path.join(DATA, f"g1_{args.dataset}_{sp}.jsonl")
        if not os.path.exists(p):
            g1_checks[sp] = None
            continue
        qs = {r["question"].strip() for r in load_jsonl(p) if r.get("question")}
        g1_checks[sp] = {
            "n_questions": len(qs),
            "overlap_with_validation": len(qs & qsets["validation"]),
            "overlap_with_test": len(qs & qsets["test"]),
            "not_in_qa_train": len(qs - qsets["train"]) if qsets["train"] else None,
        }

    report = {
        "dataset": args.dataset,
        "g1_question_checks": g1_checks,
        "missing_subgraph_splits": missing,
        "train_relation_source": train_rel_source,
        "relation_vocab": len(rels),
        "relations_per_split": {sp: len(rels_by_split[sp]) for sp in splits},
        "relations_outside_vocab": {sp: len(rel_unseen[sp]) for sp in splits},
        "relations_outside_vocab_examples": {sp: rel_unseen[sp][:10] for sp in splits},
        "edges_outside_vocab": {
            sp: sum(1 for r in rows[sp] for _, rel, _ in r["edges"] if rel not in relset)
            for sp in splits
        },
        "edges_total": {sp: sum(len(r["edges"]) for r in rows[sp]) for sp in splits},
        "relations_in_test_not_train_in_vocab": len(rel_only_test),
        "relations_in_test_not_train_examples": rel_only_test[:20],
        "relations_in_val_not_train_in_vocab": len(rel_only_val),
        "edges_with_unseen_relation": unseen_rel_edges,
        "questions_with_unseen_relation": unseen_rel_questions,
        "question_overlap": overlaps,
        "duplicate_questions_within_split": {
            sp: len(rows[sp]) - len(qsets[sp]) for sp in splits
        },
        "gold": {
            sp: gold_report(rows[sp]) for sp in splits
        },
        "verifier_questions": verifier_q,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
