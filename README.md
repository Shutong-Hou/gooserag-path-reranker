# GooseRAG: a path-conditioned reranker for KGQA (anonymized artifact)

This repository contains code, result files, logs and the paper for

> **Show the Path, Not the LLM: A 0.19B Path-Conditioned Reranker that Upgrades Any KGQA Retriever**

Everything below is designed so that a reviewer can start on a bare machine with one 24 GB GPU,
reproduce the main table in about 40 minutes, and check every significance claim from the
per-question dumps that are shipped in `results/`.

---

## 1. What is in the box

```
code/       all scripts (data prep, mining, training, evaluation, statistics, figures)
results/    every .json / .jsonl / .log produced by the runs reported in the paper
docs/       experiment ledger, full experiment summary, reproduction notes (Chinese)
paper/      LaTeX source, bibliography, figures, compiled PDF
README.md   this file
PROTOCOL.md frozen hyper-parameters and evaluation definitions
MANIFEST.md SHA-256 of every shipped result file
```

Checkpoints are **not** shipped (they are 0.6–0.8 GB each); they are regenerated in minutes by the
commands in §4. Every number in the paper can be recomputed from `results/` alone, because the
per-question dumps used for all paired tests are included.

---

## 2. Requirements

| item | value |
|---|---|
| GPU | one 24 GB card (RTX 3090 / 4090 / 5090) |
| Python | 3.11 or 3.12 |
| torch | 2.5–2.8 with CUDA |
| transformers | 4.49.0 |
| flash-linear-attention (`fla`) | 0.2.1 |
| other | `numpy`, `pandas`, `networkx`, `scikit-learn`, `pyarrow` |

The RWKV-7 kernels in `fla` require **CUDA + bf16**; float32 and CPU are not supported.

```bash
pip install -r requirements.txt
export GOOSERAG_ROGQA=$PWD/data/rog_qa          # data root (see §3)
export GOOSERAG_RWKV7=$PWD/models/RWKV7-Goose-World2.8-0.1B-HF
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 3. Data

1. **RWKV-7 backbone**: `RWKV7-Goose-World2.8-0.1B-HF` (191,034,624 parameters).
2. **WebQSP / CWQ** in RoG's processed form (`{dataset}/{train,validation,test}.jsonl`, each row has
   `question`, `q_entity`, `a_entity`, `graph`): take them from the public `rmanluo/RoG` distribution
   (HuggingFace datasets `rmanluo/RoG-webqsp`, `rmanluo/RoG-cwq`) and place the jsonl files under
   `data/rog_qa/webqsp/` and `data/rog_qa/cwq/`.
3. **MetaQA**: raw `kb.txt` + `qa_{train,dev,test}.txt` under `data/metaqa/`; convert with

```bash
python prepare_metaqa_rog.py --raw data/metaqa --out data/rog_qa/metaqa \
    --max-hops 3 --max-children 64 --max-edges 1200 --splits dev,test --stride 1
python prepare_metaqa_rog.py --raw data/metaqa --out data/rog_qa/metaqa \
    --max-hops 3 --max-children 64 --max-edges 1200 --splits train --stride 30 --dump-train
```

4. **RoG-7B** (`rmanluo/RoG`) is needed only for the portability study and the reader experiment.

All datasets keep their original licences; nothing is redistributed here.

---

## 4. Quick start: the main WebQSP table in ~40 minutes

```bash
cd code
# (1) relation head with gold paths + hard negatives        ~20 min
python build_vghn_data.py --mined mined_webqsp_train_plain.jsonl \
    --g1 ../data/g1_webqsp_train.jsonl --rels ../data/g1_webqsp_relations.json \
    --out-pos vghn_pos.jsonl --out-neg vghn_neg.jsonl --hard-frac 0.2
python train_head_vghn.py --pos vghn_pos.jsonl --neg vghn_neg.jsonl \
    --init webqsp_rwkv7_heads.pt --val-g1 ../data/g1_webqsp_val.jsonl \
    --backing rwkv7 --epochs 20 --out webqsp_rwkv7_heads_vghn.pt

# (2) path-conditioned verifier                              ~7 min
python build_path_groups.py --mined mined_webqsp_train_vghn.jsonl \
    --out path_groups_webqsp_train.jsonl --max-cand 200
python train_verifier_listwise.py --dataset webqsp --backing rwkv7 --path \
    --cands path_groups_webqsp_train.jsonl --epochs 2 --batch-groups 1 --accum 2 \
    --lr 1e-5 --max-len 128 --max-cands 64 --seed 42 \
    --out verifier_ft_webqsp_rwkv7_listwise_path_s42.pt

# (3) evaluate end-to-end once                                ~13 min
python eval_e2e.py --dataset webqsp --backing rwkv7 --split test --beam 200 \
    --topk-rel 15 --max-children 128 --max-hops 2 --ban-generic --beta 2.0 --gamma 0.5 \
    --verifier-top 200 --verifier-path \
    --heads webqsp_rwkv7_heads_vghn.pt \
    --verifier-ft verifier_ft_webqsp_rwkv7_listwise_path_s42.pt \
    --out e2e_path.json
```

`run_vghn.ps1` / `run_m4.ps1` chain these steps (Windows PowerShell); on Linux translate the
`*>>` redirections to `>> file 2>&1`. `PROTOCOL.md` lists every fixed hyper-parameter.

Expected values (see `results/` for the exact files):

| system | H@1 |
|---|---|
| plain verifier + fused score | 44.19 |
| + VGHN relation head | 47.94 |
| + path-conditioned verifier | 56.12 |
| + set-wise reranking | 57.22 |
| + backbone fine-tuning (deployed, 5 seeds) | 58.03 (n=1,627) / 58.46±0.35 (n=1,615) |

---

## 5. Reproducing each table and figure

| paper item | command(s) | expected |
|---|---|---|
| Table 1 (cumulative) | §4 above; `train_joint.py` for the last row | 44.19 / 47.94 / 56.12 / 57.22 / 58.03 |
| Table 2 (twin, WebQSP) | `run_p18.sh` (plain twin) + `run_p11.sh` (path) + `_stats_pair.py` | 46.75±1.38 vs 54.89±0.12; paired +6.58, p=2.1e-10 |
| Table 2 (twin, MetaQA) | `run_m4.ps1` (full set, local) | plain vs path full-set numbers (see results/p110_metaqa_*_full_*.json) |
| Table 3 (one pool, five rankers) | `pool_table.py`, `_p01_samepool.py` | 17.77 / 41.98 / 48.30 / 55.91 / 56.59 / 57.60 |
| Table 4 (RoG host) | `rog_plugplay.py` (n_beam 3 and 10) | 60.93 → 65.36 / 66.09; beam 10: 47.73 → 66.65 |
| Table 5 (cost) | `_breakdown.py`, `_bench_lat.py`, `_bench_plugcost.py` | 278 ms = 73 + 205; +26 / +428 ms; 2.2 GB |
| Table 6 (density) | `_dose_response.py`, `_webqsp_hop.py`, `_metaqa_hop.py` | see §5.6 of the paper |
| external rerankers (revision) | `ext_rerank.py` (BERT / T5 / small LLM) | see results/ext_*.json |
| reader end-to-end (revision) | `rog_reader_eval.py` | see results/rog_reader_*.json |
| all pairwise tests | `_stats_pair.py --a dump_A.json --b dump_B.json` | exact McNemar + 10k bootstrap CI |

---

## 6. Evaluation definitions (why the numbers are auditable)

* **H@1** = the top-1 candidate's entity string is a member of the gold answer set. Matching is
  exact on the dataset's entity names: no lower-casing, no alias list, no fuzzy match. Multiple gold
  answers count as a hit if any one is ranked first.
* **Set-hit / H@k** apply the same rule to the first *k* items (`k` = 5/10 by default, set-hit at 100).
* **In-pool oracle** = fraction of questions whose gold answer is anywhere in the candidate pool; it
  upper-bounds H@1 and is reported separately so recall and ranking are never conflated.
* Empty pools are counted as misses in the `n=1,627` denominator; tables that use the 1,615 questions
  with a non-empty pool say so in the caption.

---

## 7. Hardware consistency

Numbers were produced on RTX 4090, 5090 and 3090 machines. Re-running three stored checkpoints on a
second architecture reproduces the reported accuracies within 0.4 points; the plain baseline on a
3090 (torch 2.5.1/CUDA 12.1) gives 44.38 vs 44.19 on the 4090 (torch 2.8.0/CUDA 12.8). Every
comparison in a table is computed within one GPU and one software stack. Latency numbers are from the
4090.

---

## 8. Negative results (shipped, not hidden)

* CWQ path conditioning: 26.31 → 25.23 (−1.08), reproduced and explained by path length × vocabulary.
* Cross-KG zero-shot: WebQSP→MetaQA 29.2, WebQSP→CWQ 24.1.
* Hard-negative fraction 0.2→0.3 leaves validation in-pool recall unchanged (0.7796 → 0.7796).
* top-15→25 relations: −0.47 points over three seeds.
* Beam 400: −0.7; max-children 128→512: no change (refutes a recall-bound hypothesis).
* Long-context state reuse: slower and less accurate without multi-turn training (Appendix B).

---

## 9. Availability and licensing

Code is released under the MIT licence (see `LICENSE`). Datasets and the RWKV/RoG weights keep their
original licences and are **not** redistributed in this repository. The repository is versioned by git;
the commit history is the record of what existed at submission time. A manifest of shipped result
files with SHA-256 hashes is in `MANIFEST.md`.

Contact for questions about the artifact: see the anonymized submission.
