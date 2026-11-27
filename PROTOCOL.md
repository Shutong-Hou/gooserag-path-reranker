# Frozen protocol

This file records the evaluation protocol that was fixed before the final test runs. All values are
also the defaults of `eval_e2e.py`, so a command line that omits them still follows the protocol; any
deviation appears explicitly in the commands saved next to each result file in `results/`.

## 1. Pipeline hyper-parameters

| parameter | value |
|---|---|
| beam width | 200 |
| relations expanded per node (`topk-rel`) | 15 |
| maximum children per relation | 128 |
| maximum hops | 2 (WebQSP, CWQ), 3 (MetaQA) |
| fusion weights | beta = 2.0 (verifier), gamma = 0.5 (walker) |
| generic relations | banned (`--ban-generic`, prefix list in `eval_e2e.py`) |
| verifier top-k rescored per question | 200 |
| evaluation top-k for set-hit | 100 |
| feature normalisation | `double` for head and verifier (reproduces stored checkpoints) |

## 2. Training recipe

| model | recipe |
|---|---|
| VGHN relation head | gold paths + hard negatives (top 20% of verifier scores), 20 epochs, lr 1e-3, batch 4096, seed 42 |
| path / plain verifier | listwise softmax within a question, 2 epochs, lr 1e-5, 1 group per step × accum 2, max 64 candidates, max length 128, seed 42 (twins use identical settings, only the prompt differs) |
| backbone fine-tuning | 3 epochs, lr 2e-6 (LM) / 1e-4 (stage-1 head), 4 groups per step, max 32 candidates, max length 96, seeds 42–46 |
| MetaQA verifier | 1 epoch, otherwise identical (3,807 training questions sampled with stride 30) |
| external rerankers (revision) | same groups, same listwise objective, 2 epochs, lr 1e-5, 64 candidates, max length 192; only the backbone changes |

## 3. Data splits

* WebQSP: 2,812 / 245 / 1,628 (train/val/test); test evaluation uses 1,627 rows after dropping rows
  with empty answers.
* CWQ: 3,520 test rows (3,531 in the raw split).
* MetaQA: 3,807 train questions (stride 30 across hops, to avoid only 1-hop items), 14,274 test
  questions, plus a 2,000-question subset (stride 7) used for ablations.

## 4. Analysis plan

* Primary metric H@1; H@5/H@10 and in-pool oracle reported alongside.
* Paired comparisons: exact McNemar on per-question binary outcomes; 10,000-sample bootstrap 95% CI
  on the accuracy difference. Tests are applied to systems evaluated on the same questions.
* No multiplicity correction is applied; the three headline comparisons are at p ≤ 8.5e-7, below a
  conservative Bonferroni threshold (3e-3) for the ~15 tests reported.
* Validation is used only to check feasibility (validation pools contain ≤100 candidates and rank
  pipeline parameters in the opposite direction from test), never to choose test settings.

## 5. What is *not* tuned on test

Verifier top-k, beam width, hop limit, fusion weights and the training recipe were all fixed before
the final runs. The exploratory sweeps that informed them are kept in `results/` and are labelled
exploratory in `docs/实验台账.md`; they are not used as headline numbers.
