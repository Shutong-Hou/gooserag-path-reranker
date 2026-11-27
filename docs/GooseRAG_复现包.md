# GooseRAG 复现包（SIGIR 2027 投稿版）

> 目标：任何人在一台 24G 显存的单卡机器上，按本文档能复现论文中的所有数字。
> 所有路径以云端 `/root/autodl-tmp/GooseRAG` 为根（记为 `$ROOT`），代码在 `$ROOT/code/rog_qa`，数据在 `$ROOT/data/rog_qa`。
> 数字出处见 `实验台账.md`（E01–E44）与 `GooseRAG_全部实验汇总.md`。
> 论文主口径是 **"同数据孪生"**（同一训练组、同一配方、只改 prompt 是否带路径），复现时优先对齐它。

---

## 1. 环境

| 项 | 版本 |
|---|---|
| GPU | RTX 4090 24GB（单卡即可） |
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| transformers | 4.49.0 |
| fla | 0.2.1（RWKV-7 内核，**仅支持 CUDA + bf16**） |
| sentencepiece / protobuf / accelerate | 1.x / 7.36.1 / ≥0.26 |

必须的环境变量：

```bash
export GOOSERAG_ROGQA=$ROOT/data/rog_qa          # 数据目录
export GOOSERAG_RWKV7=$ROOT/models/RWKV7-Goose-World2.8-0.1B-HF
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

模型：`RWKV7-Goose-World2.8-0.1B-HF`（191,034,624 参数）。对照实验另需 `gpt2`、`pythia-160m`。

---

## 2. 数据

| 数据集 | 原始 | 预处理产物 | 规模（train/val/test） |
|---|---|---|---|
| WebQSP | RoG 的 WebQSP 划分 | `$ROOT/data/rog_qa/webqsp/{train,validation,test}.jsonl`（每题带 `edges` 子图） | 2,812 / 245 / 1,628（评 1,627） |
| CWQ | RoG 的 CWQ 划分 | `$ROOT/data/rog_qa/cwq/...` | 3,520 test |
| MetaQA | vanilla MetaQA（`kb.txt` + `qa_*.txt`） | `prepare_metaqa_rog.py --max-hops 3 --max-children 64 --max-edges 1200`；train 跨跳抽样 3,807（stride 30）、test 子集 2,000（stride 7）、全量 14,274 | 3,807 / — / 14,274 |

---

## 3. 训练配方（全部单卡时长实测）

| 模型 | 命令要点 | 时长 | 产物 |
|---|---|---|---|
| VGHN 关系头 | `train_head_vghn.py`（正例=金路径，负例=高判定器分错误实体） | ~20 min | `webqsp_rwkv7_heads_vghn.pt` |
| plain 判定器 | `train_verifier_listwise.py --epochs 2 --lr 1e-5 --max-cands 64 --max-len 128 --seed 42` | ~7 min | `verifier_ft_webqsp_rwkv7_listwise_beam200_s42.pt` |
| **path 判定器（主）** | 同上 + `--path`，`--cands path_groups_webqsp_train.jsonl` | ~7 min | `verifier_ft_webqsp_rwkv7_listwise_path_s42.pt` |
| 主干端到端微调（最终系统） | `train_joint.py --epochs 3 --lr-lm 2e-6 --lr-s1 1e-4 --lr-s2 1e-3 --max-cands 64` | ~60 min | `joint_full_s4{2..6}.pt` |
| SetRank（对照） | `cache_cand_states.py` + `train_setrank.py` | ~30 min | `setrank_vp_d256_s4{2..6}.pt` |
| MetaQA path 判定器 | `train_verifier_listwise.py --dataset metaqa --path --epochs 1` | ~10 min | `verifier_ft_metaqa_path_s4{2..5}.pt` |
| **plain 孪生（WebQSP）** | `run_p18.sh`：同 `path_groups_webqsp_train.jsonl`，**去掉 `--path`**，其余逐项相同 | ~7 min/seed | `verifier_ft_webqsp_p18_plain_s4{2,3,4}.pt` |
| **plain 孪生（MetaQA）** | `run_p110.sh`：同 `path_groups_metaqa_train.jsonl`，去 `--path` | ~10 min/seed | `verifier_ft_metaqa_p110_plain_s4{2,3}.pt` |

> 孪生是最严格的口径：plain 与 path 两条支路只在 prompt 上差一行，训练组数、epoch、lr、max-cands、max-len 完全一致。
> 引导 beam：`eval_e2e.py ... --guide --guide-top 200`（每跳先用 walker 取 200 个 (实体,路径)，交判定器打分后再保留 top-beam），见 `run_p111.sh`。
> 孪生补 seed 与引导 beam 全量也在 `run_p111.sh`（A/B/C/D 四段）。

路径呈现可选四种：`--path-style {rel,short,natural,entity}`（默认 `rel`；`entity` 需 `path_groups_*_ent.jsonl`，并用 `--max-len 320`）。

---

## 4. 主结果复现（WebQSP）

```bash
cd $ROOT/code/rog_qa
BASE="--dataset webqsp --backing rwkv7 --beam 200 --topk-rel 15 --max-children 128 \
      --max-hops 2 --ban-generic --head-feat-norm double --verifier-feat-norm double \
      --heads $GOOSERAG_ROGQA/webqsp_rwkv7_heads_vghn.pt --beta 2.0 --gamma 0.5"

# (1) 最终系统 5 seed（0.5846 ± 0.0035，n=1615）
#     直接用现成 runner，参数以脚本内为准
bash $ROOT/run_joint_seeds.sh          # 训练 5 seed -> joint_full_s4{2..6}.pt
bash $ROOT/run_final.sh                # 逐 seed 评测 -> final_s*.json

# (2) 同池单变量：walker / plain / fused / path / fused(path)
python pool_table.py                    # -> pool_table_test.json（0.1777 / 0.5591 / 0.5659 …）
python _plain_on_pool.py                # -> plain 0.4167 / fused 0.4794（vghn 池）

# (3) 配对显著性（+13.93, p=2.0e-29）
python _p01_samepool.py                 # -> p01_stats_webqsp_samepool.json

# (4) MetaQA（全量 87.97 / 2k 子集 88.20，4 seed 0.8898±0.0111）
python eval_e2e.py --dataset metaqa --backing rwkv7 --beam 200 --topk-rel 15 \
  --max-children 128 --max-hops 3 --ban-generic --beta 2.0 --gamma 0.5 \
  --heads $GOOSERAG_ROGQA/metaqa_rwkv7_heads.pt --split test \
  --verifier-top 200 --verifier-path \
  --verifier-ft $GOOSERAG_ROGQA/verifier_ft_metaqa_path_s42.pt \
  --out $GOOSERAG_ROGQA/metaqa_test_full_path.json
```

---

## 5. 消融复现

| 实验 | 命令 / 脚本 | 期望结果 |
|---|---|---|
| 提示词呈现 | `run_p11.sh`（`--path-style rel/short/natural`） | 54.89 / 55.87 / 56.48 |
| 训练数据规模 | `_sub_groups.py` + `run_p13.sh` | 25/50/100% → 50.46 / 52.43 / 54.89 |
| 路径内容对照 | `run_p15.sh`（`--path-shuffle` / `--entity-mask`） | 39.52 / 42.10 vs 56.18 |
| 候选数敏感性 | `eval_e2e.py --verifier-top {25,50,100,200}` | 49.23 / 51.38 / 54.33 / 56.18 |
| 同参数量主干 | `run_p02.sh`（RWKV / GPT-2 / Pythia） | 55.50 / 48.37 / 46.28 |
| 跳数分解 | `_webqsp_hop.py`、`_metaqa_hop.py` | WebQSP 1 跳 +18.22 / 2 跳 +3.47；MetaQA +18.48/+30.43/+23.95 |
| **孪生对照（WebQSP）** | `run_p18.sh` + `_stats_pair.py --a dump_p18_plain_s43.json --b dump_p11_path_s43.json` | plain 46.75±1.38（3 seed）vs path 54.89±0.12；配对 +6.58，b/c=89/196，p=2.1e-10，CI[+4.61,+8.60] |
| **孪生对照（MetaQA 2k）** | `run_p110.sh` + `_stats_pair.py` | plain 53.75±2.60（2 seed）vs path 88.98±1.11（4 seed）；配对 +32.60，b/c=61/713，p=6.2e-142，CI[+30.3,+34.9] |
| **题级剂量-反应** | `_dose_response.py --mined ... --cache ... --tag ...` | WebQSP 1/2/3/≥4 金答案：+14.93/+1.73/+6.25/+8.16；MetaQA +39.18/+33.67/+22.47/+16.04；CWQ −0.66/+1.87/−1.28/−4.06 |
| **引导 beam（换回检索预算）** | `eval_e2e.py --guide --guide-top 200`（`run_p111.sh` C/D） | beam200 无增益（60.50/60.50）；beam20 全量 53.78 → 56.12（+2.33） |
| **硬件一致性（4090→5090）** | 同一 ckpt 在第二张卡重评 | path 56.18→56.30；最终 s42 58.33→58.27；MetaQA 88.20→88.55（≤0.4pt） |
| seed 稳定性 | `run_path_seeds.sh` | 0.5489 ± 0.0013（3 seed） |

---

## 6. 可插拔复现（RoG-7B 作为底座）

```bash
# (a) 用 RoG-webqsp 7B 生成候选（n_beam = 3 / 10）
#     实际参数见 run_t4b10.sh（beam10）与 T4 首次运行的日志
bash $ROOT/run_t4b10.sh

# (b) 用我们的判定器重排（冻结 / 微调两版）：见 rog_plugplay.py 与 T4 日志中的命令行
#     冻结判定器 -> rog_plugplay_full.jsonl；微调判定器 -> rog_plugplay_full_joint.jsonl
```

期望：beam 3 → 60.93 / 65.36（冻结）/ **66.09**（微调）；beam 10 → 47.73 / **66.65**。

---

## 7. 成本测量

```bash
python _breakdown.py  --dataset webqsp --split test --limit 200 \
  --heads $GOOSERAG_ROGQA/webqsp_rwkv7_heads_vghn.pt \
  --verifier-ft $GOOSERAG_ROGQA/verifier_ft_webqsp_rwkv7_listwise_path_s42.pt \
  --out p04_breakdown.json          # beam 73.3ms + 打分 204.6ms = 277.9ms/题
python _bench_lat.py                # K=16/32/200 在 bs=1 与 bs=32 下的打分耗时
```

期望：K=16 → 26.0 ms（bs=32）/ 428.1 ms（bs=1）；K=200 → 163.9 / 3824.2 ms。

---

## 8. 图

```bash
python make_figures.py   # 输出到 figs/：fig1 Pareto、fig2 同池、fig3 可插拔、
                          # fig4 beam、fig5 MetaQA、fig6 正例密度、fig7 路径长度
```

`make_figures.py` 中的数字为该脚本内置常量，来源见第 4–5 节的 json。

---

## 9. 产物清单（判定性文件）

| 文件 | 说明 |
|---|---|
| `webqsp_rwkv7_heads_vghn.pt` | VGHN 关系头（最终） |
| `verifier_ft_webqsp_rwkv7_listwise_path_s42.pt` | path 判定器（冻结版，主消融用） |
| `joint_full_s4{2..6}.pt` | 主干端到端微调 5 seed（**最终系统**） |
| `verifier_ft_webqsp_joint_s42.pt` | 导出的可插拔判定器 |
| `verifier_ft_metaqa_path_s4{2..5}.pt` | MetaQA path 判定器（4 seed） |
| `setrank_vp_d256_s4{2..6}.pt` | SetRank 5 seed（对照） |
| `path_groups_webqsp_train.jsonl` / `_ent.jsonl` | 训练组（关系链 / 含实体链） |
| `mined_webqsp_{train,validation,test}_vghn.jsonl` | 候选池（含 walker/plain 分与路径） |
| `setrank_cache_vp_test.pt` | 同池显著性缓存（含 path 分） |

---

## 10. 预期数字对照（一键核对）

| 指标 | 期望 | 结果文件 |
|---|---|---|
| WebQSP 最终系统 H@1 | 58.03（全量）/ 58.46±0.35（n=1615, 5 seed） | `final_s*.json` |
| WebQSP 同池 path − plain | +13.93（p=2.0e-29） | `p01_stats_webqsp_samepool.json` |
| MetaQA path（全量） | 87.97 | `metaqa_test_full_path.json` |
| MetaQA 4 seed | 0.8898 ± 0.0111 | `metaqa_test_s2000_path_s4*.json` |
| 可插拔 beam3 | 66.09（+5.16, p=8.5e-7） | `rog_plugplay_full_joint.jsonl` |
| 可插拔 beam10 | 66.65（+18.92） | `rog_plugplay_full_joint_b10.jsonl` |
| 插入成本 | 26.0 ms（批量）/ 428.1 ms（串行） | `p04_latency.json` |
| SW 全系统延迟 | 277.9 ms/题（beam 73 + 打分 205） | `p04_breakdown.json` |
| 呈现消融 | 54.89 / 55.87 / 56.48 | `p11_{rel,short,natural}.json` |
| 数据规模 | 50.46 / 52.43 / 54.89 | `p13_f{0.25,0.50}.json` |
| 路径内容 | 39.52 / 42.10 | `p15_{pathshuffle,entitymask}.json` |
| 候选数曲线 | 49.23 / 51.38 / 54.33 / 56.18 | `p05b_vt*.json` |
| 主干对照 | 55.50 / 48.37 / 46.28 | `p02_*.json` |
| 跳数分解 | WebQSP 88.01/57.10；MetaQA 89.00/100.0/87.39 | `p16_webqsp_hop.json`、`p14_metaqa_hop.json` |
| 孪生 WebQSP | plain 46.75±1.38 / path 54.89±0.12（+8.14 均值；配对 +6.58, p=2.1e-10） | `p18_plain_s4{2,3,4}.json`、`p19_stats_twin.json` |
| 孪生 MetaQA | plain 53.75±2.60 / path 88.98±1.11（+35.2 均值；配对 +32.6, p=6.2e-142） | `p110_metaqa_plain_s4{2,3}.json`、`p110_stats_metaqa_twin.json` |
| 引导 beam（全量） | beam20 53.78 → 56.12（+2.33）；beam200 60.50 → 60.50 | `p17c_b20_base_full.json`、`p17c_b20_guide_full.json` |
| 剂量-反应 | 见 §5 表（三数据集四档） | `dose_*.json`（`_dose_response.py` 输出） |
| 硬件一致性 | path 56.18→56.30；s42 58.33→58.27；MetaQA 88.20→88.55 | 各 ckpt 重评日志 |

---

## 11. 已知差异与注意事项

1. 主 checkpoint（`verifier_ft_webqsp_rwkv7_listwise_path_s42.pt`，2 epoch）为 0.5618；`run_p11.sh` 的同配方复跑为 0.5489，差异来自未记录的 `max-len`/`max-cands`——两者不可混用，消融只在同一配方内比较。
2. 成本存在两套口径（按题批量 / 逐候选串行），论文同时给出，勿只引一个。
3. MetaQA 的 vanilla 版本没有分跳文件，跳数由子图 BFS 得到，论文已注明。
4. 所有 test 数字均为预注册协议下**单次评测**；seed 方差单独报告（§3 表）。
