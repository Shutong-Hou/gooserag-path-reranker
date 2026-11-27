# -*- coding: utf-8 -*-
"""端到端实体级/集合级 KGQA 评测（本地自包含版，不依赖 G 盘导入）。

输入:
  E:\\GooseRAG\\data\\rog_qa\\{dataset}\\{split}.jsonl
  E:\\GooseRAG\\data\\rog_qa\\{dataset}_{backing}_heads.pt        (闭集关系头)
  E:\\GooseRAG\\data\\rog_qa\\cond_{dataset}_{backing}_heads.pt   (--cond)
  E:\\GooseRAG\\data\\rog_qa\\verifier_{dataset}_{backing}.pt     (--verifier)
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn

DATA = os.environ.get("GOOSERAG_DATA", r"E:\GooseRAG\data")
ROG_QA = os.environ.get("GOOSERAG_ROGQA", os.path.join(DATA, "rog_qa"))
MODEL_NAMES = {
    "rwkv7": os.environ.get("GOOSERAG_RWKV7",
                            r"E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF"),
    "gpt2": os.environ.get("GOOSERAG_GPT2", r"E:\GooseRAG\models\gpt2"),
    "pythia": os.environ.get("GOOSERAG_PYTHIA", r"E:\GooseRAG\models\pythia-160m"),
}
GENERIC_PREFIX = ("common.topic.", "freebase.valuenotation.", "common.webpage.",
                  "common.document.", "type.object.")
NOTABLE_PREFIX = ("common.topic.notable_types", "freebase.valuenotation.", "type.object.")


def is_generic(rel, mode="all"):
    if mode == "none":
        return False
    if mode == "notable":
        return rel.startswith(NOTABLE_PREFIX)
    return rel.startswith(GENERIC_PREFIX)


def load_model(backing):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODEL_NAMES[backing]
    print(f"loading tokenizer {name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if backing == "rwkv7" else torch.float16
    print(f"loading model dtype={dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=dtype, local_files_only=True)
    model.eval()
    print("model loaded", flush=True)
    return model, tok, name


def load_heads(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    head = nn.Linear(blob["hidden"], len(blob["rels"]))
    head.load_state_dict(blob["rel_state"])
    return head.to(device).eval(), blob["rels"]


def _final_norm_module(model):
    """返回主干最后的归一化层（用于 --feat-norm double 的复现模式）。

    transformers 的 hidden_states[-1] 本身已在 final norm 之后；本函数仅在
    需要复现“旧版双重归一化特征”时再套一次同一模块，保证与历史关系头/判定器一致。
    """
    for path in ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    last = None
    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            last = m
    return last


class EvalFTVerifier(nn.Module):
    """评测用微调判定器；feat_norm 控制是否复现旧版双重归一化。

    历史 checkpoints 均以 double 特征训练，故默认 double 以复现已发表数值；
    single 用于 fixnorm 重训后的消融权重。
    """

    def __init__(self, model_path, feat_norm="double"):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32,
            local_files_only=True)
        self.head = nn.Linear(self.lm.config.hidden_size, 2)
        self.feat_norm = feat_norm

    def forward(self, ids, mask):
        out = self.lm(input_ids=ids, attention_mask=mask,
                      output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        if self.feat_norm == "double":
            nrm = _final_norm_module(self.lm)
            w = next(nrm.parameters(), None)
            h = nrm(h.to(w.dtype) if w is not None else h)
        last = mask.sum(dim=1) - 1
        h = h[torch.arange(h.size(0), device=h.device), last]
        return self.head(h.float())


def build_graph(edges):
    out = defaultdict(lambda: defaultdict(list))
    for h, r, t in edges:
        out[h][r].append(t)
    return out


def find_chain(g, starts, rels, target):
    """按关系序列从起点还原实体链（供含中间实体的路径呈现）。"""
    def dfs(node, i, path):
        if i == len(rels):
            return path if node == target else None
        for nxt in g.get(node, {}).get(rels[i], []):
            r = dfs(nxt, i + 1, path + [nxt])
            if r:
                return r
        return None

    for s in starts:
        r = dfs(s, 0, [s])
        if r:
            return r
    return None


@torch.no_grad()
def encode_texts(model, tok, texts, device, backing, batch_size=64, max_len=320,
                 feat_norm="double"):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        if backing == "rwkv7":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc, output_hidden_states=True, use_cache=False)
        else:
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states[-1] 已是 final norm 之后；feat_norm="double" 时再套一次同一
        # final norm 模块，用于复现历史关系头/判定器训练时的特征口径。
        h = out.hidden_states[-1]
        if feat_norm == "double":
            nrm = _final_norm_module(model)
            w = next(nrm.parameters(), None)
            h = nrm(h.to(w.dtype) if w is not None else h)
        last = enc["attention_mask"].sum(dim=1) - 1
        h = h[torch.arange(h.size(0), device=device), last]
        feats.append(h.float().cpu())
    return torch.cat(feats)


def prompt(question, prefix):
    p = " -> ".join(prefix) if prefix else "(none)"
    return f"Question: {question}\nKnown path: {p}\nNext relation:"


def prompt_cond(question, entity, prefix):
    p = " -> ".join(prefix) if prefix else "(none)"
    return f"Question: {question}\nCurrent entity: {entity}\nKnown path: {p}\nNext relation:"


def prompt_verifier(question, entity):
    return f"Question: {question}\nCandidate: {entity}\nIs this the final answer?"


PATH_STYLE = "rel"
REL_NAMES = []


def fmt_path(rels, ents=None):
    """关系路径的三种呈现方式（P1-1 消融）+ 含中间实体（P1-1b）。"""
    if not rels:
        return "(none)"
    if PATH_STYLE == "entity" and ents:
        parts = [str(ents[0])]
        for r, nxt in zip(rels, ents[1:]):
            parts.append(f" --[{r}]--> {nxt}")
        return "".join(parts)
    if PATH_STYLE == "short":
        return " -> ".join(r.split(".")[-1] for r in rels)
    if PATH_STYLE == "natural":
        return "The answer is reached via " + ", then ".join(rels) + "."
    return " -> ".join(rels)


def prompt_verifier_path(question, rels, entity, ents=None):
    p = fmt_path(rels, ents)
    return (f"Question: {question}\nRelation path: {p}\n"
            f"Candidate: {entity}\nIs this the final answer?")


@torch.no_grad()
def score_ft(ft_model, tok, texts, device, batch=32, max_len=160):
    scores = []
    ft_model.eval()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = ft_model(enc["input_ids"], enc["attention_mask"])
        scores.extend((logits[:, 1] - logits[:, 0]).float().cpu().tolist())
    return scores


def reachable_in_vocab(out, starts, gold, rel2id, max_hops):
    cur = set(starts)
    for _ in range(max_hops):
        nxt = set()
        for e in cur:
            for r, tails in out.get(e, {}).items():
                if r in rel2id:
                    nxt.update(tails)
        if not nxt:
            return False
        if nxt & gold:
            return True
        cur = nxt
    return False


GUIDER = None      # (ft_model, tok)：判定器引导 beam（P1-7）
GUIDE_TOP = 0


def _guided_prune(args, question, new, device):
    """每跳把 walker 前 GUIDE_TOP 个候选交给判定器，按判定器分保留 top-beam。

    默认（GUIDER=None）保持原行为：只按 walker 累积分截断。
    """
    if GUIDER is None or GUIDE_TOP <= 0:
        return dict(sorted(new.items(), key=lambda x: -x[1])[:args.beam])
    items = sorted(new.items(), key=lambda x: -x[1])[:GUIDE_TOP]
    ft_model, tok = GUIDER
    texts = [prompt_verifier_path(question, p, e) for (e, p), _ in items]
    vs = score_ft(ft_model, tok, texts, device, 64)
    ranked = sorted(zip(items, vs), key=lambda x: -x[1])[:args.beam]
    return {k: s for (k, s), _v in ranked}


def run_beam(args, question, starts, out, rel2id, model, tok, rel_head, prior, device):
    beams = {(e, ()): 0.0 for e in starts}
    seen_out = {}
    path_of = {}
    for hop in range(args.max_hops):
        if args.policy == "model":
            if args.cond:
                keys = list(beams.keys())
                feats = encode_texts(model, tok,
                                     [prompt_cond(question, e, p) for e, p in keys],
                                     device, args.backing, args.batch,
                                     feat_norm=args.head_feat_norm)
                logits = rel_head(feats.to(device)).cpu()
                kidx = {k: i for i, k in enumerate(keys)}
                pidx = None
            else:
                prefixes = sorted({p for _, p in beams})
                feats = encode_texts(model, tok, [prompt(question, p) for p in prefixes],
                                     device, args.backing, args.batch,
                                     feat_norm=args.head_feat_norm)
                logits = rel_head(feats.to(device)).cpu()
                pidx = {p: i for i, p in enumerate(prefixes)}
                kidx = None
        else:
            logits = prior if prior is not None else torch.zeros(len(rel2id))
            pidx = kidx = None
        new = {}
        for (e, p), s in beams.items():
            legal = [r for r in out.get(e, {}) if r in rel2id
                     and not (args.ban_generic and is_generic(r, args.generic_mode))]
            if not legal:
                continue
            idx = torch.tensor([rel2id[r] for r in legal])
            if kidx is not None:
                row = logits[kidx[(e, p)]]
            elif pidx is not None:
                row = logits[pidx[p]]
            else:
                row = logits
            lp = torch.log_softmax(row[idx], dim=0)
            k = min(args.topk_rel, len(legal))
            vals, pos = torch.topk(lp, k)
            for v, j in zip(vals.tolist(), pos.tolist()):
                r = legal[j]
                for c in out[e][r][:args.max_children]:
                    key = (c, p + (r,))
                    ns = s + v
                    if key not in new or ns > new[key]:
                        new[key] = ns
        if not new:
            break
        beams = _guided_prune(args, question, new, device)
        norm = float(hop + 1) if args.score_norm == "mean" else 1.0
        for (e, p), s in beams.items():
            sc = s / norm
            if e not in seen_out or sc > seen_out[e]:
                seen_out[e] = sc
                path_of[e] = p
    if args.length_mode == "final":
        cand = {}
        for (e, p), s in beams.items():
            norm = len(p) if args.score_norm == "mean" else 1.0
            sc = s / max(norm, 1)
            if e not in cand or sc > cand[e]:
                cand[e] = sc
                path_of[e] = p
    else:
        cand = seen_out
    return sorted(cand.items(), key=lambda x: -x[1]), beams, path_of


def run_set_beam(args, question, starts, out, rel2id, model, tok, rel_head, prior, device):
    """集合级 beam: 每个关系前缀维护实体集合, 不淘汰单个实体。"""
    beams = {(): (0.0, set(starts))}
    for hop in range(args.max_hops):
        prefixes = sorted(beams)
        if args.policy == "model":
            feats = encode_texts(model, tok, [prompt(question, p) for p in prefixes],
                                 device, args.backing, args.batch,
                                 feat_norm=args.head_feat_norm)
            logits = rel_head(feats.to(device)).cpu()
            pidx = {p: i for i, p in enumerate(prefixes)}
        else:
            logits = prior if prior is not None else torch.zeros(len(rel2id))
            pidx = None
        new = {}
        for p, (s, ents) in beams.items():
            legal = set()
            for e in ents:
                for r in out.get(e, {}):
                    if r in rel2id and not (args.ban_generic and is_generic(r, args.generic_mode)):
                        legal.add(r)
            if not legal:
                continue
            legal = sorted(legal)
            idx = torch.tensor([rel2id[r] for r in legal])
            row = logits[pidx[p]] if pidx is not None else logits
            lp = torch.log_softmax(row[idx], dim=0)
            k = min(args.topk_rel, len(legal))
            vals, pos = torch.topk(lp, k)
            for v, j in zip(vals.tolist(), pos.tolist()):
                r = legal[j]
                tails = set()
                for e in sorted(ents):
                    tails.update(out[e].get(r, ()))
                    if len(tails) > args.max_set:
                        break
                if not tails:
                    continue
                new[p + (r,)] = (s + v, set(sorted(tails)[:args.max_set]))
        if not new:
            break
        beams = dict(sorted(new.items(), key=lambda x: -x[1][0])[:args.beam])
    cand = {}
    path_of = {}
    for p, (s, ents) in beams.items():
        norm = len(p) if args.score_norm == "mean" else 1.0
        sc = s / max(norm, 1)
        for e in ents:
            if e not in cand or sc > cand[e]:
                cand[e] = sc
                path_of[e] = p
    return sorted(cand.items(), key=lambda x: -x[1]), beams, path_of


def f1(pred, gold):
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--backing", default="rwkv7", choices=["rwkv7", "gpt2", "pythia"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--topk-rel", type=int, default=5)
    ap.add_argument("--max-children", type=int, default=128)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--length-mode", choices=["final", "all"], default="all")
    ap.add_argument("--score-norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dump", default="")
    ap.add_argument("--dump-top", type=int, default=100)
    ap.add_argument("--eval-top", type=int, default=100,
                    help="set_hit / 排名统计使用的候选数（与 dump-top 解耦）")
    ap.add_argument("--ban-generic", action="store_true")
    ap.add_argument("--generic-mode", choices=["all", "notable", "none"], default="all",
                    help="all=当前全禁; notable=只禁 notable_types/valuenotation/type.object; none=不禁")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--verifier", action="store_true")
    ap.add_argument("--verifier-file", default="",
                    help="判定头权重路径; 默认 verifier_{dataset}_{backing}.pt")
    ap.add_argument("--verifier-ft", default="",
                    help="端到端微调判定器权重路径")
    ap.add_argument("--verifier-path", action="store_true",
                    help="判定器输入包含实际关系路径")
    ap.add_argument("--no-reach", action="store_true",
                    help="跳过 reachable_in_vocab（大子图上很慢）")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--verifier-top", type=int, default=200,
                    help="判定器重打分的候选数；已发表主表用 200（不是 100），改小会显著掉 top-1")
    ap.add_argument("--head-feat-norm", choices=["single", "double"], default="double",
                    help="关系头特征口径。历史 heads.pt 由 train_probe 生成，RWKV 经 sub.norm 套了一次"
                         "额外 final norm，实际是 double；默认 double 以复现已发表数值")
    ap.add_argument("--verifier-feat-norm", choices=["single", "double"], default="double",
                    help="判定器特征口径。历史 verifier_ft_*.pt 训练时套了 extra final norm，故默认 double；"
                         "fixnorm 重训的 *_fixnorm_*.pt 用 single")
    ap.add_argument("--search", choices=["entity", "set"], default="entity")
    ap.add_argument("--max-set", type=int, default=500)
    ap.add_argument("--policy", choices=["model", "random", "prior"], default="model")
    ap.add_argument("--out", default="", help="结果 JSON 路径; 默认 {dataset}_{backing}_{split}_e2e.json")
    ap.add_argument("--heads", default="", help="关系头 ckpt 路径; 留空用默认")
    ap.add_argument("--path-style", choices=["rel", "short", "natural", "entity"], default="rel",
                    help="关系路径的呈现方式：rel=全名链, short=去命名空间, natural=自然语言, "
                         "entity=含中间实体（P1-1/P1-1b）")
    ap.add_argument("--path-shuffle", action="store_true",
                    help="P1-5 对照：把候选的关系路径替换为随机关系（长度不变），检验增益是否来自路径内容")
    ap.add_argument("--verifier-max-len", type=int, default=160,
                    help="判定器输入截断长度；含中间实体的路径较长时需调大（如 320）")
    ap.add_argument("--entity-mask", action="store_true",
                    help="P1-5 对照：把候选实体名替换为 [MASK]，检验对实体名的依赖")
    ap.add_argument("--guide", action="store_true",
                    help="P1-7：用判定器引导 beam（每跳按判定器分裁剪，而不是只按 walker 分）")
    ap.add_argument("--guide-top", type=int, default=400,
                    help="引导时每跳先按 walker 取多少个候选交给判定器")
    args = ap.parse_args()
    global PATH_STYLE
    PATH_STYLE = args.path_style

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel_head = model = tok = ver_head = ft_model = None
    ft_meta = None
    prior = None
    if args.policy == "model":
        if args.heads:
            heads_path = args.heads
        elif args.cond:
            heads_path = os.path.join(ROG_QA, f"cond_{args.dataset}_{args.backing}_heads.pt")
        else:
            heads_path = os.path.join(ROG_QA, f"{args.dataset}_{args.backing}_heads.pt")
        rel_head, rels = load_heads(heads_path, device)
        model, tok, name = load_model(args.backing)
        model.to(device)
        print(f"loaded, classes={len(rels)}, cond={args.cond}, device={device}", flush=True)
        if args.verifier:
            vp = args.verifier_file or os.path.join(
                ROG_QA, f"verifier_{args.dataset}_{args.backing}.pt")
            vb = torch.load(vp, map_location="cpu", weights_only=False)
            ver_head = nn.Linear(vb["hidden"], 2)
            ver_head.load_state_dict(vb["state"])
            ver_head.to(device).eval()
            print(f"loaded verifier, val_acc={vb.get('val_acc'):.4f}", flush=True)
        if args.verifier_ft:
            ft_model = EvalFTVerifier(MODEL_NAMES[args.backing],
                                      feat_norm=args.verifier_feat_norm)
            fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
            ft_meta = fb
            ft_model.lm.load_state_dict(fb["lm_state"])
            ft_model.head.load_state_dict(fb["head_state"])
            ft_model.to(device).eval()
            print("loaded fine-tuned verifier", flush=True)
    else:
        rels = json.load(open(os.path.join(DATA, f"g1_{args.dataset}_relations.json"),
                              encoding="utf-8"))
        if args.policy == "prior":
            from collections import Counter
            cnt = Counter()
            with open(os.path.join(DATA, f"g1_{args.dataset}_train.jsonl"), encoding="utf-8") as f:
                for line in f:
                    cnt[json.loads(line)["target"]] += 1
            prior = torch.tensor([float(cnt.get(r, 0) + 1) for r in rels]).log()
        if args.verifier_ft:
            # 非 model 策略也需要判定器主干与 tokenizer，用于“先验/随机候选 + 判定器”对照
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_NAMES[args.backing],
                                                trust_remote_code=True, local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            ft_model = EvalFTVerifier(MODEL_NAMES[args.backing],
                                      feat_norm=args.verifier_feat_norm)
            fb = torch.load(args.verifier_ft, map_location="cpu", weights_only=False)
            ft_meta = fb
            ft_model.lm.load_state_dict(fb["lm_state"])
            ft_model.head.load_state_dict(fb["head_state"])
            ft_model.to(device).eval()
            print(f"loaded fine-tuned verifier (policy={args.policy})", flush=True)
    rel2id = {r: i for i, r in enumerate(rels)}
    global REL_NAMES, GUIDER, GUIDE_TOP
    REL_NAMES = list(rel2id)
    if args.guide and ft_model is not None:
        GUIDER = (ft_model, tok)
        GUIDE_TOP = args.guide_top
        print(f"verifier-guided beam: guide_top={GUIDE_TOP}", flush=True)

    src = os.path.join(ROG_QA, args.dataset, f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    tot = hit1 = set_hit = reach = 0
    f1s = {5: [], 10: [], 20: [], 100: []}
    dumps = []
    rank_counts = {"top1": 0, "top5": 0, "top20": 0, "top100": 0, "in_cand": 0, "miss": 0}
    for i, r in enumerate(rows):
        out = build_graph(r["edges"])
        starts = [e for e in r["q_entity"] if e in out]
        gold = set(r["a_entity"]) or set(r["answers"])
        if not starts or not gold:
            continue
        tot += 1
        if args.search == "set":
            ranked, beams, path_of = run_set_beam(args, r["question"], starts, out, rel2id,
                                                  model, tok, rel_head, prior, device)
        else:
            ranked, beams, path_of = run_beam(args, r["question"], starts, out, rel2id,
                                             model, tok, rel_head, prior, device)
        if (ver_head is not None or ft_model is not None) and ranked:
            cand = ranked[:args.verifier_top]
            if args.verifier_path:
                if args.path_shuffle:
                    rng = random.Random(abs(hash(r["id"])) % (2 ** 31))
                    texts = [prompt_verifier_path(
                        r["question"],
                        [rng.choice(REL_NAMES) for _ in path_of.get(e, ())], e)
                        for e, _ in cand]
                elif args.entity_mask:
                    texts = [prompt_verifier_path(r["question"], path_of.get(e, ()), "[MASK]")
                             for e, _ in cand]
                elif args.path_style == "entity":
                    texts = []
                    for e, _ in cand:
                        rels = path_of.get(e, ())
                        ch = find_chain(out, starts, rels, e) if rels else None
                        texts.append(prompt_verifier_path(r["question"], rels, e, ch))
                else:
                    texts = [prompt_verifier_path(r["question"], path_of.get(e, ()), e)
                             for e, _ in cand]
            else:
                texts = [prompt_verifier(r["question"], e) for e, _ in cand]
            if ft_model is not None:
                vs = score_ft(ft_model, tok, texts, device, args.batch,
                              max_len=args.verifier_max_len)
            else:
                feats = encode_texts(model, tok, texts, device, args.backing, args.batch,
                                     feat_norm=args.verifier_feat_norm)
                vl = ver_head(feats.to(device)).cpu()
                vs = (vl[:, 1] - vl[:, 0]).tolist()
            fused = [(e, args.gamma * s + args.beta * float(v))
                     for (e, s), v in zip(cand, vs)]
            fused.sort(key=lambda x: -x[1])
            ranked = fused + ranked[args.verifier_top:]
        pred_set = {e for e, _ in ranked[:args.eval_top]}
        if ranked and ranked[0][0] in gold:
            hit1 += 1
        if pred_set & gold:
            set_hit += 1
        for k in f1s:
            f1s[k].append(f1({e for e, _ in ranked[:k]}, gold))
        if (not args.no_reach) and reachable_in_vocab(out, starts, gold, rel2id, args.max_hops):
            reach += 1
        gold_rank = None
        for j, (e, _) in enumerate(ranked):
            if e in gold:
                gold_rank = j
                break
        if gold_rank is None:
            rank_counts["miss"] += 1
        else:
            rank_counts["in_cand"] += 1
            for k in (1, 5, 20, 100):
                if gold_rank < k:
                    rank_counts[f"top{k}"] += 1
        if args.dump:
            dumps.append({"id": r["id"], "question": r["question"],
                          "gold": sorted(gold), "pred": ranked[:args.dump_top],
                          "n_cand": len(ranked), "gold_rank": gold_rank})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} hit1={hit1/max(tot,1):.4f} "
                  f"set_hit={set_hit/max(tot,1):.4f}", flush=True)

    res = {
        "dataset": args.dataset, "backing": args.backing, "split": args.split,
        "policy": args.policy, "search": args.search, "cond": args.cond,
        "verifier": args.verifier, "beta": args.beta, "gamma": args.gamma,
        "n": tot, "hits1": hit1 / max(tot, 1), "set_hit": set_hit / max(tot, 1),
        "set_f1_top5": sum(f1s[5]) / max(len(f1s[5]), 1),
        "set_f1_top10": sum(f1s[10]) / max(len(f1s[10]), 1),
        "set_f1_top20": sum(f1s[20]) / max(len(f1s[20]), 1),
        "reachable_in_vocab": (None if args.no_reach else reach / max(tot, 1)),
        "gold_rank_rate": {k: v / max(tot, 1) for k, v in rank_counts.items()},
        "beam": args.beam, "topk_rel": args.topk_rel, "max_hops": args.max_hops,
        "length_mode": args.length_mode, "score_norm": args.score_norm,
        "eval_top": args.eval_top, "ban_generic": args.ban_generic,
        "generic_mode": args.generic_mode,
        "head_feat_norm": args.head_feat_norm,
        "verifier_feat_norm": args.verifier_feat_norm,
        "max_children": args.max_children, "verifier_top": args.verifier_top,
        "verifier_ft": args.verifier_ft or None,
        "verifier_ft_val_acc": (float(ft_meta["val_acc"])
                                if ft_meta and ft_meta.get("val_acc") is not None else None),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out_path = args.out or os.path.join(
        ROG_QA, f"{args.dataset}_{args.backing}_{args.split}_e2e.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(dumps, f, ensure_ascii=False)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
