$ErrorActionPreference = 'Continue'
$env:GOOSERAG_ROGQA = 'E:\GooseRAG\data\rog_qa'
$env:GOOSERAG_RWKV7 = 'E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$G = 'E:\GooseRAG\data'
$LOG = "$D\_vghn.log"

function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "VGHN chain start"
& $PY mine_train_candidates.py --dataset webqsp --backing rwkv7 --split train `
  --kmax 200 --beam 200 --topk-rel 15 --max-children 128 --max-hops 2 `
  --ban-generic --beta 2.0 --gamma 0.5 `
  --heads "$D\webqsp_rwkv7_heads.pt" `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_beam200.pt" `
  --dump "$D\mined_webqsp_train_plain.jsonl" *>> $LOG
Log "mine_train_plain exit=$LASTEXITCODE"

& $PY build_vghn_data.py --mined "$D\mined_webqsp_train_plain.jsonl" `
  --g1 "$G\g1_webqsp_train.jsonl" --rels "$G\g1_webqsp_relations.json" `
  --out-pos "$D\vghn_pos.jsonl" --out-neg "$D\vghn_neg.jsonl" --hard-frac 0.2 *>> $LOG
Log "vghn_data exit=$LASTEXITCODE"

& $PY train_head_vghn.py --pos "$D\vghn_pos.jsonl" --neg "$D\vghn_neg.jsonl" `
  --init "$D\webqsp_rwkv7_heads.pt" --val-g1 "$G\g1_webqsp_val.jsonl" `
  --backing rwkv7 --epochs 20 --out "$D\webqsp_rwkv7_heads_vghn.pt" *>> $LOG
Log "head_vghn exit=$LASTEXITCODE"

& $PY mine_train_candidates.py --dataset webqsp --backing rwkv7 --split train `
  --kmax 200 --beam 200 --topk-rel 15 --max-children 128 --max-hops 2 `
  --ban-generic --beta 2.0 --gamma 0.5 `
  --heads "$D\webqsp_rwkv7_heads_vghn.pt" `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_beam200.pt" `
  --dump "$D\mined_webqsp_train_vghn.jsonl" *>> $LOG
Log "mine_train_vghn exit=$LASTEXITCODE"

& $PY build_path_groups.py --mined "$D\mined_webqsp_train_vghn.jsonl" `
  --out "$D\path_groups_webqsp_train.jsonl" --max-cand 200 *>> $LOG
Log "groups exit=$LASTEXITCODE"

& $PY train_verifier_listwise.py --dataset webqsp --backing rwkv7 --path `
  --cands "$D\path_groups_webqsp_train.jsonl" --epochs 2 `
  --batch-groups 1 --accum 2 --lr 1e-5 --max-len 128 --max-cands 64 --seed 42 `
  --out "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" *>> $LOG
Log "path_verifier exit=$LASTEXITCODE"

& $PY eval_e2e.py --dataset webqsp --backing rwkv7 --split test --beam 200 `
  --topk-rel 15 --max-children 128 --max-hops 2 --ban-generic --beta 2.0 --gamma 0.5 `
  --verifier-top 200 --head-feat-norm double --verifier-feat-norm double `
  --heads "$D\webqsp_rwkv7_heads_vghn.pt" --verifier-path `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" `
  --out "$D\p11_rel_local.json" *>> $LOG
Log "eval_path exit=$LASTEXITCODE"

& $PY mine_train_candidates.py --dataset webqsp --backing rwkv7 --split test `
  --kmax 200 --beam 200 --topk-rel 15 --max-children 128 --max-hops 2 `
  --ban-generic --beta 2.0 --gamma 0.5 `
  --heads "$D\webqsp_rwkv7_heads_vghn.pt" `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_beam200.pt" `
  --dump "$D\mined_webqsp_test_vghn.jsonl" *>> $LOG
Log "mine_test_vghn exit=$LASTEXITCODE"
Log "VGHN chain done"
