$ErrorActionPreference = 'Continue'
$env:GOOSERAG_ROGQA = 'E:\GooseRAG\data\rog_qa'
$env:GOOSERAG_RWKV7 = 'E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$LOG = "$D\_m4.log"

function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "M4 start"
& $PY mine_train_candidates.py --dataset metaqa --backing rwkv7 --split train `
  --kmax 200 --beam 200 --topk-rel 15 --max-children 128 --max-hops 3 `
  --ban-generic --beta 2.0 --gamma 0.5 --verifier-feat-norm double --head-feat-norm double `
  --verifier-ft "$D\verifier_ft_metaqa_rwkv7.pt" --dump "$D\mined_metaqa_train.jsonl" *>> $LOG
Log "mine exit=$LASTEXITCODE"

& $PY build_path_groups.py --mined "$D\mined_metaqa_train.jsonl" --out "$D\path_groups_metaqa_train.jsonl" --max-cand 200 *>> $LOG
Log "groups exit=$LASTEXITCODE"

foreach ($s in 42, 43) {
  & $PY train_verifier_listwise.py --dataset metaqa --backing rwkv7 `
    --cands "$D\path_groups_metaqa_train.jsonl" --epochs 1 --batch-groups 1 --accum 2 `
    --lr 1e-5 --max-len 128 --max-cands 64 --seed $s `
    --out "$D\verifier_ft_metaqa_p110_plain_s$s.pt" *>> $LOG
  Log "plain s$s train exit=$LASTEXITCODE"
}

foreach ($s in 42, 43) {
  & $PY train_verifier_listwise.py --dataset metaqa --backing rwkv7 --path `
    --cands "$D\path_groups_metaqa_train.jsonl" --epochs 1 --batch-groups 1 --accum 2 `
    --lr 1e-5 --max-len 128 --max-cands 64 --seed $s `
    --out "$D\verifier_ft_metaqa_path_local_s$s.pt" *>> $LOG
  Log "path s$s train exit=$LASTEXITCODE"
}

$BASE = @('--dataset','metaqa','--backing','rwkv7','--split','test','--beam','200',
          '--topk-rel','15','--max-children','128','--max-hops','3','--ban-generic',
          '--beta','2.0','--gamma','0.5','--verifier-top','200','--no-reach',
          '--head-feat-norm','double','--verifier-feat-norm','double',
          '--heads',"$D\metaqa_rwkv7_heads.pt")

foreach ($s in 42, 43) {
  & $PY eval_e2e.py @BASE --verifier-ft "$D\verifier_ft_metaqa_p110_plain_s$s.pt" `
    --out "$D\p110_metaqa_plain_full_s$s.json" *>> $LOG
  Log "plain full s$s exit=$LASTEXITCODE"
}
foreach ($s in 42, 43) {
  & $PY eval_e2e.py @BASE --verifier-path --verifier-ft "$D\verifier_ft_metaqa_path_local_s$s.pt" `
    --out "$D\p110_metaqa_path_full_s$s.json" *>> $LOG
  Log "path full s$s exit=$LASTEXITCODE"
}
Log "M4 done"
