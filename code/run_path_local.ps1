$ErrorActionPreference = 'Continue'
$env:GOOSERAG_ROGQA = 'E:\GooseRAG\data\rog_qa'
$env:GOOSERAG_RWKV7 = 'E:\GooseRAG\models\RWKV7-Goose-World2.8-0.1B-HF'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$LOG = "$D\_path_local.log"
function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "path-local start"
& $PY train_verifier_listwise.py --dataset webqsp --backing rwkv7 --path `
  --cands "$D\path_groups_webqsp_train.jsonl" --epochs 2 `
  --batch-groups 1 --accum 2 --lr 1e-5 --max-len 128 --max-cands 64 --seed 42 `
  --out "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" *>> $LOG
Log "train exit=$LASTEXITCODE"

if ($LASTEXITCODE -eq 0) {
  & $PY eval_e2e.py --dataset webqsp --backing rwkv7 --split test --beam 200 `
    --topk-rel 15 --max-children 128 --max-hops 2 --ban-generic --beta 2.0 --gamma 0.5 `
    --verifier-top 200 --head-feat-norm double --verifier-feat-norm double `
    --heads "$D\webqsp_rwkv7_heads_vghn.pt" --verifier-path `
    --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" `
    --out "$D\p11_rel_local.json" *>> $LOG
  Log "eval exit=$LASTEXITCODE"
}
Log "path-local done"
