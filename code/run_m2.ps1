$ErrorActionPreference = 'Continue'
$env:GOOSERAG_ROGQA = 'E:\GooseRAG\data\rog_qa'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = ''
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$LOG = "$D\_m2.log"
function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "M2 start"
& $PY rog_reader_eval.py --limit 20 --n-beam 3 --topk 5 `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" `
  --out "$D\rog_reader_smoke.jsonl" *>> $LOG
Log "smoke exit=$LASTEXITCODE"

& $PY rog_reader_eval.py --limit 0 --n-beam 3 --topk 5 `
  --verifier-ft "$D\verifier_ft_webqsp_rwkv7_listwise_path_s42.pt" `
  --out "$D\rog_reader_full.jsonl" *>> $LOG
Log "full exit=$LASTEXITCODE"
Log "M2 done"
