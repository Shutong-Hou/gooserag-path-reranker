$ErrorActionPreference = 'Continue'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$M = 'E:\GooseRAG\models'
$LOG = "$D\_m1c.log"
function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "M1c start"
& $PY ext_rerank.py train --model "$M\t5-base" --groups "$D\path_groups_webqsp_train.jsonl" --path --epochs 2 --accum 2 --max-cands 64 --max-len 128 --out "$D\ext_t5_path.pt" *>> $LOG
Log "t5 train exit=$LASTEXITCODE"
& $PY ext_rerank.py eval --model "$M\t5-base" --ckpt "$D\ext_t5_path.pt" --mined "$D\mined_webqsp_test_vghn.jsonl" --path --max-len 160 --out "$D\ext_t5_path_test.json" *>> $LOG
Log "t5 eval exit=$LASTEXITCODE"
& $PY ext_rerank.py llm --model "$M\Qwen2.5-0.5B-Instruct" --mined "$D\mined_webqsp_test_vghn.jsonl" --path --max-pool 100 --out "$D\ext_qwen_path_test.json" *>> $LOG
Log "qwen path exit=$LASTEXITCODE"
& $PY ext_rerank.py llm --model "$M\Qwen2.5-0.5B-Instruct" --mined "$D\mined_webqsp_test_vghn.jsonl" --max-pool 100 --out "$D\ext_qwen_plain_test.json" *>> $LOG
Log "qwen plain exit=$LASTEXITCODE"
Log "M1c done"
