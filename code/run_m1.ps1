$ErrorActionPreference = 'Continue'
$env:GOOSERAG_ROGQA = 'E:\GooseRAG\data\rog_qa'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$PY = 'E:\miniconda3\envs\gooserag\python.exe'
$D = 'E:\GooseRAG\data\rog_qa'
$M = 'E:\GooseRAG\models'
$LOG = "$D\_m1b.log"
function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $LOG -Append }

Log "M1b start"
$G = "$D\path_groups_webqsp_train.jsonl"
$POOL = "$D\mined_webqsp_test_vghn.jsonl"

& $PY ext_rerank.py train --model "$M\bert-base-uncased" --groups $G --epochs 2 --accum 2 --max-cands 64 --max-len 128 --out "$D\ext_bert_plain.pt" *>> $LOG
Log "bert plain train exit=$LASTEXITCODE"
& $PY ext_rerank.py train --model "$M\bert-base-uncased" --groups $G --path --epochs 2 --accum 2 --max-cands 64 --max-len 128 --out "$D\ext_bert_path.pt" *>> $LOG
Log "bert path train exit=$LASTEXITCODE"
& $PY ext_rerank.py train --model "$M\t5-base" --groups $G --path --epochs 2 --accum 2 --max-cands 64 --max-len 128 --out "$D\ext_t5_path.pt" *>> $LOG
Log "t5 path train exit=$LASTEXITCODE"

& $PY ext_rerank.py eval --model "$M\bert-base-uncased" --ckpt "$D\ext_bert_plain.pt" --mined $POOL --max-len 160 --out "$D\ext_bert_plain_test.json" *>> $LOG
Log "bert plain eval exit=$LASTEXITCODE"
& $PY ext_rerank.py eval --model "$M\bert-base-uncased" --ckpt "$D\ext_bert_path.pt" --mined $POOL --path --max-len 160 --out "$D\ext_bert_path_test.json" *>> $LOG
Log "bert path eval exit=$LASTEXITCODE"
& $PY ext_rerank.py eval --model "$M\t5-base" --ckpt "$D\ext_t5_path.pt" --mined $POOL --path --max-len 160 --out "$D\ext_t5_path_test.json" *>> $LOG
Log "t5 path eval exit=$LASTEXITCODE"

& $PY ext_rerank.py llm --model "$M\Qwen2.5-0.5B-Instruct" --mined $POOL --path --max-pool 100 --out "$D\ext_qwen_path_test.json" *>> $LOG
Log "qwen path eval exit=$LASTEXITCODE"
& $PY ext_rerank.py llm --model "$M\Qwen2.5-0.5B-Instruct" --mined $POOL --max-pool 100 --out "$D\ext_qwen_plain_test.json" *>> $LOG
Log "qwen plain eval exit=$LASTEXITCODE"
Log "M1b done"
