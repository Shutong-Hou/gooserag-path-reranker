$ErrorActionPreference = "Continue"
$PY = "E:\miniconda3\envs\gooserag\python.exe"
$CODE = "E:\GooseRAG\code\rog_qa_1127"
$D = "E:\GooseRAG\data\rog_qa"
$LOG = "$D\_p2_cwq_train.log"

function Log($m) { $t = Get-Date -Format "HH:mm:ss"; "$t $m" | Out-File -FilePath $LOG -Append -Encoding utf8 }

Log "P2 CWQ truncation training start"

# 1) plain (no path)
& $PY "$CODE\train_verifier_listwise.py" --dataset cwq --split train `
    --cands "$D\_cwq_groups_full.jsonl" `
    --epochs 2 --batch-groups 2 --accum 2 --lr 1e-5 --max-len 160 --seed 42 `
    --out "$D\verifier_ft_cwq_p2_plain.pt" *>> $LOG
Log "plain exit=$LASTEXITCODE"

# 2) full path
& $PY "$CODE\train_verifier_listwise.py" --dataset cwq --split train `
    --cands "$D\_cwq_groups_full.jsonl" --path `
    --epochs 2 --batch-groups 2 --accum 2 --lr 1e-5 --max-len 160 --seed 42 `
    --out "$D\verifier_ft_cwq_p2_full.pt" *>> $LOG
Log "full exit=$LASTEXITCODE"

# 3) hop-1 only
& $PY "$CODE\train_verifier_listwise.py" --dataset cwq --split train `
    --cands "$D\_cwq_groups_hop1.jsonl" --path `
    --epochs 2 --batch-groups 2 --accum 2 --lr 1e-5 --max-len 160 --seed 42 `
    --out "$D\verifier_ft_cwq_p2_hop1.pt" *>> $LOG
Log "hop1 exit=$LASTEXITCODE"

# 4) coarse relation types
& $PY "$CODE\train_verifier_listwise.py" --dataset cwq --split train `
    --cands "$D\_cwq_groups_type.jsonl" --path `
    --epochs 2 --batch-groups 2 --accum 2 --lr 1e-5 --max-len 160 --seed 42 `
    --out "$D\verifier_ft_cwq_p2_type.pt" *>> $LOG
Log "type exit=$LASTEXITCODE"

Log "P2 CWQ training done"
