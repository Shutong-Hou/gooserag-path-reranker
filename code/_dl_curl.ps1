$ErrorActionPreference = 'Continue'
$root = 'E:\GooseRAG\models'
$base = 'https://hf-mirror.com'
$items = @(
  @('bert-base-uncased', 'config.json', 'config.json'),
  @('bert-base-uncased', 'vocab.txt', 'vocab.txt'),
  @('bert-base-uncased', 'tokenizer.json', 'tokenizer.json'),
  @('bert-base-uncased', 'tokenizer_config.json', 'tokenizer_config.json'),
  @('bert-base-uncased', 'model.safetensors', 'model.safetensors'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'config.json', 'config.json'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'generation_config.json', 'generation_config.json'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'tokenizer.json', 'tokenizer.json'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'tokenizer_config.json', 'tokenizer_config.json'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'vocab.json', 'vocab.json'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'merges.txt', 'merges.txt'),
  @('Qwen/Qwen2.5-0.5B-Instruct', 'model.safetensors', 'model.safetensors'),
  @('t5-base', 'config.json', 'config.json'),
  @('t5-base', 'spiece.model', 'spiece.model'),
  @('t5-base', 'tokenizer_config.json', 'tokenizer_config.json'),
  @('t5-base', 'special_tokens_map.json', 'special_tokens_map.json'),
  @('t5-base', 'model.safetensors', 'model.safetensors'),
  @('rmanluo/RoG', 'config.json', 'config.json'),
  @('rmanluo/RoG', 'generation_config.json', 'generation_config.json'),
  @('rmanluo/RoG', 'tokenizer.model', 'tokenizer.model'),
  @('rmanluo/RoG', 'tokenizer_config.json', 'tokenizer_config.json'),
  @('rmanluo/RoG', 'special_tokens_map.json', 'special_tokens_map.json'),
  @('rmanluo/RoG', 'added_tokens.json', 'added_tokens.json'),
  @('rmanluo/RoG', 'model.safetensors.index.json', 'model.safetensors.index.json'),
  @('rmanluo/RoG', 'model-00001-of-00002.safetensors', 'model-00001-of-00002.safetensors'),
  @('rmanluo/RoG', 'model-00002-of-00002.safetensors', 'model-00002-of-00002.safetensors')
)
foreach ($it in $items) {
  $repo = $it[0]; $fn = $it[1]; $name = $it[2]
  if ($repo -eq 'rmanluo/RoG') { $dir = Join-Path $root 'RoG' } else { $dir = Join-Path $root ($repo.Split('/')[-1]) }
  New-Item -ItemType Directory -Path $dir -Force | Out-Null
  $out = Join-Path $dir $name
  $url = "$base/$repo/resolve/main/$fn"
  "[$(Get-Date -Format 'HH:mm:ss')] $repo/$fn" | Tee-Object -FilePath "$root\_curl.log" -Append
  curl.exe -L --retry 5 --retry-delay 3 -C - -o $out $url 2>&1 | Select-Object -Last 1 | Tee-Object -FilePath "$root\_curl.log" -Append
}
'ALL_DONE' | Tee-Object -FilePath "$root\_curl.log" -Append
