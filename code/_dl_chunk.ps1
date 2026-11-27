param([string]$Set = "all")
$ErrorActionPreference = 'Continue'
$root = 'E:\GooseRAG\models'
$base = 'https://hf-mirror.com'
$chunk = 16MB
$log = "$root\_curl_$Set.log"

function Log($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $log -Append }

function Get-RemoteLen($url) {
  $h = curl.exe -sIL --max-time 30 $url
  $m = $h | Select-String -Pattern '^content-length:\s*(\d+)' | Select-Object -Last 1
  if ($m) { return [long]$m.Matches.Groups[1].Value }
  return 0
}

function Download-Chunked($url, $out) {
  New-Item -ItemType Directory -Path (Split-Path $out -Parent) -Force | Out-Null
  $len = Get-RemoteLen $url
  if ($len -le 0) { Log "NO_LEN $url"; return }
  $have = 0
  if (Test-Path $out) { $have = (Get-Item $out).Length }
  if ($have -eq $len) { Log "EXISTS $out $([math]::Round($len/1MB,1))MB"; return }
  if ($have -gt $len) { Remove-Item $out -Force; $have = 0 }
  Log "START $([IO.Path]::GetFileName($out)) from $([math]::Round($have/1MB,1))/$([math]::Round($len/1MB,1))MB"
  $part = "$out.part"
  while ($have -lt $len) {
    $end = [long]$have + [long]$chunk - 1
    if ($end -ge $len) { $end = [long]$len - 1 }
    $want = $end - $have + 1
    $ok = $false
    for ($try = 1; $try -le 8; $try++) {
      Remove-Item $part -Force -ErrorAction SilentlyContinue
      curl.exe -sL --max-time 150 --speed-limit 1024 --speed-time 40 --retry 0 -r "$have-$end" -o $part $url
      if ((Test-Path $part) -and ((Get-Item $part).Length -eq $want)) { $ok = $true; break }
      Log "retry $try chunk $have..$end"
      Start-Sleep -Seconds (5 * $try)
    }
    if (-not $ok) { Log "CHUNK_FAIL $([IO.Path]::GetFileName($out)) at $have"; return }
    $fs = [IO.File]::Open($out, [IO.FileMode]::Append, [IO.FileAccess]::Write)
    $ps = [IO.File]::OpenRead($part)
    $ps.CopyTo($fs); $ps.Close(); $fs.Close()
    Remove-Item $part -Force -ErrorAction SilentlyContinue
    $have = $end + 1
    Log "$([IO.Path]::GetFileName($out)) $([math]::Round(100.0*$have/$len,1))% ($([math]::Round($have/1MB,1))/$([math]::Round($len/1MB,1))MB)"
  }
  $got = (Get-Item $out).Length
  if ($got -eq $len) { Log "DONE $out $([math]::Round($len/1MB,1))MB" } else { Log "SIZE_MISMATCH $out $got/$len" }
}

$small = @(
  @('Qwen/Qwen2.5-0.5B-Instruct', "$root\Qwen2.5-0.5B-Instruct\model.safetensors"),
  @('t5-base', "$root\t5-base\model.safetensors")
)
$rog = @(
  @('rmanluo/RoG', "$root\RoG\model-00001-of-00002.safetensors"),
  @('rmanluo/RoG', "$root\RoG\model-00002-of-00002.safetensors")
)
$jobs = @()
if ($Set -eq 'small') { $jobs = $small }
elseif ($Set -eq 'rog') { $jobs = $rog }
else { $jobs = $small + $rog }
foreach ($j in $jobs) {
  $u = "$base/$($j[0])/resolve/main/$([IO.Path]::GetFileName($j[1]))"
  Download-Chunked $u $j[1]
}
Log "ALL_DONE_$Set"
