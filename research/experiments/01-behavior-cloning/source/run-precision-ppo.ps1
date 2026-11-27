param(
    [Parameter(Mandatory)][string]$RunName,
    [Parameter(Mandatory)][string]$Initialize,
    [int]$Updates=128,
    [int]$Workers=4,
    [int]$Seconds=900,
    [switch]$Preflight
)

$ErrorActionPreference='Stop'
$campaign=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$project=(Resolve-Path -LiteralPath $env:DOOM_PROJECT_ROOT).Path
$repo=Join-Path $campaign 'policy-worktree'
$python=(Resolve-Path -LiteralPath $env:DOOM_PYTHON).Path
$source=(Resolve-Path -LiteralPath $Initialize).Path
if ([IO.Path]::GetPathRoot($source) -ne ('D:' + [IO.Path]::DirectorySeparatorChar)) { throw 'The initial model must be on D:.' }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$' -or $Updates -lt 1 -or $Workers -lt 2 -or $Workers -gt 4) {
    throw 'Expected a unique run name, a positive update count and two to four workers.'
}
$hardStop=[DateTimeOffset]::Parse('2026-11-27T05:56:00+05:30')
$duration=[int][Math]::Floor(($hardStop-[DateTimeOffset]::Now).TotalSeconds)
$Seconds=[Math]::Min($Seconds,$duration)
if ($Seconds -lt 30) { throw 'Training window has passed.' }

$env:DOOM_WORK=$campaign
Set-Location -LiteralPath $repo
. .\training\usb.ps1
$env:PYTHONPATH=Join-Path $repo 'env'
$env:DOOM_INTERNAL_LIMIT_BYTES='7000000000'
$env:DOOM_STORAGE_BASELINE=Join-Path $project 'logs\storage-baseline.json'
$env:DOOM_STORAGE_BASELINE_SHA256='a158265c2cae0ad6d50818bae6e01387d6215f8a20a494115128bdc3a7fa9ccb'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'

$reading=& $python -B -c "import json; from training.storage import snapshot; print(json.dumps(snapshot()))"
if ($LASTEXITCODE -ne 0) { throw 'Could not confirm storage and memory.' }
$measure=$reading|ConvertFrom-Json
$memoryFloor=805306368L + ($Workers * 67108864L)
if ($measure.internal_delta_bytes -gt 6500000000 -or $measure.internal_free_bytes -lt 2000000000 -or
    $measure.external_free_bytes -lt 500000000 -or $measure.physical_available_bytes -lt $memoryFloor -or
    $measure.commit_available_bytes -lt 1500000000) {
    throw "Parallel training preflight blocked. $reading"
}
$stopFile=Join-Path $campaign "$RunName.stop"
if (Test-Path -LiteralPath $stopFile) { throw "Stop file already exists: $stopFile" }
$manifest=[ordered]@{
    name=$RunName
    seed=2207
    workers=$Workers
    horizon=64
    updates=$Updates
    target_transitions=$Updates*$Workers*64
    max_seconds=$Seconds
    learning_rate=0.00008
    ticks=1050
    profile='precision'
    initial_path=$source
    initial_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    preflight=$measure
}
$manifest|ConvertTo-Json -Depth 4
if ($Preflight) { return }
$logs=Join-Path $campaign 'logs'
[void][IO.Directory]::CreateDirectory($logs)
$manifest|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $logs "$RunName-launch.json") -Encoding utf8
$arguments=@(
    '-B','-m','training.train','--headless','--parallel','--no-render-buffers',
    '--memory','stack','--device','cpu','--seed','2207','--workers',"$Workers",'--horizon','64',
    '--ticks','1050','--updates',"$Updates",'--initialize',$source,'--learning-rate','0.00008',
    '--run-name',$RunName,'--reward-profile','precision','--curriculum','precision',
    '--checkpoint-seconds','90','--max-seconds',"$Seconds",'--stop-file',$stopFile,
    '--storage-stop-bytes','6800000000','--memory-stop-bytes','536870912','--commit-stop-bytes','1000000000'
)
& $python @arguments
exit $LASTEXITCODE
