param(
    [Parameter(Mandatory)][string]$ValidationFamily,
    [int]$Epochs=10
)

$ErrorActionPreference='Stop'
$campaign=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$project=(Resolve-Path -LiteralPath $env:DOOM_PROJECT_ROOT).Path
$repo=Join-Path $campaign 'policy-worktree'
$python=(Resolve-Path -LiteralPath $env:DOOM_PYTHON).Path
$parent=Join-Path $project 'checkpoints\mission-archive\stack-26-update-621-policy.pt'
if ((Get-FileHash -LiteralPath $parent).Hash -ne '5D90001F01BEB4E43560FDE4037667E7F440E797B6F86F2EE91A1E12C40D46E8') {
    throw 'Approved experiment parent SHA-256 differs.'
}
$hardStop=[DateTimeOffset]::Parse('2026-11-27T05:56:00+05:30')
$duration=[int][Math]::Floor(($hardStop-[DateTimeOffset]::Now).TotalSeconds)
if ($duration -lt 30) { throw 'Training window has passed.' }

$env:DOOM_WORK=$campaign
Set-Location -LiteralPath $repo
. .\training\usb.ps1
$env:PYTHONPATH=Join-Path $repo 'env'
$env:DOOM_INTERNAL_LIMIT_BYTES='7000000000'
$env:DOOM_STORAGE_BASELINE=Join-Path $project 'logs\storage-baseline.json'
$env:DOOM_STORAGE_BASELINE_SHA256='a158265c2cae0ad6d50818bae6e01387d6215f8a20a494115128bdc3a7fa9ccb'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'

$state=& $python -B -c "import json; from training.storage import snapshot; print(json.dumps(snapshot()))"
if ($LASTEXITCODE -ne 0) { throw 'Could not confirm storage and memory.' }
$measure=$state|ConvertFrom-Json
if ($measure.internal_delta_bytes -gt 6500000000 -or $measure.internal_free_bytes -lt 2000000000 -or
    $measure.external_free_bytes -lt 500000000 -or $measure.physical_available_bytes -lt 1073741824 -or
    $measure.commit_available_bytes -lt 1500000000) {
    throw "Supervised preflight blocked. $state"
}
$seconds=[Math]::Min(360,$duration)
& $python -B -m training.behavioral_clone --initialize $parent --validation-family $ValidationFamily --epochs $Epochs --max-seconds $seconds
exit $LASTEXITCODE
