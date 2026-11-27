param(
    [Parameter(Mandatory)][ValidateSet('ppo-control','pbrs','pbrs-anchor')][string]$Arm
)

$ErrorActionPreference = 'Stop'
$project = $env:DOOMRAY_PROJECT_ROOT
if (-not $project) { throw 'Set the private project root first.' }
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repo = Join-Path $root 'policy-worktree'
$python = Join-Path $project 'environments\competition\Scripts\python.exe'
$planPath = Join-Path $root 'training-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json -DateKind String
$until = [DateTimeOffset]::Parse($plan.training_end_latest_ist)
$remaining = [int][Math]::Floor(($until-[DateTimeOffset]::Now).TotalSeconds)
if ($remaining -lt 210) { throw 'Insufficient time to start another matched arm.' }
$weight = Join-Path $project 'submission_artifacts\final-code-staging\submission\weights\policy.pt'
$anchor = Join-Path $root 'anchors\anchor-data.npz'
$protocol = Join-Path $root 'screening\manifests\protocol.json'
$backtrack = Join-Path $root 'training-maps\red_key_backtrack.wad'
$expected = @{
    $plan.parent = $plan.parent_sha256
    $weight = $plan.release_weight_sha256
    $anchor = $plan.anchors_sha256
    $protocol = $plan.screen_protocol_sha256
    $backtrack = $plan.backtrack_sha256
}
foreach ($file in $expected.Keys) {
    if ((Get-FileHash -LiteralPath $file).Hash -ne $expected[$file]) { throw "Hash changed for $file" }
}

$env:DOOM_WORK = $root
Set-Location -LiteralPath $repo
. .\training\usb.ps1
$env:PYTHONPATH = Join-Path $repo 'env'
$env:DOOM_INTERNAL_LIMIT_BYTES = '7000000000'
$env:DOOM_STORAGE_BASELINE = Join-Path $project 'logs\storage-baseline.json'
$env:DOOM_STORAGE_BASELINE_SHA256 = 'a158265c2cae0ad6d50818bae6e01387d6215f8a20a494115128bdc3a7fa9ccb'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$reading = & $python -B -c 'import json; from training.storage import snapshot; print(json.dumps(snapshot()))'
if ($LASTEXITCODE -ne 0) { throw 'Unable to verify the actual storage baseline.' }
$storage = $reading | ConvertFrom-Json
if ($storage.internal_delta_bytes -gt 6500000000 -or $storage.internal_free_bytes -lt 2000000000 -or
    $storage.external_free_bytes -lt 500000000 -or $storage.physical_available_bytes -lt 1073741824 -or
    $storage.commit_available_bytes -lt 1500000000) {
    throw "Storage or memory preflight blocked $Arm. $reading"
}
$stop = Join-Path $root 'stop-optional.flag'
$logs = Join-Path $root 'logs'
$start = Join-Path $logs "$Arm-launch.json"
$end = Join-Path $logs "$Arm-exit.json"
if (Test-Path -LiteralPath $stop) { throw 'Organizer handoff stop is active.' }
if ((Test-Path -LiteralPath $start) -or (Test-Path -LiteralPath $end) -or
    (Test-Path -LiteralPath (Join-Path $root "checkpoints\$Arm"))) {
    throw 'This arm already has artifacts.'
}
[void][IO.Directory]::CreateDirectory($logs)
[ordered]@{
    arm = $Arm
    started_ist = [DateTimeOffset]::Now.ToOffset([TimeSpan]::FromMinutes(330)).ToString('o')
    plan_sha256 = (Get-FileHash -LiteralPath $planPath).Hash.ToLowerInvariant()
    parent_sha256 = $plan.parent_sha256
    storage = $storage
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $start -Encoding utf8
$arguments = @(
    '-B','-m','training.train','--headless','--parallel','--no-render-buffers','--memory','stack','--device','cpu',
    '--seed',"$($plan.seed)",'--workers',"$($plan.workers)",'--horizon',"$($plan.horizon)",'--updates',"$($plan.updates)",
    '--ticks',"$($plan.ticks)",'--learning-rate',"$($plan.learning_rate)",'--initialize',"$($plan.parent)",'--run-name',$Arm,
    '--curriculum','mixed','--reward-profile','progression','--checkpoint-seconds','45',
    '--max-seconds',"$([Math]::Min(420,$remaining))",'--stop-file',$stop,
    '--storage-stop-bytes','6800000000','--credit-mode',$Arm
)
if ($Arm -eq 'pbrs-anchor') {
    $arguments += @('--anchor-file',$anchor,'--anchor-sha256',"$($plan.anchors_sha256)",'--anchor-beta',"$($plan.anchor_beta)")
}
& $python @arguments
$code = $LASTEXITCODE
[ordered]@{
    arm = $Arm
    finished_ist = [DateTimeOffset]::Now.ToOffset([TimeSpan]::FromMinutes(330)).ToString('o')
    exit_code = $code
} | ConvertTo-Json | Set-Content -LiteralPath $end -Encoding utf8
exit $code
