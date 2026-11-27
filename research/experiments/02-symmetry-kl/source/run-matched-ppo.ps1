param(
    [Parameter(Mandatory)][ValidateSet('ppo-control','symmetry-kl')][string]$Arm
)

$ErrorActionPreference = 'Stop'
$campaign = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$project = (Resolve-Path -LiteralPath $env:DOOM_PROJECT_ROOT).Path
$repo = Join-Path $campaign 'policy-worktree'
$python = (Resolve-Path -LiteralPath $env:DOOM_PYTHON).Path
$config = Join-Path $campaign 'training-plan.json'
$plan = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json -DateKind String
$armPlan = $plan.arms.$Arm
$trainingCutoff = [DateTimeOffset]::Parse($plan.stop.training_cutoff_ist)
$secondsRemaining = [int][Math]::Floor(($trainingCutoff - [DateTimeOffset]::Now).TotalSeconds)
if ($secondsRemaining -lt 160) { throw 'Insufficient time before the training cutoff.' }
if ((Get-FileHash -LiteralPath $plan.parent).Hash -ne $plan.parent_sha256) {
    throw 'The C749 initialization hash differs.'
}
$frozenWeight = Join-Path $project 'submission_artifacts\final-code-staging\submission\weights\policy.pt'
if ((Get-FileHash -LiteralPath $frozenWeight).Hash -ne $plan.release_weight_sha256) {
    throw 'The released C749 weight hash differs.'
}

$env:DOOM_WORK = $campaign
Set-Location -LiteralPath $repo
. .\training\usb.ps1
$env:PYTHONPATH = Join-Path $repo 'env'
$env:DOOM_INTERNAL_LIMIT_BYTES = '7000000000'
$env:DOOM_STORAGE_BASELINE = Join-Path $project 'logs\storage-baseline.json'
$env:DOOM_STORAGE_BASELINE_SHA256 = 'a158265c2cae0ad6d50818bae6e01387d6215f8a20a494115128bdc3a7fa9ccb'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$storage = & $python -B -c 'import json; from training.storage import snapshot; print(json.dumps(snapshot()))'
if ($LASTEXITCODE -ne 0) { throw 'Unable to verify drive and memory limits.' }
$measured = $storage | ConvertFrom-Json
if ($measured.internal_delta_bytes -gt $plan.stop.launch_c_decline_bytes -or
    $measured.internal_free_bytes -lt 2000000000 -or $measured.external_free_bytes -lt 500000000 -or
    $measured.physical_available_bytes -lt 1073741824 -or $measured.commit_available_bytes -lt 1500000000) {
    throw "Storage or memory gate blocked this arm. $storage"
}

$stopFile = Join-Path $campaign 'stop-optional.flag'
$logs = Join-Path $campaign 'logs'
$startRecord = Join-Path $logs "$Arm-launch.json"
$endRecord = Join-Path $logs "$Arm-exit.json"
if (Test-Path -LiteralPath $stopFile) { throw 'The organizer handoff stop is active.' }
if ((Test-Path -LiteralPath $startRecord) -or (Test-Path -LiteralPath $endRecord) -or
    (Test-Path -LiteralPath (Join-Path $campaign "checkpoints\$Arm"))) {
    throw 'This arm already has artifacts.'
}
[void][IO.Directory]::CreateDirectory($logs)
$started = [DateTimeOffset]::Now.ToOffset([TimeSpan]::FromMinutes(330))
[ordered]@{
    arm = $Arm
    started_ist = $started.ToString('o')
    plan_sha256 = (Get-FileHash -LiteralPath $config).Hash.ToLowerInvariant()
    parent_sha256 = $plan.parent_sha256
    settings = $armPlan
    storage = $measured
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $startRecord -Encoding utf8
$arguments = @(
    '-B','-m','training.train','--headless','--parallel','--no-render-buffers','--memory','stack','--device','cpu',
    '--seed',"$($plan.seed)",'--workers',"$($plan.workers)",'--horizon',"$($plan.horizon)",'--updates',"$($plan.updates)",
    '--ticks',"$($plan.tick_cap)",'--learning-rate',"$($plan.learning_rate)",'--initialize',"$($plan.parent)",'--run-name',$Arm,
    '--curriculum',"$($plan.curriculum)",'--reward-profile',"$($plan.reward_profile)",'--checkpoint-seconds','40',
    '--max-seconds',"$([Math]::Min(270,$secondsRemaining))",'--stop-file',$stopFile,
    '--storage-stop-bytes',"$($plan.stop.soft_c_decline_bytes)",
    '--reference-coef',"$($armPlan.reference_coefficient)",'--symmetry-coef',"$($armPlan.symmetry_coefficient)"
)
& $python @arguments
$code = $LASTEXITCODE
[ordered]@{
    arm = $Arm
    finished_ist = [DateTimeOffset]::Now.ToOffset([TimeSpan]::FromMinutes(330)).ToString('o')
    exit_code = $code
} | ConvertTo-Json | Set-Content -LiteralPath $endRecord -Encoding utf8
exit $code
