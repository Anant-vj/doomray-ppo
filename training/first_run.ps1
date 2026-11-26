# Complete the first local run from the restored template on the external drive.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'usb.ps1')
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$ready = $false
if (Test-Path -LiteralPath $python) {
    & $python -B -c "import torch, vizdoom, gymnasium, doom_env"
    $ready = $LASTEXITCODE -eq 0
}
if (-not $ready) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
    if ($LASTEXITCODE) { throw 'Setup failed. Check the download error above.' }
}
& $python -B -m training.smoke_policy
if ($LASTEXITCODE) { throw 'The policy regression check failed.' }
# Start a short, real rollout immediately so the operator sees the game.
$folder = Join-Path $env:DOOM_WORK 'checkpoints\stack-17'
$resume = Join-Path $folder 'resume.pt'
$trainArgs = @('--updates','8','--workers','2','--horizon','64','--ticks','420')
if (Test-Path $resume) { $trainArgs += @('--resume',$resume) }
& $python -B -m training.train @trainArgs
if ($LASTEXITCODE) { throw 'Training failed. Check the engine or trainer error above.' }
$evalArgs = @('--random','--promote','--ticks','420','--seeds','1001','1002')
foreach ($candidate in (Get-ChildItem -LiteralPath $folder -Filter 'policy-*.pt')) {
    $evalArgs += @('--checkpoint',$candidate.FullName)
}
& $python -B -m training.evaluate @evalArgs
if ($LASTEXITCODE) { throw 'Development evaluation failed.' }
& $python -B -m training.local_validate --device cpu
exit $LASTEXITCODE
