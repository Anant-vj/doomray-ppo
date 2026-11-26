param([switch]$Preflight)

$ErrorActionPreference='Stop'
$assessment=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$project='D:\Doom\DevForge_DOOM'
$frozen=Join-Path $project 'submission_artifacts\final-code-staging'
$env:DOOM_WORK=$project
Set-Location -LiteralPath $frozen
. .\training\usb.ps1
$python=Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$env:NUMEXPR_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONPATH=Join-Path $frozen 'env'
Set-Location -LiteralPath (Join-Path $assessment 'episodes')
$mode = if ($Preflight) { 'preflight' } else { 'run' }
& $python -B (Join-Path $assessment 'evaluator\run_matched.py') $mode
exit $LASTEXITCODE
