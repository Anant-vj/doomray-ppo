$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $env:DOOM_WORK) {
    $env:DOOM_WORK = (Resolve-Path (Join-Path $repo '..\..')).Path
}
$drive = [System.IO.DriveInfo]::new([IO.Path]::GetPathRoot($env:DOOM_WORK))
if ($drive.Name -eq ([IO.Path]::GetPathRoot($env:SystemRoot))) { throw 'Choose the external drive for DOOM_WORK.' }
$settings = @{
    TEMP='temp'; TMP='temp'; TMPDIR='temp'; UV_CACHE_DIR='caches/uv';
    UV_PYTHON_INSTALL_DIR='runtimes'; UV_PROJECT_ENVIRONMENT='environments/competition';
    PIP_CACHE_DIR='caches/pip'; TORCH_HOME='caches/torch'; TORCH_EXTENSIONS_DIR='caches/torch/ext';
    HF_HOME='caches/huggingface'; HF_HUB_CACHE='caches/huggingface/hub';
    XDG_CACHE_HOME='caches/general'; CUDA_CACHE_PATH='caches/cuda';
    TRITON_CACHE_DIR='caches/triton'; TORCHINDUCTOR_CACHE_DIR='caches/inductor';
    PYTHONPYCACHEPREFIX='caches/pyc'; MPLCONFIGDIR='caches/matplotlib'
}
foreach ($pair in $settings.GetEnumerator()) {
    $target = Join-Path $env:DOOM_WORK $pair.Value
    [IO.Directory]::CreateDirectory($target) | Out-Null
    [Environment]::SetEnvironmentVariable($pair.Key,$target,'Process')
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:FARAMA_NOTIFICATIONS_DISABLED='1'
$env:PYGAME_HIDE_SUPPORT_PROMPT='1'
Set-Location -LiteralPath $repo
