$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'usb.ps1')
python -B -m training.storage
if ($LASTEXITCODE) { throw 'Storage check failed.' }
uv sync --locked --no-dev --python 3.13 --project $repo
if ($LASTEXITCODE) { throw 'Downloading Python or the locked simulator dependencies failed.' }
python -B -m training.storage
if ($LASTEXITCODE) { throw 'Storage check failed.' }
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
uv pip install --python $python torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE) { throw 'The CPU PyTorch installation failed.' }
& $python -B -m training.storage
if ($LASTEXITCODE) { throw 'Storage check failed.' }
& $python -B -c "import torch, vizdoom, gymnasium; print(torch.__version__, vizdoom.__version__, gymnasium.__version__)"
exit $LASTEXITCODE
