param([ValidateSet('smoke','train','benchmark','demo','validate','storage')][string]$Mode='demo',
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$RunArgs)
. (Join-Path $PSScriptRoot 'usb.ps1')
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
if ($env:DOOM_PYTHON) { $python = $env:DOOM_PYTHON }
if (!(Test-Path -LiteralPath $python)) { throw 'Install the project dependencies using training/setup.ps1 first.' }
switch ($Mode) {
    'smoke'     { & $python -B -m training.smoke_policy @RunArgs }
    'train'     { & $python -B -m training.train @RunArgs }
    'benchmark' { & $python -B -m training.evaluate @RunArgs }
    'demo'      { & $python -B -m training.evaluate --demo @RunArgs }
    'validate'  { & $python -B -m training.local_validate @RunArgs }
    'storage'   { & $python -B -m training.storage @RunArgs }
}
exit $LASTEXITCODE
