[CmdletBinding(DefaultParameterSetName='Menu')]
param(
    [Parameter(ParameterSetName='Menu')]
    [switch]$Menu,
    [Parameter(Mandatory,ParameterSetName='Live')]
    [ValidateSet('C749','Stack25-u174','Stack26-u621')]
    [string]$Model,
    [Parameter(ParameterSetName='Live')]
    [ValidateRange(1,4)]
    [int]$Tier,
    [Parameter(ParameterSetName='Live')]
    [ValidatePattern('^MAP[0-9]{2}$')]
    [string]$Map,
    [Parameter(ParameterSetName='Live')]
    [ValidateRange(1,2147483647)]
    [int]$Seed,
    [Parameter(ParameterSetName='Live')]
    [switch]$AllTiers,
    [Parameter(Mandatory,ParameterSetName='Replay')]
    [ValidatePattern('^[A-Za-z0-9-]+$')]
    [string]$Replay,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$assessment = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$project = (Resolve-Path -LiteralPath (Join-Path $assessment '..\..')).Path
if ([IO.Path]::GetPathRoot($assessment) -ne 'D:\') { throw 'Run the examination from D:.' }
$env:DOOM_WORK = $project
Push-Location
try { . (Join-Path $project 'workspace\devforge-doom-template-main\training\usb.ps1') }
finally { Pop-Location }
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$viewer = Join-Path $assessment 'graphical-exam\examine.py'
$request = @()
switch ($PSCmdlet.ParameterSetName) {
    'Menu' { $request = @('--menu') }
    'Replay' { $request = @('--replay', $Replay) }
    'Live' {
        $canonical = @{'c749'='C749';'stack25-u174'='Stack25-u174';'stack26-u621'='Stack26-u621'}
        $request = @('--model', $canonical[$Model.ToLowerInvariant()])
        if ($Tier) { $request += @('--tier', $Tier) }
        if ($Map) { $request += @('--map', $Map.ToUpperInvariant()) }
        if ($Seed) { $request += @('--seed', $Seed) }
        if ($AllTiers) { $request += '--all-tiers' }
    }
}

$result = & $python -B $viewer @request --preflight
if ($LASTEXITCODE -ne 0) { throw 'Examination preflight failed.' }
$result
if ($Preflight) { return }
if (-not (($result | Out-String | ConvertFrom-Json).launch_allowed)) {
    throw 'Examination storage or memory gate blocked the start.'
}

$logs = Join-Path $assessment 'graphical-exam\launch-logs'
[void][IO.Directory]::CreateDirectory($logs)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
$start = @{
    FilePath = $python
    ArgumentList = @('-B', ('"{0}"' -f $viewer)) + $request
    WorkingDirectory = $assessment
    WindowStyle = 'Normal'
    PassThru = $true
    RedirectStandardOutput = Join-Path $logs "$stamp-stdout.txt"
    RedirectStandardError = Join-Path $logs "$stamp-stderr.txt"
}
$process = Start-Process @start
Write-Output "Graphical examination launcher PID $($process.Id). Session log: $($start.RedirectStandardOutput)"
