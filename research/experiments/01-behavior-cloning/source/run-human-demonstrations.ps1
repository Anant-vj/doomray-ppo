[CmdletBinding()]
param(
    [ValidateSet('BasicAim','CenterCombat','CorridorCombat','HealthRun','KeyDoor','Backtrack','TwoRooms')]
    [string]$Layout = 'KeyDoor',
    [ValidateRange(1,2147483000)]
    [int]$Seed = 6101,
    [string]$Teammate,
    [string]$PermissionReference,
    [switch]$ConfirmTeamGenerated,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$project = (Resolve-Path -LiteralPath (Join-Path $workspace '..\..')).Path
if ([IO.Path]::GetPathRoot($workspace) -ne ('D:' + [IO.Path]::DirectorySeparatorChar)) { throw 'Run the recorder from D:.' }
$env:DOOM_WORK = $project
Push-Location
try { . (Join-Path $project 'workspace\devforge-doom-template-main\training\usb.ps1') }
finally { Pop-Location }
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$recorder = Join-Path $workspace 'record-human-demo.py'
$canonical = @{'basicaim'='BasicAim';'centercombat'='CenterCombat';'corridorcombat'='CorridorCombat';
               'healthrun'='HealthRun';'keydoor'='KeyDoor';'backtrack'='Backtrack';'tworooms'='TwoRooms'}
$chosen = $canonical[$Layout.ToLowerInvariant()]
$result = & $python -B $recorder --layout $chosen --seed $Seed --preflight
if ($LASTEXITCODE -ne 0) { throw 'Recorder preflight failed.' }
$result
if ($Preflight) { return }
if (-not (($result | Out-String | ConvertFrom-Json).launch_allowed)) { throw 'Recorder storage or memory gate blocked launch.' }
if (-not $ConfirmTeamGenerated -or [string]::IsNullOrWhiteSpace($Teammate) -or [string]::IsNullOrWhiteSpace($PermissionReference)) {
    throw 'A registered teammate and a verified organizer permission reference are required to begin recording.'
}

$env:DOOM_HUMAN_TEAMMATE = $Teammate.Trim()
$env:DOOM_HUMAN_PERMISSION = $PermissionReference.Trim()
$logs = Join-Path $workspace 'demonstrations\launch-logs'
[void][IO.Directory]::CreateDirectory($logs)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
$start = @{
    FilePath = $python
    ArgumentList = @('-B', ('"{0}"' -f $recorder), '--layout', $chosen, '--seed', $Seed, '--team-attested')
    WorkingDirectory = $workspace
    WindowStyle = 'Normal'
    PassThru = $true
    RedirectStandardOutput = Join-Path $logs "$stamp-stdout.txt"
    RedirectStandardError = Join-Path $logs "$stamp-stderr.txt"
}
$process = Start-Process @start
Write-Output "Human demonstration recorder PID $($process.Id). Log: $($start.RedirectStandardOutput)"
