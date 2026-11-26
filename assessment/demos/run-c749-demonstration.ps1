param(
    [Parameter(Mandatory)]
    [ValidateSet('Progression','Complex','Backtracking')]
    [string]$Mode,
    [ValidateRange(5,240)]
    [int]$HoldSeconds = 240,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$dir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$root = 'D:\Doom\DevForge_DOOM'
$manifest = Join-Path $dir 'c749-demonstration-manifest.json'
if ((Get-FileHash -LiteralPath $manifest).Hash -ne '1A3E02CCF1A7D1B454D26D7B16920B2837EED6B1034977B54B3579676FA4FD11') {
    throw 'Predeclared demonstration manifest SHA mismatch.'
}
$plan = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
$name = @{'progression'='Progression';'complex'='Complex';'backtracking'='Backtracking'}[$Mode.ToLowerInvariant()]
$scene = $plan.modes.$name
$checkout = (Resolve-Path -LiteralPath $plan.release.checkout).Path
$wad = (Resolve-Path -LiteralPath $scene.wad).Path
foreach ($path in @($dir,$checkout,$wad)) {
    if ([IO.Path]::GetPathRoot($path) -ne 'D:\') { throw "Use D: paths for the checkout and demonstration." }
}
if ($scene.map -in @('MAP11','MAP12')) { throw 'Choose one of the predeclared demonstration maps.' }
$safePath = $checkout.Replace('\', '/')
$commit = git --no-optional-locks -c "safe.directory=$safePath" -C $checkout rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $commit -ne $plan.release.commit) { throw 'Frozen release commit mismatch.' }
$dirty = git --no-optional-locks -c "safe.directory=$safePath" -C $checkout status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'The frozen checkout has tracked modifications.' }
$weight = (Get-FileHash -LiteralPath (Join-Path $checkout 'submission\weights\policy.pt')).Hash
$mapHash = (Get-FileHash -LiteralPath $wad).Hash
if ($weight -ne $plan.release.weight_sha256 -or $mapHash -ne $scene.wad_sha256) {
    throw 'Frozen weight or predeclared demonstration map SHA mismatch.'
}
$baselinePath = Join-Path $root 'logs\storage-baseline.json'
if ((Get-FileHash -LiteralPath $baselinePath).Hash -ne 'A158265C2CAE0AD6D50818BAE6E01387D6215F8A20A494115128BDC3A7FA9CCB') {
    throw 'Original storage baseline SHA mismatch.'
}
$baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
$delta = [Math]::Max([long]0, [long]$baseline.internal_free_bytes - [IO.DriveInfo]::new('C').AvailableFreeSpace)
$limit = 7000000000L
$reserve = 500000000L
$allowed = ($limit - $delta) -ge $reserve
$metadata = [ordered]@{
    mode = $name
    display_name = $scene.display_name
    verified_commit = $commit
    weight_sha256 = $weight
    wad = $wad
    wad_sha256 = $mapHash
    map = $scene.map
    seed = $scene.seed
    tick_cap = $scene.tick_cap
    c_net_decline_bytes = $delta
    c_decline_cap_bytes = $limit
    c_start_reserve_bytes = $reserve
    hold_seconds = $HoldSeconds
    launch_allowed = $allowed
}
$metadata | ConvertTo-Json
if ($Preflight) { return }
if (-not $allowed) {
    throw "Start requires 500,000,000 bytes of reserve under the 7,000,000,000-byte C: cap. Measured decline $delta bytes."
}

$env:DOOM_WORK = $root
. (Join-Path $root 'workspace\devforge-doom-template-main\training\usb.ps1')
$env:DOOM_DEMO_INTERNAL_LIMIT_BYTES = '7000000000'
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$viewer = Join-Path $dir 'live-audit-viewer.py'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
$session = Join-Path $dir "c749-demonstration-runs\$name-$stamp"
[void][IO.Directory]::CreateDirectory($session)
$args = @('-B', ('"{0}"' -f $viewer), '--repo', ('"{0}"' -f $checkout),
          '--manifest', ('"{0}"' -f $manifest), '--mode', $name, '--hold-seconds', $HoldSeconds, '--live')
$gameStart = @{
    FilePath = $python
    ArgumentList = $args
    WorkingDirectory = $session
    WindowStyle = 'Normal'
    PassThru = $true
    RedirectStandardOutput = Join-Path $session 'viewer-stdout.txt'
    RedirectStandardError = Join-Path $session 'viewer-stderr.txt'
}
$game = Start-Process @gameStart
$metadata['launcher_pid'] = $game.Id
$metadata['artifact_directory'] = $session
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'demo-launch.json') -Encoding utf8

$probeStart = @{
    FilePath = $python
    ArgumentList = @('-B', ('"{0}"' -f $viewer), '--probe', ('"{0}"' -f $session))
    WorkingDirectory = $session
    WindowStyle = 'Hidden'
    PassThru = $true
    RedirectStandardOutput = Join-Path $session 'probe-stdout.txt'
    RedirectStandardError = Join-Path $session 'probe-stderr.txt'
}
$probe = Start-Process @probeStart
Write-Output "Started $($scene.display_name). Viewer launcher PID $($game.Id). Probe launcher PID $($probe.Id). Artifacts: $session"
