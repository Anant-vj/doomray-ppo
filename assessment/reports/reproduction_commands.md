# Reproduction and viewing commands

Use ordinary native Windows PowerShell. All new artifacts stay on D:. The approved C: net decline limit is 7,000,000,000 bytes. The 500,000,000-byte start reserve still applies. A launch also needs sufficient memory and free space. Keep the original baseline.

## Known-seed graphical demonstrations

```powershell
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Progression
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Complex
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Backtracking
```

These are known-seed replays. The progression and backtracking success seeds were chosen after the prior outcomes were known. The launcher checks C749 identity and launch requirements before opening the viewer. Three fresh native RGB sessions were verified with zero invalid actions. The successful straight and backtracking replays had 28 and 231 legal decisions. The complex session ended in death after 201 legal decisions and showed two engine kills without resolving player credit.

Fresh recordings on the project drive [Progression](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Progression-20261127-040138964/audit-gameplay.avi) · [Complex](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Complex-20261127-040250136/audit-gameplay.avi) · [Backtracking](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Backtracking-20261127-040413648/audit-gameplay.avi)

The earlier validator-map audit uses the following launcher. Its native SDL gameplay recording is on D:.

```powershell
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-live-demo.ps1' -Audit
```

The corresponding [recorded clip](D:/Doom/DevForge_DOOM/submission_artifacts/demo-runs/20261127-024354396/audit-gameplay.avi) shows the known validator map at seed 1001. It is pre-recorded gameplay and remains available to inspect separately from a fresh launch. [Live viewer audit](D:/Doom/DevForge_DOOM/submission_artifacts/live-audit-evidence.md)

## Offline report

Open `report.html` in a normal browser or use this native PowerShell command.

```powershell
Start-Process 'D:\Doom\DevForge_DOOM\submission_artifacts\final-blind-assessment\reports\report.html'
```

The report embeds its data, CSS and JavaScript. Use the dataset filter to distinguish new assessment rows from previously completed episodes. Matching JSON, CSV and manifest files can be inspected without the browser.

## New four-tier examination

The headless matched campaign checks the D: runtime, preregistered identities and available headroom. It appends to its staged ledger and skips identities already recorded. The stop sentinel is `logs/stop-matched.flag` inside the assessment directory. The evaluator cutoff is 06:15 IST.

```powershell
$assessment = 'D:\Doom\DevForge_DOOM\submission_artifacts\final-blind-assessment'
& "$assessment\run-matched-assessment.ps1" -Preflight
& "$assessment\run-matched-assessment.ps1"
& 'D:\Doom\DevForge_DOOM\environments\competition\Scripts\python.exe' -B "$assessment\evaluator\comparison.py"
```

The headless run uses a single inference thread and skill 3. It has no gameplay video. Raw per-decision traces live in `episodes/traces`. The ledger identifies every model, layout family, seed, termination, and trace path. Use the archived JSONL to inspect what was observed without reexecuting the game. [Map manifest](map_manifest.json) · [Checkpoint manifest](checkpoint_manifest.json) · [Scoring manifest](scoring_manifest.json) · [Paired results](matched-summary.json)

## Interactive map selection and replay

The graphical examination launcher can select a single model, tier, map and seed. Use `-Preflight` to inspect the proposed launch. Close existing simulation processes before starting a new live viewer.

```powershell
$exam = 'D:\Doom\DevForge_DOOM\submission_artifacts\final-blind-assessment\run-blind-examination.ps1'
& $exam -Menu -Preflight
& $exam -Model C749 -Tier 1
& $exam -Model C749 -Tier 1 -Map MAP18 -Seed 9101
& $exam -Model C749 -Tier 3 -Map MAP22 -Seed 9102
& $exam -Model Stack26-u621 -AllTiers
& $exam -Replay Progression-20261127-040138964
& $exam -Replay Complex-20261127-040250136
& $exam -Replay Backtracking-20261127-040413648
& $exam -Replay MAP18-C749-9101-1795734095700033000
```

Selected cases and existing clips passed launcher preflight. The specific C749 MAP18 seed 9101 live command also completed with 174 presented frames and 174 legal decisions. It ended in death. Its recording is from a separate viewer check after the matched cohort. Playback reads genuine recorded footage with no policy or game load. [Graphical examiner usage](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/graphical-exam/README.md) · [MAP18 native viewer audit](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/recordings/graphical/MAP18-C749-9101-1795734095700033000/audit-report.json)
