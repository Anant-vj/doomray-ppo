# C749 live demonstration modes

The linked JSON reports are included. Original video, screenshots and prior diagnostic episodes are referenced by their local paths.

Each PowerShell command requests one live episode with the submitted C749 policy. Run the modes one at a time. The viewer shows the actual outcome, then keeps the last game frame and HUD visible for up to four minutes after the engine closes. Press Escape or close the window to dismiss it.

```powershell
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Progression
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Complex
& 'D:\Doom\DevForge_DOOM\submission_artifacts\run-c749-demonstration.ps1' -Mode Backtracking
```

Add `-Preflight` to verify the commit, weight, map hash, seed and storage without starting the game. The user raised the C: free-space decline cap to 7,000,000,000 bytes on November 27, 2026. The original baseline and 500,000,000-byte start reserve were preserved. Each mode was run once in native Windows after passing preflight. The observed decline after each engine closed was 5,044,416,512, 5,049,049,088 and 5,049,966,592 bytes, respectively. The verified replays used `-HoldSeconds 45` to run the modes one at a time. Without that switch, the viewer holds for up to four minutes.

## Verified visible runs

| Mode | Actual result | Native playback evidence |
| --- | --- | --- |
| Progression | 28 legal decisions, red keycard, one door and unlock, exit reached, 100 health, 27 distinct game frames | [Run report](c749-demonstration-runs/Progression-20261127-040138964/audit-report.json) · [Final HUD](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Progression-20261127-040138964/audit-end.png) · [3.429-second AVI](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Progression-20261127-040138964/audit-gameplay.avi) |
| Complex gameplay | 201 legal decisions, death, 108 damage received, two engine kills, no exit, 200 distinct frames | [Run report](c749-demonstration-runs/Complex-20261127-040250136/audit-report.json) · [Final HUD](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Complex-20261127-040250136/audit-end.png) · [23.086-second AVI](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Complex-20261127-040250136/audit-gameplay.avi) |
| Backtracking diagnostic | 231 legal decisions, red keycard, one door and unlock, exit reached, 100 health, 230 distinct frames | [Run report](c749-demonstration-runs/Backtracking-20261127-040413648/audit-report.json) · [Final HUD](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Backtracking-20261127-040413648/audit-end.png) · [26.514-second AVI](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Backtracking-20261127-040413648/audit-gameplay.avi) |

All runs used the native Windows SDL driver and produced 460 legal decisions with zero invalid actions. Live Windows responsiveness probes acknowledged 3/3, 8/8 and 8/8 messages, respectively. All 48 live client-area color markers matched the presented frame, and the windows were visible and foreground in those samples. The final frames remained visible when the engines closed at 04:01:52, 04:03:20 and 04:04:47 IST. All three viewers then exited after the timed hold. The short Progression run measured p99 inference of 155.35 ms. Complex and Backtracking measured 6.22 ms and 5.49 ms, respectively. The current live demonstration reads engine kills and damage received. The player hit and damage dealt below were measured in the earlier headless Complex episode. [Progression window probes](c749-demonstration-runs/Progression-20261127-040138964/audit-native-responsiveness.json) · [Complex window probes](c749-demonstration-runs/Complex-20261127-040250136/audit-native-responsiveness.json) · [Backtracking window probes](c749-demonstration-runs/Backtracking-20261127-040413648/audit-native-responsiveness.json)

Progression collected its red card at decision 6 and unlocked the door at decision 10. Backtracking collected its card at decision 177 and unlocked the door at decision 211. In the Backtracking final image, the genuine last RGB still displays the engine’s red-key message. The HUD shows the held card and one unlock, and the official wrapper records the exit at decision 231. All 17 saved HUD captures passed the direct viewport RGB comparison. [Progression unlock capture](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Progression-20261127-040138964/audit-unlock.png) · [Backtracking unlock capture](D:/Doom/DevForge_DOOM/submission_artifacts/c749-demonstration-runs/Backtracking-20261127-040413648/audit-unlock.png)

## Fixed episodes and prior outcomes

| Mode | Map, seed and tick cap | Earlier C749 episode | Why this seed was chosen |
| --- | --- | --- | --- |
| Progression demonstration | Local generated `locked_progression.wad`, `MAP01`, 3103, 1,024 | 28 legal decisions, key collected, locked door opened, exit reached, 100 health | Fastest previously successful seed on this fixture. C749 reached the exit on all three prior seeds. |
| Complex gameplay demonstration | Bundled Freedoom `freedoom2.wad`, known development `MAP08`, 3101, 2,100 | 201 legal decisions, one measured player hit, six player-attributed damage, two engine-reported kills, death with no exit | First seed in the earlier MAP08 diagnostic. Shows combat on a broader map and where the policy fails. |
| Backtracking diagnostic demonstration | Local generated `key_behind_spawn.wad`, `MAP01`, 3102, 1,400 | 231 legal decisions, key collected behind spawn, return to the locked door, exit reached, 100 health | The only prior seed that exited this fixture. The other two timed out. |

The fixed choices were recorded before any live presentation replay. Each launch uses its single declared seed, with no automatic reroll. The progression and backtracking clips are intended to demonstrate a behavior previously observed on a local fixture. The results above describe the three actual prior headless episodes. The complex map was reused from development, and the final evaluation maps MAP11 and MAP12 are outside all demonstration modes. [Fixed manifest](c749-demonstration-manifest.json) · [Progression and complex episode records](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/diagnostics-20261127-024526163/episodes.jsonl) · [Backtracking episode records](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/backtrack-20261127-025005478/episodes.jsonl)

## What the viewer displays and records

The live HUD shows health, armor, weapon and reserve ammunition, held keys, buttons selected by the neural policy and applied to the game, decision count, elapsed engine ticks, inference latency, detected door openings and unlocks, and exit status. Door counts and the most recent unlock decision come from evaluator-side sector changes. The policy continues to receive only the official observation. On termination, the official wrapper returns the last legal frame and vitals. The viewer labels them as last live and retains the latest elapsed tick when the terminal engine clock resets.

Each allowed run gets a separate `Mode-Timestamp` folder under `D:\Doom\DevForge_DOOM\submission_artifacts\c749-demonstration-runs`. It contains the exact launch metadata, live telemetry, Windows responsiveness checks, screenshots at the beginning, predeclared midpoint and end when reached, and an extra screenshot for the first observed key, door, unlock and exit. The AVI uses actual engine RGB from one frame per two decisions and may be written after the engine closes. Screenshots show the HUD. The report records the observed outcome, action validity, latency, raw terminal tick and most recent live elapsed tick. [HUD and replay source](live-audit-viewer.py) · [PowerShell launcher](run-c749-demonstration.ps1)

The launcher verifies public commit `6c2d5499f973c77b4a239f778837280edf64e57b` and materialized checkpoint SHA-256 `0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61`. The replay uses the official wrapper and unchanged CPU inference and matches the diagnostic environment and PyTorch seed order. For the viewer, the wrapper also fills its official RGB and depth buffers. C749 inference uses rays, entities, vitals and previous actions.

## Existing authentic footage

The earlier 12.114-second recording shows the submitted C749 policy on the known validator map `two_rooms.wad:MAP01`, seed 1001. That separate 525-decision episode ended at the tick cap with zero kills and exits. Its authentic clip and source as run remain available. [Two Rooms gameplay AVI](D:/Doom/DevForge_DOOM/submission_artifacts/demo-runs/20261127-024354396/audit-gameplay.avi) · [Native window evidence](D:/Doom/DevForge_DOOM/submission_artifacts/live-audit-evidence.md) · [Viewer source as run](D:/Doom/DevForge_DOOM/submission_artifacts/demo-runs/20261127-024354396/live-audit-viewer-as-run.py)
