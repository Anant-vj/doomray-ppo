# Native graphical examination

Open the lightweight Windows viewer from PowerShell. All three frozen model identities, all 16 assessment maps and the four existing recordings passed launcher preflight. Preflight checks the original storage baseline and the current memory and disk limits before any display, model or game library loads.

```powershell
$exam = 'D:\Doom\DevForge_DOOM\submission_artifacts\final-blind-assessment\run-blind-examination.ps1'
& $exam -Menu
& $exam -Model C749 -Tier 1
& $exam -Model C749 -Tier 3 -Map MAP22 -Seed 9102
& $exam -Model Stack25-u174 -Tier 2 -Map MAP17 -Seed 9101
& $exam -Model Stack26-u621 -AllTiers
& $exam -Replay Progression-20261127-040138964
& $exam -Replay Complex-20261127-040250136
& $exam -Replay Backtracking-20261127-040413648
& $exam -Replay TwoRooms
```

Append `-Preflight` to print the exact map selection, recognized recordings, and the current gate decision without launching the viewer. The default engine and policy seed is 9101. A tier or `-AllTiers` creates a queue in the frozen map order. Press Enter after the engine closes to inspect the next map. Only one game runs at a time. Close a viewer before starting another graphical or matched assessment run.

## Controls and HUD

In the menu, select fields with Up and Down or click a row. Use Left and Right to change the model, tier, map or recording. Type the seed in its field. Press Enter to open the selected recording or start the selected live run.

During game or replay, Space pauses or resumes. `−` and `+` switch among 0.25×, 0.5×, 1×, 2× and 4× speed. Escape or the close button exits. A completed live run retains the final legal frame and the outcome after the engine closes.

- **LIVE EVALUATION** displays the selected model’s actual game, neural button presses, health, armor, ammunition, keys, elapsed ticks, decision count and inference latency. The HUD labels player hit and damage counters, engine kills, 96-unit exploration cells, door activity and exit progress as evaluator data. These readings stay outside the actor’s observation.
- **COMPLETED EPISODE** displays the last legal observation and final result. Game time uses the most recent elapsed engine reading when the terminal clock resets.
- **RECORDED REPLAY** plays captured Motion JPEG frames without running an engine or loading a model. The saved clip and outcome are shown as recorded. The successful Progression and Backtracking runs and failed Complex run are available above.

## Evidence and scope

Native exploratory runs are stored under `D:\Doom\DevForge_DOOM\submission_artifacts\final-blind-assessment\recordings\graphical`. An episode ID identifies the report, screenshots and authentic game-frame AVI. Use that same ID with `-Replay`. The graphical runs remain separate from the matched assessment results. Launch logs are stored on D: under `graphical-exam\launch-logs`.

As of the first launcher checks, menu selection, single-map, single-tier, all-tier and replay syntax passed preflight. The new window and live path require a separate native check. The three fixed demonstration windows and their source footage have already been verified separately. [Launcher](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/run-blind-examination.ps1) · [Viewer source](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/graphical-exam/examine.py) · [Verified fixed demonstrations](D:/Doom/DevForge_DOOM/submission_artifacts/C749-demonstration-modes.md)
