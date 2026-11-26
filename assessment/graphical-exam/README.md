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
& $exam -Replay MAP18-C749-9101-1795734095700033000
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

## Verified in native Windows

- **Live**. The exact C749 `-Tier 1 -Map MAP18 -Seed 9101` run produced 174 legal decisions before death, with no exit. The window was visible and responsive in the Windows SDL driver. Pause and resume left the counter at decision 2. The completed frame remained visible and responsive after engine shutdown. Escape closed the window. The simulation recorded two door openings, no unlocks, zero measured player hits or damage dealt, two engine-reported kills, and 139 damage received. The exploratory result matches the matched episode's decision count and termination and sits outside the 96-episode matched ledger. [Live window log](launch-logs/20261127-043131854-stdout.txt) · [Episode result](../recordings/graphical/MAP18-C749-9101-1795734095700033000/audit-report.json) · [Final HUD](../recordings/graphical/MAP18-C749-9101-1795734095700033000/audit-end.png) · [20.114-second AVI](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/recordings/graphical/MAP18-C749-9101-1795734095700033000/audit-gameplay.avi)
- **Menu and successful replay**. The native menu opened and accepted keyboard navigation into the recorded Progression success. All 15 frames displayed. The frame counter stayed frozen during pause, then playback resumed. The 0.5× and 4× speed settings took effect and Escape closed the replay. [Menu and successful replay log](launch-logs/20261127-043207274-stdout.txt)
- **Failed replay**. The direct Complex replay command displayed all 101 authentic frames in the native player without an engine. Escape closed it. [Failed replay log](launch-logs/20261127-043216712-stdout.txt)

Single-tier, all-tier and each archived model selection passed launcher preflight. The native live check used C749 on one map. Multi-map advancement and the two archived models’ graphical paths have preflight coverage. No viewer or simulator remained active after these checks. At 04:32:34 IST, the original baseline SHA was unchanged and the C: decline was 5,080,354,816 bytes with 60,432,740,352 bytes free on D:. [Launcher](../run-blind-examination.ps1) · [Viewer source](examine.py) · [Verified fixed demonstrations](../demos/C749-demonstration-modes.md)
