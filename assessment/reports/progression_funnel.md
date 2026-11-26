# Long-horizon progression funnel

This file records what the existing measurements establish. [New tier records](tier_results.md) and the `assessment` dataset in [episodes.csv](episodes.csv) identify the outcome of each newly completed case.

| Evidence set | Key discovery and acquisition | Lock and passage | Exit and termination |
| --- | --- | --- | --- |
| Prior controlled straight route, 3 episodes | Key collected at decision 6 in every case. Initial visual discovery timing is unavailable for this summary. | Tagged lock rise occurred on decisions 12, 17 and 10. No independent lock-crossing count is quoted. | Engine-confirmed exits at decisions 165, 37 and 28. |
| Prior backtracking route, 3 episodes | One acquired the key, at decision 177. The other two hit the cap with zero keys. In seed 3101 the key entered visible-entity slots for four decisions and came within 53.01 units. | The successful case triggered the lock rise at 211. The failures ended near the lock. Separate alignment and lock-crossing counts are unavailable. | One engine-confirmed exit at 231. Two timeouts without a key or unlock. |
| Prior fresh diagnostic MAP07/08, 6 episodes | No key pickups. MAP08 exposed a key in 89 decisions, with nearest visible distance 244.44 units. | Five door rises, zero locked rises. No verified passage through an unlocked progression door. | No exits. One survivor. |
| Earlier frozen MAP11/12 holdout | Zero acquired keys for all tested policies. Key discovery was not recorded separately. | Zero unlocks for all tested policies. | Zero exits and survivors in the 24 paired episodes. |

The intended order for a new instrumented route is exploration, key discovery, pickup, locked-door discovery, unlock, crossing, exit discovery, then trigger-specific interaction and engine-confirmed exit. For a USE exit, measure range, alignment, fresh USE and actual termination only where instrumentation can do so reliably. A crossing exit has no switch-alignment stage. Death, time cap, missing events and unavailable metrics are separate cases. Policy observations exclude privileged diagnostic positions and map inspection.

The tested starting layouts have different demands. The observed short-context stack is a plausible limitation, with no controlled evidence isolating memory, visual targeting or route recovery as a cause. [Main diagnostics](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/diagnostics-20261127-024526163/suite-results.md) · [Backtracking follow-up](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/backtrack-20261127-025005478/suite-results.md)

<!-- BEGIN NEW ASSESSMENT SNAPSHOT -->
Measured on 2026-11-27 04:34:21 +05:30.

## Newly measured keycard and gauntlet routes

Tier 3 and 4 rows show distinct episodes where each event ever appeared. Counts need not decrease monotonically. Door rises, sector crossing and geometry distance are evaluator proxies. USE events only apply to static maps with a USE trigger. Engine exit is the actual terminal signal.

| Model | Episodes | Explored | Key seen / within 64 / acquired | Lock cue / rise / crossing proxy | Within 192 of exit | USE-applicable | USE range / align / fresh press | Cross-applicable | Engine exit | Death / tick cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 16 | 16 | 6 / 1 / 0 | 6 / 0 / 0 | 2 | 4 | 0 / 0 / 0 | 14 | 0 | 15 / 1 |
| Stack25-u174 | 16 | 16 | 4 / 1 / 0 | 6 / 0 / 0 | 2 | 4 | 0 / 0 / 0 | 14 | 0 | 10 / 6 |
| Stack26-u621 | 16 | 16 | 4 / 1 / 0 | 6 / 0 / 0 | 2 | 4 | 0 / 0 / 0 | 14 | 0 | 13 / 3 |

Per-episode first occurrence decision indices appear in `funnel_events`. Minimum exit geometry distance can be small across an obstruction. These observations leave the agent objective and the isolated cause of a failure unresolved.

### Per-episode observed end point

The observed end point is the most advanced available instrument signal. Multiple pathways can overlap, and lock-return timing is unresolved if an earlier lock ray was the only signal.

| Model | Tier | Map | Seed | Decisions | Termination | Most advanced available signal |
| --- | ---: | --- | ---: | ---: | --- | --- |
| C749 | 3 | MAP16 | 9101 | 1046 | death | Lock cue seen without a key entity |
| Stack25-u174 | 3 | MAP16 | 9101 | 1050 | tick_limit | Lock cue seen without a key entity |
| Stack26-u621 | 3 | MAP16 | 9101 | 349 | death | Lock cue seen without a key entity |
| C749 | 3 | MAP16 | 9102 | 331 | death | Lock cue seen without a key entity |
| Stack25-u174 | 3 | MAP16 | 9102 | 905 | death | Lock cue seen without a key entity |
| Stack26-u621 | 3 | MAP16 | 9102 | 623 | death | Lock cue seen without a key entity |
| C749 | 3 | MAP22 | 9101 | 387 | death | Key entity seen beyond 64 units |
| Stack25-u174 | 3 | MAP22 | 9101 | 1050 | tick_limit | Exploration without an observed key entity |
| Stack26-u621 | 3 | MAP22 | 9101 | 1050 | tick_limit | Exploration without an observed key entity |
| C749 | 3 | MAP22 | 9102 | 819 | death | Key entity seen beyond 64 units |
| Stack25-u174 | 3 | MAP22 | 9102 | 1050 | tick_limit | Exploration without an observed key entity |
| Stack26-u621 | 3 | MAP22 | 9102 | 1050 | tick_limit | Exploration without an observed key entity |
| C749 | 3 | MAP24 | 9101 | 299 | death | Key within 64 units without acquisition |
| Stack25-u174 | 3 | MAP24 | 9101 | 412 | death | Key within 64 units without acquisition |
| Stack26-u621 | 3 | MAP24 | 9101 | 865 | death | Key entity seen beyond 64 units |
| C749 | 3 | MAP24 | 9102 | 173 | death | Key entity seen beyond 64 units |
| Stack25-u174 | 3 | MAP24 | 9102 | 1050 | tick_limit | Key entity seen beyond 64 units |
| Stack26-u621 | 3 | MAP24 | 9102 | 323 | death | Key within 64 units without acquisition |
| C749 | 3 | MAP29 | 9101 | 105 | death | Exit geometry within 192 units without exit |
| Stack25-u174 | 3 | MAP29 | 9101 | 266 | death | Exit geometry within 192 units without exit |
| Stack26-u621 | 3 | MAP29 | 9101 | 802 | death | Exit geometry within 192 units without exit |
| C749 | 3 | MAP29 | 9102 | 743 | death | Exit geometry within 192 units without exit |
| Stack25-u174 | 3 | MAP29 | 9102 | 329 | death | Exit geometry within 192 units without exit |
| Stack26-u621 | 3 | MAP29 | 9102 | 616 | death | Exit geometry within 192 units without exit |
| C749 | 4 | MAP15 | 9101 | 453 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP15 | 9101 | 1050 | tick_limit | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP15 | 9101 | 702 | death | Exploration without an observed key entity |
| C749 | 4 | MAP15 | 9102 | 379 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP15 | 9102 | 490 | death | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP15 | 9102 | 1050 | tick_limit | Exploration without an observed key entity |
| C749 | 4 | MAP25 | 9101 | 155 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP25 | 9101 | 399 | death | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP25 | 9101 | 240 | death | Exploration without an observed key entity |
| C749 | 4 | MAP25 | 9102 | 136 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP25 | 9102 | 128 | death | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP25 | 9102 | 239 | death | Exploration without an observed key entity |
| C749 | 4 | MAP27 | 9101 | 157 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP27 | 9101 | 201 | death | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP27 | 9101 | 127 | death | Exploration without an observed key entity |
| C749 | 4 | MAP27 | 9102 | 320 | death | Exploration without an observed key entity |
| Stack25-u174 | 4 | MAP27 | 9102 | 210 | death | Exploration without an observed key entity |
| Stack26-u621 | 4 | MAP27 | 9102 | 164 | death | Exploration without an observed key entity |
| C749 | 4 | MAP28 | 9101 | 1050 | tick_limit | Lock cue seen without a key entity |
| Stack25-u174 | 4 | MAP28 | 9101 | 712 | death | Lock cue seen without a key entity |
| Stack26-u621 | 4 | MAP28 | 9101 | 1027 | death | Lock cue seen without a key entity |
| C749 | 4 | MAP28 | 9102 | 938 | death | Lock cue seen without a key entity |
| Stack25-u174 | 4 | MAP28 | 9102 | 1050 | tick_limit | Lock cue seen without a key entity |
| Stack26-u621 | 4 | MAP28 | 9102 | 934 | death | Lock cue seen without a key entity |

[Funnel and representative trajectories](report.html#current-behavior) · [Episode-level metrics](episodes.csv)
