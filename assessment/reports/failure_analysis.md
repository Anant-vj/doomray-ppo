# Gameplay failure analysis

These measured failure examples come from the earlier 36-episode independent diagnostic suite and its three-episode backtracking follow-up. The two sets contain 8,937 legal decisions and zero invalid actions. The locked MAP11/12 holdout was excluded from these diagnostic runs. Use the `assessment` dataset in [episodes.csv](episodes.csv) for new blind records and [tier results](tier_results.md) for their actual coverage.

| Observed behavior | Recorded evidence | Measurement limit |
| --- | --- | --- |
| Weak single-enemy combat | Across three trials, C749 held attack on 359 decisions. Ammo decrease was 17 units, with zero player hits, damage dealt or kills. | An attack decision describes a held button. Pickups can offset ammo decreases. The data do not establish shot count. |
| Attacks outside visible cues | In the prior main suite, 2,091 of 3,143 attack decisions had neither an enemy entity nor an enemy ray cue. 110 of 163 observed ammo-unit decreases occurred without either cue. | These are legal observation cues. They leave targets outside those cues and weapon behavior unresolved. |
| Wall pressure and stagnation | Movement was requested on 7,919 decisions. 536 gained less than one world unit. Of these, 350 also had a nearby wall ray in the requested direction. | These findings describe movement and proximity. They do not identify every collision or establish why the policy chose that action. |
| Directional skew | The prior main suite held strafe left/right on 85.8% / 2.1% and turn left/right on 63.3% / 17.3% of decisions. | Rates are held-button fractions. Independent buttons overlap. The skew appears in successful and blocked sequences. |
| Failure to reach a key and return | Straight progression succeeded in 3/3 controlled trials. The separate backtracking route succeeded in 1/3. In one failure, the key was visible for four decisions and came within 53.01 units. | Different diagnostic layouts establish different outcomes. There is no controlled ablation isolating the four-decision stack as the cause. |
| Weak larger-map progression | Six prior diagnostic MAP07/08 episodes recorded one survivor, zero keys, zero unlocks and zero exits. The engine logged nine kills. | Only one player hit and six points of recorded player damage were present. Direct attribution of the engine kills remains unresolved. |

Representative earlier success is the straight key-to-lock-to-exit case on seed 3103, which completed in 28 decisions. A representative failure is the backtracking case on seed 3101, which hit the cap without a key. The inline route graphics in [the report](report.html#behaviors) use actual sampled C749 trace positions. They are evaluator-side positions and have no actor input role. The [earlier audit](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/README.md) and [raw decision traces](D:/Doom/DevForge_DOOM/submission_artifacts/final-functional-audit/diagnostics-20261127-024526163/episodes.jsonl) remain on D:.

<!-- BEGIN NEW ASSESSMENT SNAPSHOT -->
Measured on 2026-11-27 04:34:21 +05:30.

## New blinded gameplay and reliability

Each row below uses only completed new episodes for that model. Directional percentages are held-button rates. Engine kills and recorded player hit or damage counters have separate meanings.

| Model | Episodes | Decisions | Left/right strafe | Left/right turn | Attacks / no cue | Player hits / damage | Engine kills | Ammo lower bound | Blocked / requested moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 32 | 11446 | 92.00594% / 1.94828% | 66.46864% / 12.63323% | 5280 / 1964 | 8 / 66 | 58 | 233 | 1473 / 11398 |
| Stack25-u174 | 32 | 14578 | 68.17122% / 20.00274% | 15.21471% / 68.68569% | 9374 / 5632 | 42 / 610 | 29 | 295 | 928 / 13951 |
| Stack26-u621 | 32 | 15011 | 88.69496% / 1.91193% | 26.88695% / 64.99234% | 3660 / 1747 | 30 / 343 | 74 | 226 | 1335 / 14722 |

0 execution failures recorded. 0 invalid actions recorded across the new ledger. Engine kills do not establish player credit. Ammo count is a lower bound from observed net decreases. Blocked movement is a displacement proxy.

All-action timing and exact trace-level directional counts are available in [observed-runtime-comparison.json](observed-runtime-comparison.json).

[New route traces and behavior tables](report.html#current-behavior) · [All episode metrics](episodes.csv)
