# Matched model comparisons

The following are completed historical evaluations. New assessment records are separately labeled in [episodes.csv](episodes.csv). The older wide-development suite informed checkpoint selection. The frozen holdout was evaluated after selection.

## Historical wide development

Same MAP07, MAP08, MAP13 and MAP14 layouts and four seeds per model. The earlier research used a 2,100-tick cap.

| Model | Episodes | Legacy local proxy | Survived | Exits | Engine kills | Doors | Mean 96-unit cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 16 | 11.21875 | 3/16 | 0 | 54 | 21 | 35.375 |
| Stack-25 u174 | 16 | 6.245 | 9/16 | 0 | 20 | 7 | 20.8125 |
| Stack-26 u621 | 16 | 9.85375 | 5/16 | 0 | 46 | 10 | 34 |
| Stack-26 u397 | 16 | 8.2675 | 8/16 | 0 | 54 | 6 | 24.8125 |
| Seeded random | 16 | 4.855 | 10/16 | 0 | 15 | 9 | 13.3125 |

C749 exceeded u174 by 4.97375 proxy points and survived 37.5 percentage points less often. It exceeded the preselected reference u621 by 1.365 proxy points and survived 12.5 percentage points less often. Compared with u174, C749 won 13 of 16 comparisons under the balanced profile, with no ties. All three local profiles favored C749 on this suite. These comparisons concern a development dataset and a private official scoring function. [Earlier wide ranking](D:/Doom/DevForge_DOOM/evaluations/mission-wide-dev-ranking.md)

## Earlier frozen holdout

Same MAP11 and MAP12 layouts with seeds 1001–1004 and a 2,100-tick cap, after the model was frozen.

| Model | Episodes | Legacy local proxy | Survived | Exits | Engine kills | Doors | Mean 96-unit cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 8 | 4.5700 | 0/8 | 0 | 9 | 4 | 24.125 |
| Stack-25 u174 | 8 | 2.2625 | 0/8 | 0 | 7 | 0 | 19.75 |
| Seeded random | 8 | 0.8525 | 0/8 | 0 | 6 | 1 | 10.625 |
| Stack-26 u621 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |

C749 led u174 by 2.3075 legacy proxy points and by six wins to two losses on the balanced paired comparison. Survival and exit rates were zero for all tested policies. Each had zero invalid actions, keys and unlocks. Two independent layouts give insufficient layout-level support for a useful generalization interval. [Frozen holdout evidence](D:/Doom/DevForge_DOOM/evaluations/mission-final-holdout-comparison.md)

## Scoring definitions

Historical per-episode legacy proxy is `100*exit + 8*keys + 4*locked_rises + 2*all_door_rises + 0.2*min(cells_96,100) + engine_kills + 0.02*end_HP - 0.02*damage`. Episodes are averaged. This formula includes both the locked rise term and the all-door term.

| Profile | Keys | Locked rises | Ordinary rises | Cells | Kills | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | 7 | 4 | 1.5 | .20 | 1 | 2 |
| Survival-combat | 4 | 3 | 1 | .10 | 1.5 | 5 |
| Progression | 10 | 6 | 2 | .28 | .75 | 1.5 |

Ordinary rises are max(0, all door rises minus locked rises). Cell counts are capped at 100. Safety is 1 for a living finish or tick-cap survivor, otherwise min(1, decisions / ceil(tick cap / 4)). A profile ranks eligible models by total exits first, then the mean weighted score. Zero invalid actions are required for eligibility. The prior controlled diagnostics used 64-unit cells and are kept outside these 96-unit-proxy comparisons. [Exact evaluator](D:/Doom/DevForge_DOOM/submission_artifacts/github-fresh-6c2d549-20261127-022250050/training/evaluate.py) · [Exact scoring profiles](D:/Doom/DevForge_DOOM/submission_artifacts/github-fresh-6c2d549-20261127-022250050/training/rank.py)

<!-- BEGIN NEW ASSESSMENT SNAPSHOT -->
Measured on 2026-11-27 04:34:21 +05:30.

## New matched examination

96 completed episodes. 16 fully matched independent families. 32 completed common layout/seed cases across all three models. Planned and failed records are excluded from completed-pair means.

| Model | Completed / planned | Exits | Survived | Historical | Balanced | Survival-combat | Progression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 32 / 32 | 0 | 3 | 6.93875 | 9.89485 | 8.75744 | 12.01317 |
| Stack25-u174 | 32 / 32 | 0 | 11 | 5.01063 | 7.05679 | 6.76696 | 8.5254 |
| Stack26-u621 | 32 / 32 | 0 | 8 | 8.02875 | 10.54158 | 10.08051 | 12.43353 |

| Model | Mean decisions | Keys | Unlocks | Doors | Mean 96-unit cells | Engine kills | Mean end HP | Mean damage received | Player hits / damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 357.6875 | 0 | 0 | 21 | 30.90625 | 58 | -6.3125 | 112.0625 | 8 / 66 |
| Stack25-u174 | 455.5625 | 0 | 0 | 12 | 22.34375 | 29 | 23.53125 | 79.25 | 42 / 610 |
| Stack26-u621 | 469.09375 | 0 | 0 | 17 | 31.25 | 74 | 13.875 | 93.6875 | 30 / 343 |

Each model row uses that model's measured episodes. Paired changes below use only cases completed by both models. Positive differences favor the left model for exit, survival and higher score.

| Left vs right | Pairs | Fully paired layouts | Exit pp | Survival pp | Balanced change | Relative balanced change % | Balanced W/T/L | Layout bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C749 vs Stack25-u174 | 32 | 16 | 0 | -25 | 2.83807 | 40.21754 | 23/0/9 | [0.86932, 5.48475] |
| C749 vs Stack26-u621 | 32 | 16 | 0 | -15.625 | -0.64673 | -6.135 | 13/1/18 | [-3.76978, 3.07102] |
| Stack25-u174 vs Stack26-u621 | 32 | 16 | 0 | 9.375 | -3.48479 | -33.05759 | 7/2/23 | [-6.10177, -1.36565] |

| Left vs right | Mean keys difference | Mean unlocks difference | Mean 96-unit cells difference | Mean engine kills difference | Mean end HP difference | Mean damage received difference | Mean decisions difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 vs Stack25-u174 | 0 | 0 | 8.5625 | 0.90625 | -29.84375 | 32.8125 | -97.875 |
| C749 vs Stack26-u621 | 0 | 0 | -0.34375 | -0.5 | -20.1875 | 18.375 | -111.40625 |
| Stack25-u174 vs Stack26-u621 | 0 | 0 | -8.90625 | -1.40625 | 9.65625 | -14.4375 | -13.53125 |

[Full outcome deltas and per-tier paired aggregates](matched-summary.json) · [Frozen local scoring definitions](scoring_manifest.json)
