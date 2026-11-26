# DoomRay PPO assessment

The minimum matched examination finished all 96 gameplay episodes across 16 independent layout families. All three frozen policies ran the same two seeds per layout. All 96 engine resets succeeded, and no episode was excluded. The earlier 93-episode capture is preserved in branch history.

Public main retains the approved C749 submission at `6c2d5499f973c77b4a239f778837280edf64e57b`. The official environment, validator and inference weights match that commit.

## Results

On these unfamiliar layouts, C749 averaged 9.895 on the predeclared unofficial Balanced score and survived 3/32 episodes. Stack-26 update 621 averaged 10.542 and survived 8/32. The layout-paired 95% confidence interval for C749 minus u621 was −3.770 to +3.071. Stack-25 update 174 averaged 7.057 and survived 11/32. All three policies recorded zero exits, acquired keys and unlocks.

C749 was frozen and approved from the earlier development record and exact native validation before this assessment. The three candidate policies, maps, seeds and score formulas were frozen before the new comparison. The branch provides supplementary evidence for the approved C749 handoff. The score profiles are local, and official leaderboard results remain unknown.

- [Offline examination report](reports/report.html). Download and open the file in a browser.
- [Snapshot provenance and hashes](SNAPSHOT.json)
- [Gameplay ledger](episodes/blind-episodes.jsonl), [episode CSV](reports/episodes.csv), and [matched statistics](reports/matched-summary.json)
- [Engine and model verification](reports/engine-verification.json), [score definitions](manifests/scoring_manifest.json), and [map schedule](manifests/map_manifest.json)
- [Failure analysis](reports/failure_analysis.md), [progression funnel](reports/progression_funnel.md), and [storage audit](reports/storage_audit.json)

## Reproduction and visible gameplay

Run `python -B assessment/evaluator/comparison.py` from the repository root to regenerate paired aggregates from the bundled ledger. The native launchers document the original Windows D-drive setup. Fresh runs require its matching Python environment, Freedoom WAD and archived reference checkpoints. The structured files retain their original local provenance paths for traces, videos and other evidence kept on D.

The separate [graphical examination guide](graphical-exam/README.md) records a verified native C749 session on MAP18 with an authentic [final HUD](recordings/graphical/MAP18-C749-9101-1795734095700033000/audit-end.png), a successful replay and a failed replay. Those visual checks are outside the matched ledger. The fixed [C749 demo guide](demos/C749-demonstration-modes.md) documents two local progression successes and one complex gameplay failure. See the [frozen inference audit](audit/source-trace.md) for what the official actor reads and returns.
