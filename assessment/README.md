# DoomRay PPO assessment snapshot

**Partial minimum examination.** Captured at 2026-11-27T04:29:12+05:30. The ledger contains 93 attempts and 93 completed gameplay episodes out of 96 planned. 15 of 16 independent layouts have complete results for all three models and both seeds.

This branch adds assessment tooling and evidence beside the frozen submission. C749, the official environment and validator, and the submitted files are identical to commit `6c2d5499f973c77b4a239f778837280edf64e57b`. This assessment measures local gameplay and reports unofficial scores.

- [Snapshot provenance and hashes](SNAPSHOT.json)
- [Captured gameplay ledger](episodes/blind-episodes.jsonl) and [matched comparison](reports/matched-summary.json)
- [Predeclared map and seed schedule](manifests/map_manifest.json), [checkpoint identities](manifests/checkpoint_manifest.json), and [scoring formulas](manifests/scoring_manifest.json)
- [Native examination controls](graphical-exam/README.md), [fixed C749 demonstration guide](demos/C749-demonstration-modes.md), and [frozen inference audit](audit/source-trace.md)

## Read the snapshot

Open the evidence files directly or run `python -B assessment/evaluator/comparison.py` from the repository root to regenerate the matched aggregates from the included ledger. The comparison reports each profile separately, ranks exits first, and groups repeated seeds by layout. Incomplete layout pairs remain visible. Confidence intervals require 16 fully paired layouts.

The assessment selection comes from verified static map geometry and the available project history. The four difficulty tiers are local definitions. The launcher scripts and exact manifests document the original Windows D-drive setup. Interactive or fresh matched runs require the existing local environment, local archived reference checkpoints, and installed Freedoom WAD identified in those manifests. Native replay also requires the original local captured media. The per-episode trace paths in the captured ledger refer to the original local evidence folder.

## Previously verified live C749 demonstrations

On native Windows, Progression exited with a red key and one unlock after 28 legal decisions. Complex ended in death after 201 legal decisions, with two engine-reported kills. Backtracking exited with a red key and one unlock after 231 legal decisions. These are fixed demonstrations from separate known layouts and do not count toward the assessment ledger. The three [demo evidence folders](demos/c749-demonstration-runs) contain compact measured run and responsiveness reports.
