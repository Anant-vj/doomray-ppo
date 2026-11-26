# Blind assessment map inventory

The minimum assessment was predeclared at 03:59:26 IST on November 27, 2026. It contains 16 distinct candidate layouts in the bundled Freedoom 2 WAD. Static checks passed for all 16 before any engine verification. The frozen manifest records that state at predeclaration. The episode ledger records subsequent official gameplay. No independent sealed DoomRay suite was found in the inspected workspace. [Frozen map manifest](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/manifests/map_manifest.json) · [Measured episode ledger](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/episodes/blind-episodes.jsonl)

| Tier | Fixed map order | Static basis |
| --- | --- | --- |
| 1 — Combat and Survival | MAP18, MAP19, MAP21, MAP26 | At least 12 placed Skill 3 enemies within 768 map units of spawn, measured in a straight line. Report actual combat and survival separately from exit and progression. |
| 2 — Multi-Room Exploration | MAP17, MAP20, MAP23, MAP32 | 146–539 sectors, 41–193 independent cycles in the static sector graph, and 2–13 nearby enemies. The maps also contain combat and locks. |
| 3 — Long-Horizon Progression | MAP16, MAP22, MAP24, MAP29 | At least four recognized locked-door sectors and at least two placed key colors. Reachability and lock ordering require engine verification. |
| 4 — Full Gauntlet | MAP15, MAP25, MAP27, MAP28 | 503–777 sectors, 122–194 placed monsters, all three placed key colors and three recognized locks per layout. Resource scarcity, required backtracking and distractor switches remain unverified. |

The tier labels describe a focus for this local examination. Sector cycles describe topological adjacency. Nearby monsters may be separated from spawn by a wall. Skill labels and supplies are based on placed things in the WAD. Record eventual behavior from the engine rather than treating these static indicators as gameplay measurements.

## Provenance and exclusions

- The existing training configuration includes Freedoom MAP01–06 and MAP31, official Two Rooms, the five named bundled practice maps, the official locked-key fixture and its spawn variant. Development used MAP07–08 and MAP13–14. Diagnostics used MAP09–10 and local generated layouts. The original final holdout MAP11–12 has already been measured. All these families and all bundled official test fixtures were excluded. [Runtime split source](D:/Doom/DevForge_DOOM/workspace/devforge-doom-template-main/training/game.py) · [Split record](D:/Doom/DevForge_DOOM/evaluations/mission-split-lock.md)
- Among the available training, evaluation, diagnostic and mission records searched, references to the 16 candidate maps appeared only in the earlier static map catalog, which does not report a wrapper reset for them. The 16 maps are independently held out according to the inspected project records. An undocumented run outside those records would invalidate that assumption. [Prior static catalog](D:/Doom/DevForge_DOOM/evaluations/phase2-map-audit.json)
- Freedoom MAP30 was omitted because the parser found no conventional normal exit trigger. Removing it keeps the staged protocol comparable. The suite uses the existing WAD on D: and requires no new map download or generated fixture. [Map inventory script](D:/Doom/DevForge_DOOM/submission_artifacts/final-blind-assessment/maps/inventory.py)

## Identity and episode order

The WAD SHA-256 is `a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b`. The frozen manifest SHA-256 is `5e3f7d1e96f578d4bdb37d44ce3ce78c557e990929940a0c8918fe0c4f34ab7c`. All 16 candidate families passed two distinct static fingerprint collision checks against one another and 51 excluded layout records. The first uses line endpoints, sidedness and blocking with translation, uniform scale and axis rotations/reflections normalized. The second fingerprints the sector adjacency graph independently of sector numbering and spawn or item placement. Neither hash alone proves all possible forms of geometry independence.

Use the fixed tier and map order, seeds 9101 and 9102 in that order, and matched runs of C749, Stack-25-174 and Stack-26-621. Use the same engine and policy randomness seed. Skill 3 and the official frame skip apply throughout. Local tick caps are 2,100 for tiers 1–2 and 4,200 for tiers 3–4. The minimum is 96 planned episodes. The 24- and 48-layout expansion targets have no fixed additional layouts. The same subset applies to all three policies, including failures.

## Exit measurement

The manifest lists exact static line segments and their activation type. MAP21, MAP22 and MAP25 contain a USE exit. The other selected maps contain crossing exits. For a crossing exit, report trigger proximity and actual engine-confirmed activation. A USE-exit step is inapplicable to those maps. Treat door rises and inferred passages as evaluator-side observations and use the engine’s exit flag for completion.
