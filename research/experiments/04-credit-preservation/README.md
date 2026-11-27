# Experiment 4 · mathematical credit assignment

Three matched arms began from the same frozen C749 parent with fresh Adam. Four parallel CPU workers collected 64 decisions each per update. Each arm contributed 8,192 live transitions, 32 PPO updates and 192 new optimizer steps. These figures exclude inherited C749 training counters and the separate anchor collection. The curriculum sampled equally among four families at episode resets, then uniformly within each family. Episode duration produced unequal decision exposure. [Frozen configuration](training/training-plan.json) · [Training aggregates](training/training-statistics.json)

| Arm | Native and base shaped reward | Potential | Frozen successful-history KL |
| --- | --- | --- | --- |
| PPO control | Original progression profile | Off | Off |
| PBRS | Same | Enabled on two training lessons | Off |
| PBRS + anchors | Same | Same | β = .04 |

## Reward and credit

The original dense terms remained in all arms. The additional term was `0.99 * Φ(s') - Φ(s)`. It used the red-card, matching lock, passage and exit in the two hash-verified training corridor lessons. Φ was bounded at .25 and used a zero terminal value at an actual exit, death or finite game timeout. A rollout cut retained the episode history and the true successor value. The offline test checked both WAD annotations and the discounted telescoping identity across 128 synthetic complete trajectories. Training ledgers preserve the original rewards and potential term separately. [Potential source](source/policy-worktree/training/potential.py) · [Test source](verification/check-potential.py) · [Test output](verification/potential-check.stdout.txt)

The anchor added the sum of six headwise teacher-to-learner KL terms on recorded successful histories. The policy still sampled fresh actions for PPO and its importance ratios. The public anchor data comes from 20 training-only C749 attempts, with 3,417 legal engine-read-back decisions. The compact 556 selected contexts represent 495 unique decisions and can overlap by objective. All full numeric streams remain byte-identical, including their native rewards, causal previous actions and true end flags. [Anchor collection plan](anchors/collection-manifest.json) · [All attempts](anchors/attempts.jsonl) · [Validation](anchors/validation.json) · [Anchor implementation](source/policy-worktree/training/preservation.py)

The mathematical audit separated map-wide monster deaths from actual player hits and damage. On a small complete-episode cohort, the actor-gradient cosine between native map-death reward and key-route exit reward was −0.391. Key acquisition and locked-sector rise proxy directions had cosine +0.735. These estimates have limited support, and Basic combat completion, map-death and hit/damage pulses had overlapping event support. The estimates establish neither causation nor a basis for gradient surgery. [Mathematical audit](mathematical-audit.md) · [Complete-episode diagnostic script](verification/anchor-rollout-gradients.py) · [Measured gradient directions](verification/anchor-rollout-gradients.json)

## Matched screen

All 40 episodes completed across five local layouts and two matched seeds per model. The 8,275 decisions were legal.

| Model | Exits | Survived | Keys | Unlocks | Player hits | Player damage | Balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 1 | 8 | 2 | 2 | 2 | 20 | 7.7452 |
| PPO control | 1 | 8 | 2 | 1 | 2 | 19 | 7.6583 |
| PBRS | 3 | 9 | 1 | 1 | 0 | 0 | 6.6174 |
| PBRS + anchors | 4 | 8 | 1 | 1 | 0 | 0 | 6.5348 |

Both shaped arms added exits. PBRS + anchors lost a key pickup and an unlock on the lock-recovery-return family that C749 recorded. Both shaped arms recorded zero direct player hits or damage. The frozen gate required retention of progression and combat performance, and no alternative passed. The confirmation cohort stayed sealed. C749 remained approved. [Raw episode ledger](evaluation/screening.jsonl) · [Paired outcomes](evaluation/paired-brief.md) · [Full summary](evaluation/screening-summary.json)

## Reproduce the aggregate

From this directory, run the command below with Python 3.13. It recomputes the matched scope and compares all published numerical aggregates against the recorded report without an engine.

```sh
python -B ../03-inference-decoding/verify-scores.py --bundle .
```

The historical live launch arguments, checkpoints and reward ledger paths are in [training](training/). To study the original procedure, use the archived source alongside a scratch checkout of the exact C749 base and a pre-existing compatible Python 3.13 / ViZDoom 1.3.3 runtime. Original environment guards and hash-pinned assets remain required for any authorized live reproduction. The published source is an archival snapshot. Private freeze files, binary weights and WAD assets stay out of this bundle. The known training keycard and backtrack files are identified by SHA-256 in the potential source and training plan. `${DOOMRAY_PROJECT_ROOT}` denotes a private research workspace when reading the archived native launch paths. [Training source](source/policy-worktree/training/) · [Original and published hashes](manifest.json)
