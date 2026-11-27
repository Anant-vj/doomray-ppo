# DoomRay PPO research archive

DoomRay PPO learns combat, navigation, and keycard progression from structured ViZDoom observations and emits the nine buttons in the official action contract. The frozen competition submission is [C749 on main](https://github.com/Anant-vj/doomray-ppo/tree/6c2d5499f973c77b4a239f778837280edf64e57b), Progression C, run 43, update 749. This branch documents five follow-up campaigns. It contains no alternative model weights. The [96-episode independent assessment](https://github.com/Anant-vj/doomray-ppo/tree/e9fe1740994a78b3bfb650017061b8e832d7a957) is a separate publication.

## Submitted policy and training

The [policy](../submission/policy.py) encodes 64 distance and hit-type rays with a circular 1D CNN and pools visible entities with masked means and maxima. It fuses those features with scaled vitals and the nine previously accepted buttons. C749 feeds the current embedding and three recent embeddings to a feed-forward layer for short-term temporal context. The implementation also contains a GRU variant. C749 uses the four-decision stack.

Six categorical heads sample movement, strafe, turn, attack, use, and weapon selection. Their log probabilities sum to the joint action log probability. PPO uses the exponential of the new minus the stored behavior log probability as the importance ratio, clipped at 0.2 around 1 in the policy surrogate. Training used generalized advantage estimation with discount 0.99 and decay 0.95, a value loss and entropy bonus, and Adam. The headless CPU curriculum mixed navigation, combat, survival, and progression. C749 uses the original categorical sampler at temperature 1.

## Experimental modules and selection

| Campaign | Verified result | Decision |
| --- | --- | --- |
| [Human demonstrations and behavioral cloning](experiments/01-behavior-cloning/) | 23 human episodes supplied 3,242 accepted actions. Held-out imitation cross-entropy fell from 1.348 to 0.394. Imitation plus PPO recorded 3 hits and 40 player-attributed damage, but exited 2/8 and survived 4/8 versus 5/8 and 8/8 for C749. | Supervised accuracy and improved combat did not carry over to matched progression and survival. |
| [Reflection and reference-policy KL](experiments/02-symmetry-kl/) | Absolute turn imbalance fell from 31.3 to 15.8 percentage points with symmetry KL. C749 collected two keys and opened two locks. Both continuations collected no keys and opened no locks. | The progression gate rejected both continuations. |
| [Inference decoding](experiments/03-inference-decoding/) | Temperature 0.7 cleared the matched screen against temperature 1, 1.3, and argmax. In an independent two-layout confirmation it exited 1/4 versus 2/4 for the original sampler. | The original sampler retained its place after the broken-rampart exit regression. |
| [Potential-based shaping and successful-trajectory anchors](experiments/04-credit-preservation/) | The discounted shaping identity passed a test on 128 complete synthetic trajectories. PBRS with anchors exited 4/10 versus 1/10 for C749 but lost one key and one unlock and recorded no attributable damage. | The protected progression and combat checks rejected the shaped candidates. |
| [Exact temporal-history PPO and balanced optimization](experiments/05-exact-ppo/) | Four-frame reconstruction matched sequential inference and retained stored behavior log probabilities. Balanced PBRS plus anchors exited 2/12 versus 1/12 for C749. Every continuation missed the chicanes exit C749 achieved. All four models failed progression on the larger Freedoom cases. | No continuation met the frozen gate. The GAE sensitivity comparison was offline. |

C749 was selected and frozen in the earlier development process. In the later independent assessment, Stack-26 u621 outperformed C749 on the unofficial balanced score (10.542 vs 9.895) and survival (8/32 vs 3/32). Across that broader examination, all three models recorded zero keys, unlocks, and exits. The separate original locked holdout for C749 yielded zero survivors, keys, unlocks, and exits. These results document generalization and navigation failures. Post-freeze screens were used to investigate causes and test predeclared promotion gates. No result led to a checkpoint replacement.

## Verified evidence and scope

| Evidence scope | Transitions or decisions | Rollouts or episodes | PPO optimizer steps |
| --- | ---: | ---: | ---: |
| C749 training lineage | 324,864 | 749 rollouts | 7,614 |
| All legacy learner jobs, including the C749 lineage | 618,432 | 1,452 rollouts | 14,499 |
| Later primary research learner arms | 139,264 | 544 rollouts | 3,264 |
| Two later live smoke runs, separately identified | 2,048 | 8 rollouts | 48 |
| All logged PPO learner jobs, including the later smoke runs | **759,744** | **2,004 rollouts** | **17,811** |
| Prior independent assessment | 41,035 legal | 96 episodes | |
| Five later campaigns, including eight decoder confirmation episodes | 36,668 legal | 192 episodes | |
| Canonical matched evaluations | **77,703 legal** | **288 episodes** | |

The 310 supervised cloning updates, human demonstrations, teaching rollouts, archived duplicate evidence, and all evaluation play have separate provenance. The later matched evaluation series contains zero illegal actions. The old 93-episode snapshot is a prefix of the published 96 and is counted once. The verified training total excluding the two explicitly separate later smokes is 757,696 transitions, 1,996 rollouts, and 17,763 optimizer steps. [Historical counter evidence](history/) · [Offline verification](tools/verify.py)

C749 completed the controlled keycard route in 3/3 diagnostic trials and the backtracking route in 1/3. The successful demonstration seeds replay measured, previously selected successes. The evaluation suites above used matched seeds and distinct layouts. On the broader 96-episode examination, C749 strafed left on 92.0% of its decisions and right on 1.9%. It recorded 8 verified player hits and 66 player-attributed damage in its 32 episodes. The 58 engine-recorded kills are a separate map counter. Native action latency averaged 2.22 ms over 11,446 C749 decisions in that examination. The frozen release passed its native Python 3.13 validator. Official Docker, GPU and scorer compatibility remain unverified. No figures are official competition scores.

## Reproduction and artifact integrity

From a normal checkout of this branch, `python -B research/tools/verify.py` validates the SHA-256 archive index and recomputes the published training and evaluation totals without launching ViZDoom. Add `--assessment-repo ../doomray-ppo-assessment` to read the original 96 episodes from a local clone containing the immutable assessment commit. Each experiment documents the archived source, configs, matched scope, and additional offline checks. For live reproduction, use a separate checkout of the approved commit with the matching dependencies and each experiment's archived configuration. Original and publication hashes document intentional path normalization. Files marked as private archives or traces are provenance references. Candidate hashes are retained, and sealed layouts remain private.

The original model hash is `0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61`. The two-page organizer brief is separately frozen at SHA-256 `f7510313bd1f952deee4c7020ff367a624ab1b7a07292648a4da48d82909942a`. The official release contract and validator remain in the approved tree. [Submission guide](../docs/SUBMISSION.md)

## Research references and future directions

- PPO importance ratios and the clipped surrogate. [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- Generalized advantage estimation and temporal credit. [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
- Finite episode horizons and rollout truncation. [Pardo et al., 2018](https://proceedings.mlr.press/v80/pardo18a.html)
- Potential-based reward transformations. The telescoping check supports the implemented shaping identity for the annotated training tasks, within the stated terminal handling. Transfer remains an empirical question. [Ng et al., 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Preservation through replay and curriculum sampling. CLEAR and Prioritized Level Replay informed the interpretation. The anchors used a fixed set of successful histories. The credit experiment sampled families uniformly at resets, and the balanced optimization experiment used adaptive transition feedback. [Rolnick et al., 2019](https://arxiv.org/abs/1811.11682) · [Jiang et al., 2021](https://proceedings.mlr.press/v139/jiang21b.html)

Future work could test better key and lock exploration, improve player-hit attribution, and challenge directional bias on diverse procedural layouts with fresh protected comparisons. This repository records the completed experiments and their limits.
