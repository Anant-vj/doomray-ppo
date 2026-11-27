# DoomRay PPO

**Yugaantar 2026 · DevForge Track 1 · Can It Play DOOM?**

On a small ViZDoom map, our agent learned to pick up a red keycard, return to a locked door and reach the exit. When we moved it to unfamiliar layouts, that sequence broke down. We entered **C749**, a compact policy trained entirely through live gameplay.

## One run from keycard to exit

![Seven seconds of C749 picking up the red keycard, returning to the locked door and reaching the exit. Captions show the recorded decision numbers.](docs/media/c749-key-return.gif)

| Decision 177 | Decision 211 | Decision 231 |
| --- | --- | --- |
| Collected the red keycard behind the spawn | Returned to the door and unlocked it | Reached the exit |

These 7.31 seconds come from seed **3102**, selected after the trials. C749 finished this backtracking layout once in three attempts. On the straight keycard route it finished all three. [Recorded events](https://github.com/Anant-vj/doomray-ppo/blob/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment/demos/c749-demonstration-runs/Backtracking-20261127-040413648/audit-report.json)

## How C749 chooses an action

1. **Read the geometry.** A circular convolution connects the first and last of 64 rays around the player. The policy pools up to 32 visible entities and reads player vitals and its previous action.
2. **Keep four decisions in view.** A feed-forward layer combines the current features with the previous three. This short history gives the agent motion context and resets with each game.
3. **Choose the buttons.** Six learned heads sample movement, strafe, turn, attack, use and weapon selection. Together they produce the nine-button action the simulator accepts. [Policy code](submission/policy.py)

We collected games in parallel on headless CPU workers and trained with our PyTorch implementation of proximal policy optimization. The curriculum mixed exploration, key-and-door drills, combat and resource management. Rewards included visiting new areas, collecting keys and opening locks. [PPO trainer](training/ppo.py) · [Training guide](training/LOCAL.md)

| Live transitions in C749’s lineage | PPO optimizer steps | Released weights |
| ---: | ---: | ---: |
| 324,864 | 7,614 | 386 KiB |

## What happened on unfamiliar maps?

Three frozen policies played the same **16 Freedoom layouts**, with two seeds per layout. Across all **96 episodes**, every one of the **41,035 decisions** was legal. None of the three models collected a key, unlocked a door or reached an exit.

| Checkpoint | Survived | Local balanced score |
| --- | ---: | ---: |
| **C749 · competition entry** | **3/32** | **9.895** |
| Stack-26, update 621 | 8/32 | 10.542 |
| Stack-25, update 174 | 11/32 | 7.057 |

C749 had also recorded zero survivors, keys, unlocks and exits on the original eight-game holdout. On the 32 independent runs it strafed left on 92.0% of decisions and right on 1.9%. Player counters credited eight hits and 66 damage. The engine counted 58 kills without identifying the attacker. The competition score is unknown. [Independent assessment](https://github.com/Anant-vj/doomray-ppo/tree/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment) · [Combat and movement measurements](https://github.com/Anant-vj/doomray-ppo/blob/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment/reports/observed-runtime-comparison.json)

### Why we kept C749

We selected C749 before the independent exam. Across 16 development games it led its parent, u621, by 11.219 to 9.854 on our balanced score. Their ranking reversed on unfamiliar layouts. [Development results](https://github.com/Anant-vj/doomray-ppo/blob/6c2d5499f973c77b4a239f778837280edf64e57b/README.md#checkpoint-selection)

The later U845 candidate reached the exit in **7/8** matched games versus **4/8** for C749. It finished all four key-return trials. Each agent cleared one of two guarded runs, but U845 died on the seed C749 had cleared. Our promotion rule required a replacement to preserve previously successful routes, so C749 remained the entry. [Paired runs](research/key-recovery-u845/matched-results.md) · [Research archive](research/README.md)

## Run the submitted policy

From the repository root in the competition Python 3.13 environment:

```bash
git lfs pull
python validate.py --device cpu
```

C749 passed native Windows validation. Across 11,446 independent gameplay decisions, CPU inference averaged **2.22 ms**, with **8.84 ms** at the 99th percentile. Official Linux Docker and competition GPU compatibility remain unverified. [Validator log](https://github.com/Anant-vj/doomray-ppo/blob/fc69c00d43479dd93baeb0f399e15647629d8b9f/research/key-recovery-u845/logs/validator-C749.txt) · [Environment setup](docs/SUBMISSION.md)

**Model identity:** Progression C, run 43, update 749. [Frozen release](https://github.com/Anant-vj/doomray-ppo/tree/6c2d5499f973c77b4a239f778837280edf64e57b) · [Evaluator entry point](submission/agent.py)

Git LFS weight SHA-256: `0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61`
