# DoomRay PPO

**An autonomous player for ViZDoom, with a trained PyTorch model, a PPO trainer, and tools for evaluating gameplay on your own maps.**

A locked door is easy to recognize. Getting through it can mean leaving it behind, finding the matching keycard, and returning from a different direction while staying alive. DoomRay explores whether an agent can learn that sequence from experience when it can sense only its immediate surroundings.

The released agent, **C749**, takes the player's current sensor readings and returns the buttons to press. The simulator applies them and sends back the next observation. That loop controls movement, turning, firing, weapon selection, and interaction with doors and switches. You can run the trained agent directly, inspect how it makes decisions, or use the training code to build another policy.

![Recorded ViZDoom gameplay showing C749 collecting a red keycard, unlocking a door and reaching the exit](docs/media/c749-key-return.gif)

*In this 7.31-second excerpt, C749 picks up the card at decision 177, unlocks the door at 211, and exits at 231. We selected seed 3102 after testing. It was the one successful run in three attempts on this backtracking layout.* [Replay event log](https://github.com/Anant-vj/doomray-ppo/blob/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment/demos/c749-demonstration-runs/Backtracking-20261127-040413648/audit-report.json)

## Run your first episode

For a fresh CPU setup, use Python 3.13, Git LFS, and uv. The repository includes a small two-room map and the trained weights.

```bash
git clone https://github.com/Anant-vj/doomray-ppo.git
cd doomray-ppo
git lfs pull
uv sync --locked --no-dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python validate.py --device cpu
```

The validator loads C749, plays an episode on the bundled map, and reports the outcome, action validity, and inference timing. If you already have the competition environment, run `git lfs pull` and `python validate.py --device cpu` from the checkout. To try a different level, pass its WAD file and map name, for example `--map my_level.wad:MAP01`. [Environment setup](docs/SUBMISSION.md)

## Use the agent from Python

This is the complete decision loop for one episode. C749 samples actions, so repeated runs can follow different routes.

```python
from doom_env import make_env
from submission.agent import DoomAgent

env = make_env("maps/two_rooms.wad", "MAP01", episode_ticks=2100)
try:
    agent = DoomAgent(env.observation_space, env.action_space, device="cpu")
    observation, info = env.reset(seed=0)
    agent.reset()
    while True:
        buttons = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(buttons)
        if terminated or truncated:
            print("Reached exit:", info["exit_reached"])
            break
finally:
    env.close()
```

## How DoomRay learns to play

At each decision, **64 rays** describe the geometry around the player. A circular convolution treats the first and last rays as neighbors. A second encoder summarizes up to 32 visible entities. The policy combines those signals with player vitals and the previous action. It keeps the current feature vector and the three preceding ones, giving it short-term context about movement. Six learned action heads convert those features into the simulator's nine buttons. [Policy implementation](submission/policy.py) · [Observation and action contract](docs/CONTRACT.md)

During training, parallel workers play live episodes that mix exploration, combat, resource management, and keycard routes. Proximal policy optimization (PPO) uses the resulting rewards to update the action probabilities. C749's training lineage contains **324,864 simulator transitions and 7,614 optimizer steps**. Its released weights occupy **386 KiB**. [PPO implementation](training/ppo.py) · [Training commands and curriculum](training/LOCAL.md)

## What works today

| Question | Observed result |
| --- | --- |
| Can C749 complete a keycard route? | It exited 3/3 straight-route trials and 1/3 backtracking trials on controlled layouts. |
| Does it transfer to unfamiliar maps? | On 16 unfamiliar Freedoom layouts with two seeds each, C749 survived 3/32 games and recorded zero keys, unlocks, or exits. It also survived 0/8 on the earlier locked holdout. |
| What tends to go wrong? | In those 32 games C749 strafed left on 92.0% of decisions and right on 1.9%. Player counters credited eight hits and 66 damage. The engine counted 58 kills without identifying the attacker. |
| How expensive is inference? | CPU inference averaged 2.22 ms over 11,446 decisions, with an 8.84 ms 99th percentile. C749 passed native Windows validation. Official Linux Docker and competition GPU execution remain unverified. |

C749 was selected for **Yugaantar DevForge Track 1** on development maps. In the later 96-episode comparison, Stack-26 u621 scored higher on our local balanced measure (10.542 versus 9.895) and survived 8/32 games. A subsequent key-recovery model, U845, improved matched exits from 4/8 to 7/8 but died on a guarded route C749 had cleared. C749 remains the competition entry. [Independent assessment](https://github.com/Anant-vj/doomray-ppo/tree/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment) · [U845 paired results](research/key-recovery-u845/matched-results.md)

## Explore the code

| If you want to… | Start here |
| --- | --- |
| Load the checkpoint or integrate the player | [Agent interface](submission/agent.py) |
| Change the model or train a new policy | [Network](submission/policy.py) · [Training entry point](training/train.py) |
| Compare checkpoints on matched runs | [Evaluation code](training/evaluate.py) |
| Reproduce an experiment or inspect its evidence | [Research archive](research/README.md) |
| Verify which model you downloaded | [Frozen C749 release](https://github.com/Anant-vj/doomray-ppo/tree/6c2d5499f973c77b4a239f778837280edf64e57b) |

C749 weight SHA-256: `0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61`.
