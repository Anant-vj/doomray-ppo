# DoomRay PPO

A compact neural ViZDoom player for **Yugaantar 2026 · DevForge Track 1,
Can It Play DOOM?** The task is to navigate, survive, fight, and progress in
unseen maps using the official ego-centric observations.

The policy combines an ordered circular ray encoder, masked entity pooling,
player vitals, and a four-frame history. Its six trained categorical heads
produce the official nine-button action. We trained the model from
simulator rollouts using a custom PyTorch implementation of PPO and batched
inference across separate environment workers. The evaluation agent
reads the official observation and loads the committed Git LFS weights.

## Quickstart

Clone with Git LFS enabled. For a fresh native environment with Python 3.13
and [uv](https://docs.astral.sh/uv/), run the following.

```bash
git lfs pull
uv sync --locked --no-dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python validate.py --device cpu
```

Training and evaluation commands for the existing D: environment are in
[training/LOCAL.md](training/LOCAL.md). The official Linux image can be
used with the validator’s Docker mode when that image and engine are available.

**Platforms:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+) and Windows (x86-64) all
work with `uv sync` and `validate.py`. Intel Macs have no `vizdoom` 1.3.3 wheel: use Docker or a
Linux machine. Evaluation runs on Linux, so `validate.py --docker` is the closest check.

## Repository layout

```
env/                  official ViZDoom Gymnasium wrapper (doom_env): spaces, sensors, protocol, limits
maps/                 one small test map (NOT an evaluation map); bring your own for training
submission/agent.py   your DoomAgent: the evaluation entry point
submission/weights/   model weights (Git LFS)
training/             your training pipeline
validate.py           run before submitting
tests/                tests for the wrapper and validator (uv run pytest)
docs/                 contract, rules, submission guide, evaluation details
```

## Using the environment

```python
from doom_env import make_env

env = make_env("maps/two_rooms.wad")     # test map; use your own maps for training
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Observations are a dict of `rgb`, `depth`, `vitals`, `rays` and `visible_entities`, all
ego-centric and partially observable. See [docs/CONTRACT.md](docs/CONTRACT.md).
The wrapper's `reward` is a basic default for convenience. Shape your own reward for training.

Headless training uses rays, visible entities, and vitals. Local checkpoint
selection uses matched seeds, maps, and tick caps. The reports separate
development, inspected diagnostic maps, and the final untouched holdout.

## Checkpoint selection

The selected frame-stack checkpoint is progression run 43, update 749.
Its lineage contains 324,864 gameplay decisions and 7,614 optimizer steps.
Evaluation compares policies on identical seeds, maps and 2,100-tick caps.
The local score proxy combines combat, exploration, progression and survival.
The official scoring weights are private.

| Four-map development, 16 episodes per policy | Proxy | Alive | Kills | Door events |
| --- | ---: | ---: | ---: | ---: |
| Selected progression update 749 | 11.219 | 3/16 | 54 | 21 |
| Original stack update 174 | 6.245 | 9/16 | 20 | 7 |
| Seeded random | 4.855 | 10/16 | 15 | 9 |

The selected checkpoint led the documented balanced, survival-combat
and progression profiles. The gains came with lower survival. It produced
zero invalid actions in the development suite.

The final holdout used previously untouched MAP11 and MAP12, with seeds
1001–1004. The checkpoint was frozen before the first holdout episode.

| Final holdout, 8 episodes per policy | Proxy | Alive | Kills | Door events | Mean explored cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected progression update 749 | 4.570 | 0/8 | 9 | 4 | 24.125 |
| Original stack update 174 | 2.263 | 0/8 | 7 | 0 | 19.750 |
| Seeded random | 0.853 | 0/8 | 6 | 1 | 10.625 |

C749 led on both holdout maps and won six of eight matched seed comparisons.
Across all 24 episodes, each policy recorded zero keys, unlocks, exits and
invalid actions. Full-level survival and progression remain weak.

## Hardware at evaluation

One NVIDIA **RTX 3090 (24 GB)** shared between evaluation workers. Each agent gets **2 GB VRAM,
8 GB RAM and 1 CPU thread**, so keep models compact and inference fast
(see [hardware](docs/EVALUATION.md#hardware) and [timeouts](docs/EVALUATION.md#timeouts)).
`validate.py` runs your agent under the same CPU/RAM limits and reports GPU memory.

## Documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |

## Rules in one minute

- Actions must come from **trained neural parameters**: no rule trees, no `A*`/Dijkstra, no `atan2` aimbots,
  no unmodified off-the-shelf SB3.
- Change anything you want **during training**. Evaluation uses the **vanilla engine**, the
  official wrapper and **unseen maps**.
- Create the repo during the hackathon. Libraries are fine; the core architecture and training must be yours.
