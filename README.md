# DevForge DOOM

Neural agent for **Yugaantar 2026 · DevForge Track 1, “Can It Play DOOM?”**

The policy learns through live ViZDoom gameplay using PPO. It encodes the
official raycast, entity and vital readings and combines four frames of
features before choosing a legal button combination.

## Quickstart

Use the project’s Python 3.13 environment with PyTorch and the official wrapper
installed. Fetch the trained weights through Git LFS and run the native validator.

```bash
git lfs pull
python validate.py --device cpu
```

See the local training guide for the installed package versions, commands to
resume the PPO run, and the fixed-map evaluation procedure. [Training guide](training/LOCAL.md)

For container validation, use the organizer’s image or build it from a
separate checkout of the official starter. Pass the image to
`python validate.py --docker --image IMAGE`. The official evaluation runs on Linux.

## Repository layout

```
env/                  official ViZDoom Gymnasium wrapper (doom_env): spaces, sensors, protocol, limits
maps/                 small starter validation map
submission/agent.py   official DoomAgent entry point
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

**Compute tip:** training only on raw pixels is expensive. The `rays`, `visible_entities` and
`vitals` signals run thousands of steps per second on a CPU. Hybrid pixel + spatial models are welcome.

## Hardware at evaluation

One NVIDIA **RTX 3090 (24 GB)** shared between evaluation workers. Each agent gets **2 GB VRAM,
8 GB RAM and 1 CPU thread**, so keep models compact and inference fast
(see [hardware](docs/EVALUATION.md#hardware) and [timeouts](docs/EVALUATION.md#timeouts)).
Native validation reports the resource checks available on its host. Use the official Linux
image to check the scoring environment.

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
