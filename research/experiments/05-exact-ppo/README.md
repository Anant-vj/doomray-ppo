# Experiment 5 · Exact PPO and balanced optimization

Three arms continued from the frozen C749 actor and critic with fresh Adam states. Each collected 12,288 actual transitions across 48 rollouts and made 288 new optimizer steps. Two separate live smokes collected 1,024 actual transitions and made 24 optimizer steps each. Their logs are retained separately. [Training configuration](training/plan.json) · [Run logs and exposure accounting](training/) · [Original and published hashes](manifest.json)

## What changed

- Exact PPO recomputes embeddings for the current packed frame and its three causal predecessors during each optimizer minibatch. Reset states use the original zero padding. The stored behavior action and old log probability are preserved. The ratio remains the exponential of the difference between current and stored joint log probabilities. The historical comparison passed 32 direct and 256 batched checks, with an initial absolute log ratio no larger than 1.91e-6. The exact-history live smoke's maximum was 9.54e-7.
- The balanced arms reduced the coverage coefficient from .035 to .012 and canceled the original engine-counted map-kill reward. Verified player-attributed damage supplied `.006 * min(damage_delta, 20)` per decision, capped at .12 per decision and .60 per episode. Existing progression and survival signals remained. Training records retain the distinct hit, damage, map-kill and coverage components.
- The family scheduler uses observed live transition exposure. By family, exact PPO recorded 785 / 4,616 / 96 / 6,791 decisions, balanced reward 3,554 / 2,973 / 3,591 / 2,170, and the combined arm 3,885 / 2,948 / 3,589 / 1,866. The arrays in the training bundle retain per-update receipts.
- The combined arm added the bounded shaping potential and the same 556 frozen training-only anchor contexts at KL weight .04. The combined arm recorded 11 complete annotated episodes with maximum shaping-identity residual around 8.3e-17. The offline GAE sensitivity analysis compared .95 and .97 on the archived training-only trajectories. At .97, advantage variance rose 31.5% and the 32-decision pulse weight rose from .1404 to .2735. No learning run used the alternate decay. [Archived correctness reports](verification/) · [Reward implementation](source/training/credit.py) · [Anchor inputs](../04-credit-preservation/anchors/)

The exact computation increased measured optimizer work. Mean optimization was approximately 77 ms for cached histories and 136 ms for exact histories in the short live smokes. End-to-end short-smoke rates were 86.8 and 93.9 transitions per second respectively. The measurements do not establish a steady-state throughput advantage.

## Matched screening

All 48 episodes completed over 12 matched cases, with 11,078 legal decisions and zero illegal ones.

| Model | Exits | Survived | Keys | Unlocks | Player hits | Player damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C749 | 1 | 5 | 0 | 0 | 3 | 31 |
| Exact PPO | 1 | 6 | 0 | 0 | 4 | 46 |
| Balanced reward | 0 | 5 | 0 | 0 | 3 | 35 |
| Balanced reward with PBRS and anchors | 2 | 6 | 0 | 0 | 3 | 22 |

All continuations missed C749's protected alternating-chicanes exit. All 16 plays on the larger bundled Freedoom1 E1M2 and E2M1 layouts yielded zero exits, survivors, keys, and unlocks. Recorded starting positions confirmed that both requested maps loaded. No candidate passed the gate, and the unused confirmation layouts remained sealed. [Episode ledger](evaluation/screening.jsonl) · [Paired report](evaluation/paired-results.md) · [Protocol](evaluation/protocol.json)

## Reproduction

For counts and integrity, run `python -B research/tools/verify.py` from the research checkout. With NumPy and CPU PyTorch available, the following offline commands rerun the archived numerical checks. Git LFS must materialize the frozen C749 export. The history check uses the original human numeric fixture in the first research bundle and executes its synthetic parity checks. The GAE script reads the 20 archived training-only attempt streams.

```sh
git lfs pull --include="submission/weights/policy.pt"
python -B research/experiments/05-exact-ppo/verification/check-exact-history.py
python -B research/experiments/05-exact-ppo/verification/gae-sensitivity.py
```

For an independently authorized live reproduction, apply the scoped training patch in a new detached checkout and use the locked Python 3.13 and ViZDoom 1.3.3 environment. The published comparison used the bundled Freedoom1 IWAD with SHA-256 `7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d`. Four local screening WADs and three public training variants are included in this bundle.

```sh
research_root=$(pwd)
bundle="$research_root/research/experiments/05-exact-ppo"
git worktree add --detach ../doomray-repro 6c2d5499f973c77b4a239f778837280edf64e57b
git -C ../doomray-repro apply "$bundle/source/training.patch"
export DOOM_WORK="$(dirname "$research_root")/doomray-repro-artifacts"
mkdir -p "$DOOM_WORK/training-maps"
cp "$bundle"/maps/training/*.wad "$DOOM_WORK/training-maps/"
export APPROVED_EXPORT="$research_root/submission/weights/policy.pt"
python -B - <<'PY'
import os
from pathlib import Path
import torch
saved = torch.load(os.environ["APPROVED_EXPORT"], map_location="cpu", weights_only=True)
initial = dict(memory=saved["memory"], weights=saved["weights"], updates=749, gradients=7614)
torch.save(initial, Path(os.environ["DOOM_WORK"]) / "C749-initial.pt")
PY
cd ../doomray-repro
export PYTHONPATH="$PWD/env"
python -B -m training.train --headless --parallel --no-render-buffers --workers 4 --horizon 64 \
  --updates 48 --ticks 2100 --memory stack --device cpu --seed 66143 --curriculum mixed \
  --reward-profile progression --learning-rate .00008 --initialize "$DOOM_WORK/C749-initial.pt" \
  --credit-mode exact-ppo --run-name exact-ppo
```

Repeat in separate run directories with modes and run names `balanced-reward` and `balanced-pbrs-anchor`. For the combined arm, append `--anchor-file` pointing to the original [anchor data](../04-credit-preservation/anchors/anchor-data.npz), `--anchor-sha256 fd32f3c049b0c3e221ea3ba223f176ef13cae82b7759e7440f5e228322a9d87c`, and `--anchor-beta .04`. The reconstructed initializer uses the same weights and inherited counters, with no optimizer state restored. Archive manifests give the original training-file and checkpoint hashes. Local runtime and scheduler measurements can vary across hosts.
