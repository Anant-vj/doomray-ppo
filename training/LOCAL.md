# Running the agent on the USB drive

Open PowerShell in the repository. The first-run script downloads the pinned simulator, installs CPU PyTorch, checks the policy, trains for 1,024 engine decisions in two visible windows, compares the resulting checkpoints to a uniform-random policy on two development maps, then launches the official validator with its local game window enabled.

```powershell
.\training\first_run.ps1
```

Setup, runs, caches and checkpoints use the external DevForge_DOOM directory. Set DOOM_WORK to an external directory if the repository is elsewhere. The default device is CPU. The short first run checks the full pipeline. Its sample count alone cannot establish competitive strength. The validator also checks Git and Git LFS state, so a freshly exported model can require a commit before validation passes.

## Headless training and evaluation

From the repository root, load the D: workspace configuration and use the existing
competition Python environment. This example resumes the verified Phase 1
checkpoint into a new run directory.

```powershell
. .\training\usb.ps1
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
$resume = Join-Path $env:DOOM_WORK 'checkpoints\stack-23\resume.pt'
& $python -B -m training.train --headless --memory stack --device cpu --seed 24 --workers 2 --horizon 64 --ticks 1050 --updates 512 --checkpoint-seconds 360 --resume $resume
```

Add `--parallel` for independent spawned simulation workers with batched policy
inference. `--no-render-buffers` skips the wrapper's RGB and depth observation
copies. In parallel mode it also disables native depth generation inside the
training workers. The policy uses rays, entities and vitals.
`--curriculum` adds ViZDoom Basic, My Way Home and Freedoom MAP31. Select the
worker count using local throughput measurements. `--curriculum progression`
also adds the starter's small red key, locked door and exit test layout.
`--curriculum mixed` adds the bundled Defend the Center, Health Gathering,
and Deadly Corridor scenarios for arena combat, resource collection, and
corridor encounters.

Use `--checkpoint-seconds 360` for six-minute checkpoint rotation. The trainer
saves after the first and last update as well. For a run with `--stop-file PATH`,
creating that file requests a checkpoint and a clean exit at the next update.
`--max-seconds` also saves and exits at an update boundary.
`--storage-stop-bytes` can request an earlier checkpoint when the internal
drive approaches the 2 GB budget. This measures the whole drive's free-space
change from the baseline recorded in the external workspace.

For matched reward trials, give each arm its own `--run-name` while keeping
the seed and `--initialize PATH` identical. Initialization loads the model
weights and starts a fresh optimizer. `--resume PATH` restores optimizer
state and starts fresh simulator episodes. An explicit `--learning-rate`
takes precedence over the saved learning rate.

The control reward uses the original coefficients. The survival profile
raises the damage cost from 0.003 to 0.008 per health point and reduces the
extra finish bonus on three easy practice maps from 2.0 to 0.35. Episode
logs keep the individual reward terms. The progression profile uses a damage
cost of 0.006 per health point and key and unlock bonuses of 0.45 and 0.25.
The combat profile adds a small bonus for damage actually dealt and health
actually recovered. It caps the amounts that earn reward per decision at 60
and 25 health points respectively.
Rollout logs include KL divergence,
clipping, entropy, value loss, explained variance, gradient norms, button
frequencies, and collection and learning times. PPO uses successor values
for continuing rollouts and ends bootstrapping at the game tick cap.

The headless evaluation below uses the same maps, per-map seeds and tick cap
for every policy and writes episode metrics to the D: workspace.

```powershell
$run = Join-Path $env:DOOM_WORK 'checkpoints\stack-23'
& $python -B -m training.evaluate --headless --split dev --random --seeds 1001 1002 1003 1004 --ticks 2100 --checkpoint (Join-Path $run 'policy-first.pt') --checkpoint (Join-Path $run 'policy-slot-0.pt')
& $python -B .\validate.py --device cpu
```

## Visible diagnostics

```powershell
.\training\run.ps1 train --memory stack --updates 64
.\training\run.ps1 train --memory gru --updates 64
.\training\run.ps1 benchmark --random --checkpoint ..\..\checkpoints\stack-17\policy-first.pt --checkpoint ..\..\checkpoints\gru-17\policy-first.pt --promote
.\training\run.ps1 benchmark --split holdout
.\training\run.ps1 demo --split dev --seeds 1001
.\training\run.ps1 validate --device cpu
```

Each standalone invocation starts from scratch unless given `--resume PATH` or `--initialize PATH`. Candidate checkpoints store their update number. Compare the recent `policy-slot-*.pt` files too. Development and holdout runs use the selected submission checkpoint by default. The visible commands in this section show the game. The organizer runs its own display settings on the judging hardware.

## What the agent learns

The policy combines an ordered circular ray encoder, masked mean and max pooling over visible entities, scaled vitals, and the previous nine buttons. A four-step feature history or a GRU provides memory. Six learned categorical heads drive the nine buttons. Training adds bonuses for new grid cells, distinct rising door sectors, keys and exits, and penalizes damage. Only the training environment reads privileged state. The observation encoder shared with inference sees the official dictionary.

Training interleaves the starter two-room map and Freedoom MAP01–06. Development uses MAP07–08 and the broader development suite adds MAP13–14. The earlier MAP09–10 holdout is now the diagnostic suite because its outcomes informed training changes. MAP11–12 are reserved as `final-holdout` for one post-selection check. Benchmark JSON preserves every seed, map, return, exit, health, kills, keys, damage, explored grid cells, door proxy and latency. `training.rank` compares reports from identical episodes using several documented progress weights. For local trajectory evidence on MAP09–10, run `python -B -m training.diagnose --checkpoint PATH`.

## Implementation references

The simulator and bundled game content come from ViZDoom and Freedoom. The starter supplies the wrapper, contract and unmodified validator. CleanRL's recurrent Atari PPO served as an algorithm reference. The model and trainer in this project were written for the DevForge observation and action contract and initialize from scratch.

- [Official starter at the audited revision](https://github.com/Reinforce-SST/devforge-doom-template/tree/63b90c441ca92bcdf043de9ee3930ac6370a44bc)
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
- [Freedoom](https://github.com/freedoom/freedoom)
- [CleanRL recurrent PPO reference](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
