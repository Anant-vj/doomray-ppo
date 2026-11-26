# Running the agent on the USB drive

Open PowerShell in the repository. The first-run script downloads the pinned simulator, installs CPU PyTorch, checks the policy, trains for 1,024 engine decisions in two visible windows, compares the resulting checkpoints to a uniform-random policy on two development maps, then launches the official validator with its local game window enabled.

```powershell
.\training\first_run.ps1
```

Setup, runs, caches and checkpoints use the external DevForge_DOOM directory. Set DOOM_WORK to an external directory if the repository is elsewhere. The default device is CPU. The short first run checks the full pipeline. Its sample count alone cannot establish competitive strength. The validator also checks Git and Git LFS state, so a freshly exported model can require a commit before validation passes.

## Training and evaluation

```powershell
.\training\run.ps1 train --memory stack --updates 64
.\training\run.ps1 train --memory gru --updates 64
.\training\run.ps1 benchmark --random --checkpoint ..\..\checkpoints\stack-17\policy-first.pt --checkpoint ..\..\checkpoints\gru-17\policy-first.pt --promote
.\training\run.ps1 benchmark --split holdout
.\training\run.ps1 demo --split dev --seeds 1001
.\training\run.ps1 validate --device cpu
```

Each new standalone training invocation starts from scratch unless given `--resume PATH`. Use distinct seeds for separate experiments to retain each run's candidates. Resuming restores the model, optimizer and counters and starts fresh simulator episodes. Candidate checkpoints store their actual update number. Compare the recent `policy-slot-*.pt` files too. Development and holdout runs use the selected submission checkpoint by default. All local engine entry points above show the game. The organizer runs its own display settings on the judging hardware.

## What the agent learns

The policy combines an ordered circular ray encoder, masked mean and max pooling over visible entities, scaled vitals, and the previous nine buttons. A four-step feature history or a GRU provides memory. Six learned categorical heads drive the nine buttons. Training adds bonuses for new grid cells, distinct rising door sectors, keys and exits, and penalizes damage. Only the training environment reads privileged state. The observation encoder shared with inference sees the official dictionary.

Training interleaves the starter two-room map and Freedoom MAP01–06. Development uses MAP07–08 and the final holdout uses MAP09–10. Benchmark JSON preserves every seed, map, return, exit, health, kills, keys, damage, explored grid cells, door proxy and latency. The proxy score is defined in evaluate.py. Only development comparisons can promote a model. A locked holdout check should be run once after selection.

## Implementation references

The simulator and bundled game content come from ViZDoom and Freedoom. The starter supplies the wrapper, contract and unmodified validator. CleanRL's recurrent Atari PPO served as an algorithm reference. The model and trainer in this project were written for the DevForge observation and action contract and initialize from scratch.

- [Official starter at the audited revision](https://github.com/Reinforce-SST/devforge-doom-template/tree/63b90c441ca92bcdf043de9ee3930ac6370a44bc)
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
- [Freedoom](https://github.com/freedoom/freedoom)
- [CleanRL recurrent PPO reference](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
