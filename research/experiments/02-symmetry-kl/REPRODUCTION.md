# Reproduction

## Review offline

From this experiment directory, check the published ledger hash and read the expected records and counts. The original byte hash is separately recorded in `evaluation/verified-counts.json` because the published ledger rewrites machine paths.

```powershell
Get-FileHash -Algorithm SHA256 ./evaluation/screening.jsonl
Get-Content ./evaluation/verified-counts.json
```

The reflection test uses the neighboring behavioral-cloning experiment's three valid numeric payloads and the official sensor and actor code. From the experiment directory, apply the implementation patch to a separate detached checkout at the base commit named in `artifact-manifest.json`. Set `DOOM_REPRO_CHECKOUT` to that checkout and `DOOM_PYTHON` to the existing Python 3.13 executable. This reproduced test checks structured geometry and tensor transformations offline.

```powershell
$artifact = (Resolve-Path .).Path
$checkout = (Resolve-Path $env:DOOM_REPRO_CHECKOUT).Path
git -C $checkout apply --check (Join-Path $artifact 'source/implementation.patch')
git -C $checkout apply (Join-Path $artifact 'source/implementation.patch')
$env:PYTHONPATH = "$checkout;$checkout/env"
& $env:DOOM_PYTHON -B (Join-Path $artifact 'verification/check-reflection.py')
```

The archived reflection module snapshot and original test input hashes appear in `verification/verification-evidence.json`. The test source's path-only adaptation has a separate published hash in `artifact-manifest.json`.

## Reconstruct the matched run

Use the existing Windows runtime with ViZDoom 1.3.3 and PyTorch 2.15.0 CPU, as recorded in `evaluation/runtime.json`. An exact training rerun needs the original update-749 raw checkpoint with the digest in `training/training-plan.json`. The published inference export has a distinct file digest and lacks that training metadata. Use an external-drive experiment output directory and the actual approved storage baseline for the host. Operator-supplied checkpoints and runtime paths are logical locations in the public manifests.

Run `training.train` from the detached checkout with CPU stack memory, four headless workers, horizon 64, 24 updates, 2,100 ticks, seed 34219, learning rate `6e-5`, the mixed curriculum and the progression reward. Both arms initialize the same raw parent. Set `--reference-coef 0.10 --symmetry-coef 0.02` for the symmetry arm and both values to zero for the control. Use independent run names. `source/run-matched-ppo.ps1` retains the historical command and time limits. [Recorded configurations](training/training-plan.json)

For screening use the four included WADs and the original seeds, caps and scores in `evaluation/protocol.json`. The source evaluator snapshot enforced the original byte hashes and campaign cutoff. The published protocol preserves the same scientific fields with normalized paths and its own digest. [Published artifact manifest](artifact-manifest.json)
