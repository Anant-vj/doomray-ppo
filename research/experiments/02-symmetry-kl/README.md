# Symmetry-aware PPO and a frozen C749 reference

This matched experiment tested a training loss that encourages consistent behavior on reflected observations while keeping the learner near the original C749 policy. Both arms used the same frozen C749 raw training parent, a fresh Adam optimizer, the mixed curriculum and the progression reward. They each collected **6,144 fresh decisions and 144 optimizer steps**. [Training plan](training/training-plan.json) · [Recalculated ledgers](evaluation/verified-counts.json)

## Correctness and training evidence

The reflection permutes the 64 rays with `(-i) % 64`, mirrors entity lateral bearings and velocities, and exchanges left/right movement and turn choices. The auxiliary objective compares the original and reflected policy after putting both in the same action coordinates. Symmetry uses coefficient `.02`, and the training-only `KL(C749 || learner)` reference uses `.10`. Neither auxiliary changes the inference entry point or categorical action decoder. [Source patch](source/implementation.patch) · [Reflection](source/training/reflection.py) · [Reference loss](source/training/reference.py)

Prior offline checks covered all 216 categorical actions, 419 recorded observations and 410 intact four-frame histories. They checked official sensor geometry, two reflections, previous-action alignment, entity masks and NumPy/PyTorch parity. The archived snapshot and three anonymized input payloads retain their as-tested hashes. The portable test adjusts only the data path. [Verification evidence](verification/verification-evidence.json) · [Test source](verification/check-reflection.py)

The final measured reflection KL was 3.594 nats in the symmetry arm and 4.859 in the control. The final reference KL was .103 and .093 nats, respectively. These are training diagnostics on the measured contexts. [Symmetry plus KL updates](training/symmetry-kl.jsonl) · [Matched PPO updates](training/ppo-control.jsonl)

## Matched screening

All 24 episodes on four new local layouts completed with **4,542 legal decisions**. Strafe percentages describe how often an action button was held across each model's decisions. [Ledger](evaluation/screening.jsonl) · [Protocol](evaluation/protocol.json) · [Recalculated counts](evaluation/verified-counts.json)

| Model | Exits / survived, each out of 8 | Balanced mean | Player hits / damage | Keys / unlocks | Left / right strafe |
|---|---|---:|---|---|---|
| C749 | 2 / 8 | 8.875 | 2 / 15 | 2 / 2 | 82.06% / 1.69% |
| PPO control | 4 / 8 | 5.625 | 0 / 0 | 0 / 0 | 75.70% / 1.78% |
| Symmetry + KL | 3 / 8 | 5.400 | 1 / 5 | 0 / 0 | 72.85% / 2.91% |

The symmetry arm reduced left-strafe frequency relative to the matched control, though the imbalance remained large. The two experimental arms exited more episodes, while both lost the key and unlock progress recorded by C749. C749 retained the highest Balanced mean and remained the competition model. [Candidate identities](evaluation/candidate-manifest.json)

The package contains the authored screening WADs and all 24 traces. Original source digests and published path-normalized digests are distinct in the manifest. [Artifact manifest](artifact-manifest.json) · [Reproduction](REPRODUCTION.md)
