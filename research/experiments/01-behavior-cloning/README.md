# Behavioral cloning and precision PPO

This experiment tested whether short, team-generated demonstrations improved C749-era combat and progression. The imitation and PPO-only arms started from the same archived Stack-26 update 621 lineage. C749 remained the competition model after fresh matched screening.

## Demonstration and training evidence

One registered teammate played 23 complete episodes across three known training families. The archive contains all 3,242 decisions, with gameplay metadata anonymized. Every stored button vector matched the ViZDoom readback. The verification checks the whole episode boundary, the causal previous-action chain, six-head targets, the frozen 378-value input packing and the payload SHA-256. [Dataset manifest](demonstrations/frozen-human-dataset.json) · [Recorder](source/record-human-demo.py) · [Read-only verification](verification/check-dataset.py)

| Training family | Episodes | Decisions | Outcomes |
|---|---:|---:|---|
| Red-key lesson | 14 | 2,017 | 13 exits and 1 cap |
| Basic aim | 4 | 233 | 4 exits |
| Center combat | 5 | 992 | 5 deaths |

Behavioral cloning used six categorical cross-entropy heads and intact episode history. It trained on the 18 red-key and Basic episodes with 2,250 decisions. The five Center combat episodes and 992 decisions were held out by layout family. Validation cross-entropy fell from 1.3480 to 0.3943 after 10 epochs and 310 supervised optimizer steps. The validation loss measures action agreement with those demonstrations. Closed-loop behavior was measured separately. [Supervised training receipt](training/behavioral-cloning-summary.json)

The two precision PPO arms each collected 32,768 fresh on-policy decisions and performed 768 optimizer steps. One initialized from BC and the other from archived u621. The difference isolates the initialization under the same rollout budget and precision training configuration. Their data stays separate from the human samples and the C749 training lineage. [PPO-only updater ledger](training/ppo-only-precision-1.jsonl) · [BC plus PPO updater ledger](training/bc-ppo-precision.jsonl)

## Matched screening

Forty episodes completed on four local layouts and two seeds per layout, with **6,352 legal decisions**. Balanced is the preregistered local score. [Protocol](evaluation/protocol.json) · [Full ledger](evaluation/screening.jsonl) · [Recalculated counts](evaluation/verified-counts.json)

| Model | Exits / survived, each out of 8 | Balanced mean | Player hits / damage | Keys / unlocks |
|---|---|---:|---|---|
| C749 | 5 / 8 | 6.225 | 0 / 0 | 1 / 1 |
| Stack-26 u621 | 2 / 5 | 3.988 | 1 / 8 | 0 / 0 |
| BC only | 2 / 5 | 3.854 | 0 / 0 | 0 / 0 |
| BC plus PPO | 2 / 4 | 4.812 | 3 / 40 | 0 / 0 |
| PPO only | 3 / 5 | 4.612 | 0 / 0 | 0 / 0 |

BC plus PPO recorded more player damage than the other models, alongside fewer exits and survivals than C749. BC alone reduced the frequency of left strafing, with 219 left requests in 1,173 decisions versus C749's 1,152 in 1,328, and still missed the progression gate. C749 was retained. No model from this experiment advanced to the reserved confirmation suite. [Promotion rules](evaluation/promotion-rules.json) · [Candidate identities](evaluation/candidate-manifest.json)

The original screening hash and the hash of the published, path-normalized ledger are both recorded. The four public screening WADs and all 40 gameplay traces accompany the ledger. The confirmation seal is represented by its commitment. [Artifact manifest](artifact-manifest.json) · [Reproduction](REPRODUCTION.md)
