# Key recovery PPO, update 845

U845 trained from C749 on eight short keycard recovery drills. It completed 24,576 live transitions and 576 additional PPO optimizer steps using the four-frame policy and the original categorical sampler. The original experiment commit contains the U845 weights at `submission/weights/policy.pt`. The study’s predeclared rule recommended C749. [U845 checkpoint and code](https://github.com/Anant-vj/doomray-ppo/tree/fc69c00d43479dd93baeb0f399e15647629d8b9f)

The candidate exited all four held-out backtracking trials. C749 exited one. On guarded-court engine seed 81437 and policy seed 92569, C749 exited and the candidate died. That paired loss triggered the predeclared rule to keep C749. Both models passed the official native CPU validator. The Windows validator reported warnings about CPU pinning and peak RAM measurement.

- [Layout and per-seed results](matched-results.md) · [Frozen protocol](protocol.json) · [Selection decision](decision.json)
- [C749 episode ledger](evaluation-C749.jsonl) · [Candidate episode ledger](evaluation-candidate.jsonl) · [Checkpoint identity](candidate-manifest.json)
- [Training updates](training-updates.jsonl) · [Original training and evaluation scripts](source/) · [Exact-history PPO implementation](https://github.com/Anant-vj/doomray-ppo/tree/4bfd9e2fdbf9e09da6d02fff246d40a1e1aa15a6/research/experiments/05-exact-ppo)
- [Recorded successful replay](replay-success.json) · [Recorded failed replay](replay-failure.json) · [Approved C749 release](https://github.com/Anant-vj/doomray-ppo/tree/6c2d5499f973c77b4a239f778837280edf64e57b)

The source scripts preserve the original study-directory layout and time guard. The two replay traces are from the earlier functional audit. The four reserved maps were played once per model and seed pair after the candidate was frozen. The ledgers and protocol retain their original bytes. The two public validator logs replace the host checkout paths with model labels. Their original and published checksums are recorded in the selection decision.
