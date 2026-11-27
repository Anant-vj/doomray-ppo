# Historical learner counters

The 1,452 rows in [legacy-counters.jsonl](legacy-counters.jsonl) preserve the counters from all 15 original legacy PPO learner logs. [provenance.json](provenance.json) records their source filenames, original SHA-256 hashes, available launch receipts, model settings, and starting counters. The update, gradient and run-transition counters come directly from the original log. The lineage transition count adds the inherited parent cutoff to `transitions_this_run`.

- **All recorded legacy learners** ran 618,432 gameplay transitions, 1,452 rollout updates, and 14,499 PPO optimizer steps. This includes the early live shakedowns.
- **C749 ancestry** contains 324,864 gameplay transitions, 749 rollout updates, and 7,614 PPO optimizer steps. It follows the saved Stack-24 update 110 into Stack-25. The seven additional Stack-24 updates remain in the legacy learner total.
- **Scope** is recorded learner activity through the resumed sustain run. Human recordings, supervised learning, later research campaigns, benchmarks, and evaluation decisions are counted separately.

To recompute the legacy totals, group rows by `run_id`. Add each run's last `transitions_this_run`, row count, and final `gradients` minus inherited `gradients` from `starting_counters`. To recompute C749, use the five recorded cutoffs in order. In each run select its rows through the stated last update and apply the same calculation. Counting cumulative gradients separately for each branch would count inherited optimizer work again.

The [immutable 96-episode assessment](https://github.com/Anant-vj/doomray-ppo/tree/e9fe1740994a78b3bfb650017061b8e832d7a957/assessment) publishes evaluation results and separate local metrics.
