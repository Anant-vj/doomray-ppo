# Experiment 3 · fixed inference decoding

All four decoders used the approved C749 weights and the unchanged nine-button decoder. The intervention began after the six neural heads produced logits. The control returned the original categorical agent unchanged. T07 and T13 sampled independent categorical heads after dividing their logits by 0.7 and 1.3. Argmax chose the highest logit within each head. The freeze verified identical learned tensors and 72 offline action checks. [Decoder source](source/decoding.py) · [Frozen settings](provenance/decoder-freeze.json)

## Matched screening

Four local layout families and two shared seeds yielded eight legal episodes per mode, 32 in total. The agents took 4,845 legal decisions. The table shows observed outcomes and the frozen unofficial balanced proxy. Engine and policy seeds, termination, raw telemetry and the original identity hashes are in the ledger.

| Mode | Exits | Survived | Keys | Unlocks | Player hits | Player damage | Balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 control | 2 | 6 | 1 | 0 | 3 | 17 | 6.7434 |
| T07 | 3 | 7 | 1 | 1 | 2 | 17 | 7.3882 |
| T13 | 2 | 6 | 0 | 0 | 8 | 59 | 5.9803 |
| Argmax | 2 | 6 | 2 | 0 | 4 | 40 | 7.2250 |

[Episode ledger](evaluation/screening.jsonl) · [Matched summary and frozen gates](evaluation/screening-summary.json) · [Match identities](evaluation/screening-maps.json)

## Independent local confirmation

T07 alone advanced to the two previously withheld families. Across four legal episodes per decoder, T1 recorded two exits, three survivors, two keys and one unlock. T07 recorded one exit, four survivors, two keys and one unlock. Recorded player damage rose from 0 to 8. The one lost exit on broken-rampart failed the rule to preserve exits in every family. Material gain appeared on only one family and neither family replicated it on both seeds. C749 retained the original T1 categorical configuration. Across the eight episodes there were 1,576 legal decisions and zero invalid actions. [Nomination receipt](provenance/nomination.json) · [Confirmation ledger](evaluation/confirmation.jsonl) · [Confirmation gate](evaluation/confirmation-summary.json)

The native-staging source illustrates the exact T07 deployment variant. The official validator log in this bundle applies to the frozen T1 release. The local scores and the observed behavior have no official leaderboard interpretation. [T07 staging source](source/native-t07/submission/agent.py) · [Frozen T1 native validator](validation/frozen-t1-validator.txt)

## Reproduce the aggregate

From this directory, run the following with Python 3.13. It uses the original aggregation code, reads the public logs and match identities, and checks both stages against their recorded summaries. It launches no simulator.

```sh
python -B verify-scores.py
```

The archived runner source documents how the original C749 and decoder wrappers were evaluated. It expects the original private freeze files and uses their digests. Match identities retain their hashes and seeds. Privately held layout assets, source generators and per-decision coordinates remain outside this branch. Published JSON uses relative archive references or a documented private workspace placeholder. [Original and published hashes](manifest.json) · [Scoring definitions](evaluation/scoring-manifest.json) · [Evaluator source](source/screening/evaluate_screen.py)
