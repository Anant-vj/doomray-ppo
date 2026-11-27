# C749 key recovery examination

Each cell shows counts from two episodes as **C749 / update 845**. An early death occurs within 525 of 2,100 ticks.

| Layout | Exits | Keys | Unlocks | Early deaths | Invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| west-pocket-return | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 0 | 0 / 0 |
| south-spur-return | 1 / 2 | 1 / 2 | 1 / 2 | 0 / 0 | 0 / 0 |
| forward-key-gallery | 2 / 2 | 2 / 2 | 2 / 2 | 0 / 0 | 0 / 0 |
| guarded-court | 1 / 1 | 2 / 2 | 1 / 2 | 1 / 0 | 0 / 0 |

## Seed-level results

| Layout | Engine seed | Policy seed | C749 outcome | Candidate outcome | Keys | Unlocks | Early deaths | Invalid actions |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| forward-key-gallery | 81431 | 92561 | exit | exit | 1 / 1 | 1 / 1 | False / False | 0 / 0 |
| forward-key-gallery | 81437 | 92569 | exit | exit | 1 / 1 | 1 / 1 | False / False | 0 / 0 |
| guarded-court | 81431 | 92561 | death | exit | 1 / 1 | 0 / 1 | True / False | 0 / 0 |
| guarded-court | 81437 | 92569 | exit | death | 1 / 1 | 1 / 1 | False / False | 0 / 0 |
| south-spur-return | 81431 | 92561 | tick_cap | exit | 0 / 1 | 0 / 1 | False / False | 0 / 0 |
| south-spur-return | 81437 | 92569 | exit | exit | 1 / 1 | 1 / 1 | False / False | 0 / 0 |
| west-pocket-return | 81431 | 92561 | tick_cap | exit | 0 / 1 | 0 / 1 | False / False | 0 / 0 |
| west-pocket-return | 81437 | 92569 | tick_cap | exit | 0 / 1 | 0 / 1 | False / False | 0 / 0 |

## Predeclared gameplay gate

- guarded-court seed 81437 lost exit
- guarded-court seed 81437 regressed on survival


## Recommendation

Recommended **C749**, commit `6c2d5499f973c77b4a239f778837280edf64e57b`, inference SHA-256 `0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61`.

The one candidate trained for 24,576 real transitions and 576 new optimizer steps on eight short drills. The final update was 845, starting from update 749. Throughput averaged 69.3 transitions/second.

On the archived same-seed success, the key was collected at decision 6, the lock rose at 12, and the exit occurred at 165. On the failed backtrack the agent pressed USE at the lock without the key on decisions 10 and 12. It briefly saw the key at decisions 319–322 and timed out at 350 without collecting it. This comparison motivated the eight drills.

The independent layouts used the published wrapper default cap of 2,100 ticks at skill 3. Initial observation hashes matched across all eight pairs. The official native validator returned 0 for C749 and 0 for the candidate.

[Frozen protocol](protocol.json) · [Per-seed comparison](matched-summary.json) · [C749 validator log](logs/validator-C749.txt) · [Candidate validator log](logs/validator-candidate.txt)
