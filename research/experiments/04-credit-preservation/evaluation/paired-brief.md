# Credit assignment campaign paired screen

27 November 2026 · **C749 retained**

All 40 episodes completed across 10 matched map-and-seed cases, with 8,275 legal decisions and zero invalid actions.
Each cell gives **exit/survival/keys/unlocks · player hits/damage · decisions**.

| Layout | Seed | C749 | PPO control | PBRS | PBRS + anchors |
|---|---:|---|---|---|---|
| aim-lane-cover | 63741 | 0/1/0/0 · 0/0 · 190 | 0/1/0/0 · 0/0 · 190 | 1/1/0/0 · 0/0 · 22 | 1/1/0/0 · 0/0 · 55 |
| aim-lane-cover | 63742 | 0/1/0/0 · 0/0 · 190 | 1/1/0/0 · 0/0 · 69 | 1/1/0/0 · 0/0 · 153 | 1/1/0/0 · 0/0 · 117 |
| fork-left-right | 63741 | 1/1/0/0 · 0/0 · 175 | 0/1/0/0 · 0/0 · 240 | 0/1/0/0 · 0/0 · 240 | 1/1/0/0 · 0/0 · 166 |
| fork-left-right | 63742 | 0/1/0/0 · 0/0 · 240 | 0/1/0/0 · 0/0 · 240 | 0/1/0/0 · 0/0 · 240 | 0/1/0/0 · 0/0 · 240 |
| lock-recovery-return | 63741 | 0/1/1/1 · 0/0 · 290 | 0/1/1/0 · 0/0 · 290 | 0/1/1/1 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 |
| lock-recovery-return | 63742 | 0/1/1/1 · 0/0 · 290 | 0/1/1/1 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 | 1/1/1/1 · 0/0 · 139 |
| split-shelters | 63741 | 0/0/0/0 · 0/0 · 51 | 0/0/0/0 · 1/15 · 95 | 1/1/0/0 · 0/0 · 107 | 0/0/0/0 · 0/0 · 81 |
| split-shelters | 63742 | 0/0/0/0 · 2/20 · 47 | 0/0/0/0 · 1/4 · 64 | 0/0/0/0 · 0/0 · 43 | 0/0/0/0 · 0/0 · 51 |
| two-lock-relay | 63741 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 |
| two-lock-relay | 63742 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 | 0/1/0/0 · 0/0 · 350 |

## Other observed metrics

| Model | Damage received | Ammo net drops (bullets/shells/rockets/cells) | No-cue attacks / held attacks | Blocked moves / requests | Explored cells | Max action latency |
|---|---:|---:|---:|---:|---:|---:|
| C749 | 303 | 51/0/0/0 | 680/847 | 105/2130 | 193 | 30.07 ms |
| PPO-control | 261 | 43/0/0/0 | 704/862 | 133/2142 | 191 | 26.09 ms |
| PBRS | 177 | 66/0/0/0 | 677/771 | 159/2033 | 184 | 20.55 ms |
| PBRS-anchor | 219 | 40/0/0/0 | 605/712 | 87/1777 | 181 | 15.63 ms |

Ammo net drops are a lower bound on recorded inventory consumption. Button holds are distinct from shots. Decision counts describe the observed route and stopping condition.

PBRS + anchors added both aim-lane-cover exits and one lock-recovery-return exit. On that return layout it lost a key pickup and an unlock that C749 recorded. Both shaped arms also recorded zero player hits and damage. All alternatives failed the frozen gate. Confirmation stayed sealed.

Episode ledger SHA-256 597f022736ad633de8533fcf48f5712917d556b3ad773a1799256f527c35615e.
