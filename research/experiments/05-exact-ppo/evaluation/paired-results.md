# Corrected PPO paired screening

27 November 2026. All 48 episodes completed over 12 matched cases, with 11,078 decisions and zero illegal actions.
Each cell gives **exit/survival/keys/unlocks · player hits/damage · decisions**.

| Layout | Engine seed | C749 | Exact PPO | Balanced reward | Balanced + PBRS + anchors |
|---|---:|---|---|---|---|
| offset-alley-duel | 70421 | 0/1/0/0 · 0/0 · 200 | 0/1/0/0 · 1/10 · 200 | 0/1/0/0 · 0/0 · 200 | 1/1/0/0 · 0/0 · 47 |
| offset-alley-duel | 70422 | 0/0/0/0 · 0/0 · 192 | 1/1/0/0 · 0/0 · 121 | 0/0/0/0 · 0/0 · 122 | 0/0/0/0 · 0/0 · 128 |
| alternating-chicanes | 70421 | 1/1/0/0 · 0/0 · 271 | 0/1/0/0 · 0/0 · 275 | 0/1/0/0 · 0/0 · 275 | 0/1/0/0 · 0/0 · 275 |
| alternating-chicanes | 70422 | 0/1/0/0 · 0/0 · 275 | 0/1/0/0 · 0/0 · 275 | 0/1/0/0 · 0/0 · 275 | 0/1/0/0 · 0/0 · 275 |
| return-throat-lock | 70421 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 |
| return-throat-lock | 70422 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 | 0/1/0/0 · 0/0 · 290 |
| interleaved-courts | 70421 | 0/0/0/0 · 2/15 · 125 | 0/0/0/0 · 0/0 · 97 | 0/0/0/0 · 0/0 · 111 | 1/1/0/0 · 0/0 · 147 |
| interleaved-courts | 70422 | 0/0/0/0 · 0/0 · 128 | 0/0/0/0 · 0/0 · 114 | 0/0/0/0 · 1/15 · 144 | 0/0/0/0 · 1/15 · 199 |
| freedoom1-E1M2 | 70421 | 0/0/0/0 · 0/0 · 169 | 0/0/0/0 · 2/26 · 125 | 0/0/0/0 · 0/0 · 69 | 0/0/0/0 · 0/0 · 285 |
| freedoom1-E1M2 | 70422 | 0/0/0/0 · 1/16 · 309 | 0/0/0/0 · 0/0 · 222 | 0/0/0/0 · 0/0 · 156 | 0/0/0/0 · 0/0 · 423 |
| freedoom1-E2M1 | 70421 | 0/0/0/0 · 0/0 · 277 | 0/0/0/0 · 0/0 · 243 | 0/0/0/0 · 2/20 · 243 | 0/0/0/0 · 0/0 · 504 |
| freedoom1-E2M1 | 70422 | 0/0/0/0 · 0/0 · 408 | 0/0/0/0 · 1/10 · 383 | 0/0/0/0 · 0/0 · 224 | 0/0/0/0 · 2/7 · 247 |

## Other measured totals

| Model | Damage received | Ammo net drops (bullets/shells/rockets/cells) | No-cue attacks / holds | Blocked moves / requests | Explored cells | Peak action latency |
|---|---:|---:|---:|---:|---:|---:|
| C749 | 836 | 73/4/0/0 | 492/1295 | 81/2907 | 399 | 5.22 ms |
| Exact-PPO | 749 | 45/1/0/0 | 379/946 | 84/2566 | 386 | 6.47 ms |
| Balanced-Reward | 809 | 54/0/0/0 | 551/1259 | 57/2382 | 371 | 7.81 ms |
| Balanced-PBRS-Anchor | 721 | 59/4/0/0 | 534/1459 | 146/3049 | 427 | 4.95 ms |

Ammunition net drops are lower bounds. Button holds and hit counters do not provide shot accuracy.
Decision count depends on each actual exit, death, or tick cap.

E1M2 and E2M1 are larger bundled Freedoom1 cases with no prior project gameplay references found. All 16 executed successfully through the unchanged wrapper. The first trace position for each episode matched the corresponding map spawn. These cases produced zero exits, survivals, keys, or unlocks.

Every continuation missed the alternating-chicanes exit C749 completed. The frozen gate rejected every continuation. The separate confirmation maps stayed sealed.

Episode ledger SHA-256 fe7bf7a11bb9cbd228ce3b53c038798b1846118eea9476f87c05bd105824f464.
