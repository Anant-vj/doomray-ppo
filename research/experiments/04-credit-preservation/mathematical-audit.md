# DoomRay PPO credit assignment audit

## Verified C749 setup

Run 43 resumed the Stack-26 u621 weights and Adam state at learning rate 0.0001. It used 6 parallel CPU workers, a 64-decision rollout per worker, 16-decision chunks, 8 chunks per minibatch, and 3 epochs. The final phase collected 49,152 decisions and took 1,152 optimizer steps. The 2,100-tic cap allows 525 four-tic decisions. The deployed actor uses a four-decision feature stack and six unscaled categorical heads. [Run 43 provenance](../../history/provenance.json) · [Archived learner counters](../../history/legacy-counters.jsonl) · [Frozen policy](../../../submission/policy.py#L68)

PPO computes one joint ratio `exp(Σᵢ [log πᵢ(aᵢ|h) − log πoldᵢ(aᵢ|h)])` and clips that ratio to [0.8, 1.2]. Value MSE has weight 0.5, summed policy entropy has weight 0.01, and gradient norm is capped at 0.5. Rewards are unnormalized. Advantages are centered and divided by the rollout-wide population standard deviation with floor 1e−6. After the first optimizer step, cached rollout encodings can make up to the first three decisions of each stack chunk approximate. This was observed in source and has not been linked causally to failure. [Frozen PPO code](../../../training/ppo.py#L16) · [PPO paper](https://arxiv.org/abs/1707.06347)

GAE uses `γ=.99` and `λ=.95`: `δₜ=rₜ+γV(next)−Vₜ` and `Aₜ=δₜ+γλ(1−boundaryₜ)Aₜ₊₁`. Genuine exits, deaths and the objective's finite tick cap clear the continuation value. The wrapper can return the last pre-action frame at those boundaries. Nonterminal rollout cuts use the actual successor and carried history to bootstrap. Discount and GAE geometric weight sums are 100 and 16.81 decisions. Half-lives are 68.97 and 11.30. At 100 decisions, the weights are 0.3660 and 0.00217. These scales motivate examining delayed rewards even when the value function can propagate future credit. The actor has no explicit horizon clock, so nearly identical observations can refer to different remaining budgets. [Frozen rollout code](../../../training/train.py#L141) · [GAE](https://arxiv.org/abs/1506.02438) · [Time-limit treatment](https://proceedings.mlr.press/v80/pardo18a.html)

## Historical reward evidence

The 162 completed Run 43 episodes cover 47,667 of 49,152 decisions. The next table uses signed, undiscounted totals per completed episode. The unmatched 1,485 decisions have no episode-level ledger. [Archived Run 43 counters](../../history/legacy-counters.jsonl)

| Component | Sum | Mean | Population variance | Nonzero episodes |
|---|---:|---:|---:|---:|
| Native | −36.400 | −0.2247 | 0.5401 | 132 |
| Extra exit | 12.150 | 0.0750 | 0.0410 | 30 |
| Coverage | 179.550 | 1.1083 | 0.5702 | 159 |
| Key | 5.850 | 0.0361 | 0.0149 | 13 |
| Door rise | 12.750 | 0.0787 | 0.0078 | 77 |
| Locked-door rise | 3.250 | 0.0201 | 0.0046 | 13 |
| Damage penalty | −90.864 | −0.5609 | 0.4183 | 112 |

The native reward pays +1 for an exit, −1 for death and +0.1 per map-wide kill increment. The progression profile adds +2 for ordinary exits or +0.35 for designated practice exits, +0.035 per new 96-unit cell, +0.45 per key, +0.15 per first door rise, another +0.25 for a locked rise, and −0.006 per damage received. Player damage dealt and health recovery have zero coefficients in this profile. One fatal CenterCombat episode still returned +1.849 because coverage outweighed its losses. The 14-layout mixed pool used uniform random resets after the initial worker assignments, yet the key lesson covered only 1,058 completed-episode decisions (2.22%). Sparse credit and duration-weighted exposure are plausible effects. Historical episode sums cannot establish gradient interference. [Frozen progression reward code](../../../training/game.py#L115) · [ViZDoom player and map counters](https://github.com/Farama-Foundation/ViZDoom/blob/1.3.3/src/lib/ViZDoomController.cpp#L946-L965)

## Isolated comparison

The three arms share the released C749 tensors, fresh Adam, unchanged inference, and the same per-worker map and seed streams. The scheduler chooses one of four skills with probability 0.25 at each reset. Real decisions still depend on episode duration. The base reward and actor stay matched. Training-only route potentials cover the two hash-pinned red-key lessons. The bounded potential contributes `0.99 Φ(next) − Φ(current)`, with zero successor potential at the finite objective boundary. The live and offline ledgers check the telescoping identity. The anchor arm adds a 0.04-weighted sum of six teacher-to-learner categorical KLs on stratified successful histories outside the PPO ratio. [Frozen plan](training/training-plan.json) · [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

Anchor replay borrows CLEAR's preservation idea using fixed policy targets only. It has no off-policy RL or value replay. The balanced reset scheduler uses fixed family weights and omits PLR's learning-potential prioritization. Component gradients use distinct discounted reward-to-go on observed on-policy terminal suffixes. Their finite, end-conditioned samples can give noisy or censored directions. No gradient projection is included in the comparison. [CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) · [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)

The submitted agent imports only its local policy, NumPy, PyTorch and standard-library modules. Its inputs are official rays, entity arrays, vitals and its own action history. Route geometry, simulator counters, phase labels and replay remain in training. The organizer documents exits, key/door/lock progression, unique-area exploration, enemy kills, health/time bonuses, and damage/death penalties. Exact official scoring coefficients are private. Local hits, player damage, net ammo drops, no-cue holds, blocked movement and local composite scores are diagnostics. [Official rules](../../../docs/RULES.md) · [Scoring contract](../../../docs/EVALUATION.md)
