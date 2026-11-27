"""Compare two GAE decays on verified archived C749 training-only trajectories."""

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RELEASE = EXPERIMENT.parents[2]
ARCHIVE = EXPERIMENT.parent / "04-credit-preservation/anchors"
COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"
WEIGHT = "0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61"
POLICY_SOURCE = "d8a40c277e9029b12763aa64269d3cba7260a13d6039f76e3a3ec378a5faf95f"
published_inputs = json.loads((HERE / "gae-input-digests.json").read_text(encoding="utf-8"))
LEDGER = published_inputs["attempts.jsonl"]["published_sha256"]
MANIFEST = published_inputs["collection-manifest.json"]["published_sha256"]
GAMMA = 0.99
DECAYS = (0.95, 0.97)
FAMILY = {"key-lesson": "key progression", "backtrack-lesson": "key progression",
          "basic-combat": "combat", "way-home": "navigation"}
LAGS = (0, 4, 8, 16, 32, 64, 128)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def gae(rewards, values, decay):
    """All input episodes ended in the recorded finite objective."""
    following = np.concatenate((values[1:], [0.0]))
    deltas = rewards + GAMMA * following - values
    advantages = np.empty_like(deltas, dtype=np.float64)
    carry = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        carry = deltas[t] + GAMMA * decay * carry
        advantages[t] = carry
    return advantages


def read_attempts():
    ledger = ARCHIVE / "attempts.jsonl"
    manifest = ARCHIVE / "collection-manifest.json"
    if digest(ledger) != LEDGER or digest(manifest) != MANIFEST:
        raise ValueError("Archived all-attempt ledger or original plan changed.")
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    expected = [(row["name"], seed) for row in plan["maps"] for seed in row["seeds"]]
    actual = [(row["layout"], row["seed"]) for row in episodes]
    if len(episodes) != 20 or actual != expected:
        raise ValueError("All 20 predeclared training-only attempts are required.")
    return episodes


def read_stream(row):
    path = ARCHIVE / "streams" / (row["episode_id"] + ".npz")
    n = row["decisions"]
    if (not row["finished"] or row["stop_reason"] or row["end"] not in ("exit", "death", "cap")
            or row["exact_engine_action_readbacks"] != n or digest(path) != row["stream_sha256"]):
        raise ValueError(f"Unfinalized or altered stream: {path.name}")
    with np.load(path, allow_pickle=False) as stream:
        data = {name: stream[name] for name in stream.files}
    frames, valid = data["histories"], data["valid"]
    prior, accepted = data["prev_actions"], data["accepted_actions"]
    finished = data["terminated"] | data["truncated"]
    if (frames.shape != (n, 4, 378) or valid.shape != (n, 4)
            or prior.shape != (n, 9) or accepted.shape != (n, 9) or data["rewards"].shape != (n,)
            or not np.isfinite(frames).all() or not np.isfinite(data["rewards"]).all()
            or not finished[-1] or finished[:-1].any()
            or bool(data["truncated"][-1]) != (row["end"] == "cap")
            or not np.array_equal(valid, np.arange(n)[:, None] + np.arange(-3, 1) >= 0)
            or np.any(frames[~valid]) or not np.array_equal(frames[:-1, 1:], frames[1:, :3])
            or not np.array_equal(frames[:, -1, -9:], prior)
            or np.any(prior[0]) or not np.array_equal(prior[1:], accepted[:-1])
            or not np.isin(accepted, (0, 1)).all()
            or np.any(accepted[:, 0:6:2] + accepted[:, 1:6:2] > 1)):
        raise ValueError(f"Boundary, frame, or action alignment changed: {path.name}")
    return data


def frozen_values(policy, data):
    accepted = data["accepted_actions"]
    choices = np.concatenate((accepted[:, 0:6:2] + 2 * accepted[:, 1:6:2], accepted[:, 6:]), axis=1)
    frames = torch.from_numpy(data["histories"])
    valid = torch.from_numpy(data["valid"])
    values, logits, probability = [], [], []
    with torch.inference_mode():
        for start in range(0, len(choices), 256):
            batch = slice(start, start + 256)
            action = torch.as_tensor(choices[batch], dtype=torch.long)
            output = stack_output(policy, frames[batch], valid[batch])
            _, log_prob, _, critic = policy.decision(output, action)
            values.append(critic.numpy())
            logits.append(policy.actor(output).numpy())
            probability.append(log_prob.numpy())
    logit_error = float(np.max(np.abs(np.concatenate(logits) - data["teacher_logits"])))
    logp_error = float(np.max(np.abs(np.concatenate(probability) - data["teacher_log_prob"])))
    if logit_error > 2e-5 or logp_error > 2e-5:
        raise ValueError("Archived policy outputs do not match exact frozen C749 inference.")
    critic = np.concatenate(values).astype(np.float64)
    if not np.isfinite(critic).all():
        raise ValueError("Frozen critic returned a nonfinite value.")
    return critic, logit_error, logp_error


def summarize(rows):
    advantages = [np.concatenate([row["advantages"][decay] for row in rows]) for decay in DECAYS]
    low, high = advantages
    stats = {
        "attempts": len(rows), "decisions": len(low), "ends": dict(Counter(row["end"] for row in rows)),
        "mean_native_reward": float(np.concatenate([row["rewards"] for row in rows]).mean()),
        "frozen_critic_mean": float(np.concatenate([row["values"] for row in rows]).mean()),
        "advantage": {str(decay): {
            "mean": float(a.mean()), "population_variance": float(a.var()),
            "population_std": float(a.std()), "p05": float(np.quantile(a, .05)),
            "p95": float(np.quantile(a, .95)),
            "mean_within_episode_population_variance": float(np.mean([row["advantages"][decay].var() for row in rows])),
        } for decay, a in zip(DECAYS, advantages, strict=True)},
    }
    stats["variance_difference"] = float(high.var() - low.var())
    stats["variance_relative_percent"] = float(100 * (high.var() / low.var() - 1)) if low.var() else None
    stats["advantage_rms_difference"] = float(np.sqrt(np.mean((high - low) ** 2)))
    stats["advantage_sign_change_fraction"] = float(np.mean(np.sign(high) != np.sign(low)))
    return stats


def late_reward(rows):
    """Difference due solely to the recorded final reward, holding values fixed."""
    eligible = [row for row in rows if row["rewards"][-1] != 0.]
    results = []
    for lag in LAGS:
        paired = [row for row in eligible if len(row["rewards"]) > lag]
        if not paired:
            continue
        old, new = DECAYS
        results.append({
            "lag_decisions": lag, "matched_attempts": len(paired),
            "mean_native_terminal_reward": float(np.mean([row["rewards"][-1] for row in paired])),
            "mean_terminal_pulse_contribution_lambda_.95": float(np.mean([row["pulse"][old][-1-lag] for row in paired])),
            "mean_terminal_pulse_contribution_lambda_.97": float(np.mean([row["pulse"][new][-1-lag] for row in paired])),
            "terminal_pulse_magnitude_ratio": float(((GAMMA * new) / (GAMMA * old)) ** lag),
        })
    return results


def main():
    weights = RELEASE / "submission/weights/policy.pt"
    if digest(weights) != WEIGHT:
        raise ValueError("Frozen C749 weights changed.")
    policy_file = RELEASE / "submission/policy.py"
    if Path(policy_source.__file__).resolve() != policy_file.resolve() or sha256(policy_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != POLICY_SOURCE:
        raise ValueError("Frozen C749 policy implementation changed.")
    torch.set_num_threads(1)
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    policy = policy_source.Policy(saved["memory"]).eval().requires_grad_(False)
    policy.load_state_dict(saved["weights"])
    results, logit_error, logp_error = [], 0.0, 0.0
    for row in read_attempts():
        data = read_stream(row)
        values, logit_diff, logp_diff = frozen_values(policy, data)
        logit_error, logp_error = max(logit_error, logit_diff), max(logp_error, logp_diff)
        rewards = data["rewards"].astype(np.float64)
        without_last = rewards.copy()
        without_last[-1] = 0.0
        advantages = {decay: gae(rewards, values, decay) for decay in DECAYS}
        pulse = {decay: advantages[decay] - gae(without_last, values, decay) for decay in DECAYS}
        for decay in DECAYS:
            expected = rewards[-1] * (GAMMA * decay) ** np.arange(len(rewards) - 1, -1, -1)
            np.testing.assert_allclose(pulse[decay], expected, atol=2e-12, rtol=1e-9)
        results.append(dict(layout=row["layout"], end=row["end"], rewards=rewards, values=values,
                            advantages=advantages, pulse=pulse))
    if sum(len(row["rewards"]) for row in results) != 3417:
        raise ValueError("Unexpected decisions in the verified full cohort.")
    nav_exits = [row for row in results if row["end"] == "exit" and row["layout"] != "basic-combat"]
    report = {
        "scope": "Offline finite-episode GAE sensitivity. Archived native rewards and exact frozen critic held fixed. No gameplay, optimization, or changes to experiment arms.",
        "limitations": [
            "The C749 critic was fit on shaped training rewards. Absolute advantage statistics here describe the archived native reward scale.",
            "The calculation uses complete recorded attempts, with no artificial PPO rollout cuts. It does not predict improvement on hidden maps.",
            "Wrapper-classified successful Basic combat finishes appear separately from navigational exits.",
        ],
        "release_commit": COMMIT, "weight_sha256": WEIGHT, "policy_source_sha256": POLICY_SOURCE,
        "attempt_ledger_sha256": LEDGER,
        "collection_manifest_sha256": MANIFEST, "gamma": GAMMA, "lambdas": list(DECAYS),
        "gae_weights": {str(decay): {"gamma_times_lambda": GAMMA * decay,
                                    "geometric_horizon_decisions": 1 / (1 - GAMMA * decay),
                                    "half_life_decisions": math.log(.5) / math.log(GAMMA * decay)} for decay in DECAYS},
        "live_gameplay_decisions": 0, "optimizer_steps": 0, "threads": torch.get_num_threads(),
        "frozen_policy_max_archived_logit_error": logit_error, "frozen_policy_max_archived_joint_logp_error": logp_error,
        "all": summarize(results),
        "by_training_family": {family: summarize([r for r in results if FAMILY[r["layout"]] == family])
                               for family in sorted(set(FAMILY.values()))},
        "by_recorded_layout": {layout: summarize([r for r in results if r["layout"] == layout])
                               for layout in FAMILY},
        "terminal_reward_propagation": {
            "navigational_exit": late_reward(nav_exits),
            "basic_combat_finish": late_reward([r for r in results if r["layout"] == "basic-combat" and r["end"] == "exit"]),
            "death": late_reward([r for r in results if r["end"] == "death"]),
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    sys.path[:0] = [str(RELEASE), str(EXPERIMENT / "source"), str(RELEASE / "env")]
    import numpy as np
    import torch
    from submission import policy as policy_source
    from training.exact_history import stack_output

    main()
