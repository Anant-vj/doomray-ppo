"""Offline parity and optimization timings on saved observations, with no engine."""

from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[2]
DATA = EXPERIMENT.parent / "01-behavior-cloning"
WEIGHTS = ROOT / "submission/weights/policy.pt"
FIXTURE_SHA = "7ea4fbf0ced78a7a41879c77bbcbd9bc75cd5af0e10c4d519d15945f55686d22"
sys.path[:0] = [str(EXPERIMENT / "source"), str(ROOT), str(ROOT / "env")]

import numpy as np
import torch

from submission import agent
from submission.policy import buttons, pack
from training.exact_history import stack_output
from training.ppo import optimize
from training.reward_diagnostics import current_context


torch.set_num_threads(1)
fixture = next((path for path in DATA.rglob("*.npz")
                if hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA), None)
if fixture is None:
    raise FileNotFoundError("The verified numeric keycard fixture is missing.")
with np.load(fixture, allow_pickle=False) as data:
    observed_steps = [
        {"rays": {"distance": distance, "hit_type": hits},
         "visible_entities": {"features": features, "entity_type": types, "mask": mask},
         "vitals": vitals}
        for distance, hits, features, types, mask, vitals in zip(
            data["ray_distance"], data["ray_hit_type"], data["entity_features"],
            data["entity_type"], data["entity_mask"], data["vitals"], strict=True)
    ]


def direct_parity(agent):
    carried = deque(maxlen=3)
    resets = {0, 2, 17}
    for i in range(32):
        if i in resets:
            agent.reset()
        observation = observed_steps[i]
        current = torch.from_numpy(pack(observation, agent.previous))[None]
        frames, valid = current_context(current, carried)
        expected_slots = min(4, i - max(start for start in resets if start <= i) + 1)
        assert valid.sum() == expected_slots
        with torch.no_grad():
            output = stack_output(agent.policy, frames, valid)
            torch.manual_seed(7100 + i)
            choices, _, _, _ = agent.policy.decision(output)
        torch.manual_seed(7100 + i)
        np.testing.assert_array_equal(agent.act(observation), buttons(choices[0].numpy()))
        carried.append((current, torch.tensor([float(i + 1 not in resets)])))


def frozen_rollout(policy):
    horizon, workers = 64, 4
    starts = torch.zeros(horizon + 1, workers)
    starts[0] = 1
    starts[[3, 16, 33, 51], 0] = 1
    starts[[8, 18, 20, 49], 1] = 1
    starts[[4, 30, 42, 58], 2] = 1
    starts[[15, 16, 32, 63], 3] = 1
    carried = deque(maxlen=3)
    previous = np.zeros((workers, 9), dtype=np.int8)
    hidden = policy.initial(workers, "cpu")
    rows, contexts = [], []
    with torch.no_grad():
        for i in range(horizon):
            current = torch.tensor(np.stack([
                pack(observed_steps[i * 2 + worker], previous[worker])
                for worker in range(workers)
            ]))
            history, valid = current_context(current, carried)
            sequential, after = policy.advance(policy.encode(current), hidden)
            exact = stack_output(policy, history, valid)
            torch.testing.assert_close(exact, sequential, rtol=1e-5, atol=6e-7)
            choices, old_logp, _, values = policy.decision(sequential)
            _, recomputed_logp, _, recomputed_values = policy.decision(exact, choices)
            torch.testing.assert_close(recomputed_logp, old_logp, rtol=1e-6, atol=2e-6)
            torch.testing.assert_close(recomputed_values, values, rtol=1e-5, atol=6e-7)
            rows.append((current, hidden, choices, old_logp, values))
            contexts.append((history, valid))
            continuing = 1 - starts[i + 1]
            previous = buttons(choices.numpy())
            previous[starts[i + 1].bool()] = 0
            hidden = after * continuing[:, None]
            carried.append((current, continuing))
    obs, initial, actions, old_logp, values = map(torch.stack, zip(*rows, strict=True))
    histories, valid = map(torch.stack, zip(*contexts, strict=True))
    synthetic_advantage = torch.linspace(-0.4, 0.4, horizon * workers).reshape(horizon, workers)
    return (obs, initial, starts[:-1], actions, old_logp, values + synthetic_advantage, synthetic_advantage), (histories, valid)


def check_stale_context(policy, batch, raw):
    _, initial, _, _, _, _, _ = batch
    histories, valid = raw
    learner = deepcopy(policy)
    with torch.no_grad():
        learner.fusion[0].bias[0] += .4
        valid_context = valid[16, 1:2].bool()
        assert valid_context.all()
        rebuilt = stack_output(learner, histories[16, 1:2], valid[16, 1:2])
        fresh = learner.initial(1, "cpu")
        for frame in histories[16, 1:2].unbind(1):
            sequential, fresh = learner.advance(learner.encode(frame), fresh)
        torch.testing.assert_close(rebuilt, sequential)
        stale, _ = learner.advance(learner.encode(histories[16, 1:2, -1]), initial[16, 1:2])
        drift = float((rebuilt - stale).abs().amax())
        assert drift > 1e-3
    return drift


def timings(policy, batch, raw, *, exact):
    output = []
    old = batch[4].clone()
    for trial in range(4):
        learner = deepcopy(policy)
        optimizer = torch.optim.Adam(learner.parameters(), lr=.0001, eps=1e-5)
        # If exact unexpectedly reads the cached embeddings, this test fails.
        test_batch = (batch[0], torch.full_like(batch[1], float("nan")), *batch[2:]) if exact else batch
        torch.manual_seed(2026)
        start = time.perf_counter()
        count, loss, stats = optimize(
            learner, optimizer, test_batch, diagnostics=True, raw_history=raw if exact else None,
        )
        assert count == 6 and np.isfinite(loss)
        output.append(time.perf_counter() - start)
        assert stats["initial_log_ratio_abs_max"] < 5e-6
        assert torch.equal(old, batch[4])
    return median(output[1:]), stats["initial_log_ratio_abs_max"]


if __name__ == "__main__":
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == (
        "0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61"
    )
    agent.WEIGHTS_DIR = WEIGHTS.parent
    model = agent.DoomAgent(None, SimpleNamespace(shape=(9,)), "cpu")
    direct_parity(model)
    batch, raw = frozen_rollout(model.policy)
    drift = check_stale_context(model.policy, batch, raw)
    cached, cached_ratio = timings(model.policy, batch, raw, exact=False)
    exact, exact_ratio = timings(model.policy, batch, raw, exact=True)
    print(json.dumps({
        "passed": True, "direct_agent_decisions": 32, "paired_rollout_decisions": 256,
        "threads": torch.get_num_threads(), "independent_current_encoder_shift_max": drift,
        "cached_initial_log_ratio_abs_max": cached_ratio, "exact_initial_log_ratio_abs_max": exact_ratio,
        "cached_optimize_median_s": cached, "exact_optimize_median_s": exact,
        "exact_over_cached_learner_ratio": exact / cached,
        "scope": "Offline synthetic batching on saved observations, with no live engine or trained artifact.",
    }))
