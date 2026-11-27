"""Read-only rollout analysis plus a JSON receipt. No game instance is opened."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ASSESSMENT = PROJECT / "submission_artifacts/final-blind-assessment"
RELEASE = PROJECT / "submission_artifacts/github-fresh-6c2d549-20261127-022250050"
COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"
WEIGHT = "0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61"
GAMMA = .99
OBJECTIVES = {
    "native_map_deaths_reward": "Exact native 0.1 × change in the map-wide monster death counter.",
    "native_key_route_exit": "Native +1 exit reward on the two key-route training fixtures.",
    "native_way_home_exit": "Native +1 exit reward on WayHome.",
    "native_basic_exit": "Native +1 exit reward on BasicCombat.",
    "native_death": "Exact native -1 terminal death reward.",
    "key_acquisition_event": "Unit pulse on a recorded key-count increase.",
    "lock_sector_rise_proxy": "Unit pulse on a newly raised tagged locked sector.",
    "joint_hit_damage_event": "Binary recorded simultaneous increases of player hit and damage counters. Not damage magnitude or shot attribution.",
    "net_health_loss_proxy": "Negative visible net HP loss across two consecutive pre-action snapshots. Terminal HP is unavailable.",
    "attack_without_cue_proxy": "Hypothetical -1 on ATTACK when the preceding observation has neither an enemy entity nor an enemy ray.",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_check():
    root = RELEASE.as_posix()
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={root}", "-C", root]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT or
            subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True) or
            digest(RELEASE / "submission/weights/policy.pt") != WEIGHT):
        raise ValueError("Published C749 source or materialized weights changed.")


def layout_constants():
    source = ast.parse((RELEASE / "env/doom_env/spaces.py").read_text(encoding="utf-8"))
    members = ("NUM_RAYS", "RAY_HIT_TYPES", "MAX_ENTITIES", "ENTITY_FEATURES", "ENTITY_TYPES", "VITALS_FIELDS")
    constants = {}
    for item in source.body:
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
            name = item.targets[0].id
            if name in members:
                constants[name] = ast.literal_eval(item.value)
    return [constants[k] for k in members]


def prepare(np):
    folder = ROOT / "anchors"
    manifest = read(folder / "collection-manifest.json")
    ledger = [json.loads(x) for x in (folder / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    if [(x["layout"], x["seed"]) for x in ledger] != [(m["name"], s) for m in manifest["maps"] for s in m["seeds"]]:
        raise ValueError("Diagnostic requires every frozen seed attempt in order.")
    rays, hit_types, size, features, entity_types, vitals = layout_constants()
    offset = 2 * rays + size * len(features)
    hostile = [i for i, name in enumerate(entity_types) if name.startswith("enemy_")]
    rows, exposure = [], Counter()
    reward_names = tuple(OBJECTIVES)
    native_parts = [i for i, name in enumerate(reward_names) if name.startswith("native_")]
    for ep in ledger:
        path = folder / "streams" / (ep["episode_id"] + ".npz")
        if not ep["finished"] or digest(path) != ep["stream_sha256"] or ep["stop_reason"]:
            raise ValueError(f"Unfinalized or altered stream: {path}")
        with np.load(path, allow_pickle=False) as fp:
            data = {k: fp[k] for k in fp.files}
        n = ep["decisions"]
        h, valid, action = data["histories"], data["valid"], data["accepted_actions"]
        boundary = data["terminated"] | data["truncated"]
        assert h.shape == (n, 4, 378) and action.shape == (n, 9) and valid.shape == (n, 4)
        assert boundary[-1] and not boundary[:-1].any() and ep["exact_engine_action_readbacks"] == n
        assert np.array_equal(h[:-1, 1:], h[1:, :3]) and np.array_equal(h[:, -1, -9:], data["prev_actions"])
        target = np.zeros((n, len(OBJECTIVES)), dtype=np.float64)
        location = lambda name: reward_names.index(name)
        if ep["end"] == "exit":
            label = ("native_basic_exit" if ep["layout"] == "basic-combat" else
                     "native_way_home_exit" if ep["layout"] == "way-home" else "native_key_route_exit")
            target[-1, location(label)] = 1.
        elif ep["end"] == "death":
            target[-1, location("native_death")] = -1.
        remainder = data["rewards"] - target.sum(axis=1)
        quantized = np.round(remainder * 10) / 10
        assert np.allclose(remainder, quantized, atol=1e-8, rtol=0.)
        target[:, location("native_map_deaths_reward")] = quantized
        assert np.allclose(data["rewards"], target[:, native_parts].sum(axis=1), atol=1e-8, rtol=0.)
        for name, event in (("key_acquisition_event", "key"), ("lock_sector_rise_proxy", "unlock"),
                            ("joint_hit_damage_event", "combat")):
            target[np.asarray(ep["events"][event], dtype=np.int32) - 1, location(name)] = 1.
        x = h[:, -1]
        visible = (np.isin(x[:, offset:offset+size], hostile) & (x[:, offset+size:offset+2*size] != 0)).any(axis=1)
        ray = (x[:, rays:2*rays] == hit_types.index("enemy")).any(axis=1)
        no_cue = (action[:, 6] != 0) & ~(visible | ray)
        target[no_cue, location("attack_without_cue_proxy")] = -1.
        health = 100 * x[:, offset+2*size+vitals.index("health")]
        target[:-1, location("net_health_loss_proxy")] = np.minimum(np.diff(health), 0.)
        exposure.update(visible_entities=int(visible.sum()), enemy_rays=int(ray.sum()), enemy_cues=int((visible|ray).sum()),
                        attack=int(action[:, 6].sum()), attack_without_cue=int(no_cue.sum()))
        exposure.update({f"end_{ep['end']}": 1})
        weighted = np.zeros_like(target)
        following = np.zeros(len(OBJECTIVES), dtype=np.float64)
        for k in range(n - 1, -1, -1):
            following = target[k] + GAMMA * following
            weighted[k] = (GAMMA ** k) * following
        rows.append((data, target, weighted))
    return ledger, rows, exposure


def compute(np, rows, active):
    sys.path.insert(0, str(RELEASE))
    import torch
    from submission.policy import HEADS, Policy

    torch.set_num_threads(1)
    state = torch.load(RELEASE / "submission/weights/policy.pt", map_location="cpu", weights_only=True)
    policy = Policy(state["memory"]).eval()
    policy.load_state_dict(state["weights"])
    parameters = tuple(value for name, value in policy.named_parameters() if not name.startswith("critic."))
    gradients = np.zeros((len(OBJECTIVES), sum(p.numel() for p in parameters)), dtype=np.float64)
    data = {k: np.concatenate([row[0][k] for row in rows]) for k in (
        "histories", "valid", "accepted_actions", "teacher_log_prob", "teacher_logits")}
    target = np.concatenate([row[2] for row in rows])
    max_logit = max_logp = 0.
    for start in range(0, len(target), 128):
        inds = slice(start, start + 128)
        x = torch.as_tensor(data["histories"][inds]); valid = torch.as_tensor(data["valid"][inds])
        encoded = policy.encode(x.reshape(-1, 378)).reshape(len(x), 4, 128) * valid[:, :, None]
        logits = policy.actor(policy.update(encoded.flatten(1)))
        action = torch.as_tensor(data["accepted_actions"][inds], dtype=torch.long)
        heads = torch.cat((action[:, :6:2] + 2 * action[:, 1:6:2], action[:, 6:]), dim=1)
        log_prob = sum(torch.log_softmax(part, -1).gather(-1, heads[:, i, None]).squeeze(-1)
                       for i, part in enumerate(logits.split(HEADS, -1)))
        max_logit = max(max_logit, float(np.abs(logits.detach().numpy() - data["teacher_logits"][inds]).max()))
        max_logp = max(max_logp, float(np.abs(log_prob.detach().numpy() - data["teacher_log_prob"][inds]).max()))
        for i, column in enumerate(active):
            weight = torch.as_tensor(target[inds, column], dtype=log_prob.dtype)
            grad = torch.autograd.grad((log_prob * weight).sum() / len(rows), parameters, retain_graph=i + 1 < len(active))
            gradients[column] += np.concatenate([g.detach().numpy().ravel() for g in grad])
    if max(max_logit, max_logp) > 1e-5:
        raise RuntimeError("Reconstructed exact checkpoint likelihood failed verification.")
    return gradients, max_logit, max_logp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Read streams and field support without Torch or output writes.")
    args = parser.parse_args()
    started = time.monotonic()
    sys.path.insert(0, str(ASSESSMENT / "evaluator"))
    from storage_gate import snapshot, storage_policy

    policy = storage_policy()
    storage_before = snapshot(policy, startup=True)
    source_check()
    import numpy as np

    ledger, rows, exposure = prepare(np)
    all_rewards = np.concatenate([row[1] for row in rows])
    support = {k: dict(nonzero=int(np.count_nonzero(x)), total=float(x.sum()), description=desc)
               for (k, desc), x in zip(OBJECTIVES.items(), all_rewards.T, strict=True)}
    active = [i for i, k in enumerate(OBJECTIVES) if support[k]["nonzero"]]
    names = tuple(OBJECTIVES)
    same_support = [(names[i], names[j]) for i, j in itertools.combinations(active, 2)
                    if np.array_equal(all_rewards[:, i] != 0., all_rewards[:, j] != 0.)]
    report = dict(source_commit=COMMIT, weight_sha256=WEIGHT, source_attempt_ledger_sha256=digest(ROOT / "anchors/attempts.jsonl"),
                  complete_attempts=len(ledger), decisions=len(all_rewards), exposure=exposure,
                  player_hits=sum(x["player_hits"] for x in ledger), player_damage_episode_totals_only=sum(x["player_damage_dealt"] for x in ledger),
                  objectives=support, identical_nonzero_support_pairs=same_support, gamma=GAMMA,
                  unavailable_measurements=["exact player damage or damage received per decision", "actual shots fired",
                                            "per-decision world positions and unique-cell coverage reward", "terminal post-action vitals"],
                  estimator="Mean over complete collected episodes of sum_t gamma^t grad(log pi(a_t|h_t)) G_t over actor-connected parameters; zero baseline. Descriptive, small training-only suite.")
    if args.inspect:
        print(json.dumps(report, allow_nan=False))
        return
    gradient, logit_error, logp_error = compute(np, rows, active)
    norms = np.linalg.norm(gradient, axis=1)
    pairs, unavailable = {}, []
    keys = tuple(OBJECTIVES)
    for i, j in itertools.combinations(range(len(keys)), 2):
        name = f"{keys[i]}__vs__{keys[j]}"
        denom = norms[i] * norms[j]
        if denom > 1e-10:
            pairs[name] = float(np.dot(gradient[i], gradient[j]) / denom)
        else:
            unavailable.append(name)
    report.update(actor_gradient_norms=dict(zip(keys, norms.tolist(), strict=True)), cosines=pairs,
                  unavailable_zero_support_pairs=unavailable, maximum_logit_error=logit_error,
                  maximum_chosen_log_probability_error=logp_error, storage_before=storage_before,
                  storage_after=snapshot(policy), seconds=time.monotonic()-started)
    path = ROOT / "verification/anchor-rollout-gradients.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
