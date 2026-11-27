"""Supervise the exact stacked policy on registered-team keyboard recordings."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from submission.policy import HEADS, Policy, buttons, pack
from training.storage import project_root, snapshot
from training.train import save


OFFICIAL_SOURCE = "6c2d5499f973c77b4a239f778837280edf64e57b"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_episodes(root):
    episodes = []
    for path in sorted(root.rglob("episode.json")):
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
        if not (meta.get("eligible_for_bc") and meta.get("complete_episode")):
            continue
        n = meta["decisions"]
        if (meta.get("source") != "registered_team_member_keyboard" or not meta.get("attestations") or
                not meta.get("permission_reference") or meta.get("frozen_wrapper_commit") != OFFICIAL_SOURCE or
                n <= 0 or meta.get("exact_engine_matches") != n or not meta.get("previous_action_chain_valid")):
            raise ValueError(f"Invalid human provenance or readback receipt: {path}")
        payload = path.parent / "steps.npz"
        if sha(payload) != meta["payload_sha256"]:
            raise ValueError(f"Demonstration payload digest changed: {path}")
        with np.load(payload, allow_pickle=False) as data:
            x, y = data["packed"], data["action_heads"]
            actions, previous = data["actions"], data["previous_actions"]
            boundaries = data["terminated"] | data["truncated"]
            if (x.shape != (n, 378) or y.shape != (n, 6) or actions.shape != (n, 9) or
                    not np.isfinite(x).all() or not np.all((actions == 0) | (actions == 1)) or
                    not np.array_equal(actions, data["engine_actions"]) or
                    not np.array_equal(actions, buttons(y)) or
                    np.any(y < 0) or np.any(y >= np.array(HEADS)) or
                    not np.array_equal(previous[0], np.zeros(9)) or not np.array_equal(previous[1:], actions[:-1]) or
                    boundaries.shape != (n,) or not boundaries[-1] or boundaries[:-1].any()):
                raise ValueError(f"Invalid action/history/episode boundary: {path}")
            rays = data["ray_distance"], data["ray_hit_type"]
            entities = data["entity_features"], data["entity_type"], data["entity_mask"]
            vitals = data["vitals"]
            for i in range(n):
                wrapped = {"rays": {"distance": rays[0][i], "hit_type": rays[1][i]},
                           "visible_entities": {"features": entities[0][i], "entity_type": entities[1][i],
                                                "mask": entities[2][i]}, "vitals": vitals[i]}
                if not np.array_equal(pack(wrapped, previous[i]), x[i]):
                    raise ValueError(f"Feature pack differs at row {i}: {path}")
        episodes.append((meta, path, torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)))
    return episodes


def stacked_output(policy, features, start, stop):
    left = max(0, start - 3)
    encoded = policy.encode(features[left:stop])
    missing = 3 - (start - left)
    sequence = torch.cat((encoded.new_zeros((missing, encoded.shape[-1])), encoded))
    windows = sequence.unfold(0, 4, 1).transpose(1, 2)
    history = windows[:, :3, :].reshape(len(windows), -1)
    return policy.advance(windows[:, 3, :], history)[0]


@torch.no_grad()
def verify_temporal_parity(policy, features):
    sample = features[:12]
    sequential = []
    history = policy.initial(1, sample.device)
    for row in sample:
        output, history = policy.advance(policy.encode(row.unsqueeze(0)), history)
        sequential.append(output)
    expected = torch.cat(sequential)
    torch.testing.assert_close(stacked_output(policy, sample, 0, len(sample)), expected, atol=2e-6, rtol=2e-5)
    if len(sample) > 4:
        torch.testing.assert_close(stacked_output(policy, sample, 4, len(sample)), expected[4:], atol=2e-6, rtol=2e-5)


def pass_episodes(policy, episodes, optimizer=None, batch=96):
    labels_correct = torch.zeros(len(HEADS), dtype=torch.long)
    total_loss = 0.
    decisions = updates = 0
    for _, _, x, y in episodes:
        for start in range(0, len(x), batch):
            end = min(start + batch, len(x))
            with torch.set_grad_enabled(optimizer is not None):
                output = stacked_output(policy, x, start, end)
                labels = y[start:end]
                logits = policy.actor(output).split(HEADS, dim=-1)
                loss = torch.stack([F.cross_entropy(head, labels[:, i])
                                    for i, head in enumerate(logits)]).mean()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                    optimizer.step()
                    updates += 1
            n = len(labels)
            decisions += n
            total_loss += float(loss.detach()) * n
            labels_correct += torch.stack([(head.detach().argmax(-1) == labels[:, i]).sum()
                                            for i, head in enumerate(logits)])
    return {"loss": total_loss / decisions, "decisions": decisions,
            "head_accuracy": (labels_correct / decisions).tolist(), "optimizer_updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", required=True, type=Path)
    parser.add_argument("--validation-family", required=True)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max-seconds", default=360, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2207)
    root = project_root()
    storage = snapshot()
    if (storage["internal_delta_bytes"] > 6_500_000_000 or
            storage.get("physical_available_bytes", float("inf")) < 805_306_368 or
            storage.get("commit_available_bytes", float("inf")) < 1_000_000_000):
        raise RuntimeError("Supervised startup requires additional storage or memory headroom.")
    data = load_episodes(root / "demonstrations/human")
    training = [row for row in data if row[0]["layout_family"] != args.validation_family]
    valid = [row for row in data if row[0]["layout_family"] == args.validation_family]
    if not (training and valid):
        raise RuntimeError("Record complete episodes in at least two disjoint layout families.")
    destination = root / "checkpoints/bc-precision"
    if destination.exists():
        raise RuntimeError("A supervised snapshot already exists in this campaign.")
    destination.mkdir(parents=True)
    saved = torch.load(args.initialize, map_location="cpu", weights_only=True)
    if saved["memory"] != "stack":
        raise RuntimeError("Only the existing four-decision stack is supported.")
    policy = Policy("stack")
    policy.load_state_dict(saved["weights"])
    verify_temporal_parity(policy, max(training, key=lambda row: len(row[2]))[2])
    optimizer = torch.optim.Adam(policy.parameters(), lr=.00015, eps=1e-5)
    selection = {key: [{"path": str(path), "layout_family": m["layout_family"], "seed": m["seed"],
                        "payload_sha256": m["payload_sha256"], "decisions": m["decisions"],
                        "outcome": m["outcome"]} for m, path, _, _ in episodes]
                 for key, episodes in (("train", training), ("validation", valid))}
    summary = {"parent_sha256": sha(args.initialize), "seed": 2207, "learning_rate": .00015,
               "batched_history_matches_sequential_actor": True,
               "whole_episode_layout_split": selection,
               "eligible_episode_outcomes": dict(Counter(row[0]["outcome"] for row in data)),
               "parent_train": pass_episodes(policy, training), "parent_validation": pass_episodes(policy, valid),
               "epochs": []}
    rng = random.Random(2207)
    best = float("inf")
    total_updates = 0
    began = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training)
        trained = pass_episodes(policy, training, optimizer)
        held = pass_episodes(policy, valid)
        total_updates += trained["optimizer_updates"]
        row = {"epoch": epoch, "train": trained, "validation": held, "bc_optimizer_updates": total_updates}
        summary["epochs"].append(row)
        if held["loss"] < best:
            best = held["loss"]
            summary["best_epoch"] = epoch
            summary["best_validation"] = held
            save(destination / "policy.pt",
                 {"memory": "stack", "weights": policy.state_dict(), "updates": saved.get("updates", 0),
                  "gradients": saved.get("gradients", 0), "bc_optimizer_updates": total_updates, "bc_epoch": epoch})
        temporary = destination / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination / "summary.json")
        print(json.dumps(row), flush=True)
        storage = snapshot()
        if (time.monotonic() - began >= args.max_seconds or storage["internal_delta_bytes"] >= 6_800_000_000 or
                storage.get("physical_available_bytes", float("inf")) < 536_870_912 or
                storage.get("commit_available_bytes", float("inf")) < 1_000_000_000 or
                (root / "bc-precision.stop").exists()):
            break


if __name__ == "__main__":
    main()
