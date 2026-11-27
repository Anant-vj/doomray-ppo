"""Verify anonymous human gameplay evidence from the published structured arrays."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([100, 100, 7, 100, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 20, 20, 20], dtype=np.float32)


def main():
    dataset = json.loads((ROOT / "demonstrations/frozen-human-dataset.json").read_bytes())
    decisions = 0
    for entry in dataset["episodes"]:
        path = (ROOT / entry["payload"]).resolve(strict=True)
        assert path.is_relative_to(ROOT)
        assert sha256(path.read_bytes()).hexdigest() == entry["payload_sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert all(data[k].dtype.kind in "biuf" for k in data.files)
            action, previous = data["actions"], data["previous_actions"]
            count = len(action)
            assert action.shape == (entry["decisions"], 9)
            assert np.isin(action, (0, 1)).all()
            assert (action[:, :6].reshape(count, 3, 2).sum(-1) <= 1).all()
            assert np.array_equal(action, data["engine_actions"])
            assert data["engine_action_matches"].all()
            assert not previous[0].any() and np.array_equal(previous[1:], action[:-1])
            heads = np.concatenate((action[:, :6:2] + 2 * action[:, 1:6:2], action[:, 6:]), axis=1)
            assert np.array_equal(heads, data["action_heads"])
            boundary = data["terminated"] | data["truncated"]
            assert boundary[-1] and not boundary[:-1].any()
            features = np.array(data["entity_features"], dtype=np.float32, copy=True)
            features[:, :, 3:] /= 20
            packed = np.concatenate((data["ray_distance"], data["ray_hit_type"], features.reshape(count, -1),
                                     data["entity_type"], data["entity_mask"], data["vitals"] / SCALE,
                                     previous), axis=1).astype(np.float32)
            assert packed.shape == (count, 378) and np.array_equal(packed, data["packed"])
        decisions += count
    assert len(dataset["episodes"]) == 23 and decisions == dataset["decisions"] == 3242
    print(json.dumps({"episodes": 23, "verified_decisions": decisions, "result": "PASS"}))


if __name__ == "__main__":
    main()
