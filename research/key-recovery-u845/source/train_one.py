"""Run precisely the predeclared learner once, with an allowlisted training map pool."""
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "learner"


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    config = plan["training"]
    sys.path.insert(0, str(REPO))
    import torch
    from training import game, train
    from training.storage import snapshot

    state = snapshot()
    if state["internal_delta_bytes"] >= 6_500_000_000:
        raise RuntimeError("500 MB start reserve unavailable")
    weight = REPO / "submission/weights/policy.pt"
    assert sha256(weight.read_bytes()).hexdigest() == plan["source_weight_sha256"]
    candidate_parent = torch.load(weight, weights_only=True, map_location="cpu")
    assert candidate_parent["memory"] == "stack"
    parent = ROOT / "c749-parent.pt"
    with parent.open("xb") as f:
        torch.save(dict(memory="stack", weights=candidate_parent["weights"], updates=749, gradients=7614), f)
    drill_pool = []
    for row in plan["maps"]["drills"]:
        wad = ROOT / row["file"]
        assert wad.parent == ROOT / "drills" and sha256(wad.read_bytes()).hexdigest() == row["sha256"]
        drill_pool.append((wad, row["marker"]))

    def training_maps(split):
        assert split == "mixed"
        return list(drill_pool)

    game.maps = training_maps
    until = datetime.fromisoformat(plan["hard_training_deadline"])
    seconds = int((until - datetime.now().astimezone()).total_seconds()) - 90
    if seconds < 60:
        raise RuntimeError("No safe training time left")
    sys.argv = [
        "training.train", "--headless", "--parallel", "--no-render-buffers", "--memory", "stack",
        "--device", "cpu", "--initialize", str(parent), "--seed", str(config["seed"]),
        "--workers", str(config["workers"]), "--horizon", str(config["horizon"]),
        "--updates", str(config["updates"]), "--ticks", str(config["ticks"]),
        "--learning-rate", str(config["lr"]), "--curriculum", "mixed", "--reward-profile", "progression",
        "--credit-mode", "exact-ppo", "--run-name", "key-recovery", "--storage-stop-bytes", "6500000000",
        "--max-seconds", str(seconds),
    ]
    train.main()


if __name__ == "__main__":
    main()
