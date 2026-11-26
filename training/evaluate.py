"""Compare policies on disjoint maps, save the measured episodes, and export the best."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from doom_env.remote import AgentFailure, validate_action
from submission.policy import Policy, buttons, pack
from training.game import Episode, maps, REPO
from training.storage import project_root, snapshot
from training.train import save
from training.visible import enable_windows


@torch.inference_mode()
def evaluate(path, split, seeds, ticks, delay):
    policy = None
    checkpoint = {}
    if path is not None:
        contents = Path(path).read_bytes()
        record = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        checkpoint = dict(sha256=hashlib.sha256(contents).hexdigest(), memory=record["memory"],
                          update=record.get("updates"), gradients=record.get("gradients"))
        policy = Policy(record["memory"]).eval()
        policy.load_state_dict(record["weights"])
    report, latencies = [], []
    for layout in maps(split):
        game = Episode(layout, ticks)
        try:
            no_action = np.zeros(game.env.action_space.shape, dtype=np.int8)
            for seed in seeds:
                torch.manual_seed(seed)
                game.env.action_space.seed(seed)
                obs = game.reset(seed)
                previous = np.zeros(9, np.int8)
                state = policy.initial(1, "cpu") if policy is not None else None
                invalid_actions = 0
                while True:
                    if invalid_actions:
                        action = no_action
                    else:
                        start = time.perf_counter()
                        if policy is None:
                            action = game.env.action_space.sample()
                        else:
                            frame = torch.from_numpy(pack(obs, previous))[None]
                            output, state = policy.advance(policy.encode(frame), state)
                            choice, _, _, _ = policy.decision(output)
                            action = buttons(choice[0].numpy())
                        latencies.append((time.perf_counter() - start) * 1000)
                        try:
                            action = validate_action(action, game.env.action_space)
                        except AgentFailure:
                            # The official evaluator holds no buttons for the rest of an invalid rollout.
                            invalid_actions += 1
                            action = no_action
                    previous = action
                    obs, _, term, trunc = game.step(action)
                    if delay:
                        time.sleep(delay)
                    if term or trunc:
                        break
                report.append(dict(seed=seed, invalid_actions=invalid_actions, **game.metrics()))
        finally:
            game.close()
    # Explicit local proxy: a full exit dominates; all coefficients are local choices.
    scores = [100*r["exit"] + 8*r["keys"] + 4*r["locked_sectors_raised"] +
              2*r["door_sectors_raised"] + .2*min(r["cells_96"], 100) + r["kills"] +
              .02*r["health"] - .02*r["damage"] for r in report]
    metric_names = ("wrapper_return", "steps", "kills", "health", "damage", "cells_96",
                    "door_sectors_raised", "locked_sectors_raised", "keys")
    means = {name: float(np.mean([row[name] for row in report])) for name in metric_names}
    return dict(checkpoint=str(path) if path else "uniform-random", checkpoint_metadata=checkpoint,
                split=split, seeds=list(seeds), ticks=ticks, episodes=report,
                proxy=float(np.mean(scores)), invalid_actions=sum(row["invalid_actions"] for row in report),
                totals=dict(kills=sum(row["kills"] for row in report), keys=sum(row["keys"] for row in report),
                            doors=sum(row["door_sectors_raised"] for row in report),
                            locked_doors=sum(row["locked_sectors_raised"] for row in report)),
                survival_rate=float(np.mean([row["alive"] for row in report])),
                exit_rate=float(np.mean([row["exit"] for row in report])),
                mean_per_episode=means,
                latency_ms=dict(mean=float(np.mean(latencies)), p95=float(np.percentile(latencies, 95)),
                                p99=float(np.percentile(latencies, 99))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Export the best development checkpoint.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002])
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--headless", action="store_true", help="Run the official wrapper without game windows.")
    parser.add_argument("--demo", action="store_true", help="Pace visible games at 35 engine tics/s.")
    args = parser.parse_args()
    if args.promote and (not args.checkpoint or args.split != "dev"):
        parser.error("Promotion requires a development comparison with a trained checkpoint.")
    if args.headless and args.demo:
        parser.error("Demo playback requires a game window.")
    if args.ticks <= 0:
        parser.error("Set a positive episode tick cap.")
    if not args.checkpoint and not args.random:
        args.checkpoint = [REPO / "submission/weights/policy.pt"]
    if not args.headless:
        enable_windows()
    torch.set_num_threads(1)
    paths = ([None] if args.random else []) + args.checkpoint
    reports = []
    out = project_root() / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        snapshot()
        report = evaluate(path, args.split, args.seeds, args.ticks, 4/35 if args.demo else 0.)
        report["headless"] = args.headless
        dest = out / f"{args.split}-{time.time_ns()}.json"
        dest.write_text(json.dumps(report, indent=2))
        print(dest, "proxy", report["proxy"], "latency_ms", report["latency_ms"], flush=True)
        reports.append((path, report))
    if args.promote:
        eligible = [(path, report) for path, report in reports if path is not None and not report["invalid_actions"]]
        if not eligible:
            parser.exit(1, "No candidate completed the development suite with valid actions.\n")
        best, report = max(eligible, key=lambda r: r[1]["proxy"])
        contents = best.read_bytes()
        if hashlib.sha256(contents).hexdigest() != report["checkpoint_metadata"]["sha256"]:
            parser.exit(1, "The selected checkpoint rotated during evaluation. Rerun with a fixed copy.\n")
        selected = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        save(REPO / "submission/weights/policy.pt", dict(memory=selected["memory"], weights=selected["weights"]))
        print("Exported", best, "development proxy", report["proxy"])


if __name__ == "__main__":
    main()
