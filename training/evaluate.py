"""Compare policies on disjoint maps, save the measured episodes, and export the best."""
import argparse
import json
import time

from pathlib import Path
import numpy as np
import torch

from submission.policy import Policy, buttons, pack
from training.game import Episode, maps, REPO
from training.storage import project_root, snapshot
from training.train import save
from training.visible import enable_windows


@torch.inference_mode()
def evaluate(path, split, seeds, ticks, delay):
    policy = None
    if path is not None:
        record = torch.load(path, map_location="cpu", weights_only=True)
        policy = Policy(record["memory"]).eval()
        policy.load_state_dict(record["weights"])
    report, latencies = [], []
    for layout in maps(split):
        game = Episode(layout, ticks)
        try:
            for seed in seeds:
                torch.manual_seed(seed)
                game.env.action_space.seed(seed)
                obs = game.reset(seed)
                previous = np.zeros(9, np.int8)
                state = policy.initial(1, "cpu") if policy is not None else None
                while True:
                    start = time.perf_counter()
                    if policy is None:
                        action = game.env.action_space.sample()
                    else:
                        frame = torch.from_numpy(pack(obs, previous))[None]
                        output, state = policy.advance(policy.encode(frame), state)
                        choice, _, _, _ = policy.decision(output)
                        action = buttons(choice[0].numpy())
                    latencies.append((time.perf_counter() - start) * 1000)
                    if not game.env.action_space.contains(action):
                        raise ValueError("Policy emitted an invalid action.")
                    previous = action
                    obs, _, term, trunc = game.step(action)
                    if delay:
                        time.sleep(delay)
                    if term or trunc:
                        break
                report.append(dict(seed=seed, **game.metrics()))
        finally:
            game.close()
    # Explicit local proxy: a full exit dominates; all coefficients are local choices.
    scores = [100*r["exit"] + 8*r["keys"] + 4*r["locked_sectors_raised"] +
              2*r["door_sectors_raised"] + .2*min(r["cells_96"], 100) + r["kills"] +
              .02*r["health"] - .02*r["damage"] for r in report]
    return dict(checkpoint=str(path) if path else "uniform-random", split=split, ticks=ticks,
                episodes=report, proxy=float(np.mean(scores)), invalid_actions=0,
                latency_ms=dict(mean=float(np.mean(latencies)), p95=float(np.percentile(latencies,95)),
                                p99=float(np.percentile(latencies,99))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Export the best development checkpoint.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002])
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--demo", action="store_true", help="Pace visible games at 35 engine tics/s.")
    args = parser.parse_args()
    if args.promote and (not args.checkpoint or args.split != "dev"):
        parser.error("Promotion requires a development comparison with a trained checkpoint.")
    if not args.checkpoint and not args.random:
        args.checkpoint = [REPO / "submission/weights/policy.pt"]
    enable_windows()
    torch.set_num_threads(1)
    paths = ([None] if args.random else []) + args.checkpoint
    reports = []
    out = project_root() / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        snapshot()
        report = evaluate(path, args.split, args.seeds, args.ticks, 4/35 if args.demo else 0.)
        dest = out / f"{args.split}-{time.time_ns()}.json"
        dest.write_text(json.dumps(report, indent=2))
        print(dest, "proxy", report["proxy"], "latency_ms", report["latency_ms"], flush=True)
        reports.append((path, report["proxy"]))
    if args.promote:
        best, score = max((r for r in reports if r[0] is not None), key=lambda r: r[1])
        selected = torch.load(best, map_location="cpu", weights_only=True)
        save(REPO / "submission/weights/policy.pt", dict(memory=selected["memory"], weights=selected["weights"]))
        print("Exported", best, "development proxy", score)


if __name__ == "__main__":
    main()
