"""Train PPO on the official ViZDoom environment."""
import argparse
import json
import math
import os
import time
from pathlib import Path

from training.storage import project_root, snapshot


def save(path, payload):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ticks", type=int, default=1050)
    parser.add_argument("--memory", choices=("stack", "gru"), default="stack")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--headless", action="store_true", help="Run without game windows.")
    parser.add_argument("--parallel", action="store_true", help="Run headless simulation in spawned workers.")
    parser.add_argument("--no-render-buffers", action="store_true", help="Skip unused RGB and depth buffers.")
    parser.add_argument(
        "--curriculum", nargs="?", choices=("expanded", "progression"), const="expanded",
        help="Include practice maps. Choose progression to add the starter's key-and-lock fixture.",
    )
    parser.add_argument("--checkpoint-seconds", type=float, help="Rotate checkpoints at this elapsed interval.")
    parser.add_argument("--max-seconds", type=float, help="Save and exit at the first update after this time limit.")
    parser.add_argument("--stop-file", type=Path, help="Save and exit after the next PPO update if this file appears.")
    args = parser.parse_args()
    if min(args.workers, args.horizon, args.updates, args.ticks) <= 0 or args.horizon % 16:
        parser.error("Use positive values and a horizon divisible by 16.")
    for option, duration in (
        ("--checkpoint-seconds", args.checkpoint_seconds),
        ("--max-seconds", args.max_seconds),
    ):
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            parser.error(f"{option} must be a positive finite number.")
    if args.parallel and not args.headless:
        parser.error("--parallel requires --headless.")
    if args.stop_file is not None and args.stop_file.exists():
        parser.error("Choose a stop file that does not already exist.")

    import numpy as np
    import torch

    from submission.policy import Policy, buttons, pack
    from training.game import SerialEpisodes, maps
    from training.ppo import advantages, optimize

    backend = SerialEpisodes
    if args.parallel:
        from training.parallel import ParallelEpisodes

        backend = ParallelEpisodes
    if not args.headless:
        from training.visible import enable_windows

        enable_windows()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    split = {"expanded": "curriculum", "progression": "progression"}.get(args.curriculum, "train")
    pool = maps(split)
    policy = Policy(args.memory).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    counter = gradients = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=dev, weights_only=True)
        if saved["memory"] != args.memory:
            parser.error("Choose the memory architecture of the saved checkpoint.")
        policy.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        counter, gradients = saved["updates"], saved["gradients"]
        rng = np.random.default_rng(args.seed + counter)
        torch.manual_seed(args.seed + counter)
    root = project_root()
    checkpoint = root / "checkpoints" / f"{args.memory}-{args.seed}"
    logdir = root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    snapshot()
    started = time.monotonic()
    checkpointed = started
    rotation = 0
    layouts = [pool[i % len(pool)] for i in range(args.workers)]
    seeds = [int(rng.integers(2**30)) for _ in layouts]
    with backend(layouts, args.ticks, seeds, render_buffers=not args.no_render_buffers) as games:
        obs = games.observations
        previous = np.zeros((args.workers, 9), np.int8)
        h = policy.initial(args.workers, dev)
        starts = torch.ones(args.workers, device=dev)
        episodes = []
        for update in range(counter + 1, counter + args.updates + 1):
            rows = []
            with torch.no_grad():
                for _ in range(args.horizon):
                    x = torch.as_tensor(np.stack([pack(o, p) for o, p in zip(obs, previous)]), device=dev)
                    output, h_next = policy.advance(policy.encode(x), h)
                    a, logp, _, value = policy.decision(output)
                    press = buttons(a.cpu().numpy())
                    obs, reward, term, trunc, completed = map(list, zip(*games.step(press)))
                    boundary = np.logical_or(term, trunc)
                    # Capture the final observation before resetting a timed-out worker.
                    end_x = torch.as_tensor(np.stack([pack(o, p) for o, p in zip(obs, press)]), device=dev)
                    next_output, _ = policy.advance(policy.encode(end_x), h_next)
                    following = policy.critic(next_output).squeeze(-1) * torch.tensor(np.logical_not(term), device=dev)
                    done = torch.as_tensor(boundary.astype(np.float32), device=dev)
                    rows.append((x, h, starts, a, logp, value, torch.tensor(reward, device=dev), following, done))
                    previous = press.copy()
                    pending_resets = {}
                    for i in np.flatnonzero(boundary):
                        episodes.append(completed[i])
                        layout = pool[int(rng.integers(len(pool)))]
                        pending_resets[int(i)] = (layout, int(rng.integers(2**30)))
                        previous[i] = 0
                    if pending_resets:
                        for i, observation in games.reset(pending_resets).items():
                            obs[i] = observation
                    h, starts = h_next * (1 - done[:, None]), done
            x, states, resets, actions, old_logp, values, rewards, following, done = map(torch.stack, zip(*rows))
            adv = advantages(rewards, values, following, done)
            n, loss = optimize(policy, optimizer, (x, states, resets, actions, old_logp, values + adv, adv))
            gradients += n
            payload = dict(memory=args.memory, weights=policy.state_dict(), updates=update, gradients=gradients)
            now = time.monotonic()
            time_limit = args.max_seconds is not None and now - started >= args.max_seconds
            stop_requested = args.stop_file is not None and args.stop_file.exists()
            stopping = time_limit or stop_requested
            interval_elapsed = (
                now - checkpointed >= args.checkpoint_seconds if args.checkpoint_seconds else update % 8 == 0
            )
            if update == counter + 1 or interval_elapsed or update == counter + args.updates or stopping:
                save(checkpoint / "resume.pt", dict(**payload, optimizer=optimizer.state_dict()))
                # Keep the first fallback and four recent candidate slots.
                slot_number = rotation if args.checkpoint_seconds else update // 8
                slot = "first" if update == 1 else f"slot-{slot_number % 4}"
                save(checkpoint / f"policy-{slot}.pt", payload)
                if slot != "first":
                    rotation += 1
                checkpointed = now
            transitions = (update - counter) * args.workers * args.horizon
            elapsed = time.monotonic() - started
            stats = dict(update=update, gradients=gradients, memory=args.memory, seed=args.seed, loss=loss,
                         transitions_this_run=transitions, elapsed_seconds=round(elapsed, 2),
                         sps=round(transitions / elapsed, 2), rollout_mean_shaped_reward=float(rewards.mean()),
                         backend="parallel" if args.parallel else "serial", workers=args.workers,
                         horizon=args.horizon, ticks=args.ticks, split=split,
                         render_buffers=not args.no_render_buffers, episodes=episodes)
            episodes = []
            with (logdir / f"{args.memory}-{args.seed}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            print({k:v for k,v in stats.items() if k != "episodes"}, flush=True)
            snapshot()
            if stopping:
                break


if __name__ == "__main__":
    main()
