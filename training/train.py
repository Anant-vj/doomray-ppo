"""Train from scratch on the official observations with visible game windows."""
import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from submission.policy import Policy, buttons, pack
from training.ppo import optimize, advantages
from training.storage import project_root, snapshot


def save(path, payload):
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
    args = parser.parse_args()
    if min(args.workers, args.horizon, args.updates, args.ticks) <= 0 or args.horizon % 16:
        parser.error("Use positive values and a horizon divisible by 16.")
    from training.visible import enable_windows
    from training.game import Episode, maps
    enable_windows()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    pool = maps("train")
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
    with ExitStack() as stack:
        games = []
        stack.callback(lambda: [g.close() for g in games])
        obs = []
        for i in range(args.workers):
            game = Episode(pool[i % len(pool)], args.ticks)
            games.append(game)
            obs.append(game.reset(int(rng.integers(2**30))))
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
                    results = [g.step(b) for g, b in zip(games, press)]
                    obs, reward, term, trunc = map(list, zip(*results))
                    boundary = np.logical_or(term, trunc)
                    # Capture the final observation before resetting a timed-out worker.
                    end_x = torch.as_tensor(np.stack([pack(o, p) for o, p in zip(obs, press)]), device=dev)
                    next_output, _ = policy.advance(policy.encode(end_x), h_next)
                    following = policy.critic(next_output).squeeze(-1) * torch.tensor(np.logical_not(term), device=dev)
                    done = torch.as_tensor(boundary.astype(np.float32), device=dev)
                    rows.append((x, h, starts, a, logp, value, torch.tensor(reward, device=dev), following, done))
                    previous = press.copy()
                    for i in np.flatnonzero(boundary):
                        episodes.append(games[i].metrics())
                        layout = pool[int(rng.integers(len(pool)))]
                        if (games[i].path, games[i].map_name) != layout:
                            games[i].close()
                            games[i] = Episode(layout, args.ticks)
                        obs[i] = games[i].reset(int(rng.integers(2**30)))
                        previous[i] = 0
                    h, starts = h_next * (1 - done[:, None]), done
            x, states, resets, actions, old_logp, values, rewards, following, done = map(torch.stack, zip(*rows))
            adv = advantages(rewards, values, following, done)
            n, loss = optimize(policy, optimizer, (x, states, resets, actions, old_logp, values + adv, adv))
            gradients += n
            payload = dict(memory=args.memory, weights=policy.state_dict(), updates=update, gradients=gradients)
            if update == counter + 1 or update % 8 == 0 or update == counter + args.updates:
                save(checkpoint / "resume.pt", dict(**payload, optimizer=optimizer.state_dict()))
                # Keep the first fallback and four recent candidate slots.
                slot = "first" if update == 1 else f"slot-{(update // 8) % 4}"
                save(checkpoint / f"policy-{slot}.pt", payload)
            stats = dict(update=update, gradients=gradients, memory=args.memory, seed=args.seed, loss=loss,
                         transitions_this_run=(update-counter)*args.workers*args.horizon,
                         sps=round((update-counter)*args.workers*args.horizon/(time.monotonic()-started), 2),
                         episodes=episodes)
            episodes = []
            with (logdir / f"{args.memory}-{args.seed}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            print({k:v for k,v in stats.items() if k != "episodes"}, flush=True)
            snapshot()


if __name__ == "__main__":
    main()
