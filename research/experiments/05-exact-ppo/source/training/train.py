"""Train PPO on the official ViZDoom environment."""
import argparse
import json
import math
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

from training.storage import internal_limit, project_root, snapshot


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
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Restore policy and optimizer state.")
    initialization.add_argument("--initialize", type=Path, help="Start from saved policy weights with a fresh optimizer.")
    parser.add_argument("--learning-rate", type=float, help="Override the optimizer learning rate.")
    parser.add_argument("--run-name", help="Name for logs and checkpoints, independent of the random seed.")
    parser.add_argument("--reward-profile", choices=("control", "survival", "progression", "combat"), default="control")
    parser.add_argument("--headless", action="store_true", help="Run without game windows.")
    parser.add_argument("--parallel", action="store_true", help="Run headless simulation in spawned workers.")
    parser.add_argument("--no-render-buffers", action="store_true", help="Skip unused RGB and depth buffers.")
    parser.add_argument(
        "--curriculum", nargs="?", choices=("expanded", "progression", "mixed", "varied"), const="expanded",
        help="Add practice maps, key-and-lock progression, or the mixed arena and resource lessons.",
    )
    parser.add_argument("--checkpoint-seconds", type=float, help="Rotate checkpoints at this elapsed interval.")
    parser.add_argument("--max-seconds", type=float, help="Save and exit at the first update after this time limit.")
    parser.add_argument("--stop-file", type=Path, help="Save and exit after the next PPO update if this file appears.")
    parser.add_argument("--storage-stop-bytes", type=int,
                        help="Save and exit at this soft limit on internal disk usage increase.")
    parser.add_argument("--credit-mode", choices=("ppo-control", "pbrs", "pbrs-anchor",
                        "cache-smoke", "exact-smoke", "exact-ppo", "balanced-reward", "balanced-pbrs-anchor"))
    parser.add_argument("--anchor-file", type=Path)
    parser.add_argument("--anchor-sha256")
    parser.add_argument("--anchor-beta", type=float, default=.04)
    args = parser.parse_args()
    if min(args.workers, args.horizon, args.updates, args.ticks) <= 0 or args.horizon % 16:
        parser.error("Use positive values and a horizon divisible by 16.")
    for option, duration in (
        ("--checkpoint-seconds", args.checkpoint_seconds),
        ("--max-seconds", args.max_seconds),
        ("--learning-rate", args.learning_rate),
    ):
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            parser.error(f"{option} must be a positive finite number.")
    if args.parallel and not args.headless:
        parser.error("--parallel requires --headless.")
    credit = args.credit_mode is not None
    anchored = args.credit_mode in ("pbrs-anchor", "balanced-pbrs-anchor")
    exact_history = args.credit_mode in ("exact-smoke", "exact-ppo", "balanced-reward", "balanced-pbrs-anchor")
    transition_balance = args.credit_mode in ("balanced-reward", "balanced-pbrs-anchor")
    original_curriculum = args.credit_mode in ("cache-smoke", "exact-smoke", "exact-ppo")
    if credit and not (args.parallel and args.device == "cpu" and args.memory == "stack"
                       and args.initialize and args.curriculum == "mixed" and args.reward_profile == "progression"):
        parser.error("Credit comparison uses the parallel CPU stack, frozen initialization, mixed curriculum and progression reward.")
    if anchored and not (args.anchor_file and args.anchor_sha256 and 0 < args.anchor_beta <= 1):
        parser.error("The anchor arm requires frozen anchor bytes, SHA-256 and a positive beta.")
    if args.stop_file is not None and args.stop_file.exists():
        parser.error("Choose a stop file that does not already exist.")
    if args.run_name and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.run_name):
        parser.error("Use only letters, digits, hyphens and underscores in --run-name.")
    soft_limit = args.storage_stop_bytes
    if soft_limit is not None and not 0 < soft_limit < internal_limit():
        parser.error("--storage-stop-bytes must be below the configured internal drive limit.")
    if soft_limit is not None and snapshot()["internal_delta_bytes"] >= soft_limit:
        parser.error("Internal drive usage is already above the configured soft stop limit.")

    import numpy as np
    import torch

    from doom_env.spaces import BUTTONS
    from submission.policy import Policy, buttons, pack
    from training.game import SerialEpisodes, maps
    from training.ppo import advantages, optimize

    if credit:
        from doom_env.remote import validate_action
        from training.credit import COLUMNS, SkillSchedule, family
        from training.reward_diagnostics import component_statistics, current_context, measured_gradients
        from training.transition_schedule import OriginalSchedule, TransitionSchedule

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
    split = "curriculum" if args.curriculum == "expanded" else args.curriculum or "train"
    pool = maps(split)
    policy = Policy(args.memory).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate or 3e-4, eps=1e-5)
    counter = gradients = 0
    source = args.resume or args.initialize
    if source:
        saved = torch.load(source, map_location=dev, weights_only=True)
        if saved["memory"] != args.memory:
            parser.error("Choose the memory architecture of the saved checkpoint.")
        policy.load_state_dict(saved["weights"])
        if args.resume:
            if "optimizer" not in saved:
                parser.error("--resume requires optimizer state. Use --initialize for policy weights.")
            optimizer.load_state_dict(saved["optimizer"])
            if args.learning_rate is not None:
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
        counter, gradients = saved.get("updates", 0), saved.get("gradients", 0)
        rng = np.random.default_rng(args.seed + counter)
        torch.manual_seed(args.seed + counter)
    anchor = None
    initial_anchors = None
    if anchored:
        from training.preservation import AnchorSampler

        anchor = AnchorSampler(args.anchor_file, args.anchor_sha256, args.seed + counter + 1)
        initial_anchors = anchor.measure(policy)
        if any(abs(item["kl_nats"]) > 1e-5 for item in initial_anchors.values()):
            raise ValueError("Recorded teacher and unfrozen learner differ at initialization.")
    root = project_root()
    run_name = args.run_name or f"{args.memory}-{args.seed}"
    checkpoint = root / "checkpoints" / run_name
    logdir = root / "logs"
    if args.initialize and (checkpoint.exists() or (logdir / f"{run_name}.jsonl").exists()):
        parser.error("Choose a new --run-name when initializing a separate run from weights.")
    logdir.mkdir(parents=True, exist_ok=True)
    if credit:
        credit_logs = logdir / f"{run_name}-credit"
        credit_logs.mkdir(exist_ok=False)
    snapshot()
    started = time.monotonic()
    checkpointed = started
    rotation = 0
    backend_options = {}
    if credit:
        if original_curriculum:
            schedule = OriginalSchedule(pool, rng)
        elif transition_balance:
            schedule = TransitionSchedule(pool, args.seed + counter, args.workers, (args.ticks + 3) // 4)
        else:
            schedule = SkillSchedule(pool, args.seed + counter, args.workers)
        layouts, seeds = map(list, zip(*(schedule.draw(i, initial=True) for i in range(args.workers))))
        backend_options["credit_mode"] = args.credit_mode
    else:
        layouts = [pool[i % len(pool)] for i in range(args.workers)]
        seeds = [int(rng.integers(2**30)) for _ in layouts]
    with backend(layouts, args.ticks, seeds, render_buffers=not args.no_render_buffers,
                 reward_profile=args.reward_profile, **backend_options) as games:
        obs = games.observations
        previous = np.zeros((args.workers, 9), np.int8)
        h = policy.initial(args.workers, dev)
        starts = torch.ones(args.workers, device=dev)
        stack_context = deque(maxlen=3)
        episodes = []
        for update in range(counter + 1, counter + args.updates + 1):
            rows = []
            diagnose = args.credit_mode in ("ppo-control", "exact-ppo") and (update - counter) % 8 == 0
            contexts, credit_rows, potential_rows, layout_rows = [], [], [], []
            exposure = Counter()
            skill_exposure = Counter()
            reset_exposure = Counter()
            rollout_started = time.monotonic()
            with torch.no_grad():
                for _ in range(args.horizon):
                    x = torch.as_tensor(np.stack([pack(o, p) for o, p in zip(obs, previous)]), device=dev)
                    output, h_next = policy.advance(policy.encode(x), h)
                    a, logp, _, value = policy.decision(output)
                    press = buttons(a.cpu().numpy())
                    if diagnose or exact_history:
                        contexts.append(current_context(x, stack_context))
                    if credit:
                        for row in press:
                            validate_action(row)
                    result = games.step(press)
                    if credit:
                        obs, reward, term, trunc, completed, step_credit, annotated = map(list, zip(*result))
                        credit_rows.append(np.stack(step_credit))
                        potential_rows.append(annotated)
                        identifiers = [f"{path.name}:{map_name}" for path, map_name in layouts]
                        layout_rows.append(identifiers)
                        exposure.update(identifiers)
                        skill_exposure.update(map(family, layouts))
                    else:
                        obs, reward, term, trunc, completed = map(list, zip(*result))
                    boundary = np.logical_or(term, trunc)
                    if transition_balance:
                        schedule.observe(boundary)
                    done = torch.as_tensor(boundary.astype(np.float32), device=dev)
                    ongoing = 1 - done
                    rows.append((x, h, starts, a, logp, value, torch.tensor(reward, device=dev), done))
                    if args.memory == "stack":
                        stack_context.append((x, ongoing))
                    previous = press.copy()
                    pending_resets = {}
                    for i in np.flatnonzero(boundary):
                        episodes.append(completed[i])
                        if credit:
                            layout, episode_seed = schedule.draw(int(i))
                            layouts[i] = layout
                            reset_exposure[family(layout)] += 1
                        else:
                            layout = pool[int(rng.integers(len(pool)))]
                            episode_seed = int(rng.integers(2**30))
                        pending_resets[int(i)] = (layout, episode_seed)
                        previous[i] = 0
                    if pending_resets:
                        for i, observation in games.reset(pending_resets).items():
                            obs[i] = observation
                    h, starts = h_next * ongoing[:, None], done
                end_x = torch.as_tensor(
                    np.stack([pack(o, p) for o, p in zip(obs, previous)]), device=dev
                )
                end_output, _ = policy.advance(policy.encode(end_x), h)
                final_value = policy.critic(end_output).squeeze(-1)
            x, states, resets, actions, old_logp, values, rewards, done = map(torch.stack, zip(*rows))
            # A capped game has ended. Its wrapper observation is the pre-action frame.
            # At rollout boundaries within a game, the actual successor is available.
            following = torch.cat((values[1:], final_value[None])) * (1 - done)
            adv = advantages(rewards, values, following, done)
            gradient_diagnostics = None
            if credit:
                components = np.stack(credit_rows)
                if diagnose:
                    gradient_diagnostics = measured_gradients(policy, contexts, actions, components, done.numpy())
            optimize_started = time.monotonic()
            n, loss, ppo = optimize(
                policy, optimizer, (x, states, resets, actions, old_logp, values + adv, adv), diagnostics=True,
                anchor=anchor, anchor_beta=args.anchor_beta,
                raw_history=tuple(map(torch.stack, zip(*contexts))) if exact_history else None,
            )
            optimize_seconds = time.monotonic() - optimize_started
            if stack_context:
                # Rebuild the three carried feature frames with the updated encoder.
                with torch.no_grad():
                    frames, ongoing = zip(*stack_context)
                    encoded = policy.encode(torch.cat(frames)).reshape(len(frames), args.workers, -1)
                    valid = torch.stack(ongoing).flip(0).cumprod(0).flip(0)
                    h = (encoded * valid[..., None]).transpose(0, 1).reshape(args.workers, -1)
            gradients += n
            payload = dict(memory=args.memory, weights=policy.state_dict(), updates=update, gradients=gradients)
            now = time.monotonic()
            time_limit = args.max_seconds is not None and now - started >= args.max_seconds
            stop_requested = args.stop_file is not None and args.stop_file.exists()
            storage = snapshot()
            storage_limit = soft_limit is not None and storage["internal_delta_bytes"] >= soft_limit
            memory_limit = credit and (storage.get("physical_available_bytes", math.inf) < 536_870_912
                                       or storage.get("commit_available_bytes", math.inf) < 1_000_000_000)
            unstable = credit and (not math.isfinite(loss) or ppo["approx_kl"] > .025
                                   or ppo["clip_fraction"] > .25 or ppo["initial_log_ratio_abs_max"] > .01)
            stopping = time_limit or stop_requested or storage_limit or memory_limit or unstable
            interval_elapsed = (
                now - checkpointed >= args.checkpoint_seconds if args.checkpoint_seconds else update % 8 == 0
            )
            checkpoint_write_seconds = 0.
            if update == counter + 1 or interval_elapsed or update == counter + args.updates or stopping:
                save(checkpoint / "resume.pt", dict(**payload, optimizer=optimizer.state_dict()))
                # Keep the first fallback and four recent candidate slots.
                slot_number = rotation if args.checkpoint_seconds else update // 8
                slot = "first" if update == 1 else f"slot-{slot_number % 4}"
                save(checkpoint / f"policy-{slot}.pt", payload)
                if slot != "first":
                    rotation += 1
                checkpointed = now
                checkpoint_write_seconds = time.monotonic() - now
            transitions = (update - counter) * args.workers * args.horizon
            elapsed = time.monotonic() - started
            stats = dict(update=update, gradients=gradients, memory=args.memory, seed=args.seed, loss=loss,
                         transitions_this_run=transitions, elapsed_seconds=round(elapsed, 2),
                         sps=round(transitions / elapsed, 2), rollout_mean_shaped_reward=float(rewards.mean()),
                         backend="parallel" if args.parallel else "serial", workers=args.workers,
                         horizon=args.horizon, ticks=args.ticks, split=split,
                         reward_profile=args.reward_profile, source_checkpoint=str(source) if source else None,
                         optimizer_restored=bool(args.resume), learning_rate=optimizer.param_groups[0]["lr"],
                         rollout_seconds=round(optimize_started - rollout_started, 4),
                         optimize_seconds=round(optimize_seconds, 4),
                         checkpoint_write_seconds=round(checkpoint_write_seconds, 4), ppo=ppo,
                         button_frequencies=dict(zip(
                             BUTTONS, buttons(actions.cpu().numpy()).mean(axis=(0, 1)).tolist(), strict=True
                         )),
                         internal_delta_bytes=storage["internal_delta_bytes"],
                         stopped_for_storage=storage_limit,
                         render_buffers=not args.no_render_buffers, episodes=episodes)
            if credit:
                credit_path = credit_logs / f"update-{update:04d}.npz"
                with credit_path.with_suffix(".tmp").open("xb") as receipt:
                    np.savez_compressed(receipt, columns=np.array(COLUMNS), values=components, boundary=done.numpy(),
                                        layouts=np.array(layout_rows), annotated=np.array(potential_rows))
                os.replace(credit_path.with_suffix(".tmp"), credit_path)
                stats.update(credit_mode=args.credit_mode, credit_ledger=str(credit_path),
                             sample_statistics=component_statistics(components),
                             gradient_diagnostics=gradient_diagnostics,
                             history_mode="exact_raw_four" if exact_history else "cached_chunk_start",
                             advantage_before_normalization=dict(mean=float(adv.mean()),
                                 population_std=float(adv.std(unbiased=False)), minimum=float(adv.min()),
                                 maximum=float(adv.max())),
                             layout_decisions=dict(exposure), skill_decisions=dict(skill_exposure),
                             reset_counts=dict(reset_exposure), annotated_decisions=int(np.sum(potential_rows)),
                             stopped_for_instability=unstable, stopped_for_memory=memory_limit,
                             physical_available_bytes=storage.get("physical_available_bytes"),
                             commit_available_bytes=storage.get("commit_available_bytes"))
                if transition_balance:
                    stats["cumulative_skill_decisions"] = dict(schedule.counts)
                if anchor is not None:
                    stats.update(anchor_beta=args.anchor_beta, anchor_categories=anchor.measure(policy))
                    if update == counter + 1:
                        stats["initial_anchor_categories"] = initial_anchors
            episodes = []
            with (logdir / f"{run_name}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            print({k:v for k,v in stats.items() if k != "episodes"}, flush=True)
            if stopping:
                if storage_limit:
                    print("Saved and stopped at the internal drive's soft storage limit.", flush=True)
                if unstable:
                    raise RuntimeError("Stopped at the predeclared PPO stability gate.")
                break


if __name__ == "__main__":
    main()
