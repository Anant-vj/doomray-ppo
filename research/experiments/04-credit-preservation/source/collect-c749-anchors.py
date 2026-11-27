"""Collect successful training-map windows from the unchanged published C749 actor."""

import argparse
from collections import Counter, deque
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ARTIFACTS = PROJECT / "submission_artifacts"
RELEASE = ARTIFACTS / "github-fresh-6c2d549-20261127-022250050"
ASSETS = PROJECT / "environments/competition/Lib/site-packages/vizdoom"
COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"
WEIGHT = "0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61"
IWAD = "a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b"
CUTOFF = datetime.fromisoformat("2026-11-27T06:46:30+05:30")
SCENES = (
    ("key-lesson", RELEASE / "tests/fixtures/e7ab54fb6834ffa3.wad", 1400, tuple(range(67101, 67107)),
     "ea6a0b17bca7beae8dc928e9c3b6ced51eceacf1f3c1421d7a98a698a966445e"),
    ("backtrack-lesson", PROJECT / "maps/curriculum/red_key_backtrack.wad", 1400, tuple(range(67101, 67107)),
     "2ec2f02accdda7dd0d23fc4eb43956a075781234bcce82ee7ca299ebc2d97397"),
    ("basic-combat", ASSETS / "scenarios/basic.wad", 1024, tuple(range(67201, 67205)),
     "fd0b7e5e2422e09288c786e7fc32181bb8ee4a3faa613a75ef88a3e39b5ce94e"),
    ("way-home", ASSETS / "scenarios/my_way_home.wad", 2100, tuple(range(67301, 67305)),
     "18e98ff4c9186808eb00696b799cfe88a79e7486429c1692aafbed8ef359c0b1"),
)
CATEGORIES = ("key", "unlock", "exit", "navigation", "combat")


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def identity():
    repo = RELEASE.resolve()
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != COMMIT or dirty or digest(repo / "submission/weights/policy.pt") != WEIGHT:
        raise RuntimeError("Frozen actor source or materialized weight differs.")
    if digest(ASSETS / "freedoom2.wad") != IWAD:
        raise RuntimeError("Official Freedoom IWAD identity changed.")
    for name, wad, _, _, expected in SCENES:
        if wad.resolve().drive.upper() != "D:" or digest(wad) != expected:
            raise RuntimeError(f"Training-only map changed for {name}")


def manifest():
    return dict(source="unchanged published C749, training-only maps, CPU ViZDoom official wrapper",
                release=COMMIT, weight_sha256=WEIGHT, official_iwad_sha256=IWAD,
                cutoff=CUTOFF.isoformat(), frozen_at=datetime.now().astimezone().isoformat(),
                categories=CATEGORIES, history="oldest to latest; zero invalid latent frames after encoding",
                success="Only exit-confirmed episodes. Key increases and once-raised locked sectors in the two lessons; non-combat exits; way-home exit plus two or more visited cells; BasicAim exit with positive verified hits and damage."
                        " Category windows are the 16 decisions ending at each event, with duplicate indexes removed per category.",
                maps=[dict(name=n, wad=str(w), map="MAP01", skill=3, tick_cap=c, seeds=s, wad_sha256=h)
                      for n, w, c, s, h in SCENES])


def atomic_npz(np, path, arrays):
    pending = path.with_suffix(".writing")
    with pending.open("wb") as file:
        np.savez_compressed(file, **arrays)
    pending.replace(path)


def action_heads(action):
    return [action[i] + 2 * action[i + 1] for i in (0, 2, 4)] + list(action[6:])


def collect(runtime, template, scene, cap, seed, stats, policy):
    np, torch, vizdoom, wrapper, validate_action, DoomAgent, pack, heads = runtime
    held_actions = []
    teacher_logits = []
    event = {name: [] for name in CATEGORIES}
    phases = deque(maxlen=4)
    logits_hook = []
    run = None
    previous = np.zeros(9, dtype=np.int8)
    exit_reached = False
    dead = False
    finished = False
    timeout = False
    stop_reason = None
    count_player_hits = count_player_damage = 0
    legal = 0
    started = time.monotonic()
    try:
        stats(policy, startup=True, actor_loaded=template is not None)
        run = wrapper(scene, "MAP01", episode_ticks=cap, render_buffers=False, render_mode="rgb_array")
        obs, info = run.reset(seed=seed)
        start = before = run.privileged_state()
        raised = set()
        cells = {(int(start["x"] // 96), int(start["y"] // 96))}
        if template is None:
            template = DoomAgent(run.observation_space, run.action_space, "cpu")
        torch.manual_seed(seed)
        template.reset()
        tap = template.policy.actor.register_forward_hook(lambda _layer, _inputs, value: logits_hook.append(value.clone()))
        gv = vizdoom.GameVariable
        hit, damage = (run._game.get_game_variable(v) for v in (gv.HITCOUNT, gv.DAMAGECOUNT))
        try:
            while True:
                idx = len(held_actions)
                if idx % 16 == 0:
                    stats(policy)
                    if datetime.now().astimezone() >= CUTOFF:
                        stop_reason = "collection_cutoff"
                        break
                assert np.array_equal(previous, template.previous)
                frame = pack(obs, previous)
                phases.append(frame.copy())
                history = np.zeros((4, 378), np.float32)
                valid = np.zeros(4, dtype=np.bool_)
                history[-len(phases):] = phases
                valid[-len(phases):] = True
                logits_hook.clear()
                action = np.asarray(validate_action(template.act(obs), run.action_space), dtype=np.int8)
                assert len(logits_hook) == 1
                logit = logits_hook[0][0]
                selected = action_heads(action)
                log_prob = sum(float(part[choice]) for part, choice in zip(
                    (torch.log_softmax(part, 0) for part in logit.split(heads)), selected, strict=True))
                obs, reward, terminal, truncated, info = run.step(action)
                assert np.array_equal(run._game.get_last_action(), action), f"Engine readback mismatch at {idx + 1}"
                legal += 1
                teacher_logits.append(logit.numpy().copy())
                held_actions.append((history, valid, previous.copy(), action, log_prob, reward, terminal, truncated))
                previous = action
                now = run.privileged_state()
                cells.add((int(now["x"] // 96), int(now["y"] // 96)))
                if int(now["keys"].sum()) > int(before["keys"].sum()):
                    event["key"].append(idx)
                locks, ceil = now["door_lock"], now["sector_ceiling"]
                if locks is not None and ceil is not None and start["sector_ceiling"] is not None:
                    new = set(np.flatnonzero((locks > 0) & (ceil > start["sector_ceiling"] + 4)).tolist()) - raised
                    if new:
                        event["unlock"].append(idx)
                        raised.update(new)
                next_hit, next_damage = (run._game.get_game_variable(v) for v in (gv.HITCOUNT, gv.DAMAGECOUNT))
                delta_hit, delta_damage = max(0, next_hit - hit), max(0, next_damage - damage)
                count_player_hits += int(delta_hit)
                count_player_damage += float(delta_damage)
                if delta_hit > 0 and delta_damage > 0:
                    event["combat"].append(idx)
                hit, damage, before = next_hit, next_damage, now
                if terminal or truncated:
                    exit_reached = bool(info["exit_reached"])
                    timeout = bool(truncated)
                    dead, finished = bool(now["dead"]), True
                    if exit_reached:
                        event["exit"].append(idx)
                        if len(cells) >= 2:
                            event["navigation"].append(idx)
                    break
        finally:
            tap.remove()
    except Exception as e:
        stop_reason = f"{type(e).__name__}: {e}"
    finally:
        if run is not None:
            run.close()
    arrays = {}
    parity = None
    if held_actions:
        keys = ("histories", "valid", "prev_actions", "accepted_actions", "teacher_log_prob", "rewards", "terminated", "truncated")
        arrays = {k: np.asarray(column, dtype=np.float32 if k == "teacher_log_prob" else None)
                  for k, column in zip(keys, zip(*held_actions), strict=True)}
        arrays["teacher_logits"] = np.asarray(teacher_logits, dtype=np.float32)
        assert not np.any(arrays["prev_actions"][0])
        assert np.array_equal(arrays["prev_actions"][1:], arrays["accepted_actions"][:-1])
        assert np.array_equal(arrays["prev_actions"], arrays["histories"][:, -1, -9:])
        with torch.inference_mode():
            x = torch.as_tensor(arrays["histories"])
            v = torch.as_tensor(arrays["valid"])
            encoded = template.policy.encode(x.reshape(-1, 378)).reshape(len(x), 4, 128) * v[:, :, None]
            logits = template.policy.actor(template.policy.update(encoded.flatten(1))).numpy()
            parity = float(np.max(np.abs(logits - arrays["teacher_logits"])))
            if not np.allclose(logits, arrays["teacher_logits"], atol=1e-5, rtol=1e-5):
                stop_reason = f"Frozen actor/history logit reconstruction mismatch {parity:g}"
    summary = dict(seed=seed, cap=cap, decisions=len(held_actions), end="exit" if exit_reached else "death" if dead else "cap" if timeout else "incomplete",
                   exited=exit_reached, finished=finished, stop_reason=stop_reason, exact_engine_action_readbacks=legal,
                   distinct_cells=len(cells) if held_actions else 0, player_hits=count_player_hits,
                   player_damage_dealt=count_player_damage, logit_reconstruction_max_abs=parity,
                   events={k: [j + 1 for j in v] for k, v in event.items()}, elapsed_seconds=time.monotonic() - started)
    return template, arrays, event, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ARTIFACTS / "final-blind-assessment/evaluator"))
    from storage_gate import GateClosed, snapshot, storage_policy

    identity()
    policy = storage_policy()
    print(json.dumps(dict(preflight=snapshot(policy, startup=True), source=COMMIT, weight_sha256=WEIGHT)), flush=True)
    if args.preflight:
        return
    if ROOT.drive.upper() != "D:" or Path.cwd().drive.upper() != "D:" or datetime.now().astimezone() >= CUTOFF:
        raise RuntimeError("Collection requires D: working directory and available time.")
    output = ROOT / "anchors"
    output.mkdir()
    (output / "streams").mkdir()
    (output / "engine").mkdir()
    write_json(output / "collection-manifest.json", manifest())
    os.chdir(output / "engine")
    sys.path[0:0] = [str(RELEASE / "env"), str(RELEASE)]
    import numpy as np
    import torch
    import vizdoom
    from doom_env import make_env
    from doom_env.remote import validate_action
    from submission.agent import DoomAgent
    from submission.policy import HEADS, pack

    if sys.version_info[:2] != (3, 13) or vizdoom.__version__ != "1.3.3":
        raise RuntimeError("Collection requires the official Python 3.13 and ViZDoom 1.3.3 runtime.")

    runtime = (np, torch, vizdoom, make_env, validate_action, DoomAgent, pack, HEADS)
    teacher = None
    selected = []
    attempts = []
    field = ("histories", "valid", "teacher_logits", "prev_actions", "accepted_actions", "teacher_log_prob")
    with (output / "attempts.jsonl").open("w", encoding="utf-8") as ledger:
        for name, wad, cap, seeds, expected in SCENES:
            for seed in seeds:
                if datetime.now().astimezone() >= CUTOFF:
                    break
                episode = f"{name}-{seed}"
                teacher, arrays, events, report = collect(runtime, teacher, wad, cap, seed, snapshot, policy)
                report.update(episode_id=episode, layout=name, map="MAP01", wad=str(wad), wad_sha256=expected)
                if arrays:
                    target = output / "streams" / (episode + ".npz")
                    atomic_npz(np, target, arrays)
                    report.update(stream=str(target), stream_sha256=digest(target))
                categories = (("key", "unlock", "exit") if name.endswith("lesson") else
                              ("navigation", "exit") if name == "way-home" else ("combat",))
                selected_counts = {}
                if report["exited"] and not report["stop_reason"]:
                    for k in categories:
                        indexes = sorted({i for j in events[k] for i in range(max(0, j - 15), j + 1)})
                        if indexes:
                            selected_counts[k] = len(indexes)
                            selected.append(dict(**{z: arrays[z][indexes] for z in field}, category=np.full(len(indexes), k),
                                                 episode_id=np.full(len(indexes), episode), decisions=np.array(indexes, np.int32) + 1))
                report["anchor_windows"] = selected_counts
                attempts.append(report)
                ledger.write(json.dumps(report, allow_nan=False) + "\n")
                ledger.flush()
                print(json.dumps({k: report[k] for k in ("episode_id", "end", "decisions", "anchor_windows", "stop_reason")}), flush=True)
                if report["stop_reason"]:
                    break
            if attempts and attempts[-1]["stop_reason"]:
                break
    counts = Counter()
    for row in attempts:
        counts.update(row["anchor_windows"])
    target = output / "anchor-data.npz"
    if selected:
        atomic_npz(np, target, {k: np.concatenate([r[k] for r in selected], axis=0) for k in selected[0]})
    try:
        postflight = snapshot(policy)
    except GateClosed as e:
        postflight = dict(allowed=False, detail=str(e))
    result = dict(attempted=len(attempts), finalized=sum(x["finished"] for x in attempts), exits=sum(x["exited"] for x in attempts),
                  decisions=sum(x["decisions"] for x in attempts), exact_engine_action_readbacks=sum(x["exact_engine_action_readbacks"] for x in attempts),
                  samples=sum(counts.values()), by_category={k: counts[k] for k in CATEGORIES}, missing_categories=[k for k in CATEGORIES if not counts[k]],
                  archive=str(target) if target.exists() else None, archive_sha256=digest(target) if target.exists() else None,
                  torch=torch.__version__, vizdoom=vizdoom.__version__, postflight=postflight,
                  finished_at=datetime.now().astimezone().isoformat())
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
