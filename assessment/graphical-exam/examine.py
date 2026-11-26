"""Native DoomRay examination, separate from matched assessment runs."""

import argparse
import ctypes
import hashlib
import io
import json
import math
import msvcrt
import os
from pathlib import Path
import re
import runpy
import shutil
import struct
import subprocess
import sys
import time


ASSESSMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = ASSESSMENT.parent
MANIFESTS = ASSESSMENT / "manifests"
RECORDINGS = ASSESSMENT / "recordings/graphical"
MAP_SHA = "5e3f7d1e96f578d4bdb37d44ce3ce78c557e990929940a0c8918fe0c4f34ab7c"
MODELS_SHA = "98cba1190f44c7ac7ed8bad0a7a0bc2ef6bd4e079a92843c0949857d1204c3a2"
SIZE = (1200, 720)
SPEEDS = (.25, .5, 1, 2, 4)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, expected_sha=None):
    if expected_sha and sha(path) != expected_sha:
        raise ValueError(f"Identity check failed: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Memory(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                *[(n, ctypes.c_uint64) for n in ("physical", "available", "page", "page_available",
                                                  "virtual", "virtual_available", "extended")]]


def storage():
    policy = read_json(MANIFESTS / "storage_policy.json")
    if (policy["absolute_c_decline_limit_bytes"], policy["start_reserve_bytes"],
            policy["c_decline_start_gate_bytes"]) != (7_000_000_000, 500_000_000, 6_500_000_000):
        raise ValueError("Expected the authorized 7 GB storage policy with its 500 MB reserve.")
    baseline = read_json(Path(policy["baseline"]), policy["baseline_sha256"])["internal_free_bytes"]
    c_free = shutil.disk_usage("C:/").free
    d_free = shutil.disk_usage(ASSESSMENT).free
    decline = max(0, baseline - c_free)
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("Could not measure available physical memory.")
    allowed = (ASSESSMENT.drive.upper() == "D:" and decline <= policy["c_decline_start_gate_bytes"] and
               c_free >= policy["minimum_absolute_c_free_bytes"] and d_free >= policy["minimum_d_free_bytes"] and
               memory.available >= policy["minimum_available_memory_bytes"])
    return dict(launch_allowed=allowed, c_decline=decline, c_free=c_free, d_free=d_free,
                available_memory=memory.available, commit_available=memory.page_available)


def live_stop():
    """Allow an episode to use the start reserve while retaining room for an orderly stop."""
    data = storage()
    return (data["c_decline"] >= 6_900_000_000 or data["c_free"] < 2_000_000_000 or
            data["d_free"] < 500_000_000 or data["commit_available"] < 128_000_000)


def on_d(path):
    result = Path(path).resolve(strict=True)
    if result.drive.upper() != "D:":
        raise ValueError(f"File must be on D: {result}")
    return result


def model_identity(model):
    repo = on_d(model["checkout"])
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != model["source_commit"] or changes:
        raise ValueError(f"{model['id']} checkout/commit mismatch.")
    for weight in (on_d(model["weight_path"]), on_d(repo / "submission/weights/policy.pt")):
        if sha(weight) != model["sha256"]:
            raise ValueError(f"{model['id']} weight hash mismatch.")
    return repo


def choose_maps(maps, *, tier=None, map_id=None, all_tiers=False):
    if all_tiers and (tier or map_id):
        raise ValueError("Choose AllTiers on its own, or specify a tier or map.")
    if not (all_tiers or tier or map_id):
        raise ValueError("Choose a tier, map, or AllTiers.")
    selected = [m for m in maps if (not tier or m["tier"] == tier) and (not map_id or m["id"] == map_id)]
    if not selected:
        raise ValueError("That map is absent from the selected tier.")
    known = {}
    for scene in selected:
        path = on_d(scene["wad"])
        if path not in known:
            known[path] = sha(path)
        if known[path] != scene["wad_sha256"] or scene["skill"] != 3:
            raise ValueError(f"{scene['id']} WAD or skill does not match the frozen map manifest.")
    return selected


def records():
    demo = ARTIFACTS / "demo-runs/20261127-024354396"
    paths = [("TwoRooms", demo), *[(p.name, p) for p in sorted(
        (ARTIFACTS / "c749-demonstration-runs").glob("*"))],
        *[(p.name, p) for p in sorted(RECORDINGS.glob("*"))]]
    return {name: path for name, path in paths if (path / "audit-gameplay.avi").is_file()
            and (path / "audit-report.json").is_file()}


def avi(path):
    """Read the lightweight Motion JPEG format written by the existing audit viewer."""
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"AVI ":
        raise ValueError("Replay file is not a recorded AVI.")
    stop = min(len(raw), 8 + struct.unpack_from("<I", raw, 4)[0])

    def chunks(begin, end):
        while begin + 8 <= end:
            tag, size = struct.unpack_from("<4sI", raw, begin)
            start, finish = begin + 8, begin + 8 + size
            if finish > end:
                raise ValueError("Truncated replay chunk.")
            if tag == b"LIST":
                yield from chunks(start + 4, finish)
            else:
                yield tag, raw[start:finish]
            begin = finish + (size & 1)

    microseconds, frames, codec = 0, [], None
    for tag, value in chunks(12, stop):
        if tag == b"avih":
            microseconds = struct.unpack_from("<I", value)[0]
        elif tag == b"strh" and value[:4] == b"vids":
            codec = value[4:8]
        elif tag == b"00dc":
            frames.append(value)
    if not microseconds or not frames or codec != b"MJPG":
        raise ValueError("This episode has no supported Motion JPEG video frames.")
    return microseconds / 1e6, frames


class Display:
    def __init__(self, pygame):
        self.pg = pygame
        self.window = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("DoomRay PPO | Graphical examination")
        self.heading = pygame.font.Font(None, 35)
        self.body = pygame.font.Font(None, 27)
        self.small = pygame.font.Font(None, 21)
        self.speed = 2
        self.paused = False

    def audit(self, state, **details):
        handle = self.pg.display.get_wm_info().get("window")
        print(json.dumps(dict(ui=state, pid=os.getpid(), hwnd=handle,
                              native_driver=self.pg.display.get_driver(), client_size=list(self.window.get_size()),
                              **details)), flush=True)

    def text(self, text, x, y, font=None, color=(221, 232, 242)):
        self.window.blit((font or self.body).render(text, True, color), (x, y))

    def game_frame(self, image, state, identity, rows, footer):
        self.window.fill((14, 22, 34))
        self.text(state, 18, 10, self.heading, (121, 221, 189))
        self.text(identity, 18, 43, self.small)
        if image is not None:
            self.window.blit(self.pg.transform.scale(image, (800, 600)), (0, 67))
        for i, line in enumerate(rows):
            self.text(line, 818, 81 + 24 * i, self.small)
        self.text(footer, 18, 688, self.small)
        self.pg.display.flip()

    def events(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                yield "quit"
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_ESCAPE:
                    yield "quit"
                elif event.key == self.pg.K_SPACE:
                    self.paused = not self.paused
                    yield "change"
                elif event.key in (self.pg.K_MINUS, self.pg.K_KP_MINUS):
                    self.speed = max(0, self.speed - 1)
                    yield "change"
                elif event.key in (self.pg.K_EQUALS, self.pg.K_PLUS, self.pg.K_KP_PLUS):
                    self.speed = min(len(SPEEDS) - 1, self.speed + 1)
                    yield "change"
                elif event.key in (self.pg.K_RETURN, self.pg.K_n):
                    yield "next"

    def footer(self):
        return f"Space pause/resume    - and + speed ({SPEEDS[self.speed]:g}×)    Escape close"


def play_record(display, name, directory):
    body = read_json(directory / "audit-report.json")
    recording = directory / "audit-gameplay.avi"
    delay, frames = avi(recording)
    identity = f"{body.get('model', 'C749')}  /  {body['wad']}:{body['map']}  /  Seed {body['seed']}  /  {name}"
    index, next_frame = 0, time.perf_counter()
    frame = None
    while True:
        events = set(display.events())
        if "quit" in events:
            display.audit("RECORDED REPLAY CLOSED", replay=name, displayed_frames=index)
            return
        if "change" in events:
            next_frame = time.perf_counter()
            display.audit("REPLAY CONTROL", replay=name, displayed_frames=index,
                          speed=SPEEDS[display.speed], paused=display.paused)
        count = len(frames)
        advance = frame is None or (index < count and not display.paused and time.perf_counter() >= next_frame)
        if advance:
            frame = display.pg.image.load(io.BytesIO(frames[index]), "frame.jpg").convert()
            index += 1
            next_frame = time.perf_counter() + delay / SPEEDS[display.speed]
        if advance or "change" in events:
            display.game_frame(frame, "RECORDED REPLAY", identity,
                               [f"Recorded frame {index} / {count}", f"Outcome  {body['outcome']}",
                                f"Decisions  {body['decisions']}",
                                "Playback complete" if index == count else "Paused" if display.paused else "Playing"],
                               display.footer())
        if advance and index in (1, count):
            display.audit("RECORDED REPLAY", replay=name, displayed_frames=index, frames=count,
                          recording_sha256=sha(recording), running_simulator=False)
        time.sleep(.01)


def menu(display, models, maps):
    pg = display.pg
    choices = list(models)
    model_index = tier = map_index = active = replay_index = 0
    seed = ""
    replays = list(records())
    first_present = True

    while True:
        eligible = [m for m in maps if m["tier"] == tier + 1]
        rows = [f"Model                      {choices[model_index]}",
                f"Difficulty tier           {tier + 1}",
                f"Map                         {eligible[map_index - 1]['id'] if map_index else 'All maps in this tier'}",
                f"Engine and policy seed     {seed or '9101'}",
                f"Recorded episode          {replays[replay_index] if replays else 'Unavailable'}"]
        display.window.fill((14, 22, 34))
        display.text("DoomRay PPO", 42, 38, display.heading, (121, 221, 189))
        display.text("Graphical examination", 42, 79)
        for i, row in enumerate(rows):
            highlight = i == active
            if highlight:
                pg.draw.rect(display.window, (31, 54, 74), (34, 144 + 56 * i, 930, 48), border_radius=5)
            display.text(row, 52, 156 + 56 * i, color=(121, 221, 189) if highlight else (221, 232, 242))
        display.text("Up and Down select a field. Left and Right change choices. Type to replace the seed.", 44, 479)
        display.text("Enter opens the recording when selected. Otherwise it starts live evaluation.", 44, 512)
        display.text("Escape closes this window.", 44, 545)
        pg.display.flip()
        if first_present:
            display.audit("MENU", models=choices, tiers=4, replay_count=len(replays))
            first_present = False
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                return None
            if event.type == pg.MOUSEBUTTONDOWN and 144 <= event.pos[1] < 424:
                active = (event.pos[1] - 144) // 56
                continue
            if event.type != pg.KEYDOWN:
                continue
            if event.key in (pg.K_UP, pg.K_DOWN):
                active = (active + (1 if event.key == pg.K_DOWN else -1)) % len(rows)
            elif event.key in (pg.K_LEFT, pg.K_RIGHT):
                move = 1 if event.key == pg.K_RIGHT else -1
                if active == 0:
                    model_index = (model_index + move) % len(choices)
                elif active == 1:
                    tier = (tier + move) % 4
                    map_index = 0
                elif active == 2:
                    map_index = (map_index + move) % (len(eligible) + 1)
                elif active == 4 and replays:
                    replay_index = (replay_index + move) % len(replays)
            elif active == 3 and event.key == pg.K_BACKSPACE:
                seed = seed[:-1]
            elif active == 3 and event.unicode in "0123456789" and event.unicode and len(seed) < 9:
                seed += event.unicode
            elif event.key == pg.K_RETURN:
                if active == 4 and replays:
                    return dict(replay=replays[replay_index])
                if active != 4:
                    return dict(model=choices[model_index], tier=tier + 1,
                                map_id=eligible[map_index - 1]["id"] if map_index else None,
                                seed=int(seed or 9101) or 9101)
        time.sleep(.04)


def handle_examination(display, model, scenes, seed):
    first = scenes[0]
    display.game_frame(None, "LIVE EVALUATION", f"{model['id']}  /  Tier {first['tier']}  /  {first['map']}  /  Seed {seed}",
                       ["Verifying and loading frozen model"], display.footer())
    repo = model_identity(model)
    official = model_identity(read_json(MANIFESTS / "evaluator_models.json", MODELS_SHA)["models"][0])
    if not storage()["launch_allowed"]:
        raise RuntimeError("Storage or memory headroom changed before loading the model.")
    sys.path[0:0] = [str(official / "env"), str(repo)]
    from submission.agent import DoomAgent

    source = Path(sys.modules[DoomAgent.__module__].__file__).resolve()
    if not source.is_relative_to(repo / "submission"):
        raise RuntimeError(f"Loaded unexpected DoomAgent source: {source}")
    toolkit = runpy.run_path(str(ARTIFACTS / "live-audit-viewer.py"))
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    lock_file = RECORDINGS / "engine.lock"
    with lock_file.open("a+b") as engine_lock:
        engine_lock.seek(0)
        try:
            msvcrt.locking(engine_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another graphical game already owns the simulator lock.") from exc
        try:
            for index, scene in enumerate(scenes):
                if not storage()["launch_allowed"]:
                    raise RuntimeError("Storage or memory headroom changed before this episode.")
                episode_id = f"{scene['id']}-{model['id']}-{seed}-{time.time_ns()}"
                directory = RECORDINGS / episode_id
                directory.mkdir()
                previous_directory = Path.cwd()
                os.chdir(directory)
                try:
                    keep_running = run_scene(display, model, scene, seed, episode_id, DoomAgent,
                                             toolkit, more=index + 1 < len(scenes))
                finally:
                    os.chdir(previous_directory)
                if not keep_running:
                    break
        finally:
            engine_lock.seek(0)
            msvcrt.locking(engine_lock.fileno(), msvcrt.LK_UNLCK, 1)


def run_scene(display, model, scene, seed, episode_id, agent_type, toolkit, *, more):
    import numpy as np
    import torch
    import vizdoom
    from doom_env import make_env
    from doom_env.remote import validate_action
    from doom_env.spaces import BUTTONS, FRAME_SKIP, VITALS_INDEX, VITALS_FIELDS

    pg = display.pg
    wad = Path(scene["wad"])
    identity = f"{model['id']}  /  Tier {scene['tier']}  /  {scene['map']}  /  Seed {seed}"
    pg.display.set_caption(f"DoomRay PPO | {identity}")
    display.paused = False
    display.game_frame(None, "LIVE EVALUATION", identity, ["Loading official environment"], display.footer())
    env = obs = None
    latency = tick = decision = invalid = kills = hits = dealt = damage = 0
    start_tick = last_tick = 0
    final_health = None
    frames, measured = [], []
    cells, doors, unlocks, acquired = set(), set(), set(), set()
    actions = []
    outcome, error, presented, exited = "", None, 0, False
    door_data_available = False
    before = storage()
    vzn = vizdoom.GameVariable
    v_hits, v_dealt = vzn.HITCOUNT, vzn.DAMAGECOUNT

    def draw(label):
        image = None
        keys = []
        stat = lambda field: "—"
        if obs is not None:
            image = pg.surfarray.make_surface(obs["rgb"].transpose((1, 0, 2)))
            stat = lambda field: int(obs["vitals"][VITALS_INDEX[field]])
            keys = [k[4:].replace("_", " ") for k in VITALS_FIELDS if k.startswith("key_") and stat(k)]
        progress = "Reached" if exited else "Not reached" if outcome else "Pending"
        action_names = [name.replace("_", " ").title() for name, active in zip(BUTTONS, actions) if active]
        rows = [f"Decision {decision}    Game ticks {max(0, last_tick - start_tick)} / {scene['tick_cap']}",
                f"Health {stat('health')}   Armor {stat('armor')}" + ("   [last legal]" if outcome else ""),
                f"Current weapon ammo {stat('selected_weapon_ammo')}",
                f"Bullets {stat('ammo_bullets')}   Shells {stat('ammo_shells')}",
                f"Rockets {stat('ammo_rockets')}   Cells {stat('ammo_cells')}",
                f"Held keys  {', '.join(keys) or 'None'}",
                f"Inference  {latency:.2f} ms",
                "Neural buttons applied", ", ".join(action_names[:3]) or "None"]
        if len(action_names) > 3:
            rows.append(", ".join(action_names[3:]))
        rows += ["", "Evaluator-only diagnostics", f"Player hits {hits}   Dealt {dealt}   Received {damage}",
                 f"Engine kills {kills}   Cells visited {len(cells)}",
                 f"Doors {len(doors)}   Unlocks {len(unlocks)}" if door_data_available else "Door diagnostics unavailable",
                 f"Exit  {progress}", f"Outcome  {outcome or 'In progress'}"]
        footer = display.footer() if not outcome else "Engine closed. " + ("Enter next map. " if more else "") + "Escape closes viewer."
        display.game_frame(image, label, identity, rows, footer)

    try:
        env = make_env(wad, scene["map"], episode_ticks=scene["tick_cap"], render_buffers=True, render_mode="rgb_array")
        obs, info = env.reset(seed=seed)
        start_tick = last_tick = info["tic"]
        initial = env.privileged_state()
        initial_ceilings = initial["sector_ceiling"]
        door_data_available = initial_ceilings is not None and initial["door_lock"] is not None
        cells.add((math.floor(initial["x"] / 96), math.floor(initial["y"] / 96)))
        hit_baseline = env._game.get_game_variable(v_hits)
        dealt_baseline = env._game.get_game_variable(v_dealt)
        agent = agent_type(env.observation_space, env.action_space, "cpu")
        torch.manual_seed(seed)
        agent.reset()
        next_step = time.perf_counter()
        while not outcome:
            events = set(display.events())
            if "quit" in events:
                outcome = "Viewer closed"
                break
            if "change" in events:
                draw("LIVE EVALUATION — PAUSED" if display.paused else "LIVE EVALUATION")
                next_step = time.perf_counter()
                display.audit("LIVE CONTROL", episode=episode_id, decision=decision,
                              speed=SPEEDS[display.speed], paused=display.paused)
            if display.paused or time.perf_counter() < next_step:
                time.sleep(.008)
                continue
            t0 = time.perf_counter()
            selected = agent.act(obs)
            latency = (time.perf_counter() - t0) * 1000
            measured.append(latency)
            try:
                action = validate_action(selected, env.action_space)
            except Exception:
                invalid += 1
                raise
            obs, _, ended, capped, info = env.step(action)
            decision += 1
            tick = info["tic"]
            last_tick = max(last_tick, tick)
            actions = list(action)
            evaluator = env.privileged_state()
            kills, damage = evaluator["kills"], evaluator["damage_taken"]
            final_health = evaluator["health"]
            hits = max(0, int(env._game.get_game_variable(v_hits) - hit_baseline))
            dealt = max(0, int(env._game.get_game_variable(v_dealt) - dealt_baseline))
            cells.add((math.floor(evaluator["x"] / 96), math.floor(evaluator["y"] / 96)))
            acquired.update(k[4:] for k in VITALS_FIELDS if k.startswith("key_") and obs["vitals"][VITALS_INDEX[k]])
            locks, ceiling = evaluator["door_lock"], evaluator["sector_ceiling"]
            if door_data_available and locks is not None and ceiling is not None:
                doors.update(np.flatnonzero((locks >= 0) & (ceiling > initial_ceilings + 4)).tolist())
                unlocks.update(i for i in doors if locks[i] > 0)
            else:
                door_data_available = False
            exited = bool(info["exit_reached"])
            if ended or capped:
                outcome = "Exit reached" if exited else "Tick cap reached" if capped else "Agent died"
            elif decision % 16 == 0 and live_stop():
                outcome = "Stopped at storage guard"
            draw("LIVE EVALUATION")
            presented += 1
            if decision % 2 or outcome:
                frames.append(obs["rgb"].copy())
            if decision == 1:
                pg.image.save(display.window, "audit-begin.png")
                display.audit("LIVE EVALUATION", episode=episode_id, decision=decision)
            next_step += FRAME_SKIP / (35 * SPEEDS[display.speed])
            next_step = max(time.perf_counter(), next_step)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outcome = "Episode error"
    finally:
        if env is not None:
            env.close()
    if outcome != "Viewer closed":
        draw("COMPLETED EPISODE")
        display.audit("COMPLETED EPISODE", episode=episode_id, decision=decision, outcome=outcome,
                      running_simulator=False)
        if obs is not None:
            pg.image.save(display.window, "audit-end.png")
    report = dict(episode_id=episode_id, source="graphical exploratory run", model=model["id"],
                  model_commit=model["source_commit"], weight_sha256=model["sha256"],
                  map=scene["map"], wad=wad.name, wad_sha256=scene["wad_sha256"], tier=scene["tier"], seed=seed,
                  tick_cap=scene["tick_cap"], tick=tick, last_elapsed_ticks=max(0, last_tick - start_tick),
                  outcome=outcome, error=error, decisions=decision, invalid_actions=invalid,
                  exit_reached=exited, keys_acquired=sorted(acquired),
                  doors=len(doors) if door_data_available else None, unlocks=len(unlocks) if door_data_available else None,
                  player_hits=hits, player_damage_dealt=dealt, damage_received=damage, engine_kills=kills,
                  final_health=final_health, visited_96_unit_cells=len(cells), presented_frames=presented,
                  p99_inference_ms=float(np.percentile(measured, 99)) if measured else None,
                  driver=pg.display.get_driver(), storage_before=before, storage_after=storage())
    toolkit["save_json"]("audit-report.json", report)
    if frames:
        try:
            toolkit["save_json"]("audit-clip.json", toolkit["write_clip"](pg, frames, FRAME_SKIP))
        except (pg.error, OSError, ValueError, struct.error) as exc:
            toolkit["save_json"]("audit-clip.json", {"encoding_error": str(exc)})
    print(json.dumps(report), flush=True)
    if outcome == "Viewer closed":
        return False
    while True:
        events = set(display.events())
        if "quit" in events:
            return False
        if more and "next" in events:
            return True
        time.sleep(.02)


def manifest_data():
    maps = read_json(MANIFESTS / "map_manifest.json", MAP_SHA)["maps"]
    models = {m["id"]: m for m in read_json(MANIFESTS / "evaluator_models.json", MODELS_SHA)["models"]}
    return models, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--menu", action="store_true")
    mode.add_argument("--model")
    mode.add_argument("--replay")
    parser.add_argument("--tier", type=int, choices=range(1, 5))
    parser.add_argument("--map", dest="map_id")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--all-tiers", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed <= 2_147_483_647:
        parser.error("Seed must be a positive 32-bit integer.")
    if not args.model and (args.tier or args.map_id or args.all_tiers):
        parser.error("Tier and map selection require a model.")
    models, maps = manifest_data()
    available_replays = records()
    if args.model:
        if args.model not in models:
            parser.error("Use C749, Stack25-u174, or Stack26-u621.")
        model_identity(models[args.model])
        scenes = choose_maps(maps, tier=args.tier, map_id=args.map_id, all_tiers=args.all_tiers)
    elif args.replay:
        if not re.fullmatch(r"[A-Za-z0-9-]+", args.replay) or args.replay not in available_replays:
            parser.error(f"Available replay IDs: {', '.join(available_replays)}")
        avi(available_replays[args.replay] / "audit-gameplay.avi")
    else:
        for model in models.values():
            model_identity(model)
        choose_maps(maps, all_tiers=True)
    gate = storage()
    if args.preflight:
        print(json.dumps(dict(**gate, mode="menu" if args.menu else "replay" if args.replay else "live",
                              models=list(models), maps=[s["id"] for s in (scenes if args.model else maps)],
                              replays=list(available_replays), seed=args.seed)))
        return
    if not gate["launch_allowed"]:
        raise RuntimeError(f"Storage or memory gate blocked the graphical start: {gate}")
    if os.environ.get("SDL_VIDEODRIVER", "").lower() in {"dummy", "offscreen"}:
        os.environ.pop("SDL_VIDEODRIVER")
    import pygame
    pygame.display.init()
    pygame.font.init()
    try:
        display = Display(pygame)
        if args.menu:
            picked = menu(display, models, maps)
            if picked and "replay" in picked:
                play_record(display, picked["replay"], available_replays[picked["replay"]])
            elif picked:
                scenes = choose_maps(maps, tier=picked["tier"], map_id=picked["map_id"])
                handle_examination(display, models[picked["model"]], scenes, picked["seed"])
        elif args.replay:
            play_record(display, args.replay, available_replays[args.replay])
        else:
            handle_examination(display, models[args.model], scenes, args.seed)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
