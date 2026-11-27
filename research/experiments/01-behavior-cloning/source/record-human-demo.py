"""Keyboard-only demonstration recorder for the official DoomRay environment."""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback


WORKSPACE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("DOOM_PROJECT_ROOT", WORKSPACE.parents[1])).resolve()
ARTIFACTS = PROJECT / "submission_artifacts"
RELEASE = Path(os.environ.get("DOOM_RELEASE_CHECKOUT", ARTIFACTS / "approved-release")).resolve()
COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"
POLICY_SHA = "3d759688cd9ac103624548610a9a0cad7ce551be4148b4b95dff35eb99b3ef80"
ASSETS = PROJECT / "environments/competition/Lib/site-packages/vizdoom/scenarios"
ASSESSMENT = ARTIFACTS / "final-blind-assessment"
LAYOUTS = {
    "BasicAim": (ASSETS / "basic.wad", "basic", 1024, "fd0b7e5e2422e09288c786e7fc32181bb8ee4a3faa613a75ef88a3e39b5ce94e"),
    "CenterCombat": (ASSETS / "defend_the_center.wad", "defend-center", 1352, "eb3f8d3b3f456bbef32a1d41f847189dd83cc1d4e320157ba92b75a9b0d8d5bb"),
    "CorridorCombat": (ASSETS / "deadly_corridor.wad", "deadly-corridor", 2100, "3c868f88c78df8219a2c41a2695ae04e5a19d60a32f1718282b3bbc7f20951ca"),
    "HealthRun": (ASSETS / "health_gathering.wad", "health-gathering", 1352, "c77c8ba3254ae19eaff8d00418870a7718ef8a5822e5872ab68b76205cb2bacf"),
    "KeyDoor": (RELEASE / "tests/fixtures/e7ab54fb6834ffa3.wad", "red-key-lesson", 1400, "ea6a0b17bca7beae8dc928e9c3b6ced51eceacf1f3c1421d7a98a698a966445e"),
    "Backtrack": (PROJECT / "maps/curriculum/red_key_backtrack.wad", "red-key-lesson", 1400, "2ec2f02accdda7dd0d23fc4eb43956a075781234bcce82ee7ca299ebc2d97397"),
    "TwoRooms": (RELEASE / "maps/two_rooms.wad", "two-rooms", 2100, "24f8fe53389a76c07ee09c3d25aa136d3fc95ad0e1c07357ee43082a65ac352a"),
}
MAP = "MAP01"
SIZE = (1120, 716)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def time_stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def source_check(layout):
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={RELEASE.as_posix()}", "-C", str(RELEASE)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != COMMIT or dirty or sha(RELEASE / "submission/policy.py") != POLICY_SHA:
        raise RuntimeError("Frozen wrapper or feature source differs from the published release.")
    wad, family, cap, digest = LAYOUTS[layout]
    wad = wad.resolve(strict=True)
    if WORKSPACE.drive.upper() != "D:" or wad.drive.upper() != "D:" or sha(wad) != digest:
        raise RuntimeError("Training WAD identity or D: path check failed.")
    return dict(layout=layout, layout_family=family, map=MAP, skill=3, tick_cap=cap,
                wad=str(wad), wad_sha256=digest, frozen_wrapper_commit=COMMIT, pack_source_sha256=POLICY_SHA)


def storage_tools():
    # The exhibition tool owns the existing central storage policy and original baseline check.
    helper = runpy.run_path(str(ASSESSMENT / "graphical-exam/examine.py"))
    policy = json.loads((ASSESSMENT / "manifests/storage_policy.json").read_text(encoding="utf-8"))
    return helper["storage"], policy


def load_pack(np):
    """Compile the shipped NumPy-only packing function without loading Torch for keyboard play."""
    path = RELEASE / "submission/policy.py"
    if sha(path) != POLICY_SHA:
        raise RuntimeError("Frozen feature packing source changed.")
    source = ast.parse(path.read_text(encoding="utf-8"))
    scale = next(node for node in source.body if isinstance(node, ast.Assign) and
                 any(isinstance(name, ast.Name) and name.id == "VITAL_SCALE" for name in node.targets))
    pack = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "pack")
    scope = {"np": np}
    exec(compile(ast.Module(body=[scale, pack], type_ignores=[]), str(path), "exec"), scope)
    return scope["pack"]


def keyboard_action(pg, keys):
    def axis(first, second):
        a, b = bool(keys[first]), bool(keys[second])
        return int(a and not b), int(b and not a)

    return (*axis(pg.K_w, pg.K_s), *axis(pg.K_a, pg.K_d), *axis(pg.K_LEFT, pg.K_RIGHT),
            int(keys[pg.K_SPACE]), int(keys[pg.K_e]), int(keys[pg.K_q]))


def label_heads(action):
    return (action[0] + 2 * action[1], action[2] + 2 * action[3], action[4] + 2 * action[5], *action[6:])


def input_row(np, obs, previous, accepted, pack):
    rays, entities = obs["rays"], obs["visible_entities"]
    return dict(packed=pack(obs, previous), previous_actions=np.array(previous, dtype=np.int8),
                actions=np.array(accepted, dtype=np.int8), action_heads=np.array(label_heads(accepted), dtype=np.int8),
                ray_distance=np.array(rays["distance"], copy=True), ray_hit_type=np.array(rays["hit_type"], copy=True),
                entity_features=np.array(entities["features"], copy=True), entity_type=np.array(entities["entity_type"], copy=True),
                entity_mask=np.array(entities["mask"], copy=True), vitals=np.array(obs["vitals"], copy=True))


def save_episode(np, directory, context, rows, outcome, fault, storage_before, storage_after):
    data = dict(context, outcome=outcome, fault=fault, decisions=len(rows), saved_at=time_stamp(),
                storage_before=storage_before, storage_after=storage_after)
    complete = bool(rows and (rows[-1]["terminated"] or rows[-1]["truncated"]))
    eligible = False
    if rows:
        samples = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        continuous = (not np.any(samples["previous_actions"][0]) and
                      np.array_equal(samples["previous_actions"][1:], samples["actions"][:-1]))
        verified = int(samples["engine_action_matches"].sum())
        eligible = complete and continuous and verified == len(rows) and not fault and bool(np.any(samples["actions"]))
        target = directory / "steps.npz"
        pending = directory / "steps.writing"
        with pending.open("wb") as stream:
            np.savez_compressed(stream, **samples)
        pending.replace(target)
        data.update(payload="steps.npz", payload_sha256=sha(target), exact_engine_matches=verified,
                    previous_action_chain_valid=bool(continuous), total_reward=float(samples["rewards"].sum()))
    data.update(complete_episode=complete, eligible_for_bc=bool(eligible))
    write_json(directory / "episode.json", data)
    print(json.dumps(data), flush=True)
    return data


class Screen:
    def __init__(self, pg):
        self.pg = pg
        self.screen = pg.display.set_mode(SIZE)
        pg.display.set_caption("DoomRay PPO | Team demonstration recorder")
        self.title = pg.font.Font(None, 32)
        self.font = pg.font.Font(None, 23)

    def draw(self, scene, operator, status, obs=None, decision=0, tick=0, pressed=()):
        pg = self.pg
        self.screen.fill((16, 24, 36))

        def line(text, x, y, title=False):
            font = self.title if title else self.font
            self.screen.blit(font.render(text, True, (149, 232, 194) if title else (229, 235, 245)), (x, y))

        line("TEAM DEMONSTRATION RECORDER", 15, 8, title=True)
        line(f"{scene['layout']}  /  {scene['map']}  /  Seed {scene['seed']}  /  {status}", 16, 42)
        if obs is not None:
            raw = pg.surfarray.make_surface(obs["rgb"].transpose((1, 0, 2)))
            self.screen.blit(pg.transform.scale(raw, (800, 600)), (0, 70))
        items = [f"Teammate  {operator}", f"Decision  {decision}    Tick  {tick}", "", "W / S   Forward / back",
                 "A / D   Strafe left / right", "Left / Right   Turn", "Space   Fire", "E   Use", "Q   Next weapon",
                 "P   Pause or resume", "Escape   Close and save", "", "Applied buttons", *pressed]
        for index, value in enumerate(items):
            line(value, 812, 82 + index * 23)
        line("Human keyboard input only. The official game pauses when the viewer loses focus.", 15, 685)
        pg.display.flip()


def record_episode(pg, np, screen, context, directory, storage, policy):
    from doom_env import make_env
    from doom_env.remote import validate_action
    from doom_env.spaces import BUTTONS, FRAME_SKIP

    expected = ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
                "ATTACK", "USE", "SELECT_NEXT_WEAPON")
    if tuple(BUTTONS) != expected:
        raise RuntimeError("Official button order changed.")
    storage_before = storage()
    if not storage_before["launch_allowed"]:
        raise RuntimeError("Available storage or memory changed before this episode.")
    directory.mkdir()
    write_json(directory / "start.json", dict(context, started_at=time_stamp(), buttons=list(BUTTONS),
                                             frame_skip=FRAME_SKIP, storage_before=storage_before))
    pack = load_pack(np)
    previous = np.zeros(len(BUTTONS), dtype=np.int8)
    rows = []
    last_tick = tick = 0
    env = obs = None
    outcome = fault = None
    paused = False
    closing = False
    operator = context["teammate"]
    previous_cwd = Path.cwd()
    os.chdir(directory)
    try:
        env = make_env(context["wad"], MAP, episode_ticks=context["tick_cap"], render_buffers=True, render_mode="rgb_array")
        obs, info = env.reset(seed=context["seed"])
        tick = last_tick = info["tic"]
        screen.draw(context, operator, "Press a gameplay key to begin", obs)
        next_step = time.perf_counter()
        while outcome is None:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    outcome, closing = "operator_closed", True
                elif event.type == pg.KEYDOWN and event.key == pg.K_p:
                    paused = not paused
                    screen.draw(context, operator, "Paused" if paused else "Recording", obs, len(rows), last_tick)
                    next_step = time.perf_counter()
            if closing:
                break
            if paused or not pg.key.get_focused() or time.perf_counter() < next_step:
                time.sleep(.008)
                continue
            requested = keyboard_action(pg, pg.key.get_pressed())
            if not rows and not any(requested):
                time.sleep(.008)
                continue
            accepted = tuple(validate_action(requested, env.action_space))
            row = input_row(np, obs, previous, accepted, pack)
            obs, reward, terminal, truncated, info = env.step(accepted)
            engine_action = np.asarray(env._game.get_last_action(), dtype=np.float32)
            exact = bool(engine_action.shape == (len(BUTTONS),) and np.array_equal(engine_action, row["actions"]))
            row.update(rewards=np.float32(reward), terminated=bool(terminal), truncated=bool(truncated),
                       pre_tic=np.int32(tick), post_tic=np.int32(info["tic"]),
                       engine_actions=engine_action if engine_action.shape == (len(BUTTONS),) else np.full(len(BUTTONS), -1, np.float32),
                       engine_action_matches=exact, exit_reached=bool(info["exit_reached"]))
            rows.append(row)
            previous = row["actions"]
            tick = info["tic"]
            last_tick = max(last_tick, tick)
            pressed = [name.replace("_", " ").title() for name, bit in zip(BUTTONS, accepted, strict=True) if bit]
            screen.draw(context, operator, "Recording", obs, len(rows), last_tick, pressed)
            if not exact:
                fault = f"Engine readback mismatch at decision {len(rows)}."
                outcome = "action_mismatch"
            elif terminal or truncated:
                outcome = "exit" if info["exit_reached"] else "tick_cap" if truncated else "death"
            elif len(rows) % 16 == 0:
                resources = storage()
                if (resources["c_decline"] >= 6_900_000_000 or resources["c_free"] < policy["minimum_absolute_c_free_bytes"] or
                        resources["d_free"] < policy["minimum_d_free_bytes"] or
                        resources["available_memory"] < policy["minimum_available_memory_bytes_during_episode"]):
                    outcome, closing = "storage_or_memory_stop", True
            next_step = max(time.perf_counter(), next_step + FRAME_SKIP / 35)
    except Exception as exc:
        outcome = "error"
        fault = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)
    data = save_episode(np, directory, context, rows, outcome, fault, storage_before, storage())
    if not closing:
        message = f"{outcome.replace('_', ' ').title()}  /  Saved {'eligible episode' if data['eligible_for_bc'] else 'for review'}  /  Enter starts next seed"
        screen.draw(context, operator, message, obs, len(rows), last_tick)
    return closing


def run_recorder(scene, seed, teammate, permission, storage, policy):
    import numpy as np
    import pygame as pg

    sys.path.insert(0, str(RELEASE / "env"))
    root = WORKSPACE / "demonstrations/human"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "recorder.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another native human recorder is active.") from exc
        try:
            if os.environ.get("SDL_VIDEODRIVER", "").lower() in {"dummy", "offscreen"}:
                os.environ.pop("SDL_VIDEODRIVER")
            pg.display.init()
            pg.font.init()
            screen = Screen(pg)
            if pg.display.get_driver() != "windows":
                raise RuntimeError("Recording requires the native visible Windows event driver.")
            session = root / f"{scene['layout']}-{time.time_ns()}"
            context = dict(scene, schema_version=1, source="registered_team_member_keyboard", teammate=teammate,
                           permission_reference=permission, attestations="Registered teammate confirms own game input and organizer permission reference",
                           torch_loaded=False, started_session_at=time_stamp())
            screen.draw(dict(context, seed=seed), teammate, "Registered teammate presses Enter to start")
            print(json.dumps(dict(event="waiting_for_teammate", pid=os.getpid(),
                                  hwnd=pg.display.get_wm_info().get("window"),
                                  driver=pg.display.get_driver(), layout=scene["layout"],
                                  seed=seed, engine_started=False)), flush=True)
            index = 0
            while True:
                for event in pg.event.get():
                    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                        return
                    if event.type == pg.KEYDOWN and event.key == pg.K_RETURN and pg.key.get_focused():
                        if index == 0:
                            session.mkdir()
                            write_json(session / "session.json", context)
                        current = dict(context, seed=seed + index, episode_index=index)
                        if record_episode(pg, np, screen, current, session / f"episode-{index:04d}", storage, policy):
                            return
                        index += 1
                time.sleep(.02)
        finally:
            pg.quit()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=LAYOUTS, default="KeyDoor")
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--team-attested", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed <= 2_147_483_000:
        parser.error("Choose a positive seed in range.")
    scene = source_check(args.layout)
    storage, policy = storage_tools()
    gate = storage()
    if args.preflight:
        print(json.dumps(dict(scene, seed=args.seed, **gate)))
        return
    teammate = os.environ.get("DOOM_HUMAN_TEAMMATE", "").strip()
    permission = os.environ.get("DOOM_HUMAN_PERMISSION", "").strip()
    if not (args.team_attested and teammate and permission):
        parser.error("Recording requires the registered teammate and verified organizer permission reference.")
    if not gate["launch_allowed"]:
        raise RuntimeError(f"The storage or memory launch gate refused recording: {gate}")
    run_recorder(scene, args.seed, teammate, permission, storage, policy)


if __name__ == "__main__":
    main()
