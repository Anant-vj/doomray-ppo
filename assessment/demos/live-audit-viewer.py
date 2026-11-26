"""Instrumented C749 preview, outside the submitted project."""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import zlib

SEED, CAP, TICKS_PER_SECOND = 1001, 2100, 35
SIZE, HEADER = (1120, 704), 54
MARK = (1095, 18)
COLORS = ((239, 111, 86), (118, 221, 163), (105, 177, 248), (255, 205, 101))
TITLE = "DoomRay PPO | C749 Live Audit"
COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_scene(manifest, mode, repo):
    expected = "1a3e02ccf1a7d1b454d26d7b16920b2837eed6b1034977b54b3579676fa4fd11"
    if sha256(manifest) != expected:
        raise ValueError("Predeclared demonstration manifest SHA mismatch.")
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if repo != Path(plan["release"]["checkout"]).resolve():
        raise ValueError("Replay requires the verified frozen checkout.")
    scene = plan["modes"][mode].copy()
    wad = Path(scene["wad"]).resolve()
    if wad.drive.upper() != "D:" or scene["map"] in {"MAP11", "MAP12"} or sha256(wad) != scene["wad_sha256"]:
        raise ValueError("Demonstration WAD identity mismatch.")
    scene.update(mode=mode, wad=wad, manifest=str(manifest))
    return scene


def external_storage_snapshot(*, preflight=False):
    root = Path(os.environ["DOOM_WORK"]).resolve()
    baseline_file = root / "logs/storage-baseline.json"
    if sha256(baseline_file) != "a158265c2cae0ad6d50818bae6e01387d6215f8a20a494115128bdc3a7fa9ccb":
        raise RuntimeError("Original storage baseline SHA mismatch.")
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    available = shutil.disk_usage("C:/").free
    delta = max(0, baseline["internal_free_bytes"] - available)
    if root.drive.upper() != "D:" or Path.cwd().drive.upper() != "D:":
        raise RuntimeError("Keep demonstration inputs and outputs on D:.")
    if os.environ.get("DOOM_DEMO_INTERNAL_LIMIT_BYTES") != "7000000000":
        raise RuntimeError("Use the dedicated demo launcher with its process-local budget.")
    if preflight and delta > 6_500_000_000:
        raise RuntimeError(f"Viewer start requires 500,000,000 bytes of reserve under the 7 GB C: cap. Decline {delta} bytes.")
    return dict(internal_delta_bytes=delta, internal_free_bytes=available,
                external_free_bytes=shutil.disk_usage(root).free)


def save_json(path, body):
    target = Path(path)
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)


def native_api():
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.GetForegroundWindow.restype = wintypes.HWND
    for name in ("IsWindow", "IsWindowVisible", "IsIconic", "IsHungAppWindow"):
        getattr(user, name).argtypes = [wintypes.HWND]
    user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user.GetDC.argtypes, user.GetDC.restype = [wintypes.HWND], wintypes.HDC
    user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user.SendMessageTimeoutW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM, wintypes.UINT,
        wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
    ]
    user.SendMessageTimeoutW.restype = ctypes.c_ssize_t
    return user


def native_sample(user, hwnd, expected=None, *, client_only=False):
    box = wintypes.RECT()
    user.GetClientRect(hwnd, ctypes.byref(box))
    foreground = user.GetForegroundWindow() == hwnd
    row = dict(hwnd=hwnd, visible=bool(user.IsWindowVisible(hwnd)), minimized=bool(user.IsIconic(hwnd)),
               foreground=foreground, client_pixels=[box.right, box.bottom])
    if expected is not None:
        point = wintypes.POINT(*MARK)
        owner = hwnd
        if foreground and not client_only and user.ClientToScreen(hwnd, ctypes.byref(point)):
            owner = None
        gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi.GetPixel.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        gdi.GetPixel.restype = wintypes.DWORD
        dc = user.GetDC(owner)
        try:
            color = gdi.GetPixel(dc, point.x, point.y) if dc else 0xFFFFFFFF
        finally:
            if dc:
                user.ReleaseDC(owner, dc)
        actual = None if color == 0xFFFFFFFF else [color & 255, (color >> 8) & 255, (color >> 16) & 255]
        row.update(probe_source="desktop" if owner is None else "window client",
                   marker_expected=list(expected), marker_observed=actual, marker_match=actual == list(expected))
    return row


def responsiveness(directory):
    user = native_api()
    directory = Path(directory)
    window = directory / "audit-window.json"
    until = time.monotonic() + 25
    while not window.exists() and time.monotonic() < until:
        time.sleep(.25)
    hwnd = json.loads(window.read_text(encoding="utf-8"))["hwnd"]
    samples = []
    for _ in range(12):
        result = ctypes.c_size_t()
        t0 = time.perf_counter()
        ack = user.SendMessageTimeoutW(hwnd, 0, 0, 0, 2, 300, ctypes.byref(result))
        row = native_sample(user, hwnd)
        row.update(clock=time.strftime("%H:%M:%S"), wm_null_ack=bool(ack),
                   ack_ms=1000 * (time.perf_counter() - t0), hung=bool(user.IsHungAppWindow(hwnd)))
        telemetry = directory / "audit-heartbeat.jsonl"
        if telemetry.exists():
            complete = telemetry.read_text(encoding="utf-8").splitlines()
            if complete:
                try:
                    sample = json.loads(complete[-1])
                    row.update(decision=sample["decision"], stage=sample["stage"])
                except json.JSONDecodeError:
                    pass
        samples.append(row)
        time.sleep(1)
    save_json(directory / "audit-native-responsiveness.json", samples)
    print(json.dumps(samples, indent=2))


def riff(tag, payload):
    return tag + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) % 2)


def write_clip(pygame, frames, skip):
    encoded = []
    for frame in frames:
        stream = io.BytesIO()
        pygame.image.save(pygame.surfarray.make_surface(frame.transpose((1, 0, 2))), stream, "frame.jpg")
        encoded.append(stream.getvalue())
    h, w = frames[0].shape[:2]
    rate, scale = TICKS_PER_SECOND, skip * 2
    n, largest = len(encoded), max(map(len, encoded))
    avih = struct.pack("<14I", round(1e6 * scale / rate), largest * rate // scale, 0, 16,
                       n, 0, 1, largest, w, h, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0, scale, rate,
                       0, n, largest, 0xFFFFFFFF, 0, 0, 0, w, h)
    strf = struct.pack("<IiiHH4sIiiII", 40, w, h, 1, 24, b"MJPG", w * h * 3, 0, 0, 0, 0)
    headers = riff(b"LIST", b"hdrl" + riff(b"avih", avih) +
                   riff(b"LIST", b"strl" + riff(b"strh", strh) + riff(b"strf", strf)))
    chunks, index, offset = [], [], 4
    for jpg in encoded:
        chunk = riff(b"00dc", jpg)
        chunks.append(chunk)
        index.append(struct.pack("<4sIII", b"00dc", 16, offset, len(jpg)))
        offset += len(chunk)
    Path("audit-gameplay.avi").write_bytes(riff(b"RIFF", b"AVI " + headers +
                                                riff(b"LIST", b"movi" + b"".join(chunks)) +
                                                riff(b"idx1", b"".join(index))))
    return {"path": "audit-gameplay.avi", "frames": n, "seconds": n * scale / rate,
            "pixels": [w, h], "codec": "Motion JPEG"}


def live(repo, rehearse, scene=None, hold_seconds=240):
    if scene:
        external_storage_snapshot(preflight=True)
    scenario = scene["wad"] if scene else repo / "maps/two_rooms.wad"
    map_name = scene["map"] if scene else "MAP01"
    seed = scene["seed"] if scene else SEED
    cap = scene["tick_cap"] if scene else CAP
    middle = scene["mid_decision"] if scene else 262
    title = f"DoomRay PPO | C749 {scene['mode']}" if scene else TITLE
    if rehearse:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    elif os.environ.get("SDL_VIDEODRIVER", "").lower() in {"dummy", "offscreen"}:
        os.environ.pop("SDL_VIDEODRIVER")
    sys.path[0:0] = [str(repo / "env"), str(repo)]
    import numpy as np
    import pygame
    import torch
    from doom_env import make_env
    from doom_env.remote import validate_action
    from doom_env.spaces import BUTTONS, FRAME_SKIP, VITALS_INDEX, VITALS_FIELDS
    from submission.agent import DoomAgent
    if scene:
        snapshot = external_storage_snapshot
    else:
        from training.storage import snapshot

    before = snapshot()
    pygame.display.init()
    pygame.font.init()
    user = native_api()
    probe_event = pygame.USEREVENT + 1
    pygame.time.set_timer(probe_event, 250)
    view = pygame.display.set_mode(SIZE)
    pygame.display.set_caption(title)
    hwnd = pygame.display.get_wm_info().get("window", 0)
    save_json("audit-window.json", dict(hwnd=hwnd, pid=os.getpid(), title=title,
                                        display_driver=pygame.display.get_driver(), size=list(SIZE)))
    font, detail, header = (pygame.font.Font(None, x) for x in (26, 21, 30))
    tick = decision = pulses = polls = invalid = 0
    start_tick = last_tick = 0
    opened_doors, opened_locks, acquired_keys = set(), set(), set()
    progression_events = []
    exit_reached = False
    terminal_observation = terminal_tick_regressed = False
    last_unlock_decision = None
    measured = []
    clip, captures, frame_hashes, checksums, flip_times = [], [], set(), [], []
    presses = [0] * len(BUTTONS)
    status = "Preparing official map"
    outcome = "Presenter closed viewer"
    began = time.perf_counter()

    def line(content, x, y, face=detail, color=(222, 230, 244)):
        view.blit(face.render(content, True, color), (x, y))

    def poll():
        nonlocal polls, pulses
        polls += 1
        events = pygame.event.get()
        pulses += sum(item.type == probe_event for item in events)
        return any(item.type == pygame.QUIT or
                   (item.type == pygame.KEYDOWN and item.key == pygame.K_ESCAPE) for item in events)

    def paint(obs=None, kills=0, latency=0.):
        view.fill((17, 23, 33))
        scope = scene["display_name"] if scene else "Known validator map"
        label = scenario.name if scene else "Two rooms"
        line(f"DoomRay PPO  /  Frozen C749  /  {scope}", 16, 6, header)
        line(f"{label} / {map_name} / seed {seed}    |    {status}    |    Escape closes viewer", 16, 35)
        color = COLORS[decision % len(COLORS)]
        pygame.draw.rect(view, color, pygame.Rect(MARK[0] - 10, MARK[1] - 9, 20, 18))
        if obs is not None:
            rgb = obs["rgb"]
            raw = pygame.surfarray.make_surface(rgb.transpose((1, 0, 2)))
            view.blit(pygame.transform.scale(raw, (800, 600)), (0, HEADER))
            v = obs["vitals"]
            stat = lambda name: int(v[VITALS_INDEX[name]])
            keys = " ".join(k[4:].replace("_", " ") for k in VITALS_FIELDS if k.startswith("key_") and stat(k))
            clock = f"Elapsed ticks {max(0, last_tick - start_tick)}" if scene else f"tick {tick}"
            if scene and terminal_tick_regressed:
                clock += " (last live)"
            vital_label = " (last live)" if scene and terminal_observation else ""
            row = [
                (f"Decision {decision}  /  {clock}", font if not scene else detail),
                (f"Health {stat('health')}   Armor {stat('armor')}{vital_label}", detail if vital_label else font),
                (f"Weapon ammo {stat('selected_weapon_ammo')}{vital_label}", detail),
                (f"Bullets {stat('ammo_bullets')}   Shells {stat('ammo_shells')}", detail),
                (f"Rockets {stat('ammo_rockets')}   Cells {stat('ammo_cells')}", detail),
                (f"Kills {kills}   [evaluator side]", detail),
                (f"Keys {keys or 'none'}", detail),
                (f"Inference {latency:.2f} ms", detail),
            ]
            if scene:
                last_unlock = f"   Last d{last_unlock_decision}" if last_unlock_decision else ""
                row.extend([(f"Doors {len(opened_doors)}  /  Unlocks {len(opened_locks)}{last_unlock}", detail),
                            (f"Exit {'completed' if exit_reached else 'pending'}", detail)])
            for i, (label, size) in enumerate(row):
                line(label, 814, 66 + i * (22 if scene else 25), size)
            line("Buttons applied on this decision", 814, 279, font)
            for i, (name, pressed) in enumerate(zip(BUTTONS, presses, strict=True)):
                label = ("ON   " if pressed else "off  ") + name.replace("_", " ").title()
                line(label, 814, 306 + i * 18, detail,
                     (129, 232, 171) if pressed else (151, 165, 188))
            line("Legal rays  /  forward is up", 814, 483, detail)
            center = (960, 590)
            pygame.draw.circle(view, (80, 91, 115), center, 67, 1)
            ray_colors = ((69, 79, 102), (162, 174, 192), (241, 181, 87),
                          (238, 154, 80), (236, 104, 100), (113, 219, 167))
            for i, (distance, hit) in enumerate(zip(obs["rays"]["distance"], obs["rays"]["hit_type"], strict=True)):
                angle = i * math.tau / len(obs["rays"]["distance"])
                radius = max(3, 67 * float(distance))
                pygame.draw.line(view, ray_colors[int(hit)], center,
                                 (center[0] - radius * math.sin(angle), center[1] - radius * math.cos(angle)))
            pygame.draw.circle(view, (230, 240, 250), center, 3)
            line("Pixels and vitals use the last legal observation. Kills and door events are evaluator readings.",
                 12, 669, detail)
        pygame.display.flip()
        flip_times.append(time.perf_counter())
        return color

    def capture(label, obs):
        path = f"audit-{label}.png"
        pygame.image.save(view, path)
        canvas = pygame.surfarray.array3d(view)
        roundtrip = np.array_equal(canvas[2:800:5, HEADER + 2:HEADER + 600:5], obs["rgb"].transpose((1, 0, 2)))
        captures.append(dict(path=path, phase=label, decision=decision, tic=tick, rgb_roundtrip=bool(roundtrip)))

    def heartbeat(log, stage, obs, color):
        if hwnd and pygame.display.get_driver() == "windows":
            pygame.event.pump()
            window = native_sample(user, hwnd, color, client_only=bool(scene))
        else:
            window = {}
        row = dict(stage=stage, monotonic=time.perf_counter(), decision=decision, tic=tick,
                   event_polls=polls, timer_events=pulses, presents=len(flip_times), window=window,
                   rgb_crc32=zlib.crc32(obs["rgb"].tobytes()) if obs is not None else None)
        checksums.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()

    env = None
    try:
        paint()
        env = make_env(scenario, map_name, episode_ticks=cap, render_buffers=True, render_mode="rgb_array")
        if scene:
            obs, info = env.reset(seed=seed)
            initial_ceilings = env.privileged_state()["sector_ceiling"].copy()
            agent = DoomAgent(env.observation_space, env.action_space, "cpu")
            torch.manual_seed(seed)
            agent.reset()
        else:
            agent = DoomAgent(env.observation_space, env.action_space, "cpu")
            agent.reset()
            torch.manual_seed(seed)
            env.action_space.seed(seed)
            obs, info = env.reset(seed=seed)
        tick = start_tick = last_tick = info["tic"]
        kills = 0
        captured_events = set()
        with Path("audit-heartbeat.jsonl").open("w", encoding="utf-8") as log:
            status = "Starting in 3 seconds" if not rehearse else "Running"
            color = paint(obs)
            heartbeat(log, "warmup", obs, color)
            stopped = False
            until = time.perf_counter() + (0 if rehearse else 3)
            while time.perf_counter() < until:
                if poll():
                    stopped = True
                    break
                time.sleep(.02)
            began = time.perf_counter()
            while not stopped and not poll():
                t0 = time.perf_counter()
                selected = agent.act(obs)
                latency = 1000 * (time.perf_counter() - t0)
                measured.append(latency)
                try:
                    action = validate_action(selected, env.action_space)
                except Exception:
                    invalid += 1
                    raise
                obs, _, terminal, truncated, info = env.step(action)
                decision += 1
                tick = info["tic"]
                terminal_observation = terminal or truncated
                terminal_tick_regressed = terminal_observation and tick < last_tick
                last_tick = max(last_tick, tick)
                presses = list(action)
                # Viewer-only terminal data never goes into DoomAgent.act.
                evaluator = env.privileged_state()
                kills = evaluator["kills"]
                new_events = []
                if scene:
                    keys = {k[4:] for k in VITALS_FIELDS if k.startswith("key_") and obs["vitals"][VITALS_INDEX[k]] > 0}
                    picked_up = keys - acquired_keys
                    if picked_up:
                        acquired_keys.update(picked_up)
                        new_events.append(("key", sorted(picked_up)))
                    locks, ceilings = evaluator["door_lock"], evaluator["sector_ceiling"]
                    if locks is not None and ceilings is not None:
                        raised = set(np.flatnonzero((locks >= 0) & (ceilings > initial_ceilings + 4)).tolist())
                        distinct = raised - opened_doors
                        locked = {index for index in distinct if locks[index] > 0}
                        if distinct:
                            opened_doors.update(distinct)
                            new_events.append(("door", sorted(distinct)))
                        if locked:
                            opened_locks.update(locked)
                            last_unlock_decision = decision
                            new_events.append(("unlock", sorted(locked)))
                    if info["exit_reached"] and not exit_reached:
                        new_events.append(("exit", True))
                    exit_reached = bool(info["exit_reached"])
                if terminal or truncated:
                    outcome = "Exit reached" if info["exit_reached"] else ("Tick cap reached" if truncated else "Agent died")
                status = outcome if terminal or truncated else "Running"
                color = paint(obs, kills, latency)
                pixels = obs["rgb"]
                frame_hashes.add(zlib.crc32(pixels.tobytes()))
                if ((scene and (decision % 2 or terminal_observation)) or
                        (not scene and decision <= 105 and decision % 2)):
                    clip.append(pixels.copy())
                if decision == 1:
                    capture("begin", obs)
                if decision == middle:
                    capture("mid", obs)
                for event, value in new_events:
                    progression_events.append(dict(event=event, value=value, decision=decision, raw_tic=tick,
                                                   elapsed_ticks_through_last_report=max(0, last_tick - start_tick)))
                    if event not in captured_events:
                        capture(event, obs)
                        captured_events.add(event)
                if decision == 1 or decision % 10 == 0 or terminal or truncated:
                    heartbeat(log, "finished" if terminal or truncated else "live", obs, color)
                if decision % (16 if scene else 128) == 0:
                    sample = snapshot()
                    if scene and sample["internal_delta_bytes"] >= 6_900_000_000:
                        outcome = "Stopped at storage guard"
                        break
                if terminal or truncated:
                    break
                if not rehearse:
                    target = began + decision * FRAME_SKIP / TICKS_PER_SECOND
                    time.sleep(max(0., target - time.perf_counter()))
            elapsed = time.perf_counter() - began
            data = env.privileged_state()
            env.close()
            env = None
            status = f"{outcome} / engine stopped"
            color = paint(obs, kills, measured[-1] if measured else 0.)
            capture("end", obs)
            heartbeat(log, "engine_closed", obs, color)
        after = snapshot()
        report = dict(commit=COMMIT, seed=seed, wad=scenario.name, map=map_name, decisions=decision,
                      tick=tick, outcome=outcome, kills=kills, final_health=data["health"],
                      damage=data["damage_taken"], invalid_actions=invalid, distinct_rgb_frames=len(frame_hashes),
                      valid_rgb_native_input=list(obs["rgb"].shape), display_driver=pygame.display.get_driver(),
                      display_size=list(SIZE), p99_inference_ms=float(np.percentile(measured, 99)) if measured else None,
                      gameplay_seconds=elapsed, polls=polls, poll_timer_events=pulses,
                      live_marker_samples=sum(row["stage"] == "live" for row in checksums),
                      live_marker_matches=sum(row["stage"] == "live" and bool(row["window"].get("marker_match")) for row in checksums),
                      storage_before=before["internal_delta_bytes"], storage_after=after["internal_delta_bytes"],
                      captures=captures, engine_closed_at=time.strftime("%H:%M:%S"))
        if scene:
            report.update(mode=scene["mode"], scope=scene["scope"], seed_basis=scene["seed_basis"],
                          release_checkout=str(repo), manifest=scene["manifest"], wad_sha256=scene["wad_sha256"],
                          tick_cap=cap, start_tick=start_tick,
                          elapsed_ticks_through_last_report=max(0, last_tick - start_tick),
                          terminal_tick_regressed=terminal_tick_regressed,
                          nominal_requested_ticks=decision * FRAME_SKIP,
                          last_legal_observation_vitals={name: int(obs["vitals"][VITALS_INDEX[name]])
                                                       for name in VITALS_FIELDS},
                          keys_acquired=sorted(acquired_keys), doors=len(opened_doors),
                          unlocks=len(opened_locks), last_unlock_decision=last_unlock_decision,
                          exit_reached=exit_reached, events=progression_events)
        save_json("audit-report.json", report)
        print("Engine closed.", json.dumps(report), flush=True)
        if clip:
            try:
                save_json("audit-clip.json", write_clip(pygame, clip, FRAME_SKIP))
            except (pygame.error, OSError, ValueError, struct.error) as exc:
                save_json("audit-clip.json", {"encoding_error": str(exc)})
        if not rehearse and outcome != "Presenter closed viewer":
            until = time.perf_counter() + hold_seconds
            while time.perf_counter() < until and not poll():
                time.sleep(.03)
    finally:
        if env is not None:
            env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("Progression", "Complex", "Backtracking"))
    parser.add_argument("--hold-seconds", type=int, choices=range(5, 241), default=240)
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.probe:
        responsiveness(args.probe)
    elif args.repo and (args.rehearse ^ args.live):
        if bool(args.manifest) != bool(args.mode):
            parser.error("Choose both a demonstration manifest and mode.")
        repo = args.repo.resolve()
        scene = load_scene(args.manifest.resolve(), args.mode, repo) if args.manifest else None
        live(repo, args.rehearse, scene, args.hold_seconds)
    else:
        parser.error("Choose --probe, or --repo with --rehearse or --live.")


if __name__ == "__main__":
    main()
