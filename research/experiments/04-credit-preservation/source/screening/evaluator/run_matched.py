"""One-engine, map/seed/model-ordered examination of the frozen policies."""

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from storage_gate import GateClosed, ROOT, snapshot, storage_policy


MANIFESTS = ROOT / "manifests"
EPISODES = ROOT / "episodes"
LEDGER = EPISODES / "blind-episodes.jsonl"
STOP = ROOT / "logs/stop-matched.flag"
IST = timezone(timedelta(hours=5, minutes=30))
CUTOFF = datetime(2026, 11, 27, 8, 15, tzinfo=IST)
MAP_MANIFEST_HASH = "5e3f7d1e96f578d4bdb37d44ce3ce78c557e990929940a0c8918fe0c4f34ab7c"
CHECKPOINT_MANIFEST_HASH = "832d3826010d5846762f8dc2e2e66d2d3e7b1f932cd11619d0e527fb77747a06"
SCORING_MANIFEST_HASH = "1ca7aeb19af8b5a787bd66cd86234690764689c5979876fdcb04a0b8f9bfc9fd"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_manifest(name, checksum=None):
    path = MANIFESTS / name
    if checksum is not None and digest(path) != checksum:
        raise ValueError(f"Frozen manifest mismatch: {name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_plans():
    roster = read_manifest("checkpoint_manifest.json", CHECKPOINT_MANIFEST_HASH)["candidates"]
    models = read_manifest("evaluator_models.json")["models"]
    if [m["id"] for m in models] != ["C749", "Stack25-u174", "Stack26-u621"]:
        raise ValueError("Unexpected assessor roster or order.")
    for entry, original in zip(models, roster, strict=True):
        repo = Path(entry["checkout"]).resolve()
        weight = repo / "submission/weights/policy.pt"
        if (entry["sha256"] != original["inference_sha256"] or entry["source_commit"] != original["source_commit"] or
                repo.drive.upper() != "D:" or digest(weight) != entry["sha256"] or
                digest(entry["weight_path"]) != entry["sha256"]):
            raise ValueError(f"Weight identity mismatch: {entry['id']}")
        git = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if head != entry["source_commit"] or changes:
            raise ValueError(f"Checkout identity mismatch: {entry['id']}")
        entry["repo"] = repo
    maps = read_manifest("map_manifest.json", MAP_MANIFEST_HASH)
    slots = maps["maps"]
    if [m["id"] for m in slots] != maps["stages"]["minimum"]["ordered_map_ids"]:
        raise ValueError("Map order mismatch.")
    wads = {m["wad"]: m["wad_sha256"] for m in slots}
    for path, sha in wads.items():
        if Path(path).resolve().drive.upper() != "D:" or digest(path) != sha:
            raise ValueError(f"Map archive mismatch: {path}")
    scores = read_manifest("scoring_manifest.json", SCORING_MANIFEST_HASH)
    return models, slots, scores


def import_actor(repo):
    # The three exact release exports share package names. Capture each class
    # together with its own module globals and isolate import resolution.
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("submission.agent")
        if Path(inspect.getfile(module.DoomAgent)).resolve() != repo / "submission/agent.py":
            raise ImportError("The actor did not load from its verified checkout.")
        return module.DoomAgent
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == "submission" or name.startswith("submission."):
                sys.modules.pop(name, None)


def distance_to_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator:
        t = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator))
        dx, dy = a[0] + t * dx - point[0], a[1] + t * dy - point[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)
    return math.dist(point, a), math.atan2(a[1] - point[1], a[0] - point[0])


def polygon_entry(point, segments):
    x, y = point
    inside = False
    for a, b, c, d in segments:
        if (b > y) != (d > y) and x < a + (c - a) * (y - b) / (d - b):
            inside = not inside
    return inside


def find_boundary(point, segments):
    return min(range(len(segments)), key=lambda i: distance_to_segment(point, segments[i][:2], segments[i][2:])[0])


def run_episode(spec, seed, actor, config, api):
    np, torch, make_env, validate_action, AgentFailure, space, limits, GV = api
    from comparison import episode_scores

    identity = {
        "model": config["id"], "checkpoint_sha256": config["sha256"], "source_commit": config["source_commit"],
        "tier": spec["tier"], "map_id": spec["id"], "layout_family": spec["family"],
        "wad_sha256": spec["wad_sha256"], "geometry_sha256": spec["geometry_sha256"],
        "engine_seed": seed, "policy_seed": seed, "tick_cap": spec["tick_cap"], "skill": spec["skill"],
    }
    started = time.perf_counter()
    trace, latencies, events = [], [], {}
    env = previous = None
    status, error, stage = "completed", None, "environment_initialization"
    invalid = decisions = 0
    start_tic = last_tic = reset_ms = init_ms = None
    counters = {
        "recorded_player_hits": 0, "recorded_player_damage": 0, "attack_decisions": 0, "attack_enemy_visible": 0,
        "attacks_without_enemy_cues": 0, "fresh_use": 0, "blocked_moves": 0, "stationary_decisions": 0,
        "move_requests": 0, "weapon_switch_requests": 0, "weapon_switches_without_observed_change": 0,
        "approaching_visible_enemy": 0, "retreating_visible_enemy": 0, "healing_lower_bound": 0,
        "near_wall_blocks": 0, "use_while_aligned_in_range": 0,
    }
    button_counts = [0] * len(space.BUTTONS)
    cells, doors, locks, lock_crossings, ammo = set(), set(), set(), set(), np.zeros(4, dtype=np.int64)
    entered_at, polygons = {}, {}
    target_distance = key_distance = path_travel = None
    raw_tic = 0
    received = initial_health = 0
    state_trace_file = None

    def mark(tag):
        events.setdefault(tag, decisions)

    try:
        env = make_env(spec["wad"], spec["map"], episode_ticks=spec["tick_cap"], render_buffers=False)
        stage = "environment_reset"
        obs, info = env.reset(seed=seed)
        start_tic = last_tic = raw_tic = info["tic"]
        previous = env.privileged_state()
        initial_health = previous["health"]
        polygon_segments = dict(env.sector_polygons())
        polygons = {int(i): polygon_segments[i] for i in np.flatnonzero(previous["door_lock"] > 0)
                    if i in polygon_segments and len(polygon_segments[i])}
        ceilings = previous["sector_ceiling"].copy()
        x0, y0 = previous["x"], previous["y"]
        cells.add((math.floor(x0 / 96), math.floor(y0 / 96)))
        stage = "agent_initialization"
        timer = time.perf_counter()
        agent = actor(env.observation_space, env.action_space, "cpu")
        init_ms = 1000 * (time.perf_counter() - timer)
        if init_ms > 1000 * limits.INIT_TIMEOUT_S:
            raise TimeoutError("Agent constructor exceeded the official init limit.")
        torch.manual_seed(seed)
        stage = "agent_reset"
        timer = time.perf_counter()
        agent.reset()
        reset_ms = 1000 * (time.perf_counter() - timer)
        if reset_ms > 1000 * limits.RESET_TIMEOUT_S:
            raise TimeoutError("Agent reset exceeded the official reset limit.")
        enemy_types = [i for i, name in enumerate(space.ENTITY_TYPES) if name.startswith("enemy_")]
        key_type = space.ENTITY_TYPES.index("key")
        enemy_ray = space.RAY_HIT_TYPES.index("enemy")
        locked_ray = space.RAY_HIT_TYPES.index("locked_door")
        wall_ray = space.RAY_HIT_TYPES.index("wall")
        selected = space.VITALS_INDEX["selected_weapon"]
        use, attack, weapon = [space.BUTTONS.index(name) for name in ("USE", "ATTACK", "SELECT_NEXT_WEAPON")]
        values = (GV.HITCOUNT, GV.DAMAGECOUNT, GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)
        readings = np.array([env._game.get_game_variable(v) for v in values])
        exits = spec["static_exit_triggers"]
        prior_use = False
        total_agent_time = path_travel = 0.

        while True:
            if decisions % 16 == 0:
                snapshot(STORAGE)
                if STOP.exists() or datetime.now(IST) >= CUTOFF:
                    raise GateClosed("Halted for organizer handoff or the 08:15 IST cutoff.")
            stage = "inference"
            started_action = time.perf_counter()
            chosen = agent.act(obs)
            latency = time.perf_counter() - started_action
            latencies.append(latency * 1000)
            total_agent_time += latency
            per_action_seconds = limits.FIRST_ACT_TIMEOUT_S if decisions == 0 else limits.ACT_TIMEOUT_S
            if latency > per_action_seconds or total_agent_time > limits.agent_budget_s(spec["tick_cap"]):
                raise TimeoutError("The measured actor time exceeded a contract limit.")
            stage = "action_validation"
            try:
                action = validate_action(chosen, env.action_space)
            except AgentFailure:
                invalid += 1
                raise
            for i, bit in enumerate(action):
                button_counts[i] += int(bit)

            stage = "evaluator_inputs"
            entities = obs["visible_entities"]
            types = entities["entity_type"]
            mask = entities["mask"].astype(bool)
            enemies = entities["features"][mask & np.isin(types, enemy_types)]
            key_rows = entities["features"][mask & (types == key_type)]
            ray_types = obs["rays"]["hit_type"]
            pos = (previous["x"], previous["y"])
            heading = math.radians(previous["angle"])
            if len(key_rows):
                mark("key_visible")
                distance = float(key_rows[:, 0].min() * space.RAY_RANGE)
                key_distance = min(key_distance, distance) if key_distance is not None else distance
                if distance <= 64:
                    mark("key_within_64")
            if np.any(ray_types == locked_ray):
                mark("lock_ray_detected")
            if action[attack]:
                counters["attack_decisions"] += 1
                counters["attack_enemy_visible"] += bool(len(enemies))
                counters["attacks_without_enemy_cues"] += not len(enemies) and not np.any(ray_types == enemy_ray)
            fresh_use = bool(action[use]) and not prior_use
            prior_use = bool(action[use])
            counters["fresh_use"] += fresh_use
            counters["weapon_switch_requests"] += int(action[weapon])
            for exit_line in exits:
                distance, direction = distance_to_segment(pos, exit_line["from_xy"], exit_line["to_xy"])
                target_distance = min(target_distance, distance) if target_distance is not None else distance
                if distance <= 192:
                    mark("exit_geometry_within_192")
                if exit_line["activation"] == "use" and distance <= 64:
                    mark("exit_switch_range")
                    delta = (direction - heading + math.pi) % math.tau - math.pi
                    if abs(delta) <= math.pi / 8:
                        mark("exit_switch_aligned")
                        if fresh_use:
                            mark("exit_switch_fresh_use")
                            counters["use_while_aligned_in_range"] += 1

            stage = "environment_step"
            current_obs, _, terminated, truncated, info = env.step(action)
            decisions += 1
            raw_tic = info["tic"]
            if raw_tic >= last_tic:
                last_tic = raw_tic
            current = env.privileged_state()
            current_readings = np.array([env._game.get_game_variable(v) for v in values])
            delta_counts = np.maximum(current_readings[:2] - readings[:2], 0)
            counters["recorded_player_hits"] += int(delta_counts[0])
            counters["recorded_player_damage"] += float(delta_counts[1])
            ammo += np.maximum(readings[2:] - current_readings[2:], 0).astype(np.int64)
            readings = current_readings
            received = current["damage_taken"]
            counters["healing_lower_bound"] += max(0., current["health"] - previous["health"])
            destination = (current["x"], current["y"])
            d = math.dist(pos, destination)
            path_travel += d
            cells.add(tuple(math.floor(value / 96) for value in destination))
            if len(cells) > 1:
                mark("explored_new_cell")
            if int(current["keys"].sum()) > int(previous["keys"].sum()):
                mark("key_acquired")
            rose = set(np.flatnonzero((current["door_lock"] >= 0) & (current["sector_ceiling"] > ceilings + 4)).tolist())
            doors.update(rose)
            locks.update(i for i in rose if current["door_lock"][i] > 0)
            if locks:
                mark("locked_sector_rose")
            if 0 < d < 256:
                for index in locks & polygons.keys():
                    geometry = polygons[index]
                    was_inside, now_inside = polygon_entry(pos, geometry), polygon_entry(destination, geometry)
                    if not was_inside and now_inside:
                        entered_at[index] = find_boundary(pos, geometry)
                        mark("unlocked_sector_entered")
                    if was_inside and not now_inside and index in entered_at:
                        if entered_at.pop(index) != find_boundary(destination, geometry):
                            lock_crossings.add(index)
                            mark("unlocked_sector_crossing_proxy")
            if info["exit_reached"]:
                mark("engine_exit")
            forward, left = int(action[0]) - int(action[1]), int(action[2]) - int(action[3])
            move = bool(forward or left)
            counters["move_requests"] += move
            if not (terminated or truncated):
                counters["weapon_switches_without_observed_change"] += bool(action[weapon] and obs["vitals"][selected] == current_obs["vitals"][selected])
                counters["stationary_decisions"] += d < .5
                if move and d < 1.:
                    counters["blocked_moves"] += 1
                    index = round(math.atan2(left, forward) * len(ray_types) / math.tau) % len(ray_types)
                    counters["near_wall_blocks"] += bool(ray_types[index] == wall_ray and obs["rays"]["distance"][index] * space.RAY_RANGE < 36)
            if len(enemies) and d >= 1.:
                _, sine, cosine, *_ = enemies[enemies[:, 0].argmin()]
                angle = heading + math.atan2(sine, cosine)
                projection = (destination[0] - pos[0]) * math.cos(angle) + (destination[1] - pos[1]) * math.sin(angle)
                counters["approaching_visible_enemy"] += projection >= 1.
                counters["retreating_visible_enemy"] += projection <= -1.
            trace.append([decisions, raw_tic, round(current["x"], 2), round(current["y"], 2),
                          round(current["angle"], 2), current["health"], action.tolist(), len(enemies),
                          int(current["keys"].sum()), len(doors), len(locks), int(current["kills"]),
                          round(latency * 1000, 3), float(delta_counts[1]), int(delta_counts[0])])
            previous, obs = current, current_obs
            if terminated or truncated:
                break
    except GateClosed as exc:
        status, error = "storage_or_handoff_stop", str(exc)
    except TimeoutError as exc:
        status, error = "timeout", str(exc)
    except Exception as exc:
        status = "invalid_action" if invalid else "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        armor = None
        if env is not None:
            try:
                armor = env._game.get_game_variable(GV.ARMOR)
            except Exception:
                armor = None
            try:
                env.close()
            except Exception as exc:
                status, error, stage = "failed", f"{type(exc).__name__}: {exc}", "shutdown"
    if trace:
        state_trace_file = EPISODES / "traces" / f"{config['id']}-{spec['id']}-{seed}.json.gz"
        state_trace_file.parent.mkdir(exist_ok=True)
        temporary = state_trace_file.with_suffix(".gz.tmp")
        columns = ["decision", "raw_tic", "x", "y", "angle", "hp", "action", "visible_enemies", "keys", "doors", "locks", "engine_kills", "latency_ms", "player_damage_delta", "player_hits_delta"]
        with gzip.open(temporary, "wt", encoding="utf-8") as file:
            json.dump({"identity": identity, "columns": columns, "trace": trace}, file, separators=(",", ":"))
        temporary.replace(state_trace_file)
    complete = status == "completed" and previous is not None
    alive = bool(not previous["dead"]) if complete else None
    reached = bool(previous["exit_reached"]) if complete else None
    metrics = {
        **counters, "health": previous["health"] if previous is not None else None, "armor": armor,
        "damage_received": received, "engine_kills": previous["kills"] if previous is not None else None,
        "resource_pickups": previous["items"] if previous is not None else None,
        "keys": int(previous["keys"].sum()) if previous is not None else 0, "unlocks": len(locks),
        "doors": len(doors), "lock_crossings_proxy": len(lock_crossings), "explored_cells_96": len(cells),
        "ammo_consumed_lower_bound": [int(value) for value in ammo], "initial_health": initial_health,
        "min_exit_geometry_distance": target_distance, "min_visible_key_distance": key_distance,
        "path_distance": path_travel, "button_counts": dict(zip(space.BUTTONS, button_counts, strict=True)),
    }
    result = {**identity, "status": status, "stage": stage, "error": error, "alive": alive, "exit": reached,
              "termination": ("exit" if reached else "death" if not alive else "tick_limit") if complete else status,
              "decisions": decisions, "invalid_actions": invalid, "engine_reset_ok": start_tic is not None,
              "game_ticks": (raw_tic - start_tic) if start_tic is not None and raw_tic >= last_tic else None,
              "last_live_game_tick": last_tic, "terminal_raw_tick": raw_tic,
              "wall_seconds": round(time.perf_counter() - started, 3),
              "init_ms": init_ms, "reset_ms": reset_ms, "funnel_events": events, "metrics": metrics,
              "trace": str(state_trace_file) if state_trace_file else None}
    result["latency_ms"] = dict(zip(
        ("mean", "median", "p95", "p99", "max"),
        [float(np.mean(latencies)), *map(float, np.percentile(latencies, [50, 95, 99])), max(latencies)] if latencies else [None] * 5,
        strict=True,
    ))
    result["scores"] = episode_scores(result, spec["tick_cap"]) if complete else None
    return result


def main():
    global STORAGE
    STORAGE = storage_policy()
    initial = snapshot(STORAGE, startup=True)
    models, specs, _ = verify_plans()
    cases = [(m, seed, model) for m in specs for seed in m["seeds"] for model in models]
    print(json.dumps({"storage": initial, "predeclared_layouts": len(specs), "predeclared_episodes": len(cases)}), flush=True)
    if sys.argv[1] == "preflight":
        return 0
    if STOP.exists() or datetime.now(IST) >= CUTOFF:
        raise GateClosed("Assessment is paused for organizer handoff or the 08:15 IST cutoff.")

    import numpy as np
    import torch
    from doom_env import limits, make_env, spaces, wrapper
    from doom_env.remote import AgentFailure, validate_action
    from vizdoom import GameVariable

    if wrapper.SKILL != 3:
        raise ValueError("The matching official wrapper must use skill 3.")
    torch.set_num_threads(1)
    runtime = {"python": sys.version.split()[0], "python_executable": sys.executable,
               "vizdoom": importlib.metadata.version("vizdoom"), "torch": torch.__version__,
               "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
               "wrapper": inspect.getfile(wrapper), "headless": True, "startup_storage": initial}
    log = ROOT / "logs" / f"matched-runtime-{time.strftime('%Y%m%d-%H%M%S')}.json"
    log.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    api = (np, torch, make_env, validate_action, AgentFailure, spaces, limits, GameVariable)
    actors = {entry["id"]: import_actor(entry["repo"]) for entry in models}
    existing = {
        (item["map_id"], item["engine_seed"], item["model"])
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line and (item := json.loads(line))
    } if LEDGER.exists() else set()
    with LEDGER.open("a", encoding="utf-8") as file:
        for spec, seed, model in cases:
            if STOP.exists() or datetime.now(IST) >= CUTOFF:
                break
            if (spec["id"], seed, model["id"]) in existing:
                continue
            snapshot(STORAGE, startup=True, actor_loaded=True)
            if digest(model["repo"] / "submission/weights/policy.pt") != model["sha256"]:
                raise ValueError(f"Frozen model changed before a planned episode: {model['id']}")
            result = run_episode(spec, seed, actors[model["id"]], model, api)
            file.write(json.dumps(result, ensure_ascii=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
            brief = ("model", "map_id", "engine_seed", "status", "exit", "alive", "decisions", "termination", "error")
            print(json.dumps({key: result[key] for key in brief}), flush=True)
            if result["status"] == "storage_or_handoff_stop":
                break
    print(json.dumps({"after": snapshot(STORAGE)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateClosed as exc:
        print(f"Assessment gate halted the workflow. {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
