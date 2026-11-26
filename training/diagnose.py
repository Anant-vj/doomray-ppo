"""Collect local combat trajectories from the previously inspected maps."""
import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from vizdoom import GameVariable as GV

from doom_env.remote import validate_action
from doom_env.spaces import BUTTONS, ENTITY_TYPES
from submission.policy import Policy, buttons, pack
from training.game import Episode, maps
from training.storage import project_root, snapshot


COMBAT_VARIABLES = (GV.HITCOUNT, GV.DAMAGECOUNT, GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)
ENEMY_TYPES = [i for i, name in enumerate(ENTITY_TYPES) if name.startswith("enemy_")]
INDEX = {name: i for i, name in enumerate(BUTTONS)}


def engine_counters(game):
    # These counters are for local diagnostics and never enter pack() or the actor.
    return np.array([game.env._game.get_game_variable(v) for v in COMBAT_VARIABLES], dtype=np.float64)


def episode(game, policy, seed):
    torch.manual_seed(seed)
    game.env.action_space.seed(seed)
    obs = game.reset(seed)
    previous = np.zeros(len(BUTTONS), dtype=np.int8)
    state = policy.initial(1, "cpu") if policy is not None else None
    counters = engine_counters(game)
    trajectory = []
    while True:
        if policy is None:
            action = game.env.action_space.sample()
        else:
            encoded = policy.encode(torch.from_numpy(pack(obs, previous))[None])
            output, state = policy.advance(encoded, state)
            choice, _, _, _ = policy.decision(output)
            action = buttons(choice[0].numpy())
        action = validate_action(action, game.env.action_space)
        rows = obs["visible_entities"]
        enemies = rows["features"][rows["mask"].astype(bool) & np.isin(rows["entity_type"], ENEMY_TYPES)]
        before = game.last
        obs, _, terminated, capped = game.step(action)
        after = game.last
        current = engine_counters(game)
        used = np.maximum(counters[2:] - current[2:], 0)
        counters, delta = current, np.maximum(current[:2] - counters[:2], 0)
        forward = action[0] != action[1]
        strafe = action[2] != action[3]
        moved = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
        trajectory.append({
            "step": game.steps, "action": action.tolist(), "health": after["health"],
            "damage": after["damage_taken"] - before["damage_taken"],
            "healing_lower_bound": max(after["health"] - before["health"], 0.),
            "kills": after["kills"] - before["kills"],
            "items": after["items"] - before["items"],
            "hit_events": int(delta[0]), "damage_dealt": float(delta[1]),
            "ammo_units_spent_lower_bound": used.astype(int).tolist(),
            "visible_enemy": bool(len(enemies)),
            "aligned_enemy": bool(np.any(enemies[:, 2] >= math.cos(math.radians(15)))),
            "turn_degrees": abs((after["angle"] - before["angle"] + 180) % 360 - 180),
            "move_pressed": bool(forward or strafe),
            "displacement": moved,
            "blocked_move_proxy": bool((forward or strafe) and moved < 1. and not (terminated or capped)),
        })
        previous = action
        if terminated or capped:
            break

    attacking = [r for r in trajectory if r["action"][INDEX["ATTACK"]]]
    fired_units = np.sum([r["ammo_units_spent_lower_bound"] for r in trajectory], axis=0).tolist()
    first_kill = next((r["step"] for r in trajectory if r["kills"] > 0), None)
    metrics = game.metrics()
    summary = dict(
        metrics,
        seed=seed,
        first_kill_step=first_kill,
        death_step=None if metrics["alive"] else game.steps,
        hit_events=sum(r["hit_events"] for r in trajectory),
        damage_dealt=sum(r["damage_dealt"] for r in trajectory),
        ammo_units_spent_lower_bound=dict(zip(("bullets", "shells", "rockets", "cells"), fired_units, strict=True)),
        attack_decisions=len(attacking),
        attack_without_visible_enemy=sum(not r["visible_enemy"] for r in attacking),
        aligned_attack_decisions=sum(r["aligned_enemy"] for r in attacking),
        moving_attack_decisions=sum(r["displacement"] >= 1. for r in attacking),
        move_decisions=sum(r["move_pressed"] for r in trajectory),
        blocked_move_proxy=sum(r["blocked_move_proxy"] for r in trajectory),
        turn_degrees=sum(r["turn_degrees"] for r in trajectory),
        healing_lower_bound=sum(r["healing_lower_bound"] for r in trajectory),
    )
    return summary, trajectory


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002])
    args = parser.parse_args()
    torch.set_num_threads(1)
    snapshot()
    policies = [("random", None)]
    checkpoints = {}
    for path in args.checkpoint:
        if path.stem in checkpoints:
            parser.error("Give each diagnostic checkpoint a distinct filename.")
        contents = path.read_bytes()
        record = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        policy = Policy(record["memory"]).eval()
        policy.load_state_dict(record["weights"])
        checkpoints[path.stem] = dict(path=str(path), sha256=hashlib.sha256(contents).hexdigest())
        policies.append((path.stem, policy))
    report = []
    for layout in maps("diagnostic"):
        game = Episode(layout, 2100)
        try:
            for seed in args.seeds:
                for label, actor in policies:
                    summary, trajectory = episode(game, actor, seed)
                    result = dict(policy=label, **summary)
                    report.append(dict(result, trajectory=trajectory))
                    print(json.dumps(result), flush=True)
        finally:
            game.close()
    out = project_root() / "evaluations" / f"combat-diagnostic-{time.time_ns()}.json"
    out.write_text(json.dumps({
        "checkpoints": checkpoints,
        "seeds": args.seeds, "ticks": 2100, "episodes": report,
        "ammo_measurement": "Observed resource drops are a lower bound. Pickups can mask spending.",
        "collision_measurement": "Movement requested with less than one map unit displacement per decision.",
    }), encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
