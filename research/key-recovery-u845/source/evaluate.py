"""Separate-process paired examination of the submitted agent and official wrapper."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE = PROJECT / "submission_artifacts/github-fresh-6c2d549-20261127-022250050"
MODE = sys.argv[1]
if MODE not in ("C749", "candidate"):
    raise ValueError(MODE)
REPO = BASE if MODE == "C749" else ROOT / "candidate"
sys.path[:0] = [str(REPO), str(REPO / "env")]

import numpy as np
import torch
import vizdoom
from doom_env import make_env, remote, wrapper
from submission import agent
from submission.policy import pack

protocol_file = ROOT / "protocol.json"
assert sha256(protocol_file.read_bytes()).hexdigest() == "cdfddb2aa7ab26f3fb7a500211a622481268eab698e12b7edb252d67eceb4931"
protocol = json.loads(protocol_file.read_text())
design = protocol["evaluation"]
assert wrapper.DEFAULT_EPISODE_TICKS == design["tick_cap"] and wrapper.SKILL == design["skill"]
assert Path(agent.__file__).resolve().is_relative_to(REPO)
assert Path(wrapper.__file__).resolve().is_relative_to(REPO)
weight = REPO / "submission/weights/policy.pt"
weight_sha = sha256(weight.read_bytes()).hexdigest()
expected_sha = (protocol["source_weight_sha256"] if MODE == "C749" else
                json.loads((ROOT / "candidate-manifest.json").read_text())["inference_sha256"])
assert weight_sha == expected_sha
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def run(layout, seed, policy_seed):
    wad = ROOT / layout["file"]
    assert wad.parent == ROOT / "reserved" and sha256(wad.read_bytes()).hexdigest() == layout["sha256"]
    cap = design["tick_cap"]
    env = make_env(str(wad), layout["marker"], episode_ticks=cap)
    try:
        actor = agent.DoomAgent(env.observation_space, env.action_space, "cpu")
        obs, info = env.reset(seed=seed)
        actor.reset()
        torch.manual_seed(policy_seed)
        np.random.seed(policy_seed)
        random.seed(policy_seed)
        initial_hash = sha256(pack(obs, np.zeros(9, dtype=np.int8)).tobytes()).hexdigest()
        first = env.privileged_state()
        initial_ceiling = first["sector_ceiling"].copy()
        start_tick = info["tic"]
        previous_tick = start_tick
        last_keys = first["keys"].copy()
        keys = 0
        opened = set()
        decisions = 0
        bad = 0
        reason = None
        elapsed = 0
        done = False
        first_hits = float(env._game.get_game_variable(vizdoom.GameVariable.HITCOUNT))
        first_damage = float(env._game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT))
        while not done:
            # Only the official observation is passed across the actor interface.
            action = actor.act(obs)
            try:
                accepted = remote.validate_action(action, env.action_space)
            except remote.AgentFailure:
                bad += 1
                reason = "invalid_action"
                break
            obs, _, term, trunc, info = env.step(accepted)
            decisions += 1
            state = env.privileged_state()
            keys += int(np.count_nonzero((state["keys"] > 0) & (last_keys == 0)))
            last_keys = state["keys"].copy()
            opened.update(np.flatnonzero((state["door_lock"] > 0)
                                         & (state["sector_ceiling"] > initial_ceiling + 4)).tolist())
            elapsed = int(info["tic"]) - start_tick
            done = term or trunc
            if done:
                reason = "exit" if info["exit_reached"] else ("death" if state["dead"] else "tick_cap")
                # Some exits reset the game clock. A terminal interval always ends within a frame skip.
                if info["tic"] < previous_tick:
                    lo, hi = previous_tick - start_tick + 1, previous_tick - start_tick + 4
                    if reason == "death" and lo <= cap // 4 < hi:
                        raise RuntimeError("Ambiguous early-death tick at the threshold")
                    elapsed = hi
            else:
                previous_tick = info["tic"]
            if decisions > (cap + 3) // 4:
                raise RuntimeError("Official episode cap exceeded")
        hits = float(env._game.get_game_variable(vizdoom.GameVariable.HITCOUNT)) - first_hits
        dmg = float(env._game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT)) - first_damage
        return dict(model=MODE, weight_sha256=weight_sha, layout=layout["name"], wad_sha256=layout["sha256"],
                    engine_seed=seed, policy_seed=policy_seed, initial_observation_sha256=initial_hash,
                    cap_ticks=cap, elapsed_ticks_or_terminal_upper_bound=elapsed, exit=reason == "exit",
                    keys=keys, unlocks=len(opened), death=reason == "death",
                    early_death=reason == "death" and elapsed <= cap // 4, invalid_actions=bad,
                    decisions=decisions, termination=reason, player_hits=hits, player_damage=dmg)
    finally:
        env.close()


if __name__ == "__main__":
    destination = ROOT / ("evaluation-" + MODE + ".jsonl")
    with destination.open("x", encoding="utf-8") as ledger:
        for layout in protocol["maps"]["reserved"]:
            for seed, policy_seed in zip(design["engine_seeds"], design["policy_seeds"], strict=True):
                from shutil import disk_usage
                if 34039721984 - disk_usage("C:/").free >= 6500000000:
                    raise RuntimeError("500 MB reserve unavailable before episode")
                result = run(layout, seed, policy_seed)
                ledger.write(json.dumps(result) + "\n")
                ledger.flush()
                print(layout["name"], seed, result["termination"], result["decisions"], flush=True)
