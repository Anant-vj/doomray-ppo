"""Offline geometry and return-identity check for the training-only potential."""

import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

from doom_env.mapinfo import load_mapinfo
from doom_env.wad import read_wad
from training.potential import GAMMA, POTENTIAL_BOUND, make_potential


PROJECT = Path(os.environ["DOOMRAY_PROJECT_ROOT"])
WORKTREE = Path(__file__).resolve().parents[1] / "policy-worktree"
LAYOUTS = (
    (WORKTREE / "tests/fixtures/e7ab54fb6834ffa3.wad", 96.0),
    (PROJECT / "maps/curriculum/red_key_backtrack.wad", 288.0),
)


def state(x, *, y=96.0, has_key=False, clearance=0.0):
    return {
        "x": x,
        "y": y,
        "keys": [0, 0, int(has_key), 0, 0, 0],
        "sector_ceiling": [128.0, clearance, 128.0, 128.0],
        "sector_floor": [0.0] * 4,
        "finished": False,
    }


def keep(potential, successor):
    return potential.step(successor, terminated=False, truncated=False)


def check_fixture(path, spawn):
    map_info = load_mapinfo(str(path), "MAP01")
    env = SimpleNamespace(map_name="MAP01", wad=str(path), mapinfo=map_info)
    potential = make_potential(env, path, "MAP01")
    assert potential is not None
    assert make_potential(env, path, "MAP02") is None
    assert make_potential(None, "unsupported.wad", "MAP01") is None

    text = next(lump.data.decode() for lump in read_wad(path) if lump.name == "TEXTMAP")
    assert f"thing {{ x = {spawn}; y = 96.0; angle = 0; type = 1;" in text
    assert "thing { x = 224.0; y = 96.0; angle = 180; type = 13;" in text
    assert "special = 243; playercross = true;" in text
    assert map_info.door_sectors == {1: 129}
    lines = map_info.lines
    for x, sectors in ((384, {0, 1}), (448, {1, 2}), (640, {2, 3})):
        match = (lines.p1[:, 0] == x) & (lines.p2[:, 0] == x)
        assert int(match.sum()) == 1
        i = match.nonzero()[0][0]
        assert {lines.p1[i, 1], lines.p2[i, 1]} == {64, 128}
        assert {lines.front[i], lines.back[i]} == sectors
        assert not lines.impassable[i]

    init = potential.reset(state(spawn))
    nearer_key = spawn + (224 - spawn) / 2
    assert 0 < init < POTENTIAL_BOUND
    assert keep(potential, state(nearer_key)) > 0 and potential.phase == "key"
    keep(potential, state(224, has_key=True))
    assert potential.phase == "lock"
    assert keep(potential, state(352, has_key=True)) > 0
    keep(potential, state(352, has_key=True, clearance=55))
    assert potential.phase == "lock"
    assert keep(potential, state(352, has_key=True, clearance=56)) > 0 and potential.phase == "crossing"
    assert keep(potential, state(420, has_key=True, clearance=80)) > 0
    assert keep(potential, state(448, has_key=True, clearance=80)) > 0 and potential.phase == "crossing"
    keep(potential, state(449, has_key=True, clearance=80))
    assert potential.phase == "exit"
    assert keep(potential, state(560, has_key=True)) > 0
    closeness = keep(potential, state(620, has_key=True))
    assert 0 < closeness <= POTENTIAL_BOUND
    assert potential.step(None, terminated=True, truncated=False) < 0
    assert potential.phase == "terminal"
    try:
        keep(potential, state(spawn))
    except RuntimeError:
        pass
    else:
        raise AssertionError("The finished episode's state leaked through a second step.")
    assert potential.reset(state(spawn)) == init

    # A closing/reopening after the first opening keeps the same potential at
    # the same pose. The discounted closed loop cannot earn positive shaping.
    anchor = potential.reset(state(352, has_key=True, clearance=80))
    loop = [
        keep(potential, state(352, has_key=True, clearance=0)),
        keep(potential, state(352, has_key=True, clearance=80)),
    ]
    assert math.isclose(loop[0] + GAMMA * loop[1], (GAMMA ** 2 - 1) * anchor, abs_tol=1e-12)
    assert all(x <= 0 for x in loop)

    samples = random.Random(int(spawn))
    for cutoff in range(1, 65):
        anchor = potential.reset(state(spawn))
        rewards = []
        stages = [state(224), state(224, has_key=True), state(352, has_key=True),
                  state(400, has_key=True, clearance=96), state(449, has_key=True, clearance=96)]
        for i in range(cutoff - 1):
            successor = stages[i] if i < len(stages) else state(samples.uniform(450, 639), has_key=True)
            rewards.append(keep(potential, successor))
        # The wrapper's terminal observation is deliberately absent. This also
        # checks both actual cutoffs without using a PPO rollout boundary.
        rewards.append(potential.step(None, terminated=cutoff % 2 == 0, truncated=cutoff % 2 == 1))
        assert max(map(abs, rewards)) <= POTENTIAL_BOUND
        shaped_return = sum((GAMMA ** i) * reward for i, reward in enumerate(rewards))
        assert math.isclose(shaped_return, -anchor, abs_tol=1e-12)


if __name__ == "__main__":
    for fixture, player_x in LAYOUTS:
        check_fixture(fixture, player_x)
    print("PASS: 2 hashed training WADs, actual key/lock/exit geometry, 128 discounted trajectories, "
          "terminal/truncation, reset, and same-door loop. No engine launched.")
