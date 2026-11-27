"""Training-only potential for two verified red-key corridor lessons."""

import hashlib
import math
from pathlib import Path


GAMMA = 0.99
POTENTIAL_BOUND = 0.25
_WAD_SHA256 = {
    "e7ab54fb6834ffa3.wad": "ea6a0b17bca7beae8dc928e9c3b6ced51eceacf1f3c1421d7a98a698a966445e",
    "red_key_backtrack.wad": "2ec2f02accdda7dd0d23fc4eb43956a075781234bcce82ee7ca299ebc2d97397",
}

# Both WADs share the same straight, flat corridor. The backtracking lesson
# only moves the player start. Key type 13 is the red card (keys index 2).
_KEY_X, _CENTER_Y = 224.0, 96.0
_RED_CARD = 2
_DOOR_SECTOR = 1
_RED_LOCK = 129
_OPENING_HEIGHT = 56.0  # Official movement sensor's player height.
_APPROACH_X = 352.0  # Inside the west room, 32 units from the USE face.
_DOOR_EAST_X = 448.0
_EXIT_X = 640.0  # Actual player-cross exit trigger spans the hallway.
_DOOR_INTERACTION_COST = 32.0  # A one-time remaining-cost margin until passage is clear.
_DISTANCE_SCALE = 640.0


class RedKeyPotential:
    """Potential over the corridor state and the attained progression milestones."""

    def __init__(self):
        self._previous = None
        self._opened = False
        self._crossed = False
        self.phase = "inactive"

    def _evaluate(self, state):
        if state["finished"]:
            raise ValueError("Potential needs a live state before an episode boundary.")
        x, y = state["x"], state["y"]
        if not state["keys"][_RED_CARD]:
            self.phase = "key"
            to_key = math.hypot(x - _KEY_X, y - _CENTER_Y)
            remaining = to_key + (_EXIT_X - _KEY_X) + _DOOR_INTERACTION_COST
        else:
            clearance = state["sector_ceiling"][_DOOR_SECTOR] - state["sector_floor"][_DOOR_SECTOR]
            self._opened |= clearance >= _OPENING_HEIGHT
            if not self._opened:
                self.phase = "lock"
                to_approach = math.hypot(max(0.0, _APPROACH_X - x), y - _CENTER_Y)
                remaining = to_approach + (_EXIT_X - _APPROACH_X) + _DOOR_INTERACTION_COST
            else:
                self._crossed |= x > _DOOR_EAST_X
                if self._crossed:
                    self.phase = "exit"
                    remaining = max(0.0, _EXIT_X - x)
                else:
                    self.phase = "crossing"
                    remaining = max(0.0, _DOOR_EAST_X - x) + (_EXIT_X - _DOOR_EAST_X)
        return POTENTIAL_BOUND * max(0.0, 1.0 - remaining / _DISTANCE_SCALE)

    def reset(self, state):
        """Start a new episode at its own potential and clear all past milestones."""
        self._opened = False
        self._crossed = False
        self._previous = self._evaluate(state)
        return self._previous

    def step(self, state, *, terminated, truncated):
        """Return the signed reward term for one genuine environment transition."""
        if self._previous is None:
            raise RuntimeError("Reset the potential before starting an episode.")
        previous = self._previous
        if terminated or truncated:
            # Both are true finite-horizon game ends here. The wrapper's final
            # observation can be stale. PPO rollout boundaries never call this.
            self._previous = None
            self.phase = "terminal"
            return -previous
        self._previous = self._evaluate(state)
        return GAMMA * self._previous - previous


def make_potential(env, path, map_name):
    """Return a potential only for the two existing, byte-verified training MAP01s."""
    path = Path(path)
    expected = _WAD_SHA256.get(path.name)
    if expected is None or map_name.upper() != "MAP01":
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected training fixture bytes: {path}")
    if Path(env.wad).resolve() != path.resolve() or env.map_name != "MAP01":
        raise ValueError("The training fixture and wrapper map must match.")
    info = env.mapinfo
    if info is None or info.num_sectors != 4 or info.door_sectors != {_DOOR_SECTOR: _RED_LOCK}:
        raise ValueError("The wrapper did not recognize the annotated red-key passage.")
    return RedKeyPotential()
