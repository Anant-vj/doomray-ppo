"""Local map pools, episode metrics, and training-only rewards."""
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from doom_env import make_env
import vizdoom

REPO = Path(__file__).resolve().parents[1]
# The bundles use the same IWAD as evaluation. Map splits are disjoint.
SPLITS = {
    "train": (1, 2, 3, 4, 5, 6),
    "dev": (7, 8),
    "dev-wide": (7, 8, 13, 14),
    "holdout": (9, 10),
    "diagnostic": (9, 10),
    "final-holdout": (11, 12),
}
class RewardProfile(NamedTuple):
    damage_cost: float
    lesson_finish: float
    key: float
    unlock: float
    dealt: float = 0.
    healed: float = 0.


REWARD_PROFILES = {
    "control": RewardProfile(.003, 2., .3, .15),
    "survival": RewardProfile(.008, .35, .3, .15),
    "progression": RewardProfile(.006, .35, .45, .25),
    "combat": RewardProfile(.006, .35, .45, .25, dealt=.002, healed=.004),
}
REWARD_TERMS = (
    "native", "exit_bonus", "coverage", "keys", "doors", "unlocks", "damage", "dealt", "healed",
)
PRACTICE_WADS = {"basic.wad", "my_way_home.wad", "e7ab54fb6834ffa3.wad"}
MIXED_SCENARIOS = ("defend_the_center", "health_gathering", "deadly_corridor")


def maps(split):
    assets = Path(vizdoom.__file__).parent
    iwad = assets / "freedoom2.wad"
    expanded = split in ("curriculum", "progression", "mixed")
    map_numbers = SPLITS["train" if expanded else split]
    items = [(iwad, f"MAP{i:02}") for i in map_numbers]
    if split == "train" or expanded:
        items.insert(0, (REPO / "maps/two_rooms.wad", "MAP01"))
    if expanded:
        items.extend([
            (assets / "scenarios" / "basic.wad", "MAP01"),
            (assets / "scenarios" / "my_way_home.wad", "MAP01"),
            (iwad, "MAP31"),
        ])
    if split in ("progression", "mixed"):
        # The official key-pickup/locked-door/exit regression layout.
        items.append((REPO / "tests/fixtures/e7ab54fb6834ffa3.wad", "MAP01"))
    if split == "mixed":
        items.extend((assets / "scenarios" / f"{name}.wad", "MAP01") for name in MIXED_SCENARIOS)
    return items


class Episode:
    def __init__(self, layout, ticks, *, render_buffers=True, reward_profile="control"):
        self.path, self.map_name = layout
        self.env = make_env(str(self.path), self.map_name, episode_ticks=ticks, render_buffers=render_buffers)
        self.reward_profile = REWARD_PROFILES[reward_profile]
        self.exit_bonus = self.reward_profile.lesson_finish if self.path.name in PRACTICE_WADS else 2.

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.last = self.env.privileged_state()
        self.cells = {self.cell(self.last)}
        self.raised = set()
        self.unlocked = set()
        self.initial_ceilings = self.last["sector_ceiling"].copy()
        self.native = 0.
        self.shaped = 0.
        self.reward_terms = np.zeros(len(REWARD_TERMS), dtype=np.float64)
        self.damage_dealt = self._damage_dealt() if self.reward_profile.dealt else 0.
        self.steps = 0
        return obs

    @staticmethod
    def cell(s):
        return math.floor(s["x"] / 96), math.floor(s["y"] / 96)

    def _damage_dealt(self):
        return float(self.env._game.get_game_variable(vizdoom.GameVariable.DAMAGECOUNT))

    def step(self, action):
        obs, reward, term, trunc, _ = self.env.step(action)
        s = self.env.privileged_state()
        cell = self.cell(s)
        new_cell = cell not in self.cells
        self.cells.add(cell)
        # A ceiling rising above its start height within a tagged door sector.
        locks = s["door_lock"]
        opened = set(np.flatnonzero((locks >= 0) & (s["sector_ceiling"] > self.initial_ceilings + 4)).tolist())
        distinct = opened - self.raised
        self.raised.update(opened)
        self.unlocked.update(i for i in distinct if locks[i] > 0)
        keys = float(s["keys"].sum() - self.last["keys"].sum())
        damage = max(0, s["damage_taken"] - self.last["damage_taken"])
        profile = self.reward_profile
        dealt = 0.
        if profile.dealt:
            current = self._damage_dealt()
            dealt = min(max(current - self.damage_dealt, 0.), 60.)
            self.damage_dealt = current
        healed = min(max(s["health"] - self.last["health"], 0.), 25.)
        terms = (
            reward, self.exit_bonus * s["exit_reached"], .035 * new_cell, profile.key * keys,
            .15 * len(distinct), profile.unlock * sum(locks[i] > 0 for i in distinct),
            -profile.damage_cost * damage, profile.dealt * dealt, profile.healed * healed,
        )
        shaped = sum(terms)
        self.reward_terms += terms
        self.native += reward
        self.shaped += shaped
        self.steps += 1
        self.last = s
        return obs, float(shaped), term, trunc

    def metrics(self):
        s = self.last
        return dict(wad=self.path.name, map=self.map_name, steps=self.steps, exit=bool(s["exit_reached"]),
                    alive=not s["dead"], health=s["health"], kills=s["kills"], damage=s["damage_taken"],
                    keys=int(s["keys"].sum()), door_sectors_raised=len(self.raised),
                    locked_sectors_raised=len(self.unlocked), cells_96=len(self.cells),
                    wrapper_return=self.native, shaped_return=float(self.shaped),
                    reward_terms=dict(zip(REWARD_TERMS, self.reward_terms.tolist(), strict=True)))

    def close(self):
        self.env.close()


class SerialEpisodes:
    """Run the rollout environments in the learner process."""

    def __init__(self, layouts, ticks, seeds, *, render_buffers=True, reward_profile="control"):
        self._ticks = ticks
        self._render_buffers = render_buffers
        self._reward_profile = reward_profile
        self._games = [None] * len(layouts)
        try:
            starts = dict(enumerate(zip(layouts, seeds, strict=True)))
            self.observations = list(self.reset(starts).values())
        except BaseException:
            self.close()
            raise

    def step(self, actions):
        results = []
        for game, action in zip(self._games, actions, strict=True):
            obs, reward, terminated, truncated = game.step(action)
            metrics = game.metrics() if terminated or truncated else None
            results.append((obs, reward, terminated, truncated, metrics))
        return results

    def reset(self, starts):
        obs = {}
        for index, (layout, seed) in starts.items():
            game = self._games[index]
            if game is None or (game.path, game.map_name) != layout:
                if game is not None:
                    game.close()
                game = Episode(layout, self._ticks, render_buffers=self._render_buffers,
                               reward_profile=self._reward_profile)
                self._games[index] = game
            obs[index] = game.reset(seed)
        return obs

    def close(self):
        for game in self._games:
            if game is not None:
                game.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
