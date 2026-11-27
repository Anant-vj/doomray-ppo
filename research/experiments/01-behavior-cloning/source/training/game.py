"""Local map pools, episode metrics, and training-only rewards."""
import math
import os
import re
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
from doom_env import make_env
from doom_env.spaces import BUTTONS, ENTITY_TYPES, RAY_HIT_TYPES, VITALS_INDEX
from doom_env.wad import Lump, read_wad, write_wad
import vizdoom

from training.storage import project_root

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
    blind_attack: float = 0.
    unchanged_switch: float = 0.
    neutralize_map_deaths: bool = False


REWARD_PROFILES = {
    "control": RewardProfile(.003, 2., .3, .15),
    "survival": RewardProfile(.008, .35, .3, .15),
    "progression": RewardProfile(.006, .35, .45, .25),
    "combat": RewardProfile(.006, .35, .45, .25, dealt=.002, healed=.004),
    "precision": RewardProfile(.008, .35, .45, .25, dealt=.006, healed=.003,
                               blind_attack=.006, unchanged_switch=.002, neutralize_map_deaths=True),
}
REWARD_TERMS = (
    "native", "exit_bonus", "coverage", "keys", "doors", "unlocks", "damage", "dealt", "healed",
    "map_death_adjustment", "blind_attack", "unchanged_switch",
)
PRACTICE_WADS = {"basic.wad", "my_way_home.wad", "e7ab54fb6834ffa3.wad", "red_key_backtrack.wad"}
MIXED_SCENARIOS = ("defend_the_center", "health_gathering", "deadly_corridor")
ENEMY_ENTITIES = [i for i, name in enumerate(ENTITY_TYPES) if name.startswith("enemy_")]
ENEMY_RAY = RAY_HIT_TYPES.index("enemy")
ATTACK = BUTTONS.index("ATTACK")
SWITCH = BUTTONS.index("SELECT_NEXT_WEAPON")
WEAPON = VITALS_INDEX["selected_weapon"]


def backtrack_lesson(source):
    """Move the original lesson's player spawn between the key and locked door."""
    start = b"thing { x = 96.0; y = 96.0; angle = 0; type = 1;"
    relocated = b"thing { x = 288.0; y = 96.0; angle = 0; type = 1;"
    lumps = read_wad(source)
    if [lump.name for lump in lumps] != ["MAP01", "TEXTMAP", "ENDMAP"] or lumps[1].data.count(start) != 1:
        raise ValueError("The supplied key lesson does not have the expected player spawn.")
    lumps[1] = Lump("TEXTMAP", lumps[1].data.replace(start, relocated, 1))
    path = project_root() / "maps" / "curriculum" / "red_key_backtrack.wad"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    write_wad(temporary, lumps)
    os.replace(temporary, path)
    return path


def training_yaw(source, angle):
    """Change just the player-one start angle in a training-only copy."""
    lumps = read_wad(source)
    indices = [i for i, lump in enumerate(lumps) if lump.name in ("TEXTMAP", "THINGS")]
    if len(indices) != 1:
        raise ValueError(f"Expected one map definition in {source.name}.")
    index = indices[0]
    lump = lumps[index]
    if lump.name == "TEXTMAP":
        starts = [block for block in re.findall(rb"\bthing\s*\{[^{}]*\}", lump.data)
                  if re.search(rb"\btype\s*=\s*1\s*;", block)]
        angle_field = rb"\bangle\s*=\s*-?\d+(?:\.\d+)?\s*;"
        if len(starts) != 1 or len(re.findall(angle_field, starts[0])) > 1:
            raise ValueError(f"Cannot identify the player-one angle in {source.name}.")
        replacement = f"angle = {angle};".encode()
        updated, replaced = re.subn(angle_field, replacement, starts[0])
        if not replaced:
            # UDMF can omit the angle to indicate the default east-facing start.
            updated = starts[0][:-1] + b"\n" + replacement + b"\n}"
        data = lump.data.replace(starts[0], updated, 1)
    else:
        hexen = any(lump.name == "BEHAVIOR" for lump in lumps)
        stride, angle_offset, type_offset = (20, 8, 10) if hexen else (10, 4, 6)
        if len(lump.data) % stride:
            raise ValueError(f"Malformed thing records in {source.name}.")
        starts = [i for i in range(0, len(lump.data), stride)
                  if struct.unpack_from("<h", lump.data, i + type_offset)[0] == 1]
        if len(starts) != 1:
            raise ValueError(f"Cannot identify the player-one start in {source.name}.")
        writable = bytearray(lump.data)
        struct.pack_into("<h", writable, starts[0] + angle_offset, angle)
        data = bytes(writable)
    lumps[index] = Lump(lump.name, data)
    dest = project_root() / "maps" / "curriculum" / f"train-only-{source.stem}-yaw{angle}.wad"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(f".{os.getpid()}.tmp")
    write_wad(temporary, lumps)
    os.replace(temporary, dest)
    return dest


def maps(split):
    assets = Path(vizdoom.__file__).parent
    iwad = assets / "freedoom2.wad"
    if split == "precision":
        existing = maps("varied")
        varied_sources = [assets / "scenarios/basic.wad", REPO / "tests/fixtures/e7ab54fb6834ffa3.wad"]
        return existing + [(training_yaw(path, angle), "MAP01") for path in varied_sources for angle in (90, 270)]
    expanded = split in ("curriculum", "progression", "mixed", "varied")
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
    if split in ("progression", "mixed", "varied"):
        # The official key-pickup/locked-door/exit regression layout.
        key_lesson = REPO / "tests/fixtures/e7ab54fb6834ffa3.wad"
        items.append((key_lesson, "MAP01"))
        if split == "varied":
            items.append((backtrack_lesson(key_lesson), "MAP01"))
    if split in ("mixed", "varied"):
        items.extend((assets / "scenarios" / f"{name}.wad", "MAP01") for name in MIXED_SCENARIOS)
    return items


class Episode:
    def __init__(self, layout, ticks, *, render_buffers=True, reward_profile="control"):
        self.path, self.map_name = layout
        self.env = make_env(str(self.path), self.map_name, episode_ticks=ticks, render_buffers=render_buffers)
        self.reward_profile = REWARD_PROFILES[reward_profile]
        practice = self.path.name in PRACTICE_WADS or self.path.name.startswith("train-only-")
        self.exit_bonus = self.reward_profile.lesson_finish if practice else 2.

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.observation = obs
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
        before = self.observation
        profile = self.reward_profile
        obs, reward, term, trunc, _ = self.env.step(action)
        self.observation = obs
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
        dealt = 0.
        if profile.dealt:
            current = self._damage_dealt()
            dealt = min(max(current - self.damage_dealt, 0.), 60.)
            self.damage_dealt = current
        healed = min(max(s["health"] - self.last["health"], 0.), 25.)
        map_death_adjustment = -.1 * (s["kills"] - self.last["kills"]) if profile.neutralize_map_deaths else 0.
        blind_attack = 0.
        if profile.blind_attack and action[ATTACK] and dealt == 0:
            entities = before["visible_entities"]
            has_entity = np.any(entities["mask"].astype(bool) & np.isin(entities["entity_type"], ENEMY_ENTITIES))
            has_ray = np.any(before["rays"]["hit_type"] == ENEMY_RAY)
            blind_attack = -profile.blind_attack if not (has_entity or has_ray) else 0.
        unchanged_switch = -profile.unchanged_switch if (
            action[SWITCH] and not (term or trunc) and before["vitals"][WEAPON] == obs["vitals"][WEAPON]
        ) else 0.
        terms = (
            reward, self.exit_bonus * s["exit_reached"], .035 * new_cell, profile.key * keys,
            .15 * len(distinct), profile.unlock * sum(locks[i] > 0 for i in distinct),
            -profile.damage_cost * damage, profile.dealt * dealt, profile.healed * healed,
            map_death_adjustment, blind_attack, unchanged_switch,
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
