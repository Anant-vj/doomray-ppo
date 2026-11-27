"""Training-only reward accounting and matched skill exposure."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import vizdoom

from doom_env import spaces
from training.game import Episode, REWARD_TERMS
from training.storage import project_root


GAMMA = .99
FAMILIES = ("combat", "navigation", "progression", "mixed-survival")
REWARDS = ("native_exit", "native_death", "native_map_kills", *REWARD_TERMS[1:], "potential")
MEASUREMENTS = (
    "player_hits", "player_damage", "damage_received", "health_gain_lower_bound",
    "attack_held", "attacks_without_enemy_cues", "ammo_net_drop", "enemy_visible",
    "enemy_alignment_abs_rad",
)
COLUMNS = REWARDS + MEASUREMENTS
PLAYER_FIELDS = tuple(getattr(vizdoom.GameVariable, name)
                      for name in ("HITCOUNT", "DAMAGECOUNT", "AMMO2", "AMMO3", "AMMO5", "AMMO6"))
ENEMIES = [i for i, name in enumerate(spaces.ENTITY_TYPES) if name.startswith("enemy_")]
ENEMY_RAY = spaces.RAY_HIT_TYPES.index("enemy")


def family(layout):
    path, name = layout
    if path.name in ("basic.wad", "defend_the_center.wad", "deadly_corridor.wad"):
        return "combat"
    if path.name in ("e7ab54fb6834ffa3.wad", "red_key_backtrack.wad"):
        return "progression"
    if path.name in ("two_rooms.wad", "my_way_home.wad") or (
            path.name == "freedoom2.wad" and name in ("MAP01", "MAP02")):
        return "navigation"
    return "mixed-survival"


class SkillSchedule:
    def __init__(self, pool, seed, workers):
        backtrack = project_root() / "training-maps/red_key_backtrack.wad"
        pool = [*pool, (backtrack, "MAP01")]
        groups = defaultdict(list)
        for layout in pool:
            groups[family(layout)].append(layout)
        if not backtrack.is_file() or set(groups) != set(FAMILIES):
            raise ValueError("Balanced training pool does not contain the frozen four skill families.")
        self.groups = groups
        self.rngs = [np.random.default_rng(np.random.SeedSequence([seed, i])) for i in range(workers)]

    def draw(self, worker, initial=False):
        rng = self.rngs[worker]
        skill = FAMILIES[worker % len(FAMILIES)] if initial else FAMILIES[int(rng.integers(len(FAMILIES)))]
        pool = self.groups[skill]
        layout = pool[int(rng.integers(len(pool)))]
        return layout, int(rng.integers(2**30))


class MeasuredEpisode(Episode):
    def __init__(self, layout, ticks, *, render_buffers, reward_profile, credit_mode):
        super().__init__(layout, ticks, render_buffers=render_buffers, reward_profile=reward_profile)
        self.potential = None
        if credit_mode in ("pbrs", "pbrs-anchor"):
            from training.potential import make_potential

            self.potential = make_potential(self.env, self.path, self.map_name)

    def reset(self, seed):
        observation = super().reset(seed)
        self.initial_potential = self.potential.reset(self.last) if self.potential is not None else 0.
        self.current_observation = observation
        self.counters = self.read_counters()
        self.credit_total = np.zeros(len(COLUMNS), dtype=np.float64)
        self.discounted_reward = np.zeros(len(REWARDS), dtype=np.float64)
        return observation

    def read_counters(self):
        game = self.env._game
        return np.array([game.get_game_variable(variable) for variable in PLAYER_FIELDS], dtype=np.float64)

    def step(self, action):
        before = self.last
        previous_terms = self.reward_terms.copy()
        previous_observation = self.current_observation
        observation, reward, terminated, truncated = super().step(action)
        terms = self.reward_terms - previous_terms
        native_exit = float(self.last["exit_reached"])
        native_death = -float(self.last["dead"])
        map_kills = .1 * (self.last["kills"] - before["kills"])
        if abs(terms[0] - native_exit - native_death - map_kills) > 1e-7:
            raise RuntimeError("Training reward ledger does not match the native reward.")
        shaping = 0.
        if self.potential is not None:
            shaping = self.potential.step(self.last, terminated=terminated, truncated=truncated)
        counters = self.read_counters()
        hits, damage = np.maximum(counters[:2] - self.counters[:2], 0.)
        ammo = float(np.maximum(self.counters[2:] - counters[2:], 0.).sum())
        self.counters = counters
        visible = previous_observation["visible_entities"]
        enemies = np.flatnonzero(visible["mask"] & np.isin(visible["entity_type"], ENEMIES))
        enemy_cue = bool(len(enemies)) or bool(np.any(previous_observation["rays"]["hit_type"] == ENEMY_RAY))
        alignment = 0.
        if len(enemies):
            sin, cos = visible["features"][enemies[0], 1:3]
            alignment = abs(float(np.arctan2(sin, cos)))
        row = np.array((
            native_exit, native_death, map_kills, *terms[1:], shaping,
            hits, damage, max(0., self.last["damage_taken"] - before["damage_taken"]),
            max(0., self.last["health"] - before["health"]), action[6], action[6] and not enemy_cue,
            ammo, bool(len(enemies)), alignment,
        ), dtype=np.float64)
        self.credit_total += row
        self.discounted_reward += (GAMMA ** (self.steps - 1)) * row[:len(REWARDS)]
        self.current_observation = observation
        self.credit_row = row
        return observation, reward + shaping, terminated, truncated

    def metrics(self):
        result = super().metrics()
        result["skill"] = family((self.path, self.map_name))
        result["credit_totals"] = dict(zip(COLUMNS, self.credit_total.tolist(), strict=True))
        result["discounted_reward"] = dict(zip(REWARDS, self.discounted_reward.tolist(), strict=True))
        result["potential_enabled_on_map"] = self.potential is not None
        if self.potential is not None:
            residual = self.discounted_reward[REWARDS.index("potential")] + self.initial_potential
            if abs(residual) > 1e-9:
                raise RuntimeError("Live terminal potential did not telescope.")
            result["initial_potential"] = self.initial_potential
            result["terminal_potential_residual"] = float(residual)
        return result
