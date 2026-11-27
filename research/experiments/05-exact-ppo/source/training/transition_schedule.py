"""Matched layout scheduling using measured skill-decision exposure."""

from collections import defaultdict

import numpy as np

from training.credit import FAMILIES, family
from training.storage import project_root


TARGET = dict(zip(FAMILIES, (.30, .25, .30, .15), strict=True))
EXPECTED_LENGTH = dict(zip(FAMILIES, (172., 391., 148., 227.), strict=True))
VARIANTS = ("red_key_backtrack.wad", "train-only-basic-yaw90.wad", "train-only-basic-yaw270.wad")


class OriginalSchedule:
    """The original uniform-layout reset distribution."""

    def __init__(self, pool, rng):
        self.pool = pool
        self.rng = rng

    def draw(self, worker, initial=False):
        layout = self.pool[worker % len(self.pool)] if initial else self.pool[int(self.rng.integers(len(self.pool)))]
        return layout, int(self.rng.integers(2**30))


class TransitionSchedule:
    """At genuine episode resets, forecast how each choice affects skill exposure."""

    def __init__(self, pool, seed, workers, episode_cap):
        path = project_root() / "training-maps"
        layouts = [*pool, *((path / name, "MAP01") for name in VARIANTS)]
        if not all(wad.is_file() for wad, _ in layouts):
            raise ValueError("A fixed training map or variant is missing.")
        self.groups = defaultdict(list)
        for layout in layouts:
            self.groups[family(layout)].append(layout)
        if set(self.groups) != set(FAMILIES):
            raise ValueError("The training pool does not contain all four skill families.")
        self.counts = dict.fromkeys(FAMILIES, 0)
        self.expected = dict(EXPECTED_LENGTH)
        self.workers = [None] * workers
        self.cap = episode_cap
        self.rng = [np.random.default_rng(np.random.SeedSequence([seed, worker])) for worker in range(workers)]

    def observe(self, boundary):
        for worker, ended in enumerate(boundary):
            skill, length = self.workers[worker]
            self.counts[skill] += 1
            length += 1
            if ended:
                self.expected[skill] = .8 * self.expected[skill] + .2 * length
                self.workers[worker] = None
            else:
                self.workers[worker] = (skill, length)

    def draw(self, worker, initial=False):
        if self.workers[worker] is not None:
            raise RuntimeError("A skill may only be reassigned at an episode boundary.")
        if initial:
            skill = FAMILIES[worker % len(FAMILIES)]
        else:
            projected = {name: float(count) for name, count in self.counts.items()}
            for assignment in self.workers:
                if assignment is not None:
                    name, elapsed = assignment
                    projected[name] += max(0., min(self.cap - elapsed, max(32., self.expected[name] - elapsed)))
            total = sum(projected.values())

            def error(candidate):
                next_length = min(self.cap, self.expected[candidate])
                after = total + next_length
                return sum((projected[name] + (next_length if name == candidate else 0.) -
                            TARGET[name] * after) ** 2 / TARGET[name] for name in FAMILIES)

            skill = min(FAMILIES, key=error)
        self.workers[worker] = (skill, 0)
        options = self.groups[skill]
        rng = self.rng[worker]
        return options[int(rng.integers(len(options)))], int(rng.integers(2**30))
