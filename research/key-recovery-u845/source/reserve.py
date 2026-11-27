"""Freeze four fresh comparison maps, a disjoint training pool, and the decision rule."""
from collections import deque
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PUB = PROJECT / "experiments/publication-verification-20261127/fresh-public"
SOURCE = PUB / "research/experiments/02-symmetry-kl/source/screening/udmf.py"
layout = runpy.run_path(str(SOURCE))["layout"]
official = runpy.run_path(str(ROOT / "learner/env/doom_env/wad.py"))
inspect_geometry = runpy.run_path(str(PROJECT / "experiments/symmetry-reference-20261127/screening/geometry.py"))["inspect"]
assert not (ROOT / "protocol.json").exists()

# Cell rows follow south-to-north order in UDMF. A narrow door sector blocks the exit.
# Name, row strings, initial heading, column widths, row heights (world units).
reserve = [
    ("west-pocket-return", ["#k#####", "#..####", "..P..DX"], 0,
     [106, 84, 105, 92, 104, 32, 109], [83, 75, 106]),
    ("south-spur-return", ["k.#####", "#..####", "##P..##", "####.DX"], 18,
     [92, 74, 106, 88, 116, 32, 98], [86, 94, 107, 89]),
    ("forward-key-gallery", ["###..##", "P.k..DX"], 8,
     [81, 75, 93, 83, 96, 32, 109], [72, 94]),
    ("guarded-court", ["#.z.###", "P..k.DX", "##..z##"], 0,
     [101, 83, 117, 93, 105, 32, 91], [94, 112, 87]),
]
drills = [
    ("reverse-short", ["kPDX"], 0, [78, 88, 32, 92], [94]),
    ("reverse-wide", ["k.PDX"], 32, [74, 72, 88, 32, 88], [126]),
    ("reverse-mirror", ["XD.Pk"], 166, [87, 32, 76, 89, 82], [102]),
    ("reverse-north", ["X", "D", "P", ".", "k"], 270, [101], [86, 32, 80, 72, 79]),
    ("oblique-south", ["k", "P", "D", "X"], 54, [89], [74, 85, 32, 84]),
    ("side-card", ["k###", ".PDX"], 352, [79, 98, 32, 85], [72, 92]),
    ("opposite-card", ["XDP.", "###k"], 202, [91, 32, 84, 77], [95, 84]),
    ("forward-retention", ["Pk.DX"], 343, [85, 76, 82, 32, 89], [88]),
]


def reachable(rows, walk_through_door):
    cells = {(x, y): c for y, row in enumerate(rows) for x, c in enumerate(row)
             if c != "#" and (walk_through_door or c != "D")}
    start = next(p for p, c in cells.items() if c == "P")
    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if p in cells and p not in visited:
                queue.append(p)
                visited.add(p)
    return {cells[p] for p in visited}


known = set()
files = list((PUB / "research/experiments").glob("*/evaluation/*.jsonl"))
files.append(PROJECT / "submission_artifacts/final-blind-assessment/manifests/map_manifest.json")
for file in files:
    known.update(re.findall(r'"geometry_sha256"\s*:\s*"([a-f0-9]{64})"', file.read_text(encoding="utf-8-sig")))
maps = {}
held_shapes = set()
for group, cases in (("reserved", reserve), ("drills", drills)):
    entries = []
    for name, rows, angle, xs, ys in cases:
        assert len(xs) == len(rows[0]) and len(ys) == len(rows)
        assert "k" in reachable(rows, False) and "X" not in reachable(rows, False)
        assert "X" in reachable(rows, True)
        path = ROOT / group / (name + ".wad")
        layout(path, rows, angle, dict(enumerate(xs)), dict(enumerate(ys)), official)
        fingerprint = inspect_geometry(path, "MAP01")["geometry_sha256"]
        assert fingerprint not in held_shapes
        if group == "reserved":
            assert fingerprint not in known
            held_shapes.add(fingerprint)
        entries.append(dict(name=name, file=str(path.relative_to(ROOT)), marker="MAP01",
                            sha256=sha256(path.read_bytes()).hexdigest(), geometry_sha256=fingerprint))
    maps[group] = entries
protocol = dict(
    created=datetime.now().astimezone().isoformat(), start="2026-11-27T07:54:09+05:30",
    hard_training_deadline="2026-11-27T08:44:09+05:30",
    official_base="6c2d5499f973c77b4a239f778837280edf64e57b",
    source_weight_sha256="0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61",
    known_published_geometry_fingerprints_compared=len(known),
    evaluation=dict(engine_seeds=[81431, 81437], policy_seeds=[92561, 92569], pairs="zip",
                    tick_cap=2100, skill=3, early_death="Confirmed death at or before 525 elapsed game ticks (first quarter).",
                    target=["west-pocket-return", "south-spur-return"],
                    retention=["forward-key-gallery", "guarded-court"],
                    gate="Require at least one additional successful exit or key pickup on each recovery layout. "
                         "Reject any loss of a baseline exit, survival, key or unlock on the same map/seed, "
                         "any extra early death, any invalid action, or a failed validator. No rerolls."),
    training=dict(updates=96, workers=4, horizon=64, transitions=24576, lr=0.00002, seed=83453,
                  ticks=448, sampler="Original six categorical heads", history="Archived exact raw four",
                  selection="Only the final update 845; retain baseline on interrupted or incomplete run.",
                  reward="Original progression reward profile, unchanged."),
    maps=maps)
with (ROOT / "protocol.json").open("x", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2)
print("Reserved four layouts and eight disjoint drills. Published geometry fingerprints checked:", len(known))
