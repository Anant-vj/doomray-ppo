"""Read-only map inventory and family checks. It imports no game or policy runtime."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from functools import cache
from math import gcd, hypot
from pathlib import Path
import re
import runpy
import struct


ASSESSMENT = Path(__file__).resolve().parents[3] / "submission_artifacts/final-blind-assessment"
ARTIFACTS = ASSESSMENT.parent
PROJECT = ARTIFACTS.parent
RELEASE = ARTIFACTS / "github-fresh-6c2d549-20261127-022250050"
OFFICIAL_WAD = runpy.run_path(str(RELEASE / "env/doom_env/wad.py"))
ASSETS = PROJECT / "environments/competition/Lib/site-packages/vizdoom"
FREEDOOM = ASSETS / "freedoom2.wad"
FREEDOOM_SHA = "a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b"
TIERS = {1: (18, 19, 21, 26), 2: (17, 20, 23, 32), 3: (16, 22, 24, 29), 4: (15, 25, 27, 28)}
SEEDS = (9101, 9102)
KEYS = {5: "blue", 6: "yellow", 13: "red", 38: "red", 39: "yellow", 40: "blue"}
DOOM_EXIT_SPECIALS = {11: ("use", "normal"), 51: ("use", "secret"), 52: ("cross", "normal"),
                      124: ("cross", "secret"), 197: ("gun", "normal"), 198: ("gun", "secret")}
MONSTERS = {7, 9, 16, 58, 64, 65, 66, 67, 68, 69, 71, 84, 3001, 3002, 3003, 3004, 3005, 3006}
RESOURCES = {17, 82, 83, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2010, 2011, 2012,
             2013, 2014, 2015, 2018, 2019, 2046, 2047, 2048, 2049}


def hash_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


@cache
def wad_file(path):
    return OFFICIAL_WAD["read_wad"](path)


def binary(data):
    vertices = list(struct.iter_unpack("<hh", data["VERTEXES"]))
    sides = [values[-1] for values in struct.iter_unpack("<hh8s8s8sH", data["SIDEDEFS"])]
    hexen = "BEHAVIOR" in data
    fmt = "<HHH6BHH" if hexen else "<7H"
    lines = [(v[0], v[1], sides[v[-2]] if v[-2] != 65535 else -1,
              sides[v[-1]] if v[-1] != 65535 else -1, bool(v[2] & 1))
             for v in struct.iter_unpack(fmt, data["LINEDEFS"])]
    things = [] if hexen else [(x, y, kind) for x, y, _, kind, flags in struct.iter_unpack("<hhHHH", data["THINGS"])
                               if kind == 1 or (flags & 2 and not flags & 16)]
    return vertices, lines, len(data["SECTORS"]) // 26, things, "hexen" if hexen else "doom"


def udmf(data):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", data["TEXTMAP"].decode("utf-8"), flags=re.S)
    blocks = {}
    for kind, body in re.findall(r"(vertex|sector|sidedef|linedef|thing)\s*\{([^{}]*)\}", text):
        blocks.setdefault(kind, []).append(dict(re.findall(r"([A-Za-z0-9_]+)\s*=\s*([^;]+?)\s*;", body)))
    vertices = [(round(float(v["x"]) * 1024), round(float(v["y"]) * 1024)) for v in blocks["vertex"]]
    sides = [int(v["sector"]) for v in blocks["sidedef"]]
    lines = [(int(v["v1"]), int(v["v2"]), sides[int(v["sidefront"])],
              sides[int(v["sideback"])] if int(v.get("sideback", -1)) >= 0 else -1,
              v.get("blocking") == "true") for v in blocks["linedef"]]
    return vertices, lines, len(blocks["sector"]), [], "udmf"


def shape_hash(vertices, lines):
    """Normalize translation, uniform scale, axis rotations, reflection, and line ordering."""
    origin = vertices[0]
    vertices = [(x - origin[0], y - origin[1]) for x, y in vertices]
    scale = 0
    for x, y in vertices:
        scale = gcd(scale, gcd(x, y))
    if scale == 0:
        raise ValueError("Map has no nonzero geometry")
    hashes = []
    for a, b in ((0, 1), (1, 0)):
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            points = [(sx * p[a] // scale, sy * p[b] // scale) for p in vertices]
            min_x = min(x for x, _ in points)
            min_y = min(y for _, y in points)
            points = [(x - min_x, y - min_y) for x, y in points]
            segments = [(*sorted((points[i], points[j])), back >= 0, block) for i, j, _, back, block in lines]
            hashes.append(hash_json(sorted(segments)))
    return min(hashes)


def sector_graph(lines, sectors):
    neighbors = [set() for _ in range(sectors)]
    for _, _, a, b, _ in lines:
        if a >= 0 and b >= 0 and a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    labels = [str(len(ns)) for ns in neighbors]
    for _ in range(8):
        labels = [hash_json([labels[i], sorted(labels[j] for j in ns)]) for i, ns in enumerate(neighbors)]
    unseen, components = set(range(sectors)), 0
    while unseen:
        components += 1
        queue = [unseen.pop()]
        while queue:
            linked = neighbors[queue.pop()] & unseen
            queue.extend(linked)
            unseen.difference_update(linked)
    edges = sum(map(len, neighbors)) // 2
    return hash_json(sorted(labels)), edges, components, edges - sectors + components


def inspect(path, map_name):
    data = OFFICIAL_WAD["map_lumps"](wad_file(path), map_name)
    vertices, lines, sectors, things, fmt = udmf(data) if "TEXTMAP" in data else binary(data)
    graph, edges, components, cycles = sector_graph(lines, sectors)
    starts = [(x, y) for x, y, kind in things if kind == 1]
    enemies = [(x, y) for x, y, kind in things if kind in MONSTERS]
    near_monsters = (sum(hypot(x - starts[0][0], y - starts[0][1]) <= 768 for x, y in enemies)
                     if len(starts) == 1 else None)
    exit_triggers = []
    if fmt == "doom":
        for index, (v1, v2, _, special, _, _, _) in enumerate(struct.iter_unpack("<7H", data["LINEDEFS"])):
            if special in DOOM_EXIT_SPECIALS:
                activation, destination = DOOM_EXIT_SPECIALS[special]
                exit_triggers.append(dict(linedef=index, special=special, activation=activation,
                                          destination=destination, from_xy=vertices[v1], to_xy=vertices[v2]))
    return dict(geometry_sha256=shape_hash(vertices, lines), sector_graph_sha256=graph, format=fmt,
                sectors=sectors, lines=len(lines), graph_edges=edges, graph_components=components,
                graph_cycle_rank=cycles, key_colors=sorted({KEYS[t] for _, _, t in things if t in KEYS}),
                player_starts=len(starts) if things else None, placed_monsters_skill3=len(enemies) if things else None,
                placed_monsters_within_768_euclidean=near_monsters,
                placed_resources_skill3=sum(t in RESOURCES for _, _, t in things) if things else None,
                static_exit_triggers=exit_triggers)


def run():
    previous_static = PROJECT / "evaluations/phase2-map-audit.json"
    previous = {r["map"]: r for r in json.loads(previous_static.read_text(encoding="utf-8-sig"))
                if r.get("wad") == "freedoom2.wad" and r.get("split") == "freedoom_candidate"}
    excluded_freedoom = {**{i: "training" for i in range(1, 7)}, **{i: "development" for i in (7, 8, 13, 14)},
                          **{i: "diagnostic" for i in (9, 10)}, **{i: "completed prior holdout" for i in (11, 12)},
                          30: "boss ending without a standard exit-lump signature", 31: "training curriculum"}
    known = [(FREEDOOM, f"MAP{i:02}", category) for i, category in sorted(excluded_freedoom.items())]
    known += [(ASSETS / f"scenarios/{name}.wad", "MAP01", "training scenario")
              for name in ("basic", "my_way_home", "defend_the_center", "health_gathering", "deadly_corridor")]
    known += [(RELEASE / "maps/two_rooms.wad", "MAP01", "training and official validator")]
    known += [(p, "MAP01", "official fixture and potential prior runtime exposure")
              for p in sorted((RELEASE / "tests/fixtures").glob("*.wad"))]
    known += [(p, "MAP01", "prior generated diagnostic")
              for p in sorted((ARTIFACTS / "final-functional-audit").glob("*/layouts/*.wad"))]
    known += [(p, "MAP01", "prior spawn-altered training curriculum")
              for p in sorted((PROJECT / "maps/curriculum").glob("*.wad"))]
    known_entries = []
    for path, marker, reason in known:
        info = inspect(path, marker)
        known_entries.append(dict(wad=path.as_posix(), map=marker, reason=reason,
                                  geometry_sha256=info["geometry_sha256"],
                                  sector_graph_sha256=info["sector_graph_sha256"]))
    ash = hashlib.sha256(FREEDOOM.read_bytes()).hexdigest()
    if ash != FREEDOOM_SHA:
        raise ValueError(f"Bundled Freedoom identity changed: {ash}")
    maps = []
    for tier, numbers in TIERS.items():
        for number in numbers:
            marker = f"MAP{number:02}"
            old = previous[marker]
            static = inspect(FREEDOOM, marker)
            if old.get("wrapper_reset") or not old["standard_exit_special"] or static["player_starts"] != 1:
                raise ValueError(f"Previous exposure, missing known exit, or ambiguous start: {marker}")
            if not any(e["destination"] == "normal" for e in static["static_exit_triggers"]):
                raise ValueError(f"No conventional normal exit trigger: {marker}")
            if (static["sectors"], static["lines"]) != (old["sectors"], old["lines"]):
                raise ValueError(f"Current geometry inventory disagrees with the prior static audit: {marker}")
            maps.append(dict(id=marker, wad=FREEDOOM.as_posix(), map=marker, tier=tier, skill=3,
                             tick_cap=2100 if tier <= 2 else 4200,
                             family=f"freedoom2-{marker.lower()}-{static['sector_graph_sha256'][:12]}",
                             wad_sha256=ash, seeds=SEEDS, verification="PASSED_STATIC_ENGINE_PENDING", **static,
                             ordinary_door_sectors=old["unlocked_doors"], locked_door_sectors=old["locked_doors"],
                             static_standard_exit_line=old["standard_exit_special"],
                             phase2_layout_sha256=old["layout_sha256"]))
    for prop in ("geometry_sha256", "sector_graph_sha256"):
        counts = Counter(m[prop] for m in maps)
        prior = {m[prop] for m in known_entries}
        if len(counts) != len(maps) or counts.keys() & prior:
            raise ValueError(f"Candidate family collision on {prop}")
    return dict(
        predeclared_at_local=datetime.now().astimezone().isoformat(timespec="seconds"),
        scope="New local competitive-style assessment on previously unplayed layouts in reviewed project records.",
        provenance=dict(
            prior_sealed_suite="No independently sealed DoomRay suite was found in the inspected local inventory.",
            candidate_status="The training split and available training, development, holdout and diagnostic records do not report candidate gameplay. Earlier static audit rows exist, with no wrapper reset recorded.",
            limitations="Project records are not an exhaustive history of activity outside this workspace. Geometry and graph fingerprints are static and establish no engine reachability or difficulty outcomes.",
            previous_static_catalog=previous_static.as_posix(),
            prior_split=(PROJECT / "evaluations/mission-split-lock.md").as_posix(),
            prior_runtime_source=(PROJECT / "workspace/devforge-doom-template-main/training/game.py").as_posix()),
        tier_definitions={
            1: "Combat and Survival: 12–26 placed monsters within 768 map units of spawn by straight-line distance. Measure survival, damage and real combat. Distance is not line-of-sight evidence.",
            2: "Multi-Room Exploration: 146–539 sectors, static sector-graph cycle ranks 41–193 and 2–13 nearby placed monsters. Focus on exploration and doors. Keys and combat can still occur.",
            3: "Long-Horizon Progression: at least four detected locked door sectors and two or more placed key colors. Record key, unlock and exit behavior. The required order of locks is unverified.",
            4: "Full Gauntlet: 503–777 sectors, 122–194 placed Skill 3 monsters, all three placed key colors and three detected locks. Resource scarcity and required backtracking are unverified."},
        episode_order="Tier order, map order as declared below, seed order, then C749, Stack-25-174 and Stack-26-621. Same engine and policy randomness seed. Preserve all attempted outcomes.",
        stages={
            "minimum": dict(status="PREDECLARED_PENDING_ENGINE_VERIFICATION", ordered_map_ids=[m["id"] for m in maps],
                            layout_count=16, seeds=SEEDS, policies=3, planned_episodes=96),
            "expanded": dict(status="NOT_PREDECLARED_NO_VERIFIED_ADDITIONAL_LAYOUTS", target_layouts=24, target_episodes=216),
            "full": dict(status="NOT_PREDECLARED_NO_VERIFIED_ADDITIONAL_LAYOUTS", target_layouts=48, target_episodes=432)},
        official_engine_verified_layouts=0,
        geometry_method="Axis rotation, reflection, translation and uniform scale normalized SHA-256 of raw line endpoints, sidedness and blocking. An eight-round sector adjacency graph fingerprint provides a second screen for re-indexing, minor wall/scale edits, and spawn variants. Both exclude things. These checks cannot exclude every transformation.",
        tick_cap_basis="Predeclared local horizons of 2100 ticks for the first two tiers and 4200 for progression and gauntlet. The organizer's private map limits are unknown.",
        maps=maps,
        excluded_layouts=known_entries)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
