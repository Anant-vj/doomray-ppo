"""Freeze four new topologies and matched settings without loading the game."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import struct

import geometry
from udmf import layout


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
PROJECT = ROOT.parents[2]
ARTIFACTS = PROJECT / "submission_artifacts"
EARLIER = PROJECT / "experiments/final-competitive-20261127"
FROZEN_REPO = ARTIFACTS / "github-fresh-6c2d549-20261127-022250050"
SEEDS = (58231, 58232)
RELEASE_COMMIT = "6c2d5499f973c77b4a239f778837280edf64e57b"
C749_HASH = "0b8f290cfe36a25f8593da925d4a4a0ddb8790b37f9319a019aed276a73f5e61"
SCORE_HASH = "1ca7aeb19af8b5a787bd66cd86234690764689c5979876fdcb04a0b8f9bfc9fd"

# Each form has different sector connectivity as well as a different fixed spawn/route.
LAYOUTS = (
    ("north-dogleg", 2, 920, 0, ("#P.####", "##.##X#", "##.##.#", "#.....#", "###.###"), {}, {}),
    ("south-dogleg", 2, 980, 0, ("##.X###", "##.####", "##....#", "#####.#", "..P...#", "##..###"), {}, {}),
    ("deadend-key-retrace", 3, 1040, 0, ("#######X", "#######D", "##....P.", "...#..##", "k..##.##"), {}, {1: 32}),
    ("offset-single-enemy", 1, 760, 32, ("##.##", "##.z#", "P...#", "#.###", "#...X"), {}, {}),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reference_maps():
    """Use an explicit allowlist that excludes the prior withheld WAD directory."""
    folders = (PROJECT / "maps", FROZEN_REPO / "maps", FROZEN_REPO / "tests/fixtures",
               geometry.ASSETS / "scenarios", ARTIFACTS / "final-functional-audit",
               EARLIER / "screening_maps", EARLIER / "maps", EXPERIMENT / "maps")
    files = {geometry.FREEDOOM, *(path for folder in folders for path in folder.rglob("*.wad"))}
    shapes, graphs, count, issues = set(), set(), 0, []
    for path in sorted(files):
        names = geometry.OFFICIAL_WAD["list_maps"](geometry.wad_file(path))
        for name in names:
            if name in {"MAP11", "MAP12"}:
                continue
            try:
                result = geometry.inspect(path, name)
            except (OSError, KeyError, ValueError, TypeError, IndexError, struct.error) as exc:
                issues.append([str(path), name, type(exc).__name__])
                continue
            shapes.add(result["geometry_sha256"])
            graphs.add(result["sector_graph_sha256"])
            count += 1
    return shapes, graphs, count, issues


def main():
    protocol = ROOT / "manifests/protocol.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    baseline = FROZEN_REPO / "submission/weights/policy.pt"
    scorer = ROOT / "manifests/scoring_manifest.json"
    if digest(baseline) != C749_HASH or digest(scorer) != SCORE_HASH:
        raise ValueError("Archived control or preregistered scores changed")

    shapes, graphs, count, issues = reference_maps()
    maps = []
    for name, tier, ticks, angle, rows, x_overrides, y_overrides in LAYOUTS:
        path = ROOT / "maps" / f"{name}.wad"
        exits = layout(path, rows, angle, x_overrides, y_overrides, geometry.OFFICIAL_WAD)
        facts = geometry.inspect(path, "MAP01")
        shape, graph = facts["geometry_sha256"], facts["sector_graph_sha256"]
        if shape in shapes or graph in graphs or facts["graph_components"] != 1:
            raise ValueError(f"Static novelty or connectivity check failed for {name}")
        shapes.add(shape)
        graphs.add(graph)
        maps.append({**facts, "id": name, "map": "MAP01", "family": f"local-{name}-{graph[:12]}",
                     "tier": tier, "skill": 3, "seeds": SEEDS, "tick_cap": ticks, "wad": str(path),
                     "wad_sha256": digest(path), "static_exit_triggers": exits,
                     "scope": "New static topology with engine solvability pending"})

    data = {
        "frozen_at_ist": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "Fresh synthetic local matched screening; no old hidden/withheld cases.",
        "screening_maps": maps,
        "prior_static_reference_count": count,
        "prior_static_issues": issues,
        "novelty_check": "Different rotation/reflection/scale-normalized line geometry and sector-graph hashes versus every readable prior reference and across this suite. Previous sealed folder and MAP11/MAP12 excluded.",
        "order": "Listed map order, listed identical engine/policy seeds, then exact frozen roster order. One attempt per identity.",
        "scoring_manifest_sha256": SCORE_HASH,
        "expected_episodes": len(maps) * len(SEEDS) * 3,
        "roster": [
            {"id": "C749", "source_commit": RELEASE_COMMIT, "checkout": str(FROZEN_REPO),
             "weight_source": str(baseline), "expected_sha256": C749_HASH},
            {"id": "PPO-control", "source_commit": RELEASE_COMMIT, "checkout": str(FROZEN_REPO),
             "weight_source": str(EXPERIMENT / "checkpoints/ppo-control/resume.pt")},
            {"id": "Symmetry-KL", "source_commit": RELEASE_COMMIT, "checkout": str(FROZEN_REPO),
             "weight_source": str(EXPERIMENT / "checkpoints/symmetry-kl/resume.pt")},
        ],
        "promotion_gate": {
            "legal": "Eight completed legal episodes per model, paired with C749; zero invalid actions or timeouts.",
            "preserve": "Match or exceed C749's exits and survivals, with neither reduced on any family. No net loss of measured player damage, acquired keys or verified unlocks.",
            "material_outcome": "On two independent families improve a measured outcome: exits, keys, unlocks, player-attributed damage, or at least one extra visited 96-unit cell together with a lower blocked-move fraction. Lower directional bias alone is diagnostic.",
            "decision": "If the behavior or safety criterion fails, retain C749. Even passing is local screening evidence; preserve public main and require separate validation and explicit approval for any replacement.",
        },
    }
    with protocol.open("xb") as output:
        output.write((json.dumps(data, indent=2) + "\n").encode())
    print(json.dumps({"protocol": str(protocol), "sha256": digest(protocol), "prior_reference_layouts": count,
                      "new_layouts": len(maps), "prior_static_issues": issues}, indent=2))


if __name__ == "__main__":
    main()
