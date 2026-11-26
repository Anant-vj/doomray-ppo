"""Compare local evaluation reports across several predeclared score profiles."""

import argparse
import json
from math import ceil
from pathlib import Path
from statistics import mean


# The competition's scoring coefficients are private. This range tests whether
# a development result depends on one trade-off. Exit count takes priority
# in every profile. The profiles exclude HP/damage and use one death measure.
PROFILES = {
    "balanced": (7.0, 4.0, 1.5, 0.20, 1.00, 2.0),
    "survival-combat": (4.0, 3.0, 1.0, 0.10, 1.50, 5.0),
    "progression": (10.0, 6.0, 2.0, 0.28, 0.75, 1.5),
}
TICKS_PER_DECISION = 4


def load(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    episodes = {
        (row["wad"], row["map"], row["seed"]): row for row in report["episodes"]
    }
    if len(episodes) != len(report["episodes"]) or not episodes:
        raise ValueError(f"Duplicate or missing episode keys in {path}")
    if "headless" not in report or not report["headless"]:
        raise ValueError(f"Use matched headless evaluations: {path}")
    return report, episodes


def secondary(row, weights, decision_cap):
    locked = row["locked_sectors_raised"]
    normal = max(0, row["door_sectors_raised"] - locked)
    safety = 1.0 if row["alive"] else min(1.0, row["steps"] / decision_cap)
    features = (row["keys"], locked, normal, min(100, row["cells_96"]), row["kills"], safety)
    return sum(weight * value for weight, value in zip(weights, features, strict=True))


def name(report):
    checkpoint = report.get("checkpoint_metadata", {})
    update = checkpoint.get("update")
    return (
        f"{Path(report['checkpoint']).stem} (u{update})"
        if update is not None
        else Path(report["checkpoint"]).stem
    )


def rank(reference, paths):
    base, original = load(reference)
    decision_cap = ceil(base["ticks"] / TICKS_PER_DECISION)
    profiles = {label: [secondary(original[k], w, decision_cap) for k in original]
                for label, w in PROFILES.items()}
    lines = [
        "# Matched checkpoint score sensitivity",
        "",
        f"Reference is `{name(base)}`. Each candidate used {len(original)} identical map/seed episodes "
        f"with a {base['ticks']:,}-tick cap. Differences use paired episodes.",
        "",
        "The historical proxy is shown for continuity. Each new profile ranks by total exits first "
        "and then the mean weighted score. The profiles distinguish locked and ordinary door events "
        "and express survival with one bounded term. The competition's weights are private.",
        "",
        "| Policy | Eligible | Exits | Survival | Kills | Keys | Locks | Doors | Cells | Legacy | "
        "Balanced change | Survival-combat change | Progression change | Balanced W/T/L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    entries = []
    for path in [reference, *paths]:
        report, episodes = load(path)
        if report["split"] != base["split"] or report["ticks"] != base["ticks"] or episodes.keys() != original.keys():
            raise ValueError(f"Evaluation suite mismatch in {path}")
        values = {label: [secondary(episodes[k], w, decision_cap) for k in original]
                  for label, w in PROFILES.items()}
        balanced_pairs = list(zip(values["balanced"], profiles["balanced"], strict=True))
        diffs = [(int(episodes[k]["exit"]) - int(original[k]["exit"]), round(a - b, 9))
                 for k, (a, b) in zip(original, balanced_pairs, strict=True)]
        win = sum(d > (0, 0) for d in diffs)
        tie = sum(d == (0, 0) for d in diffs)
        loss = len(diffs) - win - tie
        rows = list(episodes.values())
        totals = report["totals"]
        exits = sum(row["exit"] for row in rows)
        alive = sum(row["alive"] for row in rows)
        valid = report["invalid_actions"] == 0
        delta = [mean(new - old for new, old in zip(values[label], scores, strict=True))
                 for label, scores in profiles.items()]
        entries.append((valid, exits, mean(values["balanced"]), (
            f"| {name(report)} | {'yes' if valid else 'no'} | {exits} | {alive}/{len(rows)} | "
            f"{totals['kills']} | {totals['keys']} | {totals['locked_doors']} | {totals['doors']} | "
            f"{report['mean_per_episode']['cells_96']:.2f} | {report['proxy']:.3f} | "
            f"{delta[0]:+.3f} | {delta[1]:+.3f} | {delta[2]:+.3f} | {win}/{tie}/{loss} |"
        )))
    lines.extend(row for _, _, _, row in sorted(entries, reverse=True))
    lines.extend([
        "",
        "The secondary profile weights, in feature order, are shown below.",
        "",
        "| Profile | Key | Locked door | Ordinary door | Grid cell | Kill | Safety |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(f"| {label} | " + " | ".join(f"{v:g}" for v in weights) + " |"
          for label, weights in PROFILES.items()),
        "",
        "Safety is one for a living finish or tick-limit survivor. For a death it is the share of the tick "
        "budget reached. Grid coverage is capped at 100 cells. The small, shared set of maps gives a noisy "
        "estimate of generalization, so prefer gains across profiles and on both maps.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    print(rank(args.reference, args.reports))


if __name__ == "__main__":
    main()
