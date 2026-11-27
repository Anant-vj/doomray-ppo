"""Aggregate frozen, paired DoomRay blind examination episode records."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
from math import isfinite
from pathlib import Path
import random
from statistics import fmean


BASE = Path(__file__).resolve().parents[1]
MODELS = ("C749", "Stack25-u174", "Stack26-u621")
UNIT_FIELDS = ("tier", "map_id", "layout_family", "engine_seed", "policy_seed")
WEIGHTS = {
    "balanced": (7.0, 4.0, 1.5, 0.20, 1.00, 2.0),
    "survival-combat": (4.0, 3.0, 1.0, 0.10, 1.50, 5.0),
    "progression": (10.0, 6.0, 2.0, 0.28, 0.75, 1.5),
}
PROFILES = ("historical", *WEIGHTS)
METRICS = (
    "keys", "unlocks", "doors", "explored_cells_96", "engine_kills", "health",
    "damage_received", "recorded_player_hits", "recorded_player_damage",
)
REQUIRED_METRICS = METRICS[:7]
RAW_OUTCOMES = ("exit", "alive", "decisions", *METRICS)
MIN_LAYOUTS = 16
MIN_SEED_PAIRS = 2
BOOTSTRAP_RESAMPLES = 4096
BOOTSTRAP_SEED = 20261127


def episode_scores(row, tick_cap):
    """Compute the frozen unofficial scores for one completed episode."""
    if isinstance(tick_cap, bool) or not isinstance(tick_cap, int) or tick_cap <= 0:
        raise ValueError("tick_cap must be a positive number of engine ticks")
    m = row["metrics"]
    cells = min(m["explored_cells_96"], 100)
    keys = m["keys"]
    unlocks = m["unlocks"]
    doors = m["doors"]
    kills = m["engine_kills"]
    safety = 1.0 if row["alive"] else min(1.0, row["decisions"] / ((tick_cap + 3) // 4))
    features = (keys, unlocks, max(0, doors - unlocks), cells, kills, safety)
    return {
        "historical": (100 * row["exit"] + 8 * keys + 4 * unlocks + 2 * doors
                       + 0.2 * cells + kills + 0.02 * m["health"] - 0.02 * m["damage_received"]),
        **{name: sum(weight * value for weight, value in zip(weights, features, strict=True))
           for name, weights in WEIGHTS.items()},
    }


def number(value):
    return type(value) in (int, float, bool) and isfinite(value)


def mean(values):
    return fmean(values) if values else None


def unit(row):
    return int(row["tier"]), row["map_id"], row["layout_family"], int(row["engine_seed"]), int(row["policy_seed"])


def frozen_units(manifest):
    """Read the planned minimum tier/map/seed schedule and engine settings."""
    stage = manifest["stages"]["minimum"]
    selected = set(stage["ordered_map_ids"])
    planned = {}
    for entry in manifest["maps"]:
        if entry["id"] not in selected:
            continue
        for seed in entry["seeds"]:
            key = (int(entry["tier"]), entry["id"], entry["family"], seed, seed)
            if key in planned:
                raise ValueError(f"Duplicate planned matchup {key}")
            planned[key] = entry
    if len(planned) * len(MODELS) != stage["planned_episodes"]:
        raise ValueError("Frozen matchup count does not match the planned episode count")
    return planned


def inspect_row(row, plan):
    """Keep the source status and independently check comparability."""
    if row.get("status") != "completed":
        return str(row.get("status") or "missing-status"), None
    m = row.get("metrics")
    scores = row.get("scores")
    if not isinstance(m, dict) or not isinstance(scores, dict):
        return "missing-completed-metrics-or-scores", None
    if (any(row.get(name) not in (0, 1) for name in ("exit", "alive"))
            or type(row.get("decisions")) is not int or row["decisions"] < 0
            or any(not number(m.get(name)) for name in REQUIRED_METRICS)
            or any(not number(scores.get(name)) for name in PROFILES)):
        return "invalid-completed-metrics-or-scores", None
    tick_cap = row.get("tick_cap", plan.get("tick_cap"))
    if (tick_cap is None or tick_cap != plan.get("tick_cap", tick_cap)
            or row.get("wad_sha256", plan.get("wad_sha256")) != plan.get("wad_sha256")
            or row.get("skill", plan.get("skill")) != plan.get("skill")):
        return "frozen-setting-mismatch", None
    try:
        expected_scores = episode_scores(row, tick_cap)
    except ValueError:
        return "invalid-tick-cap", None
    if any(abs(scores[name] - expected_scores[name]) > 1e-8 for name in PROFILES):
        return "frozen-score-mismatch", None
    reliability = row.get("reliability") or {}
    invalid = row.get("invalid_actions", m.get("invalid_actions", reliability.get("invalid_actions", 0)))
    if not number(invalid) or invalid != 0:
        return "invalid-actions", None
    return "completed", row


def index_attempts(rows, planned):
    attempt_index = {model: defaultdict(list) for model in MODELS}
    issues = []
    for line, row in rows:
        try:
            model = row["model"]
            key = unit(row)
            hash(key)
        except (KeyError, TypeError, ValueError):
            issues.append({"line": line, "reason": "missing-or-invalid-attempt-identity"})
            continue
        if model not in MODELS:
            issues.append({"line": line, "reason": "unexpected-model", "model": model})
            continue
        attempt_index[model][key].append((line, row))

    completed = {model: {} for model in MODELS}
    status = {model: {} for model in MODELS}
    for model, attempts in attempt_index.items():
        for key, records in attempts.items():
            if key not in planned:
                state, result = "unplanned-matchup", None
            elif len(records) != 1:
                state, result = "duplicate-attempt-key", None
            else:
                state, result = inspect_row(records[0][1], planned[key])
            status[model][key] = state
            if result is not None:
                completed[model][key] = result
                continue
            issues.append({
                "model": model, "unit": dict(zip(UNIT_FIELDS, key, strict=True)), "reason": state,
                "rows": [{"line": line, "status": row.get("status"), "stage": row.get("stage"),
                          "error": row.get("error")}
                         for line, row in records],
            })
    return attempt_index, completed, status, issues


def layout_groups(keys):
    grouped = defaultdict(set)
    for key in keys:
        grouped[key[2]].add(key)
    return grouped


def complete_layouts(keys, planned):
    return {family: sorted(units) for family, units in sorted(layout_groups(planned).items())
            if len({key[3:] for key in units}) >= MIN_SEED_PAIRS and units <= keys}


def metric(row, name):
    if name in ("exit", "alive", "decisions"):
        return row.get(name)
    if name in PROFILES:
        return row["scores"].get(name)
    return row["metrics"].get(name)


def model_totals(planned, attempts, completed, statuses):
    rows = list(completed.values())
    valid = len(rows)
    raw_completed = sum(row.get("status") == "completed" for records in attempts.values() for _, row in records)
    counts = {name: sum(bool(row[name]) for row in rows) for name in ("exit", "alive")}
    metrics = {}
    for name in METRICS:
        values = [metric(row, name) for row in rows if number(metric(row, name))]
        metrics[name] = {"measured": len(values), "total": sum(values) if values else None, "mean": mean(values)}
    failed_statuses = Counter(state for state in statuses.values() if state != "completed")
    return {
        "planned": len(planned), "attempted": sum(map(len, attempts.values())),
        "attempted_unique": len(attempts), "completed": valid, "recorded_completed": raw_completed,
        "unattempted": len(planned.keys() - attempts.keys()), "failed": sum(failed_statuses.values()),
        "failure_status_counts": dict(sorted(failed_statuses.items())),
        "attempted_layouts": len(layout_groups(attempts)), "completed_layouts": len(layout_groups(completed)),
        "fully_completed_layouts": len(complete_layouts(completed.keys(), planned)),
        "exits": counts["exit"], "survived": counts["alive"], "deaths": valid - counts["alive"],
        "exit_rate": counts["exit"] / valid if valid else None,
        "survival_rate": counts["alive"] / valid if valid else None,
        "mean_decisions": mean([row["decisions"] for row in rows]),
        "metrics": metrics, "scores": {name: mean([metric(row, name) for row in rows]) for name in PROFILES},
    }


def percentile(ordered, probability):
    offset = (len(ordered) - 1) * probability
    index = int(offset)
    fraction = offset - index
    return ordered[index] + fraction * (ordered[min(index + 1, len(ordered) - 1)] - ordered[index])


def bootstrap(differences, groups):
    means = [fmean(differences[key] for key in group) for group in groups.values()
             if all(key in differences for key in group)]
    count = len(means)
    result = {"status": "UNVERIFIED_INSUFFICIENT_COMPLETE_LAYOUTS", "complete_layouts": count,
              "minimum_complete_layouts": MIN_LAYOUTS, "confidence": 0.95, "interval": None}
    if count < MIN_LAYOUTS:
        return result
    generator = random.Random(BOOTSTRAP_SEED)
    draws = sorted(fmean(generator.choices(means, k=count)) for _ in range(BOOTSTRAP_RESAMPLES))
    result.update(status="ESTIMATED", interval=[percentile(draws, 0.025), percentile(draws, 0.975)],
                  equal_layout_mean_difference=fmean(means), resamples=BOOTSTRAP_RESAMPLES)
    return result


def compare_measure(name, keys, left, right, groups):
    pairs = {}
    for key in sorted(keys):
        values = metric(left[key], name), metric(right[key], name)
        if all(map(number, values)):
            pairs[key] = values
    differences = {key: x - y for key, (x, y) in pairs.items()}
    left_mean = mean([x for x, _ in pairs.values()])
    right_mean = mean([y for _, y in pairs.values()])
    difference = mean(list(differences.values()))
    relative = difference / right_mean * 100 if right_mean and right_mean > 0 and left_mean >= 0 else None
    result = {"measured_pairs": len(pairs), "left_mean": left_mean, "right_mean": right_mean,
              "mean_difference": difference, "relative_change_percent": relative,
              "layout_bootstrap_95": bootstrap(differences, groups)}
    if name in ("exit", "alive"):
        result["percentage_point_difference"] = difference * 100 if difference is not None else None
    if name in PROFILES:
        outcomes = [(int(left[key]["exit"]) - int(right[key]["exit"]), round(delta, 9))
                    for key, delta in differences.items()]
        result["win_tie_loss"] = {"wins": sum(value > (0, 0) for value in outcomes),
                                  "ties": sum(value == (0, 0) for value in outcomes),
                                  "losses": sum(value < (0, 0) for value in outcomes)}
    return result


def compare_pair(left, right, planned, attempts, completed, status):
    lhs, rhs = completed[left], completed[right]
    paired = lhs.keys() & rhs.keys()
    groups = complete_layouts(paired, planned)
    attempted = (attempts[left].keys() | attempts[right].keys()) & planned.keys()
    left_failed = {key for key, state in status[left].items() if key in planned and state != "completed"}
    right_failed = {key for key, state in status[right].items() if key in planned and state != "completed"}
    return {
        "left": left, "right": right, "planned_pairs": len(planned), "attempted_pairs": len(attempted),
        "completed_pairs": len(paired), "paired_layouts": len(layout_groups(paired)),
        "fully_paired_layouts": len(groups), "unattempted_pairs": len(planned.keys() - attempted),
        "unpaired_attempted_pairs": len(attempted - paired),
        "pairs_with_failed_attempt": len(left_failed | right_failed),
        "left_failed_attempts": len(left_failed), "right_failed_attempts": len(right_failed),
        "left_completed_right_failed": len(lhs.keys() & right_failed),
        "right_completed_left_failed": len(rhs.keys() & left_failed),
        "left_completed_right_unattempted": len(lhs.keys() - attempts[right].keys()),
        "right_completed_left_unattempted": len(rhs.keys() - attempts[left].keys()),
        "outcomes": {name: compare_measure(name, paired, lhs, rhs, groups) for name in RAW_OUTCOMES},
        "profiles": {name: compare_measure(name, paired, lhs, rhs, groups) for name in PROFILES},
    }


def common_rankings(common, completed):
    if not common:
        return {name: [] for name in PROFILES}
    episodes = {model: [completed[model][key] for key in sorted(common)] for model in MODELS}
    rankings = {}
    for name in PROFILES:
        rows = [{"model": model, "exits": sum(bool(row["exit"]) for row in results),
                 "mean_score": fmean(metric(row, name) for row in results)}
                for model, results in episodes.items()]
        rankings[name] = sorted(rows, key=lambda row: (-row["exits"], -row["mean_score"], row["model"]))
    return rankings


def scoped_summary(planned, attempts, completed, status):
    by_model = {model: model_totals(planned, attempts[model], completed[model], status[model]) for model in MODELS}
    common = set.intersection(*(set(completed[model]) for model in MODELS))
    expected = len(planned)
    return {
        "counts": {"planned_episodes": expected * len(MODELS),
                   "attempted_episodes": sum(row["attempted"] for row in by_model.values()),
                   "recorded_completed_episodes": sum(row["recorded_completed"] for row in by_model.values()),
                   "completed_episodes": sum(row["completed"] for row in by_model.values()),
                   "failed_or_excluded_units": sum(row["failed"] for row in by_model.values()),
                   "unattempted_episodes": sum(row["unattempted"] for row in by_model.values()),
                   "expected_layouts": len(layout_groups(planned)),
                   "attempted_layouts": len(layout_groups(set().union(*(set(index) for index in attempts.values())))),
                   "completed_layouts": len(layout_groups(set().union(*(set(index) for index in completed.values())))),
                   "fully_matched_layouts": len(complete_layouts(common, planned))},
        "models": by_model,
        "matched": {"status": "COMPLETE" if len(common) == expected and expected else "PARTIAL",
                    "completed_units": len(common), "planned_units": expected,
                    "full_rankings": common_rankings(common, completed)},
        "pairwise": [compare_pair(left, right, planned, attempts, completed, status)
                     for left, right in combinations(MODELS, 2)],
    }


def build_summary(rows, planned):
    attempts, completed, status, issues = index_attempts(rows, planned)
    data = scoped_summary(planned, attempts, completed, status)
    tiers = {}
    for tier in sorted({key[0] for key in planned}):
        limits = {key: info for key, info in planned.items() if key[0] == tier}
        scoped = [{model: {key: info for key, info in index[model].items() if key[0] == tier}
                   for model in MODELS} for index in (attempts, completed, status)]
        tiers[str(tier)] = scoped_summary(limits, *scoped)
    return {"schema_version": 1, "classification": "Unofficial local proxies and measured gameplay",
            "models_order": list(MODELS), **data, "tiers": tiers, "attempt_issues": issues}


def read_rows(path):
    rows, issues = [], []
    if not path.is_file():
        return rows, [{"reason": "episode-log-missing", "path": str(path)}]
    with path.open(encoding="utf-8-sig") as source:
        for line, text in enumerate(source, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                issues.append({"line": line, "reason": "invalid-json-line"})
                continue
            if not isinstance(row, dict):
                issues.append({"line": line, "reason": "non-object-json-line"})
                continue
            rows.append((line, row))
    return rows, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=BASE / "episodes/blind-episodes.jsonl")
    parser.add_argument("--maps", type=Path, default=BASE / "manifests/map_manifest.json")
    parser.add_argument("--output", type=Path, default=BASE / "reports/matched-summary.json")
    args = parser.parse_args()
    planned = frozen_units(json.loads(args.maps.read_text(encoding="utf-8-sig")))
    rows, issues = read_rows(args.episodes)
    summary = build_summary(rows, planned)
    summary["episode_source"] = str(args.episodes.resolve())
    summary["map_manifest"] = str(args.maps.resolve())
    summary["input_issues"] = issues
    output = args.output.with_suffix(".tmp.json")
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.replace(args.output)
    print(args.output, summary["counts"]["completed_episodes"], "completed episodes")


if __name__ == "__main__":
    main()
