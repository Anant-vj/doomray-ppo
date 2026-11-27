"""Rebuild the published counters using Python's standard library only."""

import argparse
import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = "e9fe1740994a78b3bfb650017061b8e832d7a957"
LAYERS = (
    ("01-behavior-cloning", "screening", 40, 6352),
    ("02-symmetry-kl", "screening", 24, 4542),
    ("03-inference-decoding", "screening", 32, 4845),
    ("03-inference-decoding", "confirmation", 8, 1576),
    ("04-credit-preservation", "screening", 40, 8275),
    ("05-exact-ppo", "screening", 48, 11078),
)
LEARNERS = {
    "01-behavior-cloning": (6462, {"bc-ppo-precision": 128, "ppo-only-precision-1": 128}),
    "02-symmetry-kl": (7614, {"ppo-control": 24, "symmetry-kl": 24}),
    "04-credit-preservation": (7614, {"ppo-control": 32, "pbrs": 32, "pbrs-anchor": 32}),
    "05-exact-ppo": (7614, {"exact-ppo": 48, "balanced-reward": 48, "balanced-pbrs-anchor": 48}),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def document(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ledger(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_index():
    index = ROOT / "SHA256SUMS"
    listed = {}
    for entry in index.read_text(encoding="utf-8").splitlines():
        check, name = entry.split("  ", maxsplit=1)
        require(name not in listed, f"Duplicate index path {name}")
        listed[name] = check
    files = {
        item.relative_to(ROOT).as_posix(): item
        for item in ROOT.rglob("*") if item.is_file() and item != index
    }
    require(listed.keys() == files.keys(), "Index and archive file sets differ")
    for name, path in files.items():
        require(digest(path) == listed[name], f"SHA-256 mismatch for {name}")
        if path.suffix == ".py":
            ast.parse(path.read_bytes(), filename=name)
    return len(files)


def episodes(rows, expected_count, expected_decisions, suite):
    require(len(rows) == expected_count, f"{suite} episode count")
    models = defaultdict(list)
    for row in rows:
        require(
            row["status"] == "completed" and not row["error"] and row["engine_reset_ok"]
            and row["decisions"] > 0 and row["invalid_actions"] == 0,
            f"{suite} incomplete or illegal episode",
        )
        models[row["model"]].append(row)
    common = None
    for model, items in models.items():
        units = {(row["map_id"], row["engine_seed"], row["policy_seed"]) for row in items}
        require(len(units) == len(items), f"{suite} duplicate episode for {model}")
        if common is None:
            common = units
        require(units == common, f"{suite} unmatched suite for {model}")
    decisions = sum(row["decisions"] for row in rows)
    require(decisions == expected_decisions, f"{suite} decision count")
    return models, decisions


def check_report(report, models):
    require(set(report["models"]) == set(models), "Report/model roster mismatch")
    require(report["counts"]["completed_episodes"] == sum(map(len, models.values())),
            "Report/ledger episode count mismatch")
    for model, items in models.items():
        saved = report["models"][model]
        require(saved["completed"] == len(items), f"{model} completed episodes")
        require(saved["exits"] == sum(row["exit"] for row in items), f"{model} exits")
        require(saved["survived"] == sum(row["alive"] for row in items), f"{model} survival")
        for metric, summary in saved["metrics"].items():
            samples = [row["metrics"].get(metric) for row in items]
            samples = [value for value in samples if value is not None]
            require(len(samples) == summary["measured"], f"{model} {metric} support")
            require(abs(sum(samples) - summary["total"]) < 1e-8, f"{model} {metric} total")
        for score, average in saved["scores"].items():
            observed = sum(row["scores"][score] for row in items) / len(items)
            require(abs(observed - average) < 1e-8, f"{model} {score} average")


def historical():
    base = ROOT / "history"
    index = document(base / "provenance.json")
    source = base / index["ledger"]["file"]
    require(digest(source) == index["ledger"]["sha256"], "Legacy ledger hash mismatch")
    rows = defaultdict(list)
    for entry in ledger(source.read_bytes()):
        rows[entry["run_id"]].append(entry)
    runs = {item["run_id"]: item for item in index["runs"]}
    require(rows.keys() == runs.keys() and len(runs) == 15, "Legacy run roster mismatch")
    totals = []
    for run, spec in runs.items():
        start, items = spec["starting_counters"], rows[run]
        previous = start
        for row in items:
            require(
                row["update"] == previous["update"] + 1
                and row["gradients"] > previous["gradients"]
                and row["transitions_this_run"] > previous["transitions_this_run"]
                and row["transitions_cumulative"] == row["transitions_this_run"]
                + start["transitions_cumulative"],
                f"Invalid counters in {run}",
            )
            previous = row
        totals.append((previous["transitions_this_run"], len(items),
                       previous["gradients"] - start["gradients"]))
        require(totals[-1] == (spec["new_transitions"], spec["updates_logged"],
                               spec["new_optimizer_steps"]), f"Reported legacy count for {run}")
        parent = spec["parent"]
        if parent:
            inherited = next(row for row in rows[parent["run_id"]] if row["update"] == parent["update"])
            for field in ("update", "gradients", "transitions_cumulative"):
                require(inherited[field] == start[field], f"{run} inherited {field}")
    lineage = []
    for cut in index["selected_c749"]["verified_cutoffs"]:
        run = cut["run_id"]
        stop = [row for row in rows[run] if row["update"] <= cut["last_update"]]
        start = runs[run]["starting_counters"]
        require(stop[0]["update"] == cut["first_update"]
                and stop[-1]["update"] == cut["last_update"], f"C749 cutoff for {run}")
        lineage.append((stop[-1]["transitions_this_run"], len(stop),
                        stop[-1]["gradients"] - start["gradients"]))
    return add(totals), add(lineage)


def add(totals):
    return tuple(map(sum, zip(*totals)))


def job(bundle, name, inherited, updates):
    path = ROOT / "experiments" / bundle / "training" / (name + ".jsonl")
    rows = ledger(path.read_bytes())
    require(len(rows) == updates, f"{bundle} {name} rollout count")
    start = 621 if bundle == "01-behavior-cloning" else 749
    for i, row in enumerate(rows, 1):
        require(row["update"] == start + i and row["gradients"] == inherited + 6 * i
                and row["transitions_this_run"] == 256 * i, f"Invalid counters in {path.name}")
    return rows[-1]["transitions_this_run"], len(rows), rows[-1]["gradients"] - inherited


def report_number(label, actual, expected):
    require(actual == expected, f"{label} {actual!r} != {expected!r}")
    print(f"{label}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-repo", type=Path,
                        help="A local Git repository with the immutable assessment commit")
    args = parser.parse_args()
    print(f"Archive: {check_index()} files matched SHA-256")
    legs = []
    for bundle, stage, expected_count, expected_decisions in LAYERS:
        base = ROOT / "experiments" / bundle / "evaluation"
        models, count = episodes(ledger((base / f"{stage}.jsonl").read_bytes()),
                                 expected_count, expected_decisions, f"{bundle}/{stage}")
        specific = base / f"{stage}-summary.json"
        check_report(document(specific if specific.is_file() else base / "summary.json"), models)
        print(f"{bundle}/{stage}: {expected_count} episodes, {count} legal decisions")
        legs.append((expected_count, count))
    later_eval = add(legs)
    report_number("Later matched episodes and legal decisions", later_eval, (192, 36668))
    if args.assessment_repo:
        old = subprocess.check_output([
            "git", "-c", f"safe.directory={args.assessment_repo.resolve().as_posix()}",
            "-C", str(args.assessment_repo), "show",
            f"{ASSESSMENT}:assessment/episodes/blind-episodes.jsonl",
        ])
        old_rows = ledger(old)
        old_models, old_decisions = episodes(old_rows, 96, 41035, "immutable assessment")
        require(len(old_models) == 3 and len({row["map_id"] for row in old_rows}) == 16,
                "Immutable assessment roster mismatch")
        report_number("Canonical matched episodes and legal decisions",
                      add((later_eval, (len(old_rows), old_decisions))), (288, 77703))
    legacy, lineage = historical()
    report_number("Legacy transitions, rollouts and new optimizer steps", legacy,
                  (618432, 1452, 14499))
    report_number("C749 transitions, rollouts and new optimizer steps", lineage,
                  (324864, 749, 7614))
    primary = add([job(bundle, name, inherited, count)
                   for bundle, (inherited, runs) in LEARNERS.items()
                   for name, count in runs.items()])
    smokes = add([job("05-exact-ppo", name, 7614, 4) for name in ("cache-smoke", "exact-smoke")])
    report_number("Later primary transitions, rollouts and new optimizer steps", primary,
                  (139264, 544, 3264))
    report_number("Separate live smoke transitions, rollouts and new optimizer steps", smokes,
                  (2048, 8, 48))
    report_number("All logged transitions, rollouts and new optimizer steps",
                  add((legacy, primary, smokes)), (759744, 2004, 17811))
    bc = document(ROOT / "experiments/01-behavior-cloning/training/behavioral-cloning-summary.json")
    supervised = sum(row["train"]["optimizer_updates"] for row in bc["epochs"])
    report_number("Separate supervised optimizer steps", supervised, 310)
    verifier = ROOT / "experiments/03-inference-decoding/verify-scores.py"
    for bundle in ("03-inference-decoding", "04-credit-preservation"):
        subprocess.run([sys.executable, "-B", str(verifier), "--bundle",
                        str(ROOT / "experiments" / bundle)], check=True)


if __name__ == "__main__":
    main()
