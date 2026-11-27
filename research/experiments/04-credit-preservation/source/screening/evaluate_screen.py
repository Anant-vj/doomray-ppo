"""Matched credit-preservation screen with a fixed actor and immutable weight roster."""

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
MANIFESTS = ROOT / "manifests"
OUTPUT = ROOT / "results"
LEDGER = OUTPUT / "episodes.jsonl"
PROTOCOL_SHA = "da14fff9531a75678bf0b178377bbd200237d8ed26ba6622ebfc780b0b60cc57"
SOURCE_SHA = {
    "comparison.py": "f54b51f360416ff1773518b9aace5fd547b241767dc886fa01fa2d6aca845e08",
    "run_matched.py": "f10a00ddd5f5ebf763846065baa13e91d20f124397866b80da4e4074b0a17413",
    "storage_gate.py": "5f0b7bfe666873478bf3ffb521b2bcd88c2983e8101d265097a03f5bf352066a",
}
sys.path.insert(0, str(ROOT / "evaluator"))
import comparison
import run_matched as audit
from storage_gate import GateClosed, snapshot, storage_policy


def read_json(path):
    return json.loads(path.read_bytes())


def encode(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()


def freeze(path, contents):
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"Already frozen input differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(contents)


def check_gate():
    if audit.STOP.exists() or datetime.now(audit.IST) >= audit.CUTOFF:
        raise GateClosed("Organizer handoff or 08:15 IST cutoff")
    return snapshot(audit.STORAGE, startup=True)


def verify_repo(model):
    repo = Path(model["checkout"]).resolve(strict=True)
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if repo.drive.upper() != "D:" or head != model["source_commit"] or changed:
        raise ValueError("The frozen actor checkout changed")
    return repo


def verify_protocol():
    path = MANIFESTS / "protocol.json"
    if audit.digest(path) != PROTOCOL_SHA:
        raise ValueError("Frozen protocol bytes changed")
    protocol = read_json(path)
    if datetime.fromisoformat(protocol["experimental_cutoff_ist"]) != audit.CUTOFF:
        raise ValueError("Frozen protocol and runner cutoff differ")
    if [entry["id"] for entry in protocol["roster"]] != ["C749", "PPO-control", "PBRS", "PBRS-anchor"]:
        raise ValueError("The frozen roster changed")
    for name, expected in SOURCE_SHA.items():
        if audit.digest(ROOT / "evaluator" / name) != expected:
            raise ValueError(f"The copied evaluator source changed: {name}")
    if audit.digest(MANIFESTS / "scoring_manifest.json") != protocol["scoring_manifest_sha256"]:
        raise ValueError("Frozen scoring weights changed")
    repo = verify_repo(protocol["roster"][0])
    baseline = repo / "submission/weights/policy.pt"
    if audit.digest(baseline) != protocol["roster"][0]["expected_sha256"]:
        raise ValueError("Frozen C749 weight did not match")
    for spec in protocol["screening_maps"]:
        wad = Path(spec["wad"]).resolve(strict=True)
        if not wad.is_relative_to(ROOT / "maps") or audit.digest(wad) != spec["wad_sha256"]:
            raise ValueError(f"Frozen map differs: {spec['id']}")
    return protocol, repo


def freeze_roster(roster):
    exact = []
    for model in roster:
        source = Path(model["weight_source"]).resolve(strict=True)
        if source.drive.upper() != "D:" or not 0 < source.stat().st_size <= 20_000_000:
            raise ValueError(f"Expected a compact trusted checkpoint on D: {source}")
        if model["id"] != "C749" and not source.is_relative_to(EXPERIMENT / "checkpoints"):
            raise ValueError(f"Unexpected experimental weight path: {source}")
        contents = source.read_bytes()
        checksum = sha256(contents).hexdigest()
        if model.get("expected_sha256", checksum) != checksum or audit.digest(source) != checksum:
            raise ValueError(f"Weight changed while freezing: {source}")
        target = source if model["id"] == "C749" else OUTPUT / "weights" / model["id"] / "policy.pt"
        if target != source:
            freeze(target, contents)
        exact.append({**model, "weight_path": str(target), "sha256": checksum, "bytes": len(contents)})
    freeze(OUTPUT / "roster.json", encode({"protocol_sha256": PROTOCOL_SHA, "models": exact}))
    return exact


def row_key(row):
    return row["map_id"], row["engine_seed"], row["model"]


def outcomes(rows):
    sums = Counter()
    fields = ("recorded_player_damage", "recorded_player_hits", "keys", "unlocks", "explored_cells_96",
              "move_requests", "blocked_moves", "attack_decisions", "attacks_without_enemy_cues", "stationary_decisions")
    for row in rows:
        sums.update({field: row["metrics"][field] for field in fields})
        sums.update({"exits": int(row["exit"]), "survivors": int(row["alive"])})
    return {field: sums[field] for field in ("exits", "survivors", *fields)}


def blocked_fraction(metrics):
    return metrics["blocked_moves"] / metrics["move_requests"] if metrics["move_requests"] else None


def material_gain(base, new):
    if any(new[field] > base[field] for field in ("exits", "keys", "unlocks", "recorded_player_damage")):
        return True
    old_blocked, new_blocked = blocked_fraction(base), blocked_fraction(new)
    return (new["explored_cells_96"] > base["explored_cells_96"] and old_blocked is not None and
            new_blocked is not None and new_blocked < old_blocked)


def measure_gate(reference, candidate, maps, legal):
    family_deltas, gain_families, replicated = {}, [], []
    for item in maps:
        name = item["id"]
        base_rows, new_rows = [[row for row in group if row["map_id"] == name] for group in (reference, candidate)]
        base, new = outcomes(base_rows), outcomes(new_rows)
        delta = {field: new[field] - base[field] for field in new}
        old_blocked, new_blocked = blocked_fraction(base), blocked_fraction(new)
        per_seed = {}
        for seed in item["seeds"]:
            before, after = [[row for row in group if row["engine_seed"] == seed] for group in (base_rows, new_rows)]
            per_seed[str(seed)] = bool(before and after and material_gain(outcomes(before), outcomes(after)))
        if material_gain(base, new):
            gain_families.append(name)
            if all(per_seed.values()):
                replicated.append(name)
        family_deltas[name] = {**delta, "reference_blocked_fraction": old_blocked,
                               "candidate_blocked_fraction": new_blocked, "material_gain_by_seed": per_seed}

    preserves_survival = all(row["survivors"] >= 0 for row in family_deltas.values())
    preserves_exits = all(row["exits"] >= 0 for row in family_deltas.values())
    preserves_progress = all(delta[field] >= 0 for delta in family_deltas.values() for field in ("keys", "unlocks"))
    qualifies = legal and preserves_survival and preserves_exits and preserves_progress and len(gain_families) >= 2 and bool(replicated)
    return {"complete_paired_legal_cases": legal, "preserves_survival_per_family": preserves_survival,
            "preserves_exits_per_family": preserves_exits, "preserves_key_unlock_per_family": preserves_progress,
            "material_gain_families": gain_families, "replicated_gain_families": replicated, "passes_local_screen": qualifies,
            "family_deltas": family_deltas}


def summarize(protocol, models):
    maps = protocol["screening_maps"]
    comparison.MODELS = tuple(model["id"] for model in models)
    schedule = {"stages": {"minimum": {"ordered_map_ids": [spec["id"] for spec in maps],
                                      "planned_episodes": protocol["expected_episodes"]}}, "maps": maps}
    rows, issues = comparison.read_rows(LEDGER)
    summary = comparison.build_summary(rows, comparison.frozen_units(schedule))
    summary["input_issues"] = issues
    summary["protocol_sha256"] = PROTOCOL_SHA
    summary["source_ledger_sha256"] = audit.digest(LEDGER)
    bad_lines = {entry["line"] for issue in summary["attempt_issues"] for entry in issue.get("rows", [])}
    by_model, behavior = {}, {}
    for name in comparison.MODELS:
        selected = [row for line, row in rows if line not in bad_lines and row["model"] == name and row["status"] == "completed"]
        by_model[name] = selected
        counts = Counter()
        for row in selected:
            counts.update(row["metrics"]["button_counts"])
        acted = sum(row["decisions"] for row in selected)
        totals = outcomes(selected)
        behavior[name] = {**totals, "legal_decisions": acted, "buttons": dict(counts),
                          "absolute_strafe_bias": abs(counts["MOVE_LEFT"] - counts["MOVE_RIGHT"]) / acted if acted else None,
                          "absolute_turn_bias": abs(counts["TURN_LEFT"] - counts["TURN_RIGHT"]) / acted if acted else None,
                          "max_action_latency_ms": max((row["latency_ms"]["max"] for row in selected), default=None),
                          "worst_episode_p99_ms": max((row["latency_ms"]["p99"] for row in selected), default=None),
                          "recorded_gameplay_wall_seconds": sum(row["wall_seconds"] for row in selected)}
    summary["behavior"] = behavior
    expected_per_model = protocol["expected_episodes"] // len(models)
    summary["promotion_screen"] = {
        name: measure_gate(by_model["C749"], by_model[name], maps,
                           len(by_model["C749"]) == len(by_model[name]) == expected_per_model)
        for name in comparison.MODELS if name != "C749"
    }
    temporary = OUTPUT / "summary.tmp.json"
    temporary.write_bytes(encode(summary))
    temporary.replace(OUTPUT / "summary.json")
    print(json.dumps({"counts": summary["counts"], "storage_after": snapshot(audit.STORAGE),
                      "summary": str(OUTPUT / "summary.json")}), flush=True)


def run(protocol, repo):
    models = freeze_roster(protocol["roster"])
    states = OUTPUT / "engine"
    states.mkdir(exist_ok=True)
    os.chdir(states)
    for name, value in {"TEMP": str(states), "TMP": str(states), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
        os.environ[name] = value
    sys.path.insert(0, str(repo / "env"))
    import numpy as np
    import torch
    from doom_env import limits, make_env, spaces, wrapper
    from doom_env.remote import AgentFailure, validate_action
    from vizdoom import GameVariable

    if wrapper.SKILL != 3:
        raise ValueError("The official wrapper must use Skill 3")
    torch.set_num_threads(1)
    api = (np, torch, make_env, validate_action, AgentFailure, spaces, limits, GameVariable)
    actors = {}
    for model in models:
        actor = audit.import_actor(repo)
        actor.__init__.__globals__["WEIGHTS_DIR"] = Path(model["weight_path"]).parent
        actors[model["id"]] = actor
    runtime = {"python": sys.version.split()[0], "python_executable": sys.executable, "torch": torch.__version__,
               "vizdoom": importlib.metadata.version("vizdoom"), "cuda_available": torch.cuda.is_available(),
               "wrapper": inspect.getfile(wrapper), "cpu_threads": torch.get_num_threads(), "headless": True}
    freeze(OUTPUT / "runtime.json", encode(runtime))
    print(json.dumps(runtime), flush=True)
    previous, issues = comparison.read_rows(LEDGER)
    if any(issue["reason"] != "episode-log-missing" for issue in issues):
        raise ValueError(f"Invalid prior episode records: {issues}")
    attempted = {row_key(row): row for _, row in previous}
    if len(attempted) != len(previous):
        raise ValueError("Duplicate prior attempted case")
    audit.EPISODES = OUTPUT
    try:
        with LEDGER.open("a", encoding="utf-8", newline="\n") as ledger:
            for spec in protocol["screening_maps"]:
                for seed in spec["seeds"]:
                    for model in models:
                        key = spec["id"], seed, model["id"]
                        prior = attempted.get(key)
                        if prior:
                            if prior["checkpoint_sha256"] != model["sha256"] or prior["wad_sha256"] != spec["wad_sha256"]:
                                raise ValueError(f"Prior case identity changed: {key}")
                            continue
                        check_gate()
                        if audit.digest(model["weight_path"]) != model["sha256"]:
                            raise ValueError(f"Frozen weight file changed: {model['id']}")
                        result = audit.run_episode(spec, seed, actors[model["id"]], model, api)
                        ledger.write(json.dumps(result, allow_nan=False) + "\n")
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        keys = ("model", "map_id", "engine_seed", "status", "decisions", "alive", "exit", "error")
                        print(json.dumps({field: result[field] for field in keys}), flush=True)
                        if result["status"] == "storage_or_handoff_stop":
                            raise GateClosed(result["error"])
    except GateClosed as exc:
        print(json.dumps({"halted": str(exc)}), flush=True)
    summarize(protocol, models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve().drive.upper() != "D:":
        raise ValueError("Run with the resident competition Python on D:")
    audit.STOP = EXPERIMENT / "stop-optional.flag"
    audit.STORAGE = storage_policy()
    storage = check_gate()
    protocol, repo = verify_protocol()
    print(json.dumps({"protocol_sha256": PROTOCOL_SHA, "episodes": protocol["expected_episodes"], "storage": storage,
                      "cutoff_ist": protocol["experimental_cutoff_ist"],
                      "weight_sources_exist": {entry["id"]: Path(entry["weight_source"]).is_file()
                                               for entry in protocol["roster"]}}), flush=True)
    if args.run:
        run(protocol, repo)


if __name__ == "__main__":
    main()
