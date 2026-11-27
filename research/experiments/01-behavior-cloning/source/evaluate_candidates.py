"""Run frozen local screening/confirmation using the unchanged competition actor and wrapper."""

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ASSESSMENT = PROJECT / "submission_artifacts/final-blind-assessment"
PROTOCOL = "e9ec321a8f28bc5381f130b09ef2ded9cfb4c2f3d418a08575d4ab4f0844f7dd"
sys.path.insert(0, str(ASSESSMENT / "evaluator"))
import comparison
import run_matched as audit
from storage_gate import GateClosed, snapshot, storage_policy


def read_json(path):
    return json.loads(path.read_bytes())


def encode(obj):
    return (json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=True) + "\n").encode()


def write_exact(path, contents):
    if path.is_file():
        if path.read_bytes() != contents:
            raise ValueError(f"Previously frozen contents changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(contents)


def check_gate(storage):
    result = snapshot(storage, startup=True)
    if audit.STOP.exists() or datetime.now(audit.IST) >= audit.CUTOFF:
        raise GateClosed("Stopped for organizer handoff or 06:15 IST cutoff.")
    return result


def checked_checkout(entry):
    repo = Path(entry["checkout"]).resolve()
    if repo.drive.upper() != "D:":
        raise ValueError("Actor checkout must remain on D:")
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != entry["source_commit"] or changes:
        raise ValueError(f"Actor checkout identity changed: {entry['id']}")
    if not entry.get("continuation") and audit.digest(repo / "submission/weights/policy.pt") != entry["sha256"]:
        raise ValueError(f"Archived checkout weight differs: {entry['id']}")
    return repo


def register_continuation(spec, controls, screening):
    name, delimiter, source = spec.partition("=")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,47}", name) or not delimiter or name in controls:
        raise ValueError("Use an unused candidate identity and absolute D checkpoint: ID=operator-local-reference/policy.pt")
    path = Path(source).resolve(strict=True)
    if path.drive.upper() != "D:" or path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"Candidate must be a .pt checkpoint on D: {path}")
    if not 1 <= path.stat().st_size <= 20_000_000:
        raise ValueError("Use small inference exports or a compatible local training checkpoint")
    contents = path.read_bytes()
    digest = sha256(contents).hexdigest()
    materialized = ROOT / "evaluation-weights" / name / digest / "policy.pt"
    entry = {
        "id": name, "checkout": controls["C749"]["checkout"], "source_commit": controls["C749"]["source_commit"],
        "sha256": digest, "weight_path": str(materialized), "continuation": True, "archive_source": str(path),
        "weight_bytes": len(contents),
    }
    registry = screening / "candidates" / f"{name}.json"
    if registry.exists() and read_json(registry)["sha256"] != digest:
        raise ValueError(f"Candidate identity {name} already refers to a different checkpoint")
    write_exact(materialized, contents)
    if not registry.exists():
        write_exact(registry, encode(entry))
    return read_json(registry)


def checked_maps(protocol, nominee):
    if nominee:
        sealed = ROOT / "sealed_confirmation/manifest.json"
        if audit.digest(sealed) != protocol["sealed_confirmation_manifest_sha256"]:
            raise ValueError("Previously sealed layout manifest changed")
        maps = read_json(sealed)["maps"]
    else:
        maps = protocol["screening_maps"]
    for item in maps:
        path = Path(item["wad"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or audit.digest(path) != item["wad_sha256"]:
            raise ValueError(f"New map bytes changed: {item['id']}")
    return maps


def row_key(row):
    return row["map_id"], row["engine_seed"], row["model"]


def behavior(rows, models, issues):
    excluded = {entry["line"] for issue in issues for entry in issue.get("rows", [])}
    names = ("attack_decisions", "attacks_without_enemy_cues", "attack_enemy_visible", "stationary_decisions",
             "move_requests", "blocked_moves", "near_wall_blocks", "weapon_switch_requests",
             "weapon_switches_without_observed_change")
    result = {}
    for model in models:
        runs = [row for line, row in rows if line not in excluded and row["model"] == model["id"]
                and row["status"] == "completed"]
        counts, buttons = Counter(), Counter()
        for row in runs:
            counts.update({name: row["metrics"][name] for name in names})
            buttons.update(row["metrics"]["button_counts"])
        decisions = sum(row["decisions"] for row in runs)
        wall = sum(row["wall_seconds"] for row in runs)
        result[model["id"]] = {
            "legal_decisions": decisions, "wall_seconds_in_episode_records": wall,
            "decisions_per_recorded_wall_second": decisions / wall if wall else None,
            "button_counts": dict(buttons), "counters": {name: counts[name] for name in names},
            "no_cue_fraction_of_attack_decisions": (counts["attacks_without_enemy_cues"] / counts["attack_decisions"]
                                                    if counts["attack_decisions"] else None),
            "stationary_fraction_of_decisions": counts["stationary_decisions"] / decisions if decisions else None,
            "mean_inference_ms_over_decisions": (sum(row["latency_ms"]["mean"] * row["decisions"] for row in runs)
                                                / decisions if decisions else None),
            "worst_episode_p99_inference_ms": max((row["latency_ms"]["p99"] for row in runs), default=None),
            "max_inference_ms": max((row["latency_ms"]["max"] for row in runs), default=None),
        }
    return result


def finish_summary(output, maps, models):
    comparison.MODELS = tuple(entry["id"] for entry in models)
    total = sum(len(item["seeds"]) for item in maps) * len(models)
    plan = {"stages": {"minimum": {"ordered_map_ids": [item["id"] for item in maps], "planned_episodes": total}},
            "maps": maps}
    ledger = output / "episodes.jsonl"
    rows, issues = comparison.read_rows(ledger)
    summary = comparison.build_summary(rows, comparison.frozen_units(plan))
    summary["input_issues"] = issues
    summary["source_ledger_sha256"] = audit.digest(ledger)
    summary["suite"] = "Fresh local fixtures with frozen model/map/seed order"
    summary["behavior"] = behavior(rows, models, summary["attempt_issues"])
    temporary = output / "summary.tmp.json"
    temporary.write_bytes(encode(summary))
    temporary.replace(output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Refresh evidence from a closed ledger without a game")
    parser.add_argument("--candidate", action="append", default=[], metavar="ID=operator-local-reference/checkpoint.pt")
    parser.add_argument("--confirm", metavar="SCREENED_NOMINEE")
    args = parser.parse_args()
    if args.summarize and (args.preflight or args.candidate):
        parser.error("Summary-only mode cannot preflight or register candidates")
    if args.confirm and args.candidate:
        parser.error("Register and complete screening before nominating a candidate for confirmation")
    if Path(sys.executable).resolve().drive.upper() != "D:":
        raise ValueError("Use the existing competition Python on D:")
    audit.STOP = ROOT / "stop-optional.flag"
    audit.STORAGE = storage_policy()
    before = None if args.summarize else check_gate(audit.STORAGE)
    protocol_path = ROOT / "evaluation-protocol.json"
    if audit.digest(protocol_path) != PROTOCOL:
        raise ValueError("Frozen screening protocol changed")
    protocol = read_json(protocol_path)
    if audit.digest(protocol["score_source"]) != protocol["score_source_sha256"]:
        raise ValueError("Frozen score manifest changed")
    screening = ROOT / "screening-results"
    controls = {entry["id"]: entry for entry in protocol["archive_roster"]}
    for spec in args.candidate:
        register_continuation(spec, controls, screening)
    registrants = [read_json(path) for path in sorted((screening / "candidates").glob("*.json"))]
    models = [*controls.values(), *registrants]
    output = screening
    if args.confirm:
        if args.confirm == "C749" or args.confirm not in {entry["id"] for entry in models}:
            raise ValueError("Confirmation requires a screened alternative nominee")
        output = ROOT / "confirmation-results"
        roster = {"C749", "Stack26-u621", args.confirm}
        models = [entry for entry in models if entry["id"] in roster]
        verified = read_json(screening / "summary.json")["models"]
        if not all(verified.get(name, {}).get("completed") == 8 for name in roster):
            raise ValueError("All confirmation policies must have eight complete screening episodes")
        write_exact(output / "nomination.json", encode({"nominee": args.confirm, "models": models,
                                                        "screening_sha256": audit.digest(screening / "episodes.jsonl")}))
    maps = checked_maps(protocol, args.confirm)
    for entry in models:
        if audit.digest(entry["weight_path"]) != entry["sha256"]:
            raise ValueError(f"Frozen candidate bytes changed: {entry['id']}")
        checked_checkout(entry)
    batch = {"protocol_sha256": PROTOCOL, "map_ids": [item["id"] for item in maps], "models": models,
             "confirmed_nominee": args.confirm}
    batch_content = encode(batch)
    batch_hash = sha256(batch_content).hexdigest()
    if args.summarize:
        summary = finish_summary(output, maps, models)
        print(json.dumps({"counts": summary["counts"], "source_sha256": summary["source_ledger_sha256"]}), flush=True)
        return
    write_exact(output / "batches" / f"{batch_hash}.json", batch_content)
    print(json.dumps({"storage": before, "batch_sha256": batch_hash, "models": [entry["id"] for entry in models],
                      "layout_count": len(maps), "scope": "confirmation" if args.confirm else "screening"}), flush=True)
    if args.preflight:
        return

    engine = output / "engine"
    engine.mkdir(exist_ok=True)
    os.chdir(engine)
    for name, value in {"TEMP": str(engine), "TMP": str(engine), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
        os.environ[name] = value
    sys.path.insert(0, str(Path(controls["C749"]["checkout"]) / "env"))
    import numpy as np
    import torch
    from doom_env import limits, make_env, spaces, wrapper
    from doom_env.remote import AgentFailure, validate_action
    from vizdoom import GameVariable

    if wrapper.SKILL != 3:
        raise ValueError("Expected official Skill 3")
    torch.set_num_threads(1)
    api = (np, torch, make_env, validate_action, AgentFailure, spaces, limits, GameVariable)
    actors = {}
    for entry in models:
        actor = audit.import_actor(checked_checkout(entry))
        if entry.get("continuation"):
            actor.__init__.__globals__["WEIGHTS_DIR"] = Path(entry["weight_path"]).parent
        actors[entry["id"]] = actor
    print(json.dumps({"python": sys.version.split()[0], "exe": sys.executable, "torch": torch.__version__,
                      "vizdoom": importlib.metadata.version("vizdoom"), "cuda_available": torch.cuda.is_available(),
                      "wrapper": inspect.getfile(wrapper), "threads": torch.get_num_threads(), "headless": True}), flush=True)
    audit.EPISODES = output
    ledger = output / "episodes.jsonl"
    previous, issues = comparison.read_rows(ledger)
    if [issue for issue in issues if issue["reason"] != "episode-log-missing"]:
        raise ValueError(f"Unrecognized prior evidence: {issues}")
    attempted = {row_key(row): row for _, row in previous}
    if len(attempted) != len(previous):
        raise ValueError("Duplicate historical attempts in the frozen ledger")
    with ledger.open("a", encoding="utf-8", newline="\n") as file:
        for spec in maps:
            for seed in spec["seeds"]:
                for entry in models:
                    identity = spec["id"], seed, entry["id"]
                    prior = attempted.get(identity)
                    if prior:
                        if prior["checkpoint_sha256"] != entry["sha256"] or prior["wad_sha256"] != spec["wad_sha256"]:
                            raise ValueError(f"Previously attempted identity changed: {identity}")
                        continue
                    check_gate(audit.STORAGE)
                    if audit.digest(entry["weight_path"]) != entry["sha256"]:
                        raise ValueError(f"Checkpoint changed before simulation: {entry['id']}")
                    result = audit.run_episode(spec, seed, actors[entry["id"]], entry, api)
                    file.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                    brief = ("model", "map_id", "engine_seed", "status", "decisions", "alive", "exit", "error")
                    print(json.dumps({key: result[key] for key in brief}), flush=True)
                    if result["status"] == "storage_or_handoff_stop":
                        finish_summary(output, maps, models)
                        return
    summary = finish_summary(output, maps, models)
    print(json.dumps({"counts": summary["counts"], "summary": str(output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
