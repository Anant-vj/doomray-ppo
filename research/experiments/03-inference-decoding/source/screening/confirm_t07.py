"""Confirm the single approved T07 nominee on the separately frozen local families."""

from datetime import datetime
from pathlib import Path
import sys

import evaluate_screen as evaluation


ROOT = evaluation.ROOT
OUT = ROOT / "confirmation-results"
SCREEN_SUMMARY_SHA = "7e8d14b6d8decbf673219f5cb86436b2aee10bbbff062d611313e7478cb2bd11"
SCREEN_LEDGER_SHA = "31954cb637341aa01000202073271471df6a904874f5bc6a7546f025d0792dc9"


def summarize(protocol, models):
    maps = protocol["screening_maps"]
    comparison = evaluation.comparison
    comparison.MODELS = tuple(model["id"] for model in models)
    plans = {"stages": {"minimum": {"ordered_map_ids": [spec["id"] for spec in maps],
                                   "planned_episodes": protocol["expected_episodes"]}}, "maps": maps}
    ledger = OUT / "episodes.jsonl"
    rows, issues = comparison.read_rows(ledger)
    summary = comparison.build_summary(rows, comparison.frozen_units(plans))
    candidates = {model["id"]: [row for _, row in rows if row["model"] == model["id"] and row["status"] == "completed"]
                  for model in models}
    legal = not issues and not summary["attempt_issues"] and summary["counts"]["completed_episodes"] == 8
    gate = evaluation.measure_gate(candidates["T1-control"], candidates["T07"], maps, legal)
    gate["complete_four_legal_cases_per_mode"] = gate.pop("complete_eight_legal_cases")
    gate["passes_confirmation"] = gate.pop("passes_local_screen") and len(gate["material_gain_families"]) == len(maps)
    summary.update(input_issues=issues, confirmation_gate=gate, source_ledger_sha256=evaluation.audit.digest(ledger),
                   screening_summary_sha256=SCREEN_SUMMARY_SHA, screening_ledger_sha256=SCREEN_LEDGER_SHA,
                   screening_protocol_sha256=evaluation.PROTOCOL_SHA,
                   confirmation_manifest_sha256=protocol["withheld_confirmation_manifest_sha256"])
    temporary = OUT / "summary.tmp.json"
    temporary.write_bytes(evaluation.encode(summary))
    temporary.replace(OUT / "summary.json")
    print(evaluation.encode({"counts": summary["counts"], "confirmation_pass": gate["passes_confirmation"],
                             "storage_after": evaluation.snapshot(evaluation.audit.STORAGE),
                             "summary": str(OUT / "summary.json")}).decode(), flush=True)


def main():
    if Path(sys.executable).resolve().drive.upper() != "D:":
        raise ValueError("Use the resident competition environment on D:")
    if len(sys.argv) != 2 or sys.argv[1] != "--run-approved-t07":
        raise ValueError("T07 confirmation requires an explicit approval invocation")
    evaluation.audit.STOP = evaluation.EXPERIMENT / "stop-optional.flag"
    evaluation.audit.STORAGE = evaluation.storage_policy()
    opening = evaluation.check_gate()
    protocol, repo = evaluation.verify_protocol()
    summary_path = ROOT / "results/summary.json"
    if evaluation.audit.digest(summary_path) != SCREEN_SUMMARY_SHA:
        raise ValueError("Closed screening summary changed")
    if evaluation.audit.digest(ROOT / "results/episodes.jsonl") != SCREEN_LEDGER_SHA:
        raise ValueError("Closed screening ledger changed")
    screening = evaluation.read_json(summary_path)
    if not screening["promotion_screen"]["T07"]["passes_local_screen"]:
        raise ValueError("T07 did not qualify under the frozen screening gate")
    roster = [entry for entry in protocol["roster"] if entry["id"] in ("T1-control", "T07")]
    if [entry["id"] for entry in roster] != ["T1-control", "T07"]:
        raise ValueError("Unexpected confirmation roster")
    decision = {
        "selected_at_ist": datetime.now(evaluation.audit.IST).isoformat(timespec="seconds"),
        "nominee": "T07",
        "selection_rationale": "T07 adds one exit, one survivor and one verified unlock, preserving keys and recorded player damage versus T1-control.",
        "roster": roster,
        "screening_protocol_sha256": evaluation.PROTOCOL_SHA,
        "screening_summary_sha256": SCREEN_SUMMARY_SHA,
        "screening_ledger_sha256": SCREEN_LEDGER_SHA,
        "decoder_sha256": protocol["decoder_sha256"],
        "confirmation_manifest_sha256": protocol["withheld_confirmation_manifest_sha256"],
        "gate": "Complete legal sample; preserve exits, survival, keys, unlocks on each family and total player damage. Material gain on both families and replicated on both seeds in at least one of them.",
        "further_nominees": False,
    }
    if (OUT / "nomination.json").exists():
        previous = evaluation.read_json(OUT / "nomination.json")
        decision["selected_at_ist"] = previous["selected_at_ist"]
    evaluation.freeze(OUT / "nomination.json", evaluation.encode(decision))
    print(evaluation.encode({"nomination": str(OUT / "nomination.json"), "storage_at_open": opening,
                             "screening_summary_sha256": SCREEN_SUMMARY_SHA}).decode(), flush=True)

    seal_dir = ROOT / "withheld_confirmation"
    seal = seal_dir / "manifest.json"
    if evaluation.audit.digest(seal) != decision["confirmation_manifest_sha256"]:
        raise ValueError("The preregistered new confirmation manifest changed")
    maps = evaluation.read_json(seal)["maps"]
    if len(maps) != 2 or any(len(spec["seeds"]) != 2 for spec in maps):
        raise ValueError("Expected exactly two separately frozen families with two seeds each")
    for spec in maps:
        wad = Path(spec["wad"]).resolve(strict=True)
        if not wad.is_relative_to(seal_dir) or evaluation.audit.digest(wad) != spec["wad_sha256"]:
            raise ValueError("A new confirmation map changed or points outside its seal")
    exact = {**protocol, "screening_maps": maps, "roster": roster, "expected_episodes": 8}
    evaluation.OUTPUT = OUT
    evaluation.LEDGER = OUT / "episodes.jsonl"
    evaluation.summarize = summarize
    evaluation.run(exact, repo)


if __name__ == "__main__":
    main()
