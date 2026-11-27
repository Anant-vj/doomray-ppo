"""Recompute archived screening and confirmation aggregates with no engine."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    source = bundle / "source/screening/evaluator/comparison.py"
    spec = importlib.util.spec_from_file_location("archived_comparison", source)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    evaluation = bundle / "evaluation"
    for stage in ("screening", "confirmation"):
        saved_path = evaluation / f"{stage}-summary.json"
        if not saved_path.is_file():
            continue
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        match = json.loads((evaluation / f"{stage}-maps.json").read_text(encoding="utf-8"))
        maps = match["maps"]
        comparison.MODELS = tuple(saved["models_order"])
        frozen = {"maps": maps, "stages": {"minimum": {
            "ordered_map_ids": [item["id"] for item in maps],
            "planned_episodes": sum(len(item["seeds"]) for item in maps) * len(comparison.MODELS),
        }}}
        rows, issues = comparison.read_rows(evaluation / f"{stage}.jsonl")
        actual = comparison.build_summary(rows, comparison.frozen_units(frozen))
        for key in ("counts", "models", "matched", "pairwise", "tiers", "attempt_issues"):
            if actual[key] != saved[key]:
                raise ValueError(f"{stage} disagrees with the archived {key}.")
        if issues != saved["input_issues"]:
            raise ValueError(f"{stage} mismatch in input issues.")
        print(f"{stage}: {actual['counts']['completed_episodes']} completed, "
              f"{sum(row['decisions'] for _, row in rows)} recorded decisions. Aggregates match.")


if __name__ == "__main__":
    main()
