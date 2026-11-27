"""Compute matched results and apply the predeclared promotion rules once."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parent
plan = json.loads((root / "protocol.json").read_text())
metrics = ("exit", "keys", "unlocks", "early_death", "invalid_actions")
target = plan["evaluation"]["target"]
expected = {(m["name"], seed, policy) for m in plan["maps"]["reserved"]
            for seed, policy in zip(plan["evaluation"]["engine_seeds"], plan["evaluation"]["policy_seeds"], strict=True)}
all_rows, paired, issues = {}, [], []
for model in ("C749", "candidate"):
    ledger = root / ("evaluation-" + model + ".jsonl")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 8
    all_rows[model] = {(r["layout"],r["engine_seed"],r["policy_seed"]):r for r in rows}
    assert all_rows[model].keys() == expected
for pair in sorted(expected):
    a,b = [r[pair] for r in all_rows.values()]
    assert a["wad_sha256"] == b["wad_sha256"] and a["initial_observation_sha256"] == b["initial_observation_sha256"]
    assert a["cap_ticks"] == b["cap_ticks"] == plan["evaluation"]["tick_cap"]
    if b["invalid_actions"] or a["invalid_actions"]:
        issues.append(f"{pair[0]} seed {pair[1]} recorded an invalid action")
    for metric in ("exit", "keys", "unlocks"):
        if b[metric] < a[metric]:
            issues.append(f"{pair[0]} seed {pair[1]} lost {metric}")
    if b["death"] > a["death"] or b["early_death"] > a["early_death"]:
        issues.append(f"{pair[0]} seed {pair[1]} regressed on survival")
    paired.append(dict(layout=pair[0], engine_seed=pair[1], policy_seed=pair[2], C749=a,candidate=b))
totals = {}
for layout in [r["name"] for r in plan["maps"]["reserved"]]:
    totals[layout] = {}
    for model, rows in all_rows.items():
        matched = [r for (name,_,_), r in rows.items() if name == layout]
        assert len(matched) == 2
        totals[layout][model] = {k:sum(r[k] for r in matched)
                                 for k in (*metrics, "death", "decisions", "player_hits", "player_damage")}
    a,b = totals[layout].values()
    if layout in target and b["exit"] <= a["exit"] and b["keys"] <= a["keys"]:
        issues.append(f"{layout} showed no additional exit or key pickup")
summary = dict(per_layout=totals, matched=paired, gate_failures=issues,
               gameplay_gate_passed=not issues, ledger_sha256={model: sha256((root / ("evaluation-"+model+".jsonl")).read_bytes()).hexdigest()
                                                               for model in all_rows})
with (root / "matched-summary.json").open("x", encoding="utf-8") as f:
    json.dump(summary,f,indent=2)
lines = ["# C749 key recovery examination", "",
         "Each cell shows counts from two episodes as **C749 / update 845**. An early death occurs within 525 of 2,100 ticks.",
         "", "| Layout | Exits | Keys | Unlocks | Early deaths | Invalid actions |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
for name, by_model in totals.items():
    a,b = by_model.values()
    lines.append("| "+name+" | "+" | ".join(f"{a[k]} / {b[k]}" for k in metrics)+" |")
lines.extend(["", "## Seed-level results", "", "| Layout | Engine seed | Policy seed | C749 outcome | Candidate outcome | Keys | Unlocks | Early deaths | Invalid actions |",
              "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |"])
for row in paired:
    a,b = row["C749"],row["candidate"]
    lines.append(f"| {row['layout']} | {row['engine_seed']} | {row['policy_seed']} | {a['termination']} | {b['termination']} | "
                 + " | ".join(f"{a[k]} / {b[k]}" for k in metrics[1:])+" |")
lines.extend(["", "## Predeclared gameplay gate", "", *(["- "+msg for msg in issues] or ["Passed the matched gameplay gate."])])
(root / "matched-results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))
