"""Export only the prespecified terminal update to an independent C749 checkout."""
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import subprocess
import torch

root = Path(__file__).resolve().parent
read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
plan = read(root / "protocol.json")
assert sha256((root / "protocol.json").read_bytes()).hexdigest() == "cdfddb2aa7ab26f3fb7a500211a622481268eab698e12b7edb252d67eceb4931"
receipt = read(root / "logs/learner-exit.json")
assert receipt["exit_code"] == 0 and datetime.fromisoformat(receipt["finished"]) < datetime.fromisoformat(plan["hard_training_deadline"])
rows = [json.loads(line) for line in (root / "logs/key-recovery.jsonl").read_text().splitlines()]
assert [row["update"] for row in rows] == list(range(750, 846))
allowlist = {entry["name"] + ".wad:MAP01" for entry in plan["maps"]["drills"]}
for i, row in enumerate(rows, 1):
    assert (row["transitions_this_run"], row["gradients"]) == (i*256, 7614+i*6)
    assert row["history_mode"] == "exact_raw_four" and not row["stopped_for_instability"]
    assert set(row["layout_decisions"]) <= allowlist and sum(row["layout_decisions"].values()) == 256
    assert not row["stopped_for_memory"] and not row["stopped_for_storage"]
saved = root / "checkpoints/key-recovery/resume.pt"
model = torch.load(saved, weights_only=True, map_location="cpu")
assert model["memory"] == "stack" and model["updates"] == 845 and model["gradients"] == 8190
repo = root / "candidate"
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == plan["official_base"]
parent = torch.load(root / "c749-parent.pt", weights_only=True, map_location="cpu")
assert parent["weights"].keys() == model["weights"].keys()
changed_tensors = sum(not torch.equal(v, parent["weights"][k]) for k,v in model["weights"].items())
assert changed_tensors > 0
export = repo / "submission/weights/policy.pt"
assert len(export.read_bytes()) < 200 and plan["source_weight_sha256"] in export.read_text()
torch.save(dict(memory="stack", weights=model["weights"]), export)
restored = torch.load(export, weights_only=True, map_location="cpu")
assert all(torch.equal(restored["weights"][k], v) for k, v in model["weights"].items())
data = dict(parent_sha256=plan["source_weight_sha256"], init="published inference weights, fresh Adam",
            local_update=845, new_rollouts=len(rows), live_transitions=rows[-1]["transitions_this_run"],
            new_optimizer_steps=rows[-1]["gradients"]-7614, changed_tensors=changed_tensors,
            max_initial_log_ratio_error=max(row["ppo"]["initial_log_ratio_abs_max"] for row in rows),
            measured_training_sps=rows[-1]["sps"], trainer=receipt,
            training_sha256=sha256(saved.read_bytes()).hexdigest(), training_path=str(saved),
            inference_sha256=sha256(export.read_bytes()).hexdigest(), inference_path=str(export))
with (root / "candidate-manifest.json").open("x") as f:
    json.dump(data, f, indent=2)
print(json.dumps(data, indent=2))
