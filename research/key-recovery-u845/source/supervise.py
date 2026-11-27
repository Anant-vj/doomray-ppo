"""Supervise the one declared learner with the wall-clock deadline."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
plan = json.loads((root / "protocol.json").read_text())
deadline = datetime.fromisoformat(plan["hard_training_deadline"]).timestamp()
started = datetime.now().astimezone().isoformat()
with (root / "logs/learner-console.txt").open("x", encoding="utf-8") as log:
    proc = subprocess.Popen([sys.executable, "-B", "-u", str(root / "train_one.py")],
                            stdout=log, stderr=subprocess.STDOUT, cwd=root / "temp")
    try:
        proc.wait(timeout=max(0, deadline - time.time() - 10))
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=True)
        proc.wait(timeout=8)
    end = datetime.now().astimezone().isoformat()
    receipt = dict(started=started, finished=end, exit_code=proc.returncode)
    (root / "logs/learner-exit.json").write_text(json.dumps(receipt, indent=2))
    print(receipt, flush=True)
    sys.exit(proc.returncode)
