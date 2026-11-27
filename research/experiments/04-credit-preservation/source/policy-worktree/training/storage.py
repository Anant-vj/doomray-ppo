"""USB workspace usage and internal-drive budget, in bytes."""
import json
import ctypes
import hashlib
import os
import shutil
import sys
from pathlib import Path


def project_root():
    return Path(os.environ.get("DOOM_WORK", Path(__file__).resolve().parents[3]))


def internal_limit():
    limit = int(os.environ.get("DOOM_INTERNAL_LIMIT_BYTES", "2000000000"))
    if not 500_000_000 < limit <= 7_000_000_000:
        raise ValueError("Choose an internal storage budget between 0.5 and 7 GB.")
    return limit


def _windows_memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (key, ctypes.c_uint64) for key in
            ("physical", "available", "pagefile", "commit_available", "virtual", "free_virtual", "extended")
        ]

    state = Status(length=ctypes.sizeof(Status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise RuntimeError("Windows memory measurement is unavailable.")
    return {"physical_available_bytes": state.available, "commit_available_bytes": state.commit_available}


def snapshot(detailed=False):
    root = project_root()
    disk = os.environ.get("SystemDrive", root.anchor) if os.name == "nt" else root.anchor
    free = shutil.disk_usage(disk + "/" if disk.endswith(":") else disk).free
    external_baseline = os.environ.get("DOOM_STORAGE_BASELINE")
    path = Path(external_baseline) if external_baseline else root / "logs" / "storage-baseline.json"
    if not external_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"internal_free_bytes": free}), encoding="utf-8")
    baseline = path.read_bytes()
    if external_baseline and hashlib.sha256(baseline).hexdigest() != os.environ.get("DOOM_STORAGE_BASELINE_SHA256"):
        raise RuntimeError("Original drive baseline hash mismatch.")
    delta = max(0, json.loads(baseline)["internal_free_bytes"] - free)
    limit = internal_limit()
    data = dict(internal_free_bytes=free, internal_delta_bytes=delta,
                external_free_bytes=shutil.disk_usage(root).free, internal_limit_bytes=limit)
    if os.name == "nt":
        data.update(_windows_memory())
    if detailed:
        for name in ("checkpoints", "logs", "."):
            data[name] = sum(p.stat().st_size for p in (root / name).rglob("*") if p.is_file())
    warning = limit - 500_000_000
    if delta >= warning:
        print(f"Internal free space has fallen by at least {warning / 1e9:g} GB.", file=sys.stderr)
    if delta >= limit:
        raise RuntimeError(f"Internal free space fell by {limit / 1e9:g} GB. The run has stopped.")
    return data


if __name__ == "__main__":
    print(json.dumps(snapshot(True)))
