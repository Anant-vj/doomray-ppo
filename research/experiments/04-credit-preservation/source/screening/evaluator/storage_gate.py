"""Storage and memory gates for the external assessment."""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "manifests/storage_policy.json"


class GateClosed(RuntimeError):
    pass


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in ("physical_total", "physical_available", "commit_total", "commit_available",
                     "virtual_total", "virtual_available", "extended_virtual_available")
    ]


def storage_policy():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline = Path(config["baseline"]).read_bytes()
    if hashlib.sha256(baseline).hexdigest() != config["baseline_sha256"]:
        raise GateClosed("Original storage baseline hash mismatch.")
    limit, reserve = config["absolute_c_decline_limit_bytes"], config["start_reserve_bytes"]
    if limit != 7_000_000_000 or reserve != 500_000_000 or config["c_decline_start_gate_bytes"] != limit - reserve:
        raise GateClosed("Unexpected cap or reserve in assessment storage policy.")
    return config, json.loads(baseline)["internal_free_bytes"]


def snapshot(policy, *, startup=False, actor_loaded=False):
    config, baseline = policy
    c, d = shutil.disk_usage("C:/"), shutil.disk_usage("D:/")
    memory = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise GateClosed("Windows memory measurement unavailable.")
    decline = max(0, baseline - c.free)
    gate = config["c_decline_start_gate_bytes"] if startup else config["absolute_c_decline_limit_bytes"] - 200_000_000
    if startup:
        memory_floor = config["minimum_available_memory_bytes_with_actor_loaded" if actor_loaded else "minimum_available_memory_bytes"]
    else:
        memory_floor = config["minimum_available_memory_bytes_during_episode"]
    result = {
        "decline_bytes": decline, "threshold_bytes": gate, "c_free_bytes": c.free, "d_free_bytes": d.free,
        "physical_available_bytes": memory.physical_available, "commit_available_bytes": memory.commit_available,
        "required_available_memory_bytes": memory_floor,
        "original_baseline_free_bytes": baseline,
        "allowed": decline <= gate and c.free >= config["minimum_absolute_c_free_bytes"] and
        d.free >= config["minimum_d_free_bytes"] and memory.commit_available >= 1_000_000_000 and
        memory.physical_available >= memory_floor,
    }
    if not result["allowed"]:
        raise GateClosed(json.dumps(result))
    return result
