"""Run the official validator with local desktop presentation enabled."""
import runpy
from pathlib import Path

from training.visible import enable_windows

if __name__ == "__main__":
    enable_windows()
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "validate.py"), run_name="__main__")
