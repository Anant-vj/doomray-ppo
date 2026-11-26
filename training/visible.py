"""Local game windows. The competition worker imports only submission/."""
import os
import types

import vizdoom
from doom_env import wrapper


class WindowedGame(vizdoom.DoomGame):
    def set_window_visible(self, visible):
        super().set_window_visible(True)


def enable_windows():
    if os.environ.get("SDL_VIDEODRIVER", "").lower() in {"dummy", "offscreen"}:
        raise RuntimeError("A desktop SDL video driver is required to show the game.")
    # The starter creates and initializes its own DoomGame in one constructor.
    # Change only the constructor used by the local wrapper; keep ViZDoom itself intact.
    windowed = types.SimpleNamespace(**vars(vizdoom))
    windowed.DoomGame = WindowedGame
    wrapper.vd = windowed
