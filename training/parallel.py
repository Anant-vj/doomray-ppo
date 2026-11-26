"""Parallel headless rollouts without sending image buffers through worker pipes."""

import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from training.storage import project_root


def _policy_observation(obs):
    return {key: obs[key] for key in ("rays", "visible_entities", "vitals")}


def _disable_native_depth():
    from doom_env import wrapper

    factory = wrapper.vd

    class NoDepthGame(factory.DoomGame):
        def set_depth_buffer_enabled(self, enabled):
            super().set_depth_buffer_enabled(False)

    # The wrapper's render_buffers flag skips returning images, but still asks
    # ViZDoom to generate depth. Override that request only in this worker.
    engine = SimpleNamespace(**vars(factory))
    engine.DoomGame = NoDepthGame
    wrapper.vd = engine


def _episode_worker(connection, ticks, render_buffers, workdir, reward_profile):
    from training.game import Episode

    game = None
    try:
        os.chdir(workdir)
        if not render_buffers:
            _disable_native_depth()
        while True:
            command, data = connection.recv()
            if command == "close":
                break
            if command == "reset":
                layout, seed = data
                if game is None or (game.path, game.map_name) != layout:
                    if game is not None:
                        game.close()
                    game = Episode(layout, ticks, render_buffers=render_buffers, reward_profile=reward_profile)
                result = _policy_observation(game.reset(seed))
            else:
                obs, reward, terminated, truncated = game.step(data)
                metrics = game.metrics() if terminated or truncated else None
                result = _policy_observation(obs), reward, terminated, truncated, metrics
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        try:
            if game is not None:
                game.close()
        finally:
            connection.close()


class ParallelEpisodes:
    """Keep an independent ViZDoom episode in each Windows-spawned worker."""

    def __init__(self, layouts, ticks, seeds, *, render_buffers=False, reward_profile="control", timeout=45):
        ctx = mp.get_context("spawn")
        self._workers = []
        self._timeout = timeout
        temp_root = (project_root() / "temp").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self._workdir = tempfile.TemporaryDirectory(prefix="doom-workers-", dir=temp_root)
        try:
            for index, _ in enumerate(layouts):
                workdir = Path(self._workdir.name) / str(index)
                workdir.mkdir()
                parent, child = ctx.Pipe()
                worker_args = (child, ticks, render_buffers, workdir, reward_profile)
                process = ctx.Process(target=_episode_worker, args=worker_args, daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._workers.append((parent, process))
            starts = dict(enumerate(zip(layouts, seeds, strict=True)))
            self.observations = list(self.reset(starts).values())
        except BaseException:
            self.close()
            raise

    def _request(self, command, messages):
        deadline = time.monotonic() + self._timeout
        result = {}
        for index, message in messages.items():
            self._workers[index][0].send((command, message))
        for index in messages:
            connection, _ = self._workers[index]
            if not connection.poll(max(0, deadline - time.monotonic())):
                raise RuntimeError(f"ViZDoom worker {index} timed out during {command}.")
            try:
                ok, reply = connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError(f"ViZDoom worker {index} disconnected during {command}.") from exc
            if not ok:
                raise RuntimeError(f"ViZDoom worker {index} failed during {command}.\n{reply}")
            result[index] = reply
        return result

    def step(self, actions):
        if len(actions) != len(self._workers):
            raise ValueError("Provide exactly one action per ViZDoom worker.")
        return list(self._request("step", dict(enumerate(actions))).values())

    def reset(self, starts):
        layouts = {
            index: ((Path(wad).resolve(), name), seed) for index, ((wad, name), seed) in starts.items()
        }
        return self._request("reset", layouts)

    def close(self):
        workers, self._workers = self._workers, []
        for connection, _ in workers:
            try:
                connection.send(("close", None))
            except (EOFError, OSError):
                pass
            finally:
                connection.close()
        deadline = time.monotonic() + 5
        for _, process in workers:
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        self._workdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
