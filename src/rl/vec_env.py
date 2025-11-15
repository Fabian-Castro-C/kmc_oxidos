"""
Vectorized parallel environments for faster training.

Uses multiprocessing to run multiple environments in parallel.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from src.data.tio2_parameters import TiO2Parameters
from src.rl.agent_env import AgentBasedTiO2Env


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: Any,
) -> None:
    """Worker process for running an environment."""
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    # Auto-reset on episode end
                    obs, reset_info = env.reset()
                    info["final_observation"] = obs
                    info["final_info"] = reset_info
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send(
                    (
                        env.single_agent_observation_space,
                        env.action_space,
                        env.global_feature_space,
                    )
                )
            elif cmd == "get_action_mask":
                mask = env.get_action_mask()
                remote.send(mask)
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print("Worker interrupted")
    finally:
        env.close()


class CloudpickleWrapper:
    """Wrapper to make functions picklable for multiprocessing."""

    def __init__(self, x: Any) -> None:
        self.x = x

    def __getstate__(self) -> Any:
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob: Any) -> None:
        import pickle

        self.x = pickle.loads(ob)


class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environments in parallel subprocesses.

    This dramatically improves sample collection efficiency by running N environments
    simultaneously, effectively multiplying your SPS by N (up to CPU core limits).
    """

    def __init__(
        self,
        lattice_size: tuple[int, int, int],
        num_envs: int = 8,
        max_steps: int = 256,
    ) -> None:
        """
        Initialize vectorized environments.

        Args:
            lattice_size: Size of the lattice (nx, ny, nz)
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode
        """
        self.num_envs = num_envs
        self.lattice_size = lattice_size
        self.max_steps = max_steps
        self.waiting = False
        self.closed = False

        # Create environment factory
        def make_env() -> AgentBasedTiO2Env:
            params = TiO2Parameters()
            return AgentBasedTiO2Env(
                lattice_size=lattice_size,
                tio2_parameters=params,
                max_steps=max_steps,
            )

        # Start worker processes
        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(num_envs)], strict=False
        )
        self.ps = [
            mp.Process(
                target=_worker,
                args=(work_remote, remote, CloudpickleWrapper(make_env)),
                daemon=True,
            )
            for (work_remote, remote) in zip(self.work_remotes, self.remotes, strict=False)
        ]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # Get spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        (
            self.single_agent_observation_space,
            self.action_space,
            self.global_feature_space,
        ) = self.remotes[0].recv()

    def step_async(self, actions: list[Any]) -> None:
        """Send step commands to all environments (non-blocking)."""
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {len(actions)}"
            )
        for remote, action in zip(self.remotes, actions, strict=False):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        """Wait for step commands to complete and return results."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminateds, truncateds, infos = zip(*results, strict=False)
        return list(obs), list(rews), list(terminateds), list(truncateds), list(infos)

    def step(self, actions: list[Any]) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        """Step all environments synchronously."""
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> tuple[list[dict], list[dict]]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results, strict=False)
        return list(obs), list(infos)

    def get_action_masks(self) -> list[np.ndarray]:
        """Get action masks from all environments."""
        for remote in self.remotes:
            remote.send(("get_action_mask", None))
        return [remote.recv() for remote in self.remotes]

    def close(self) -> None:
        """Close all environment processes."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if not self.closed:
            self.close()
