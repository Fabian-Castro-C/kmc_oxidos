"""
Gymnasium environment for TiO2 thin film growth with KMC.

This module integrates the KMC simulator as a Gymnasium environment
for reinforcement learning.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ..kmc.lattice import SpeciesType
from ..kmc.simulator import KMCSimulator

logger = logging.getLogger(__name__)


class TiO2GrowthEnv(gym.Env):  # type: ignore[misc]
    """
    Gymnasium environment for TiO2 thin film growth.

    Observation Space:
        Local neighborhood of each active site (species types of neighbors).

    Action Space:
        Discrete selection of events from the KMC event catalog.

    Reward:
        Negative energy change (-ΔE) to drive system to lower energy states.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lattice_size: tuple[int, int, int] = (10, 10, 10),
        temperature: float = 600.0,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        neighborhood_radius: int = 1,
    ) -> None:
        """
        Initialize environment.

        Args:
            lattice_size: Lattice dimensions.
            temperature: Temperature (K).
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            neighborhood_radius: Radius for local observations.
        """
        super().__init__()

        self.lattice_size = lattice_size
        self.temperature = temperature
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.neighborhood_radius = neighborhood_radius

        # Initialize KMC simulator
        self.simulator = KMCSimulator(
            lattice_size=lattice_size,
            temperature=temperature,
            deposition_rate=deposition_rate,
        )

        # Define observation space: global statistics
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(7,),
            dtype=np.int32,
        )

        # Action space with masking support
        self.max_action_space = 1000
        self.action_space = spaces.Discrete(self.max_action_space)

        self.current_step = 0
        self.episode_reward = 0.0

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """Return mask of valid actions for current state."""
        n_events = len(self.simulator.event_catalog)
        mask = np.zeros(self.max_action_space, dtype=np.bool_)
        if n_events > 0:
            mask[:n_events] = True
        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[npt.NDArray[np.int32], dict[str, Any]]:
        """
        Reset environment.

        Args:
            seed: Random seed.
            options: Additional options (reserved for future use).

        Returns:
            Initial observation and info dict.
        """
        super().reset(seed=seed)

        self.simulator.reset()
        self.current_step = 0
        self.episode_reward = 0.0

        obs = self._get_observation()
        info = self._get_info()
        info["action_mask"] = self.action_masks()

        return obs, info

    def step(self, action: int) -> tuple[npt.NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with action masking support."""
        if not self.action_masks()[action]:
            obs = self._get_observation()
            info = self._get_info()
            info["action_mask"] = self.action_masks()
            info["invalid_action"] = True
            return obs, -10.0, False, False, info

        event = self.simulator.event_catalog.events[action]

        delta_e = self.simulator.rate_calculator.calculate_energy_change(
            self.simulator.lattice, event.site_index, event.target_index
        )

        self.simulator.execute_event(event)
        self.simulator.advance_time()
        self.simulator.step += 1

        reward = -delta_e
        self.episode_reward += reward

        obs = self._get_observation()

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = len(self.simulator.event_catalog) == 0

        info = self._get_info()
        info["action_mask"] = self.action_masks()
        info["invalid_action"] = False

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> npt.NDArray[np.int32]:
        """
        Get observation for current state.

        For multi-agent, we'd return observations for all agents.
        Here we return a simplified global observation.

        Returns:
            Observation array.
        """
        # Simplified: return composition as observation
        composition = self.simulator.lattice.get_composition()
        obs = np.array(
            [
                composition[SpeciesType.TI],
                composition[SpeciesType.O],
                composition[SpeciesType.VACANT],
                len(self.simulator.event_catalog),
                self.simulator.step,
                self.current_step,
                0,  # Padding
            ],
            dtype=np.int32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary."""
        return {
            "step": self.current_step,
            "kmc_step": self.simulator.step,
            "kmc_time": self.simulator.time,
            "n_events": len(self.simulator.event_catalog),
            "episode_reward": self.episode_reward,
            **self.simulator.get_statistics(),
        }

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode == "human":
            logger.info(f"Step: {self.current_step}")
            logger.info(f"KMC Time: {self.simulator.time:.2e}s")
            logger.info(f"Composition: {self.simulator.lattice.get_composition()}")
            logger.info(f"Episode Reward: {self.episode_reward:.2f}")
