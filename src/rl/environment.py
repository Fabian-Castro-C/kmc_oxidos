"""
Gymnasium environment for TiO2 thin film growth with KMC.

This module integrates the KMC simulator as a Gymnasium environment
for reinforcement learning.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ..kmc.lattice import SpeciesType
from ..kmc.simulator import KMCSimulator


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

        # Define observation space: local neighborhood for each agent
        # For simplicity, we use a flattened representation
        # Each agent observes: species of 6 nearest neighbors + own species
        n_neighbor_features = 7  # 6 neighbors + self
        self.observation_space = spaces.Box(
            low=0, high=len(SpeciesType), shape=(n_neighbor_features,), dtype=np.int32
        )

        # Action space: select from available events
        # We'll dynamically handle this, but define a large enough space
        self.action_space = spaces.Discrete(1000)

        self.current_step = 0
        self.episode_reward = 0.0

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

        # Reset simulator
        self.simulator.reset()
        self.current_step = 0
        self.episode_reward = 0.0

        # Build initial event list
        self.simulator.build_event_list()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[npt.NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action index (event selection).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Get event from catalog
        if action >= len(self.simulator.event_catalog):
            # Invalid action, penalize
            obs = self._get_observation()
            return obs, -1.0, False, False, self._get_info()

        event = self.simulator.event_catalog.events[action]

        # Calculate energy change before event
        delta_e = self.simulator.rate_calculator.calculate_energy_change(
            self.simulator.lattice, event.site_index, event.target_index
        )

        # Execute event
        self.simulator.execute_event(event)
        self.simulator.advance_time()
        self.simulator.step += 1

        # Rebuild event list
        self.simulator.build_event_list()

        # Calculate reward (negative energy change)
        reward = -delta_e
        self.episode_reward += reward

        # Get new observation
        obs = self._get_observation()

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = len(self.simulator.event_catalog) == 0

        info = self._get_info()

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
            print(f"Step: {self.current_step}")
            print(f"KMC Time: {self.simulator.time:.2e}s")
            print(f"Composition: {self.simulator.lattice.get_composition()}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
