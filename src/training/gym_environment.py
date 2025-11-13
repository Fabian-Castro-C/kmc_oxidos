"""
Gymnasium environment for training SwarmThinkers policies.

This module wraps the KMC simulator with SwarmEngine in a Gymnasium-compatible
environment for Stable-Baselines3 training.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ..analysis import calculate_roughness
from ..kmc.lattice import SpeciesType
from ..kmc.simulator import KMCSimulator
from ..rl import (
    SwarmEngine,
    create_adsorption_swarm_policy,
    create_desorption_swarm_policy,
    create_diffusion_swarm_policy,
    create_reaction_swarm_policy,
)

logger = logging.getLogger(__name__)


class TiO2GrowthEnv(gym.Env):  # type: ignore[misc]
    """
    Gymnasium environment for TiO2 thin film growth with SwarmThinkers.

    This environment integrates:
    - KMC simulator for physical dynamics
    - SwarmEngine for multi-policy proposals
    - Morphology-based rewards (roughness, coverage)

    Observation Space:
        Box: [height_mean, height_std, roughness, coverage, n_Ti, n_O, n_vacant]
        - height_mean: Mean surface height
        - height_std: Standard deviation of height
        - roughness: Surface roughness (nm)
        - coverage: Surface coverage fraction
        - n_Ti: Number of Ti atoms
        - n_O: Number of O atoms
        - n_vacant: Number of vacant sites

    Action Space:
        Discrete(n_swarm): Select which SwarmPolicy proposal to execute
        - 0: Adsorption policy proposal
        - 1: Diffusion policy proposal
        - 2: Reaction policy proposal
        - 3: Desorption policy proposal

    Reward:
        Multi-objective reward combining:
        - Negative roughness (smoother is better)
        - Positive coverage (more growth is better)
        - Penalty for high roughness/coverage ratio (discourage islands)
        - Penalty for low ESS (encourage diverse sampling)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lattice_size: tuple[int, int, int] = (8, 8, 5),
        temperature: float = 600.0,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        n_swarm: int = 4,
        n_proposals: int = 32,
        reward_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Gymnasium environment for SwarmThinkers training.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            temperature: Temperature (K).
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            n_swarm: Number of swarm policies (default 4).
            n_proposals: Number of proposals per policy (default 32).
            reward_weights: Weights for reward components:
                - roughness_weight: Weight for roughness penalty (default -1.0)
                - coverage_weight: Weight for coverage reward (default +0.5)
                - ratio_weight: Weight for roughness/coverage ratio penalty (default -0.3)
                - ess_weight: Weight for ESS penalty (default -0.1)
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.lattice_size = lattice_size
        self.temperature = temperature
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.n_swarm = n_swarm
        self.n_proposals = n_proposals
        self._seed = seed

        # Reward weights
        default_weights = {
            "roughness_weight": -1.0,
            "coverage_weight": 0.5,
            "ratio_weight": -0.3,
            "ess_weight": -0.1,
        }
        self.reward_weights = {**default_weights, **(reward_weights or {})}

        # Initialize KMC simulator
        self.simulator = KMCSimulator(
            lattice_size=lattice_size,
            temperature=temperature,
            deposition_rate=deposition_rate,
            seed=seed,
        )

        # Initialize swarm policies (using default observation_dim=51)
        diffusion_policy = create_diffusion_swarm_policy()
        adsorption_policy = create_adsorption_swarm_policy()
        desorption_policy = create_desorption_swarm_policy()
        reaction_policy = create_reaction_swarm_policy()

        # Initialize SwarmEngine with all policies
        self.swarm_engine = SwarmEngine(
            diffusion_policy=diffusion_policy,
            adsorption_policy=adsorption_policy,
            desorption_policy=desorption_policy,
            reaction_policy=reaction_policy,
            rate_calculator=self.simulator.rate_calculator,
            device="cpu",
        )

        # Observation space: global morphology statistics
        # [height_mean, height_std, roughness, coverage, n_Ti, n_O, n_vacant]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        # Action space: select which swarm policy to use
        self.action_space = spaces.Discrete(n_swarm)

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_roughness: list[float] = []
        self.episode_coverage: list[float] = []
        self.episode_ess: list[float] = []

        logger.info(
            f"Initialized TiO2GrowthEnv: lattice={lattice_size}, "
            f"n_swarm={n_swarm}, n_proposals={n_proposals}, max_steps={max_steps}"
        )

    def _get_observation(self) -> npt.NDArray[np.float32]:
        """
        Get current observation from simulator state.

        Returns:
            Observation array: [height_mean, height_std, roughness, coverage, n_Ti, n_O, n_vacant]
        """
        # Calculate surface heights
        heights = self.simulator.lattice.get_height_profile()
        height_mean = float(np.mean(heights))
        height_std = float(np.std(heights))

        # Calculate roughness (nm)
        roughness = calculate_roughness(heights)

        # Calculate coverage
        nx, ny, _ = self.lattice_size
        total_surface_sites = nx * ny
        occupied_sites = np.sum(heights > 0)
        coverage = float(occupied_sites / total_surface_sites)

        # Count species
        composition = self.simulator.lattice.get_composition()
        n_ti = float(composition.get(SpeciesType.TI, 0))
        n_o = float(composition.get(SpeciesType.O, 0))
        n_vacant = float(composition.get(SpeciesType.VACANT, 0))

        obs = np.array(
            [height_mean, height_std, roughness, coverage, n_ti, n_o, n_vacant],
            dtype=np.float32,
        )

        return obs

    def _compute_reward(self, prev_obs: npt.NDArray[np.float32]) -> float:
        """
        Compute multi-objective reward.

        Reward = w_r * Δroughness + w_c * Δcoverage + w_ratio * (roughness/coverage) + w_ess * ESS_penalty

        Args:
            prev_obs: Previous observation for computing deltas.

        Returns:
            Scalar reward value.
        """
        current_obs = self._get_observation()

        # Extract metrics
        prev_roughness = prev_obs[2]
        prev_coverage = prev_obs[3]
        current_roughness = current_obs[2]
        current_coverage = current_obs[3]

        # Compute deltas
        delta_roughness = current_roughness - prev_roughness  # Negative is better
        delta_coverage = current_coverage - prev_coverage  # Positive is better

        # Roughness/coverage ratio (penalize high roughness with low coverage)
        ratio_penalty = 0.0
        if current_coverage > 0.01:
            ratio_penalty = current_roughness / current_coverage

        # ESS penalty (from last swarm step)
        ess_penalty = 0.0
        if (
            hasattr(self.swarm_engine, "last_ess")
            and self.swarm_engine.last_ess is not None
            and self.swarm_engine.last_ess < 0.5
        ):
            # ESS should be close to 1.0; penalize if < 0.5
            ess_penalty = 0.5 - self.swarm_engine.last_ess

        # Weighted sum
        reward = (
            self.reward_weights["roughness_weight"] * delta_roughness
            + self.reward_weights["coverage_weight"] * delta_coverage
            + self.reward_weights["ratio_weight"] * ratio_penalty
            + self.reward_weights["ess_weight"] * ess_penalty
        )

        return float(reward)

    def _get_info(self) -> dict[str, Any]:
        """
        Get info dict with episode diagnostics.

        Returns:
            Info dictionary with current metrics.
        """
        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "time": self.simulator.time,
            "height_mean": obs[0],
            "height_std": obs[1],
            "roughness": obs[2],
            "coverage": obs[3],
            "n_Ti": obs[4],
            "n_O": obs[5],
            "n_vacant": obs[6],
        }

        # Add ESS if available
        if hasattr(self.swarm_engine, "last_ess") and self.swarm_engine.last_ess is not None:
            info["ess"] = self.swarm_engine.last_ess

        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options (reserved for future use).

        Returns:
            Initial observation and info dict.
        """
        super().reset(seed=seed)

        # Reset simulator
        self.simulator.reset()

        # Reset SwarmEngine (re-initialize policies)
        diffusion_policy = create_diffusion_swarm_policy()
        adsorption_policy = create_adsorption_swarm_policy()
        desorption_policy = create_desorption_swarm_policy()
        reaction_policy = create_reaction_swarm_policy()

        self.swarm_engine = SwarmEngine(
            diffusion_policy=diffusion_policy,
            adsorption_policy=adsorption_policy,
            desorption_policy=desorption_policy,
            reaction_policy=reaction_policy,
            rate_calculator=self.simulator.rate_calculator,
            device="cpu",
        )

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_roughness = []
        self.episode_coverage = []
        self.episode_ess = []

        obs = self._get_observation()
        info = self._get_info()

        logger.debug(f"Environment reset: obs={obs}")

        return obs, info

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Number of proposals to use for swarm step (action is interpreted as n_swarm multiplier).
                    action=0 -> n_swarm=4, action=1 -> n_swarm=8, action=2 -> n_swarm=16, action=3 -> n_swarm=32

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Store previous observation for reward computation
        prev_obs = self._get_observation()

        # Map action to n_swarm (number of proposals per policy)
        # action 0: minimal swarm (4 proposals)
        # action 1: small swarm (8 proposals)
        # action 2: medium swarm (16 proposals)
        # action 3: large swarm (32 proposals)
        n_swarm = 4 * (2**action)  # 4, 8, 16, 32

        # Execute one swarm step
        try:
            event, importance_weight = self.swarm_engine.run_step(
                self.simulator.lattice, n_swarm=n_swarm
            )

            if event is None:
                # No valid events available
                logger.warning("No valid events available, episode truncated")
                obs = self._get_observation()
                info = self._get_info()
                info["truncated_reason"] = "no_events"
                return obs, -10.0, False, True, info

            # Execute event in simulator
            self.simulator.execute_event(event)
            self.simulator.advance_time()
            self.simulator.step += 1

            # Store ESS for reward computation
            # ESS = 1 / (1 + var(weights)) ≈ 1 / weight for single-step
            # For simplicity, we'll use 1/weight as a proxy
            self.swarm_engine.last_ess = min(1.0, 1.0 / importance_weight)

        except Exception as e:
            logger.error(f"SwarmEngine step failed: {e}")
            # Return terminal state on error
            obs = self._get_observation()
            info = self._get_info()
            info["error"] = str(e)
            return obs, -10.0, True, False, info

        # Compute reward
        reward = self._compute_reward(prev_obs)
        self.episode_reward += reward

        # Get new observation
        obs = self._get_observation()

        # Track metrics
        self.episode_roughness.append(obs[2])
        self.episode_coverage.append(obs[3])
        if hasattr(self.swarm_engine, "last_ess") and self.swarm_engine.last_ess is not None:
            self.episode_ess.append(self.swarm_engine.last_ess)

        # Check termination conditions
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = len(self.simulator.event_catalog) == 0  # No more events

        # Add episode summary to info on episode end
        info = self._get_info()
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step,
                "final_roughness": obs[2],
                "final_coverage": obs[3],
                "mean_ess": np.mean(self.episode_ess) if self.episode_ess else 0.0,
            }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (placeholder for visualization)."""
        if self.current_step % 100 == 0:
            obs = self._get_observation()
            logger.info(
                f"Step {self.current_step}: roughness={obs[2]:.3f}, "
                f"coverage={obs[3]:.3f}, reward={self.episode_reward:.3f}"
            )

    def close(self) -> None:
        """Clean up resources."""
        pass
