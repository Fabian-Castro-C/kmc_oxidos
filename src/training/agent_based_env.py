"""
Agent-based Gymnasium environment for SwarmThinkers.

This environment implements the faithful SwarmThinkers architecture:
- Per-particle agents with local observations
- Shared policy network
- Global softmax event selection
- Physical rate reweighting

Observation: Variable [N_agents, 58] with padding
Action: Single integer selecting agent×action pair
Reward: Morphology-based (roughness, coverage, Ti:O ratio)
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from src.analysis.roughness import calculate_roughness
from src.kmc.lattice import Lattice, SpeciesType
from src.rl.action_space import N_ACTIONS
from src.rl.particle_agent import create_agents_from_lattice
from src.rl.rate_calculator import ActionRateCalculator
from src.rl.shared_policy import SharedPolicyNetwork
from src.rl.swarm_coordinator import SwarmCoordinator

logger = logging.getLogger(__name__)


class AgentBasedTiO2Env(gym.Env):  # type: ignore[misc]
    """
    Agent-based Gymnasium environment for SwarmThinkers training.

    Implements faithful paper architecture:
    - Particle agents observe local neighborhoods (58-dim)
    - Shared policy network outputs action logits
    - Global softmax selects ONE event from ALL agent×action pairs
    - Physical rates reweight policy probabilities

    Observation Space:
        Dict:
        - "agents": Box(shape=(max_agents, 58), dtype=float32)
          Local observations for each agent (padded)
        - "mask": Box(shape=(max_agents,), dtype=bool)
          Valid agent mask (True = real agent, False = padding)
        - "global": Box(shape=(7,), dtype=float32)
          Global state features [height_mean, height_std, roughness,
                                 coverage, n_Ti, n_O, n_vacant]

    Action Space:
        Discrete(max_agents * N_ACTIONS):
        Flattened index into [agent_idx, action_idx] pairs
        Invalid actions are masked during sampling

    Reward:
        Multi-objective:
        - Smooth surface: -roughness
        - Growth: +coverage
        - Stoichiometry: penalty for Ti:O ratio != 1:2
        - ESS penalty: encourage diverse sampling
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lattice_size: tuple[int, int, int] = (8, 8, 5),
        temperature: float = 600.0,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        max_agents: int = 128,
        use_reweighting: bool = True,
        reward_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize agent-based environment.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            temperature: Temperature (K).
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            max_agents: Maximum number of agents (for padding).
            use_reweighting: Use physical rate reweighting (default True).
            reward_weights: Weights for reward components:
                - roughness_weight: Roughness penalty (default -1.0)
                - coverage_weight: Coverage reward (default +0.5)
                - stoichiometry_weight: Ti:O ratio penalty (default -0.3)
                - ess_weight: ESS penalty (default -0.1)
            seed: Random seed.
        """
        super().__init__()

        self.lattice_size = lattice_size
        self.temperature = temperature
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.max_agents = max_agents
        self.use_reweighting = use_reweighting

        # Reward weights
        if reward_weights is None:
            reward_weights = {
                "roughness_weight": -1.0,
                "coverage_weight": 0.5,
                "stoichiometry_weight": -0.3,
                "ess_weight": -0.1,
            }
        self.reward_weights = reward_weights

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # Define observation space
        # agents: [max_agents, 58] local observations (padded)
        # mask: [max_agents] valid agent mask
        # global: [7] global features
        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_agents, 58),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(max_agents,),
                    dtype=np.float32,
                ),
                "global": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
            }
        )

        # Define action space
        # Flattened [agent_idx, action_idx] pairs
        self.action_space = spaces.Discrete(max_agents * N_ACTIONS)

        # Initialize components (will be created in reset())
        self.lattice: Lattice | None = None
        self.rate_calculator: ActionRateCalculator | None = None
        self.policy: SharedPolicyNetwork | None = None
        self.coordinator: SwarmCoordinator | None = None
        self.agents: list = []

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info: dict[str, Any] = {}

        # Metrics tracking
        self.prev_roughness = 0.0
        self.prev_coverage = 0.0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            observation: Initial observation dict
            info: Episode information dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize lattice
        self.lattice = Lattice(size=self.lattice_size)

        # Initialize rate calculator
        self.rate_calculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )

        # Initialize policy and coordinator
        self.policy = SharedPolicyNetwork(obs_dim=58, action_dim=N_ACTIONS)
        self.coordinator = SwarmCoordinator(self.policy)

        # Create initial agents (all sites)
        self.agents = create_agents_from_lattice(self.lattice)

        # Reset episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info = {
            "episode_length": 0,
            "episode_reward": 0.0,
            "final_roughness": 0.0,
            "final_coverage": 0.0,
            "n_ti": 0,
            "n_o": 0,
        }

        # Initial metrics
        self.prev_roughness = self._calculate_roughness()
        self.prev_coverage = self._calculate_coverage()

        observation = self._get_observation()
        info = {"step": 0}

        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, npt.NDArray], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Flattened [agent_idx, action_idx] index

        Returns:
            observation: New observation dict
            reward: Step reward
            terminated: Episode ended (success)
            truncated: Episode ended (max steps)
            info: Step information
        """
        # Decode action
        agent_idx = action // N_ACTIONS
        action_idx = action % N_ACTIONS

        # Increment step count first
        self.step_count += 1

        # Validate action
        if agent_idx >= len(self.agents):
            # Invalid agent index (padding)
            reward = -1.0  # Penalty for invalid action
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = {"invalid_action": True, "step": self.step_count}
            return self._get_observation(), reward, terminated, truncated, info

        # Execute event (simplified - just for structure)
        # In real implementation, would execute the actual lattice update
        # For now, use coordinator to select event (ignoring action input)
        if self.use_reweighting:
            selected_event, ess = self.coordinator.select_event_with_reweighting(
                self.agents, self.lattice, self.rate_calculator, temperature=1.0
            )
        else:
            selected_event = self.coordinator.select_event(
                self.agents, temperature=1.0
            )
            ess = None

        # TODO: Execute the selected event on lattice
        # This would involve:
        # 1. Get agent and action from selected_event
        # 2. Execute action on lattice (diffusion, adsorption, etc.)
        # 3. Update agents list
        # For now, just continue (step_count already incremented above)

        # Calculate reward
        reward = self._calculate_reward(ess)
        self.total_reward += reward

        # Check termination
        terminated = False  # Success condition (e.g., target thickness)
        truncated = self.step_count >= self.max_steps

        # Update metrics
        self.prev_roughness = self._calculate_roughness()
        self.prev_coverage = self._calculate_coverage()

        # Info dict
        info = {
            "step": self.step_count,
            "roughness": self.prev_roughness,
            "coverage": self.prev_coverage,
            "n_agents": len(self.agents),
        }

        if ess is not None:
            info["ess"] = ess

        # Episode end
        if terminated or truncated:
            self.episode_info["episode_length"] = self.step_count
            self.episode_info["episode_reward"] = self.total_reward
            self.episode_info["final_roughness"] = self.prev_roughness
            self.episode_info["final_coverage"] = self.prev_coverage
            self.episode_info["n_ti"] = self._count_species(SpeciesType.TI)
            self.episode_info["n_o"] = self._count_species(SpeciesType.O)
            info.update(self.episode_info)

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> dict[str, npt.NDArray]:
        """
        Get current observation with padding.

        Returns:
            Observation dict with agents, mask, and global features
        """
        n_agents = len(self.agents)

        # Collect agent observations
        agent_obs = np.zeros((self.max_agents, 58), dtype=np.float32)
        mask = np.zeros(self.max_agents, dtype=np.float32)

        for i, agent in enumerate(self.agents[: self.max_agents]):
            obs = agent.observe()
            agent_obs[i] = obs.to_vector()
            mask[i] = 1.0

        # Global features
        global_features = np.array(
            [
                self._calculate_mean_height(),
                self._calculate_height_std(),
                self._calculate_roughness(),
                self._calculate_coverage(),
                self._count_species(SpeciesType.TI),
                self._count_species(SpeciesType.O),
                self._count_species(SpeciesType.VACANT),
            ],
            dtype=np.float32,
        )

        return {
            "agents": agent_obs,
            "mask": mask,
            "global": global_features,
        }

    def _calculate_reward(self, ess: float | None = None) -> float:
        """
        Calculate multi-objective reward.

        Args:
            ess: Effective sample size (optional)

        Returns:
            Scalar reward
        """
        roughness = self._calculate_roughness()
        coverage = self._calculate_coverage()

        # Roughness penalty (smoother is better)
        roughness_reward = self.reward_weights["roughness_weight"] * roughness

        # Coverage reward (more growth is better)
        coverage_reward = self.reward_weights["coverage_weight"] * coverage

        # Stoichiometry penalty
        n_ti = self._count_species(SpeciesType.TI)
        n_o = self._count_species(SpeciesType.O)
        ideal_ratio = 0.5  # Ti:O = 1:2
        actual_ratio = n_ti / (n_ti + n_o + 1e-6)
        stoich_penalty = (
            self.reward_weights["stoichiometry_weight"]
            * abs(actual_ratio - ideal_ratio)
        )

        # ESS penalty (encourage diverse sampling)
        ess_penalty = 0.0
        if ess is not None and self.use_reweighting:
            # Low ESS means concentrated weights (bad)
            # Normalize by number of valid actions
            n_valid = sum(len(a.get_valid_actions()) for a in self.agents)
            ess_normalized = ess / (n_valid + 1e-6)
            ess_penalty = self.reward_weights["ess_weight"] * (1.0 - ess_normalized)

        total_reward = (
            roughness_reward + coverage_reward + stoich_penalty + ess_penalty
        )

        return float(total_reward)

    def _calculate_roughness(self) -> float:
        """Calculate surface roughness."""
        if self.lattice is None:
            return 0.0
        try:
            roughness = calculate_roughness(self.lattice)
            return float(roughness)
        except Exception:
            return 0.0

    def _calculate_coverage(self) -> float:
        """Calculate surface coverage fraction."""
        if self.lattice is None:
            return 0.0
        n_atoms = self._count_species(SpeciesType.TI) + self._count_species(
            SpeciesType.O
        )
        nx, ny, _ = self.lattice_size
        total_surface_sites = nx * ny
        return n_atoms / total_surface_sites

    def _calculate_mean_height(self) -> float:
        """Calculate mean surface height."""
        if self.lattice is None:
            return 0.0
        heights = [site.position[2] for site in self.lattice.sites if site.is_occupied()]
        return float(np.mean(heights)) if heights else 0.0

    def _calculate_height_std(self) -> float:
        """Calculate height standard deviation."""
        if self.lattice is None:
            return 0.0
        heights = [site.position[2] for site in self.lattice.sites if site.is_occupied()]
        return float(np.std(heights)) if heights else 0.0

    def _count_species(self, species: SpeciesType) -> int:
        """Count atoms of given species."""
        if self.lattice is None:
            return 0
        return sum(1 for site in self.lattice.sites if site.species == species)

    def render(self) -> None:
        """Render environment (text output)."""
        if self.lattice is None:
            print("Environment not initialized")
            return

        print(f"\nStep: {self.step_count}/{self.max_steps}")
        print(f"Agents: {len(self.agents)}")
        print(f"Roughness: {self._calculate_roughness():.3f} nm")
        print(f"Coverage: {self._calculate_coverage():.3f}")
        print(
            f"Ti: {self._count_species(SpeciesType.TI)}, "
            f"O: {self._count_species(SpeciesType.O)}"
        )

    def close(self) -> None:
        """Clean up resources."""
        pass
