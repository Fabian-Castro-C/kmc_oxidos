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
from src.training.energy_calculator import SystemEnergyCalculator

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
        temperature: float | None = None,
        temperature_range: tuple[float, float] | None = None,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        max_agents: int = 128,
        use_reweighting: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize agent-based environment.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            temperature: Fixed temperature (K). Mutually exclusive with temperature_range.
            temperature_range: Temperature range (T_min, T_max) for curriculum.
                Sample random temperature each episode.
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            max_agents: Maximum number of agents (for padding).
            use_reweighting: Use physical rate reweighting (default True).
            seed: Random seed.

        Note:
            Reward follows SwarmThinkers paper: r_t = -ΔE (energy-based, no weights needed).
        """
        super().__init__()

        # Temperature handling
        if temperature is not None and temperature_range is not None:
            raise ValueError("Specify either temperature or temperature_range, not both")
        if temperature is None and temperature_range is None:
            temperature = 600.0  # Default

        self.lattice_size = lattice_size
        self.temperature_fixed = temperature
        self.temperature_range = temperature_range
        self.temperature = temperature if temperature is not None else temperature_range[0]
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.max_agents = max_agents
        self.use_reweighting = use_reweighting

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
        self.energy_calculator: SystemEnergyCalculator | None = None

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info: dict[str, Any] = {}
        self.prev_energy: float = 0.0  # Track energy for reward calculation

        # Metrics tracking
        self.prev_roughness = 0.0
        self.prev_coverage = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
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

        # Sample temperature if using range
        if self.temperature_range is not None:
            self.temperature = np.random.uniform(
                self.temperature_range[0], self.temperature_range[1]
            )
        else:
            self.temperature = self.temperature_fixed

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

        # Initialize energy calculator for reward
        self.energy_calculator = SystemEnergyCalculator()

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

        # Calculate initial energy for reward computation
        self.prev_energy = self.energy_calculator.calculate_system_energy(self.lattice)

        # Initial metrics
        self.prev_roughness = self._calculate_roughness()
        self.prev_coverage = self._calculate_coverage()

        observation = self._get_observation()
        info = {"step": 0}

        return observation, info

    def step(self, action: int) -> tuple[dict[str, npt.NDArray], float, bool, bool, dict[str, Any]]:
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

        # Execute event using coordinator
        if self.use_reweighting:
            selected_event, ess = self.coordinator.select_event_with_reweighting(
                self.agents, self.lattice, self.rate_calculator, temperature=1.0
            )
        else:
            selected_event = self.coordinator.select_event(self.agents, temperature=1.0)
            ess = None

        # Execute the selected event on lattice
        success = self._execute_event(selected_event)

        # Update agents list after lattice modification
        if success:
            self.agents = create_agents_from_lattice(self.lattice)

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

    def _execute_event(self, selected_event: Any) -> bool:
        """
        Execute the selected event on the lattice.

        Args:
            selected_event: SelectedEvent from coordinator

        Returns:
            True if event executed successfully, False otherwise
        """
        from src.rl.particle_agent import ActionType

        if self.lattice is None or len(self.agents) == 0:
            return False

        agent_idx = selected_event.agent_idx
        action = selected_event.action

        # Validate agent index
        if agent_idx >= len(self.agents):
            return False

        agent = self.agents[agent_idx]
        site = agent.site

        # Execute action based on type
        if action == ActionType.ADSORB_TI:
            # Adsorb Ti atom - find vacant site above current position
            x, y, z = site.position
            # Try to adsorb on top (z+1)
            if z + 1 < self.lattice_size[2]:
                target_site = self.lattice.get_site(x, y, z + 1)
                if target_site is not None and not target_site.is_occupied():
                    target_site.species = SpeciesType.TI
                    return True
            return False

        elif action == ActionType.ADSORB_O:
            # Adsorb O atom - find vacant site above current position
            x, y, z = site.position
            # Try to adsorb on top (z+1)
            if z + 1 < self.lattice_size[2]:
                target_site = self.lattice.get_site(x, y, z + 1)
                if target_site is not None and not target_site.is_occupied():
                    target_site.species = SpeciesType.O
                    return True
            return False

        elif action == ActionType.DESORB:
            # Desorb atom
            if site.is_occupied():
                site.species = SpeciesType.VACANT
                return True

        elif action in [
            ActionType.DIFFUSE_X_POS,
            ActionType.DIFFUSE_X_NEG,
            ActionType.DIFFUSE_Y_POS,
            ActionType.DIFFUSE_Y_NEG,
            ActionType.DIFFUSE_Z_POS,
            ActionType.DIFFUSE_Z_NEG,
        ]:
            # Diffusion event
            if not site.is_occupied():
                return False

            # Get target position
            x, y, z = site.position
            if action == ActionType.DIFFUSE_X_POS:
                target_pos = (x + 1, y, z)
            elif action == ActionType.DIFFUSE_X_NEG:
                target_pos = (x - 1, y, z)
            elif action == ActionType.DIFFUSE_Y_POS:
                target_pos = (x, y + 1, z)
            elif action == ActionType.DIFFUSE_Y_NEG:
                target_pos = (x, y - 1, z)
            elif action == ActionType.DIFFUSE_Z_POS:
                target_pos = (x, y, z + 1)
            else:  # DIFFUSE_Z_NEG
                target_pos = (x, y, z - 1)

            # Check if target is within bounds
            nx, ny, nz = self.lattice_size
            if not (
                0 <= target_pos[0] < nx and 0 <= target_pos[1] < ny and 0 <= target_pos[2] < nz
            ):
                return False

            # Get target site
            target_site = self.lattice.get_site(target_pos[0], target_pos[1], target_pos[2])
            if target_site is None or target_site.is_occupied():
                return False

            # Execute diffusion
            target_site.species = site.species
            site.species = SpeciesType.VACANT
            return True

        elif action == ActionType.REACT_TIO2:
            # Simplified reaction (not fully implemented)
            # In full implementation, would check for O neighbors and form TiO2
            return False

        return False

    def _get_observation(self) -> dict[str, npt.NDArray]:
        """
        Get current observation with padding.

        Returns:
            Observation dict with agents, mask, and global features
        """
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

    def _calculate_reward(self, ess: float | None = None) -> float:  # noqa: ARG002
        """
        Calculate SwarmThinkers reward for an open system: r_t = -ΔΩ.

        A negative change in grand potential (system becomes more stable)
        results in a positive reward.

        Args:
            ess: Effective sample size (unused, kept for compatibility)

        Returns:
            Reward r_t = -ΔΩ (eV)
        """
        if self.lattice is None or self.energy_calculator is None:
            return 0.0

        # Calculate current grand potential
        current_omega = self.energy_calculator.calculate_grand_potential(self.lattice)

        # Grand potential change
        delta_omega = current_omega - self.prev_energy  # prev_energy is now prev_omega

        # Reward = -ΔΩ (favor stability)
        reward = -delta_omega

        # Update previous grand potential for next step
        self.prev_energy = current_omega

        return float(reward)

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
        n_atoms = self._count_species(SpeciesType.TI) + self._count_species(SpeciesType.O)
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
        print(f"Ti: {self._count_species(SpeciesType.TI)}, O: {self._count_species(SpeciesType.O)}")

    def close(self) -> None:
        """Clean up resources."""
        pass
