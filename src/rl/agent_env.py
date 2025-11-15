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
from gymnasium import spaces

from src.analysis.roughness import calculate_roughness
from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import Lattice, SpeciesType
from src.rl.action_space import N_ACTIONS, ActionType
from src.rl.particle_agent import create_agents_from_lattice
from src.rl.rate_calculator import ActionRateCalculator

from .energy_calculator import SystemEnergyCalculator

logger = logging.getLogger(__name__)


class AgentBasedTiO2Env(gym.Env):  # type: ignore[misc]
    """
    Agent-based Gymnasium environment for SwarmThinkers training.

    This version is refactored for true scalability, removing the `max_agents`
    constraint. The environment adapts to the lattice size, allowing models
    to be trained on small lattices and run on large ones.

    Observation Space:
        A dictionary containing the observations for all active agents and
        global system features. This is handled by a custom training loop,
        not fed directly into a standard SB3 model.
        - "agent_observations": List of local observations.
        - "global_features": Numpy array of global features.

    Action Space:
        A `MultiDiscrete` space representing `[agent_index, action_index]`.
        The number of agents is dynamic.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lattice_size: tuple[int, int, int] = (8, 8, 5),
        tio2_parameters: TiO2Parameters | None = None,
        temperature: float | None = None,
        temperature_range: tuple[float, float] | None = None,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        use_reweighting: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the scalable agent-based environment.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            tio2_parameters: Dataclass with physical parameters.
            temperature: Fixed temperature (K). Mutually exclusive with temperature_range.
            temperature_range: Temperature range (T_min, T_max) for curriculum.
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            use_reweighting: Use physical rate reweighting (default True).
            seed: Random seed.
        """
        super().__init__()

        # Temperature handling
        if temperature is not None and temperature_range is not None:
            raise ValueError("Specify either temperature or temperature_range, not both")
        if temperature is None and temperature_range is None:
            temperature = 600.0  # Default

        self.lattice_size = lattice_size
        self.tio2_params = tio2_parameters if tio2_parameters else TiO2Parameters()
        self.temperature_fixed = temperature
        self.temperature_range = temperature_range
        self.temperature = temperature if temperature is not None else (temperature_range[0] if temperature_range else 600.0)
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.use_reweighting = use_reweighting

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # The observation space is a dictionary containing observations for all agents
        # and global system features. Since the number of agents is dynamic,
        # we define the agent observation space and will handle the list of
        # observations in the custom training loop.
        self.single_agent_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32
        )
        self.global_feature_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "agent_observations": self.single_agent_observation_space, # Placeholder for shape
            "global_features": self.global_feature_space,
        })

        # The action space is now decoupled from a fixed max_agents value
        # It represents choosing one agent and one action for that agent.
        # The actual size will be set in reset() based on lattice size.
        self.n_possible_agents = self.lattice_size[0] * self.lattice_size[1] * self.lattice_size[2]
        self.action_space = spaces.MultiDiscrete([self.n_possible_agents, N_ACTIONS])


        # Initialize components (will be (re)created in reset())
        self.lattice: Lattice = Lattice(size=self.lattice_size)
        self.rate_calculator: ActionRateCalculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator: SystemEnergyCalculator = SystemEnergyCalculator(params=self.tio2_params)
        self.agents: list = []

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info: dict[str, Any] = {}
        self.prev_omega: float = 0.0  # Track grand potential for reward calculation

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Can contain 'lattice_size' to dynamically change it.

        Returns:
            observation: Initial observation dict
            info: Episode information dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Allow dynamic resizing of the lattice on reset
        if options and "lattice_size" in options:
            self.lattice_size = options["lattice_size"]

        # Sample temperature if using range
        if self.temperature_range is not None:
            self.temperature = np.random.uniform(
                self.temperature_range[0], self.temperature_range[1]
            )
        else:
            self.temperature = self.temperature_fixed if self.temperature_fixed is not None else 600.0

        # Initialize lattice
        self.lattice = Lattice(size=self.lattice_size)

        # Update dynamic action space size
        self.n_possible_agents = self.lattice_size[0] * self.lattice_size[1] * self.lattice_size[2]
        self.action_space = spaces.MultiDiscrete([self.n_possible_agents, N_ACTIONS])

        # Initialize rate and energy calculators with current temperature
        self.rate_calculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator = SystemEnergyCalculator(params=self.tio2_params)


        # Create initial agents (surface sites)
        self._update_agents()

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

        # Calculate initial grand potential for reward computation
        self.prev_omega = self.energy_calculator.calculate_grand_potential(self.lattice)

        observation = self._get_observation()
        info = {"step": 0, "n_agents": len(self.agents)}

        return observation, info

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: A tuple of (agent_index, action_index)

        Returns:
            observation: New observation dict
            reward: Step reward
            terminated: Episode ended (e.g., full coverage)
            truncated: Episode ended (max steps)
            info: Step information
        """
        agent_idx, action_idx = action
        self.step_count += 1

        # Validate action
        if agent_idx >= len(self.agents):
            # Invalid agent index (can happen if agent list changes)
            reward = -1.0  # Penalty for invalid action
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = {"invalid_action": True, "step": self.step_count, "n_agents": len(self.agents)}
            return self._get_observation(), reward, terminated, truncated, info

        # Get the agent and the action to perform
        agent = self.agents[agent_idx]

        # Execute the event on the lattice
        success = self._execute_event(agent, action_idx)

        # Update agents list after lattice modification
        if success:
            self._update_agents()

        # Calculate reward based on change in grand potential
        reward = self._calculate_reward()
        self.total_reward += reward

        # Check termination conditions
        terminated = False  # Define a success condition, e.g., target thickness
        truncated = self.step_count >= self.max_steps

        # Prepare info dict
        info = {
            "step": self.step_count,
            "roughness": self._calculate_roughness(),
            "coverage": self._calculate_coverage(),
            "n_agents": len(self.agents),
            "executed_action": (agent_idx, action_idx),
            "reward": reward,
        }

        # At the end of an episode, populate final stats
        if terminated or truncated:
            self.episode_info.update({
                "episode_length": self.step_count,
                "episode_reward": self.total_reward,
                "final_roughness": info["roughness"],
                "final_coverage": info["coverage"],
                "n_ti": self._count_species(SpeciesType.TI),
                "n_o": self._count_species(SpeciesType.O),
            })
            info.update(self.episode_info)

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def _update_agents(self) -> None:
        """
        Efficiently update the list of active agents.
        Agents are top-most atoms or vacant surface sites.
        """
        self.agents = create_agents_from_lattice(self.lattice)

    def _execute_event(self, agent: Any, action_idx: int) -> bool:
        """
        Execute the selected event on the lattice.

        Args:
            agent: The agent performing the action.
            action_idx: The index of the action to perform.

        Returns:
            True if event executed successfully, False otherwise.
        """
        site = agent.site

        # Adsorption Events
        if action_idx == ActionType.ADSORB_TI or action_idx == ActionType.ADSORB_O:
            species_to_adsorb = SpeciesType.TI if action_idx == ActionType.ADSORB_TI else SpeciesType.O
            # Adsorption happens on vacant sites
            if site.is_occupied():
                return False
            site.species = species_to_adsorb
            return True

        # Desorption Event
        elif action_idx == ActionType.DESORB:
            if not site.is_occupied():
                return False
            site.species = SpeciesType.VACANT
            return True

        # Diffusion Events
        elif action_idx in [
            ActionType.DIFFUSE_X_POS, ActionType.DIFFUSE_X_NEG,
            ActionType.DIFFUSE_Y_POS, ActionType.DIFFUSE_Y_NEG,
            ActionType.DIFFUSE_Z_POS, ActionType.DIFFUSE_Z_NEG,
        ]:
            if not site.is_occupied():
                return False

            x, y, z = site.position
            dx, dy, dz = 0, 0, 0
            if action_idx == ActionType.DIFFUSE_X_POS:
                dx = 1
            elif action_idx == ActionType.DIFFUSE_X_NEG:
                dx = -1
            elif action_idx == ActionType.DIFFUSE_Y_POS:
                dy = 1
            elif action_idx == ActionType.DIFFUSE_Y_NEG:
                dy = -1
            elif action_idx == ActionType.DIFFUSE_Z_POS:
                dz = 1
            else:
                dz = -1  # Z_NEG

            target_pos = (x + dx, y + dy, z + dz)

            # Periodic boundary conditions for x and y
            nx, ny, nz = self.lattice_size
            target_x, target_y = target_pos[0] % nx, target_pos[1] % ny
            target_z = target_pos[2]

            # Check z-boundary
            if not (0 <= target_z < nz):
                return False

            target_site = self.lattice.get_site(target_x, target_y, target_z)
            if target_site is None or target_site.is_occupied():
                return False

            # Execute diffusion
            target_site.species = site.species
            site.species = SpeciesType.VACANT
            return True

        return False

    def _get_observation(self) -> dict[str, Any]:
        """
        Get current observation.

        Returns:
            A dictionary containing a list of agent observations and global features.
            This structure is meant for a custom training loop that can handle
            variable numbers of agents per step.
        """
        # Collect local observations from all active agents
        agent_observations = [agent.observe().to_vector() for agent in self.agents]

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
            "agent_observations": agent_observations,
            "global_features": global_features,
        }

    def _calculate_reward(self) -> float:
        """
        Calculate SwarmThinkers reward for an open system: r_t = -ΔΩ.

        A negative change in grand potential (system becomes more stable)
        results in a positive reward.

        Returns:
            Reward r_t = -ΔΩ (eV)
        """
        # Calculate current grand potential
        current_omega = self.energy_calculator.calculate_grand_potential(self.lattice)

        # Grand potential change
        delta_omega = current_omega - self.prev_omega

        # Reward = -ΔΩ (favor stability)
        reward = -delta_omega

        # Update previous grand potential for next step
        self.prev_omega = current_omega

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
