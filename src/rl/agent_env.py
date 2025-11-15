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
from src.rl.action_space import N_ACTIONS, ActionType, create_action_mask
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
        lattice_size: tuple[int, int, int] = (8, 8, 50),
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
        self.temperature = (
            temperature
            if temperature is not None
            else (temperature_range[0] if temperature_range else 600.0)
        )
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
        self.observation_space = spaces.Dict(
            {
                "agent_observations": self.single_agent_observation_space,  # Placeholder for shape
                "global_features": self.global_feature_space,
            }
        )

        # The action space is now decoupled from a fixed max_agents value.
        # The training loop will select between agent actions and global actions.
        self.action_space = spaces.Discrete(N_ACTIONS)  # For a single agent

        # Initialize components (will be (re)created in reset())
        self.lattice: Lattice = Lattice(size=self.lattice_size)
        self.rate_calculator: ActionRateCalculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator: SystemEnergyCalculator = SystemEnergyCalculator(
            params=self.tio2_params
        )
        self.agents: list = []
        self.site_to_agent_map: dict[int, int] = {}  # Maps site_idx to agent index in self.agents

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info: dict[str, Any] = {}
        self.prev_omega: float = 0.0  # Track grand potential for reward calculation
        self.step_info: list[dict[str, Any]] = []  # Store info for logging

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
            self.temperature = (
                self.temperature_fixed if self.temperature_fixed is not None else 600.0
            )

        # Initialize lattice
        self.lattice = Lattice(size=self.lattice_size)

        # The action space is for a single agent; the training loop handles the rest.
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Initialize rate and energy calculators with current temperature
        self.rate_calculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator = SystemEnergyCalculator(params=self.tio2_params)

        # Create initial agents (surface sites) and build the map
        self._rebuild_agents()

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

    def step(self, action: tuple[int, int] | str) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        The action can be one of two things:
        1. A tuple (agent_idx, action_idx) for an agent-driven event.
        2. A string "DEPOSIT_TI" or "DEPOSIT_O" for a global deposition event.
        """
        self.step_count += 1

        is_deposition_action = isinstance(action, str)

        if is_deposition_action:
            if action == "DEPOSIT_TI":
                success, reason = self._execute_deposition(species=SpeciesType.TI)
            elif action == "DEPOSIT_O":
                success, reason = self._execute_deposition(species=SpeciesType.O)
            else:
                success = False
                reason = f"Unknown deposition action: {action}"
        else:
            agent_idx, action_idx = action

            # Validate agent index
            if agent_idx >= len(self.agents):
                reward = -0.01  # Small penalty for invalid agent
                terminated = False
                truncated = self.step_count >= self.max_steps
                reason = f"Agent index {agent_idx} out of bounds ({len(self.agents)} agents)."
                info = {
                    "failure_reason": reason,
                    "step": self.step_count,
                    "n_agents": len(self.agents),
                }
                return self._get_observation(), reward, terminated, truncated, info

            _agent = self.agents[agent_idx]
            _action_enum = ActionType(action_idx)
            success, reason = self._execute_agent_action(agent_idx, action_idx)

        # Calculate reward based on change in grand potential
        reward = self._calculate_reward()
        if not success:
            # If the action was not successful, the grand potential will not have changed.
            # We apply a small penalty to discourage the agent from choosing invalid actions.
            reward = -0.01
        self.total_reward += reward

        # Check termination conditions
        terminated = False  # Define a success condition, e.g., target thickness
        truncated = self.step_count >= self.max_steps

        # Prepare info dict
        if is_deposition_action:
            executed_action = action  # This will be "DEPOSIT_TI" or "DEPOSIT_O"
        else:
            action_name = str(ActionType(action_idx).name)
            executed_action = (agent_idx, action_name)

        info = {
            "step": self.step_count,
            "roughness": self._calculate_roughness(),
            "coverage": self._calculate_coverage(),
            "n_agents": len(self.agents),
            "executed_action": executed_action,
            "reward": reward,
            "success": success,
            "failure_reason": reason,
        }
        self.step_info.append(info)

        # At the end of an episode, populate final stats
        if terminated or truncated:
            self.episode_info.update(
                {
                    "episode_length": self.step_count,
                    "episode_reward": self.total_reward,
                    "final_roughness": info["roughness"],
                    "final_coverage": info["coverage"],
                    "n_ti": self._count_species(SpeciesType.TI),
                    "n_o": self._count_species(SpeciesType.O),
                }
            )
            info.update(self.episode_info)

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for all valid actions for the current agents.

        Returns:
            A numpy array of shape (n_agents, N_ACTIONS) where True indicates a
            valid action.
        """
        return create_action_mask(self.agents, self.lattice.size, self.lattice)

    def _rebuild_agents(self) -> None:
        """
        Rebuild the full list of active agents and the site-to-agent map.
        Only called at reset, not during step execution.
        """
        self.agents = create_agents_from_lattice(self.lattice)
        self.site_to_agent_map = {agent.site_idx: idx for idx, agent in enumerate(self.agents)}

    def _add_agent(self, site_idx: int) -> None:
        """
        Add a new agent for a given site.
        
        Args:
            site_idx: Index of the site that became an agent.
        """
        from src.rl.particle_agent import ParticleAgent
        
        new_agent = ParticleAgent(site_idx=site_idx, lattice=self.lattice)
        agent_idx = len(self.agents)
        self.agents.append(new_agent)
        self.site_to_agent_map[site_idx] = agent_idx

    def _remove_agent(self, site_idx: int) -> None:
        """
        Remove an agent at a given site.
        
        Args:
            site_idx: Index of the site that is no longer an agent.
        """
        if site_idx not in self.site_to_agent_map:
            return
        
        agent_idx = self.site_to_agent_map[site_idx]
        
        # Remove from agents list (swap with last element for O(1) removal)
        last_agent = self.agents[-1]
        if agent_idx < len(self.agents) - 1:
            self.agents[agent_idx] = last_agent
            self.site_to_agent_map[last_agent.site_idx] = agent_idx
        
        self.agents.pop()
        del self.site_to_agent_map[site_idx]

    def _update_agent_at_site(self, site_idx: int) -> None:
        """
        Update or create/remove agent at a specific site based on current lattice state.
        
        Args:
            site_idx: Index of the site to update.
        """
        site = self.lattice.sites[site_idx]
        should_be_agent = site.is_occupied()
        is_agent = site_idx in self.site_to_agent_map
        
        if should_be_agent and not is_agent:
            self._add_agent(site_idx)
        elif not should_be_agent and is_agent:
            self._remove_agent(site_idx)

    def _update_affected_agents(self, affected_site_indices: list[int]) -> None:
        """
        Update agents for affected sites and their neighbors.
        This is called after deposition, desorption, or diffusion events.
        
        Args:
            affected_site_indices: Indices of sites that changed.
        """
        sites_to_check = set(affected_site_indices)
        
        # Add neighbors of affected sites
        for site_idx in affected_site_indices:
            site = self.lattice.sites[site_idx]
            sites_to_check.update(site.neighbors)
        
        # Update each affected site
        for site_idx in sites_to_check:
            self._update_agent_at_site(site_idx)

    def _execute_deposition(self, species: SpeciesType) -> tuple[bool, str]:
        """
        Executes a global deposition event for a given species.
        Tries random columns until finding an available spot or exhausting all options.
        """
        nx, ny, nz = self.lattice.size

        # Create list of all columns and shuffle them
        all_columns = [(x, y) for x in range(nx) for y in range(ny)]
        np.random.shuffle(all_columns)

        # Try each column until we find one with space
        for x, y in all_columns:
            # Search upward from z=1 (skip z=0, it's the substrate) to find first vacant site with support
            for z in range(1, nz):  # Start from z=1, not z=0
                site_idx = x + y * nx + z * nx * ny
                site = self.lattice.sites[site_idx]

                if site.species == SpeciesType.VACANT:
                    # Check if site below is occupied (provides support)
                    # Support can be SUBSTRATE, TI, or O (anything except VACANT)
                    site_below_idx = x + y * nx + (z - 1) * nx * ny
                    site_below = self.lattice.sites[site_below_idx]

                    if site_below.species != SpeciesType.VACANT:
                        # Found valid deposition site with support (below is SUBSTRATE, TI, or O)
                        success, reason = self.lattice.deposit_atom(site_idx, species)
                        if success:
                            # Update only affected agents incrementally
                            self._update_affected_agents([site_idx])
                        return success, reason
                    # This vacant site doesn't have support, check next z level

        # All columns exhausted - lattice is completely full
        return False, "Deposition failed: all columns are full"

    def _execute_agent_action(self, agent_idx: int, action_idx: int) -> tuple[bool, str]:
        """
        Execute the selected event on the lattice for a specific agent.
        This was previously the _execute_event method.
        """
        agent = self.agents[agent_idx]

        # Diffusion actions
        if action_idx in [
            ActionType.DIFFUSE_X_POS.value,
            ActionType.DIFFUSE_X_NEG.value,
            ActionType.DIFFUSE_Y_POS.value,
            ActionType.DIFFUSE_Y_NEG.value,
            ActionType.DIFFUSE_Z_POS.value,
            ActionType.DIFFUSE_Z_NEG.value,
        ]:
            action_enum = ActionType(action_idx)
            target_site = agent.get_neighbor_site(action_enum, self.lattice.size)

            if target_site is None:
                return False, f"Diffusion failed: invalid move for action {action_enum.name}"

            from_site_idx = agent.site_idx
            success, reason = self.lattice.diffuse_atom(from_site_idx, target_site)
            if success:
                # Update only affected agents incrementally (both source and destination)
                self._update_affected_agents([from_site_idx, target_site])

        elif action_idx == ActionType.DESORB.value:
            # Desorption action
            from_site_idx = agent.site_idx
            success, reason = self.lattice.desorb_atom(from_site_idx)
            if success:
                # Update only affected agents incrementally
                self._update_affected_agents([from_site_idx])
        else:
            return False, f"Unknown action index: {action_idx}"

        return success, reason

    def _get_observation(self) -> dict[str, Any]:
        """
        Get current observation. Optimized to pre-allocate numpy array.

        Returns:
            A dictionary containing a list of agent observations and global features.
            This structure is meant for a custom training loop that can handle
            variable numbers of agents per step.
        """
        num_agents = len(self.agents)

        # Pre-allocate numpy array for agent observations
        # Shape: (num_agents, 58) where 58 is the observation vector size
        agent_observations = np.zeros((num_agents, 58), dtype=np.float32)

        # Populate the array
        for i, agent in enumerate(self.agents):
            agent_observations[i] = agent.observe().to_vector()

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
            height_profile = self.lattice.get_height_profile()
            roughness = calculate_roughness(height_profile)
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
